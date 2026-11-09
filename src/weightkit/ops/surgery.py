"""Structural surgery: layer stacking/removal, structured pruning, KV pooling,
vocabulary resize, tie/untie, noise, dtype cast."""
from __future__ import annotations

from typing import List, Optional

import torch

from ..arch import parse_layers
from ..base import Context, Op, Selection, register


def _restack(ctx: Context, indices: List[int]) -> None:
    a, s = ctx.arch, ctx.state
    rx = a.layer_regex()
    old = s.snapshot()
    by_layer = {}
    for n in s.names():
        m = rx.match(n)
        if m:
            by_layer.setdefault(int(m.group(1)), []).append(n)
    bad = [i for i in indices if i not in by_layer]
    if bad:
        raise ValueError(f"layer index/indices {bad} do not exist (model has {len(by_layer)} layers)")
    for names in by_layer.values():
        for n in names:
            del s[n]
    for j, i in enumerate(indices):
        po, pn = a.layer_prefix.format(i=i), a.layer_prefix.format(i=j)
        for n in by_layer[i]:
            s[pn + n[len(po):]] = old[n]
    s.config["num_hidden_layers"] = len(indices)
    lt = s.config.get("layer_types")
    if isinstance(lt, list) and len(lt) == len(by_layer):
        s.config["layer_types"] = [lt[i] for i in indices]


@register("layer_stack", "surgery")
class LayerStack(Op):
    """Rebuild the layer stack from slices of the current one ('frankenmerge' /
    depth up-scaling / layer removal).  ``ranges``: [[start, end), ...]
    (end-exclusive, may overlap or repeat), or ``indices``: explicit list."""

    def __init__(self, ranges: Optional[list] = None, indices: Optional[list] = None):
        if (ranges is None) == (indices is None):
            raise ValueError("give exactly one of `ranges` or `indices`")
        self.indices = list(indices) if indices is not None else [i for a, b in ranges for i in range(a, b)]

    def run(self, ctx: Context) -> None:
        _restack(ctx, self.indices)


@register("drop_layers", "surgery")
class DropLayers(Op):
    """Remove decoder layers, e.g. ``layers: "20-24"`` (inclusive; negatives allowed)."""

    def __init__(self, layers):
        self.layers = layers

    def run(self, ctx: Context) -> None:
        n = ctx.arch.n_layers(ctx.state)
        drop = set(parse_layers(self.layers, n))
        _restack(ctx, [i for i in range(n) if i not in drop])


@register("prune_mlp", "surgery")
class PruneMLP(Op):
    """Structured pruning: physically remove the least important MLP neurons
    (importance = |gate row| * |up row| * |down col|), same count in every
    layer, and update ``intermediate_size``.  Give ``keep_ratio`` or ``keep``."""

    def __init__(self, keep_ratio: Optional[float] = None, keep: Optional[int] = None):
        if (keep_ratio is None) == (keep is None):
            raise ValueError("give exactly one of keep_ratio / keep")
        self.keep_ratio, self.keep = keep_ratio, keep

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        ids = a.layer_ids(s)
        n = s[a.layer_name(ids[0], "up")].shape[0]
        k = self.keep if self.keep is not None else max(1, int(round(n * self.keep_ratio)))
        for i in ids:
            g, u, d = (ctx.get(a.layer_name(i, r)) for r in ("gate", "up", "down"))
            score = g.norm(dim=1) * u.norm(dim=1) * d.norm(dim=0)
            idx = score.topk(k).indices.sort().values.cpu()
            for role in ("gate", "up"):
                ctx.write(a.layer_name(i, role), s[a.layer_name(i, role)][idx])
                bn = a.layer_name(i, role, "bias")
                if bn in s:
                    ctx.write(bn, s[bn][idx])
            ctx.write(a.layer_name(i, "down"), s[a.layer_name(i, "down")][:, idx])
        s.config["intermediate_size"] = k


@register("prune_heads", "surgery")
class PruneHeads(Op):
    """Structured pruning: remove whole KV groups (a K/V head with its q heads)
    with the smallest |o_proj cols| * |v rows|; same count in every layer.
    ``keep``: number of KV groups to keep."""

    def __init__(self, keep: int):
        self.keep = keep

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        D = a.dims(s)
        kv, rep, hd = D["kv_heads"], D["rep"], D["head_dim"]
        if not 1 <= self.keep <= kv:
            raise ValueError(f"keep must be in [1, {kv}]")
        for i in a.layer_ids(s):
            qn, kn, vn, on = (a.layer_name(i, r) for r in "qkvo")
            O = s[on]
            V = s[vn]
            O4 = O.float().reshape(O.shape[0], kv, rep, hd)
            score = O4.pow(2).sum(dim=(0, 2, 3)).sqrt() * V.float().reshape(kv, -1).norm(dim=1)
            idx = score.topk(self.keep).indices.sort().values
            q = s[qn]
            s[qn] = q.reshape(kv, rep, hd, -1)[idx].reshape(self.keep * rep * hd, -1)
            for n in (kn, vn):
                W = s[n]
                s[n] = W.reshape(kv, hd, -1)[idx].reshape(self.keep * hd, -1)
            s[on] = O.reshape(O.shape[0], kv, rep, hd)[:, idx].reshape(O.shape[0], -1)
            for n, shape in ((qn, (kv, rep, hd)), (kn, (kv, hd)), (vn, (kv, hd))):
                bn = n[: -len("weight")] + "bias"
                if bn in s:
                    s[bn] = s[bn].reshape(shape)[idx].reshape(-1)
        s.config["num_key_value_heads"] = self.keep
        s.config["num_attention_heads"] = self.keep * rep
        s.config["head_dim"] = hd


@register("kv_pool", "surgery")
class KVPool(Op):
    """Convert MHA/GQA to GQA with fewer KV heads by mean-pooling consecutive
    K/V heads (Ainslie et al. 2023).  Approximate; usually needs brief
    fine-tuning.  ``kv_heads`` must divide the current number of KV heads."""

    def __init__(self, kv_heads: int):
        self.kv_heads = kv_heads

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        D = a.dims(s)
        kv, hd = D["kv_heads"], D["head_dim"]
        if kv % self.kv_heads:
            raise ValueError(f"{self.kv_heads} must divide current kv heads {kv}")
        g = kv // self.kv_heads
        for i in a.layer_ids(s):
            for role in ("k", "v"):
                n = a.layer_name(i, role)
                W = ctx.get(n)
                ctx.write(n, W.reshape(self.kv_heads, g, hd, -1).mean(1).reshape(self.kv_heads * hd, -1))
                bn = a.layer_name(i, role, "bias")
                if bn in s:
                    ctx.write(bn, ctx.get(bn).reshape(self.kv_heads, g, hd).mean(1).reshape(-1))
        s.config["num_key_value_heads"] = self.kv_heads
        s.config["head_dim"] = hd


@register("untie_embeddings", "surgery")
class UntieEmbeddings(Op):
    """Give the model its own lm_head (copy of the embedding)."""

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        if a.lm_head not in s:
            s[a.lm_head] = s[a.embed].clone()
        s.config["tie_word_embeddings"] = False


@register("tie_embeddings", "surgery")
class TieEmbeddings(Op):
    """Drop lm_head and share the embedding matrix."""

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        if a.lm_head in s:
            if not torch.equal(s[a.lm_head], s[a.embed]):
                ctx.log("tie_embeddings: lm_head differs from embedding; discarding lm_head")
            del s[a.lm_head]
        s.config["tie_word_embeddings"] = True


@register("resize_vocab", "surgery")
class ResizeVocab(Op):
    """Truncate or extend the vocabulary to ``size`` rows.  New rows are the
    mean embedding (+ optional gaussian ``noise`` x std)."""

    def __init__(self, size: int, noise: float = 0.0):
        self.size, self.noise = size, noise

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        for n in (a.embed, a.lm_head):
            if n not in s:
                continue
            W = ctx.get(n)
            if self.size <= W.shape[0]:
                new = W[: self.size]
            else:
                extra = self.size - W.shape[0]
                rows = W.mean(0, keepdim=True).repeat(extra, 1)
                if self.noise:
                    rows = rows + self.noise * W.std() * torch.randn(rows.shape, generator=ctx.generator()).to(W)
                new = torch.cat([W, rows], 0)
            ctx.write(n, new)
        s.config["vocab_size"] = self.size


@register("perturb", "surgery")
class Perturb(Op):
    """Add gaussian noise with std = ``std`` x std(W) (robustness probing,
    re-initialisation, breaking symmetry)."""

    def __init__(self, std: float = 0.01, **sel):
        self.std = std
        self.sel = Selection.from_kwargs(sel)

    def run(self, ctx: Context) -> None:
        for n in ctx.select(self.sel):
            W = ctx.get(n)
            noise = torch.randn(W.shape, generator=ctx.generator()).to(W)
            ctx.write(n, W + self.std * W.std() * noise)


@register("cast", "surgery")
class Cast(Op):
    """Convert every floating-point tensor to ``dtype`` (float32/float16/bfloat16)."""

    def __init__(self, dtype: str = "bfloat16"):
        self.dtype = getattr(torch, dtype)

    def run(self, ctx: Context) -> None:
        for n in ctx.state.names():
            t = ctx.state[n]
            if t.is_floating_point() and t.dtype != self.dtype:
                ctx.state[n] = t.to(self.dtype)
        ctx.state.config["torch_dtype"] = str(self.dtype).replace("torch.", "")
