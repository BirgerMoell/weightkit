"""Function-preserving weight transforms ("computational invariances").

Every op here leaves the network's input->output function unchanged (up to
floating point) while changing the weights.  They are useful by themselves
(outlier suppression before quantization, canonical forms) and as building
blocks (permutation alignment before merging).
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch

from ..arch import ArchSpec
from ..base import Context, Op, Selection, register
from ..store import TensorStore


# ------------------------------------------------------------------ norm fold
def _fold_one(ctx: Context, norm_name: str, followers) -> None:
    off = ctx.arch.norm_offset
    g = ctx.get(norm_name) + off
    for wn in followers:
        ctx.write(wn, ctx.get(wn) * g[None, :])
    ctx.write(norm_name, torch.full_like(g, 1.0 - off))


@register("fold_norm", "transform")
class FoldNorm(Op):
    """Fold every RMSNorm scale into the linear layers that read its output, so
    all norms become parameter-free.  Prerequisite for rotations; exact."""

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        for i in a.layer_ids(s):
            for norm_role, followers in (("attn_norm", ("q", "k", "v")), ("mlp_norm", ("gate", "up"))):
                nn_ = a.layer_name(i, norm_role)
                if nn_ in s:
                    _fold_one(ctx, nn_, [a.layer_name(i, r) for r in followers if a.layer_name(i, r) in s])
        if a.final_norm in s:
            if a.lm_head in s:
                _fold_one(ctx, a.final_norm, [a.lm_head])
            elif not torch.allclose(ctx.get(a.final_norm) + a.norm_offset, torch.ones(1), atol=1e-6):
                raise ValueError("tied embeddings: cannot fold the final norm; run untie_embeddings first")


def _norms_are_identity(ctx: Context) -> bool:
    a = ctx.arch
    for n in ctx.state.names():
        if n.endswith("layernorm.weight") or n == a.final_norm:
            if not torch.allclose(ctx.get(n) + a.norm_offset, torch.ones(1), atol=1e-6):
                return False
    return True


# ------------------------------------------------------------------- rotation
def hadamard(n: int, gen: torch.Generator) -> torch.Tensor:
    """Randomised Hadamard (random signs x Sylvester) if n is a power of two,
    else a random orthogonal matrix."""
    if n & (n - 1) == 0:
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        signs = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).float()
        return signs[:, None] * H / math.sqrt(n)
    return random_orthogonal(n, gen)


def random_orthogonal(n: int, gen: torch.Generator) -> torch.Tensor:
    Q, R = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return Q * torch.sign(torch.diagonal(R))[None, :]


@register("rotate", "transform")
class Rotate(Op):
    """Rotate the residual stream by an orthogonal matrix R (QuaRot / SpinQuant
    style).  Embedding, o_proj, down_proj write into the stream (rows rotate);
    q/k/v/gate/up/lm_head read from it (columns counter-rotate).  The function
    is unchanged but activation/weight outliers are spread out.

    kind: hadamard (default) | random | file (``matrix``: .pt with an orthogonal [d,d])
    fold: fold norm scales first (required; set false only if already folded).
    """

    def __init__(self, kind: str = "hadamard", matrix: Optional[str] = None, fold: bool = True):
        if kind not in ("hadamard", "random", "file"):
            raise ValueError("kind must be hadamard, random or file")
        self.kind, self.matrix, self.fold = kind, matrix, fold

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        if self.fold:
            FoldNorm().run(ctx)
        elif not _norms_are_identity(ctx):
            raise ValueError("norm scales are not folded; use fold: true")
        d = s[a.embed].shape[1]
        if self.kind == "file":
            R = torch.load(self.matrix, map_location="cpu").float()
        elif self.kind == "random":
            R = random_orthogonal(d, ctx.generator())
        else:
            R = hadamard(d, ctx.generator())
        R = R.to(ctx.device, ctx.dtype)
        if R.shape != (d, d) or not torch.allclose(R @ R.T, torch.eye(d, device=R.device, dtype=R.dtype), atol=1e-4):
            raise ValueError("rotation matrix must be orthogonal [d, d]")
        ctx.write(a.embed, ctx.get(a.embed) @ R)
        if a.lm_head in s:
            ctx.write(a.lm_head, ctx.get(a.lm_head) @ R)
        for i in a.layer_ids(s):
            for role in ("q", "k", "v", "gate", "up"):
                n = a.layer_name(i, role)
                if n in s:
                    ctx.write(n, ctx.get(n) @ R)
            for role in ("o", "down"):
                n = a.layer_name(i, role)
                if n in s:
                    ctx.write(n, R.T @ ctx.get(n))
                bn = a.layer_name(i, role, "bias")
                if bn in s:
                    ctx.write(bn, ctx.get(bn) @ R)


# --------------------------------------------------------------- permutations
def permute_mlp(store: TensorStore, arch: ArchSpec, layer: int, perm: torch.Tensor) -> None:
    """Reorder the intermediate neurons of one MLP (exact)."""
    for role in ("gate", "up"):
        n = arch.layer_name(layer, role)
        store[n] = store[n][perm]
        bn = arch.layer_name(layer, role, "bias")
        if bn in store:
            store[bn] = store[bn][perm]
    dn = arch.layer_name(layer, "down")
    store[dn] = store[dn][:, perm]


def permute_heads(
    store: TensorStore, arch: ArchSpec, layer: int, perm_kv: torch.Tensor, perm_rep: torch.Tensor
) -> None:
    """Reorder KV groups (perm_kv) and q heads inside each group (perm_rep)."""
    D = arch.dims(store)
    kv, rep, hd = D["kv_heads"], D["rep"], D["head_dim"]
    q, k, v, o = (arch.layer_name(layer, r) for r in "qkvo")
    W = store[q]
    store[q] = W.reshape(kv, rep, hd, -1)[perm_kv][:, perm_rep].reshape(W.shape)
    for n in (k, v):
        W = store[n]
        store[n] = W.reshape(kv, hd, -1)[perm_kv].reshape(W.shape)
    W = store[o]
    store[o] = W.reshape(W.shape[0], kv, rep, hd)[:, perm_kv][:, :, perm_rep].reshape(W.shape)
    for role, n, shape in (("q", q, (kv, rep, hd)), ("k", k, (kv, hd)), ("v", v, (kv, hd))):
        bn = n[: -len("weight")] + "bias"
        if bn in store:
            b = store[bn].reshape(shape)
            b = b[perm_kv][:, perm_rep] if role == "q" else b[perm_kv]
            store[bn] = b.reshape(-1)


@register("permute", "transform")
class Permute(Op):
    """Randomly permute MLP neurons and/or attention heads (exact).  Mostly for
    testing alignment/merging pipelines or de-correlating checkpoints."""

    def __init__(self, mlp: bool = True, heads: bool = True, layers=None):
        self.mlp, self.heads, self.layers = mlp, heads, layers

    def run(self, ctx: Context) -> None:
        from ..arch import parse_layers

        a, s = ctx.arch, ctx.state
        gen = ctx.generator()
        for i in parse_layers(self.layers, a.n_layers(s)):
            if self.mlp:
                n = s[a.layer_name(i, "up")].shape[0]
                permute_mlp(s, a, i, torch.randperm(n, generator=gen))
            if self.heads:
                D = a.dims(s)
                permute_heads(s, a, i, torch.randperm(D["kv_heads"], generator=gen), torch.randperm(D["rep"], generator=gen))


# ------------------------------------------------------------------- rescale
@register("rescale", "transform")
class Rescale(Op):
    """Per-channel rescaling across a linear pair (exact): MLP up/down
    (up *= c, down /= c) and attention v/o (v *= c, o /= c).

    mode: equalize  balance weight norms of each pair (cross-layer equalization;
                    reduces per-channel range mismatch before quantization)
          random    log-uniform random c in [e^-spread, e^spread] (testing)
    """

    def __init__(self, mode: str = "equalize", target=("mlp", "vo"), spread: float = 1.0, layers=None):
        if mode not in ("equalize", "random"):
            raise ValueError("mode must be equalize or random")
        self.mode, self.target, self.spread, self.layers = mode, tuple(target), spread, layers

    def _c(self, ctx, a_norm, b_norm):
        if self.mode == "random":
            u = torch.rand(a_norm.shape, generator=ctx.generator()) * 2 - 1
            return torch.exp(u * self.spread).to(ctx.device, a_norm.dtype)
        return (b_norm.clamp_min(1e-8) / a_norm.clamp_min(1e-8)).sqrt()

    def run(self, ctx: Context) -> None:
        from ..arch import parse_layers

        a, s = ctx.arch, ctx.state
        for i in parse_layers(self.layers, a.n_layers(s)):
            if "mlp" in self.target:
                un, dn = a.layer_name(i, "up"), a.layer_name(i, "down")
                U, Dm = ctx.get(un), ctx.get(dn)
                c = self._c(ctx, U.norm(dim=1), Dm.norm(dim=0))
                ctx.write(un, U * c[:, None])
                ctx.write(dn, Dm / c[None, :])
                bn = a.layer_name(i, "up", "bias")
                if bn in s:
                    ctx.write(bn, ctx.get(bn) * c)
            if "vo" in self.target:
                D = a.dims(s)
                kv, rep, hd = D["kv_heads"], D["rep"], D["head_dim"]
                vn, on = a.layer_name(i, "v"), a.layer_name(i, "o")
                V, O = ctx.get(vn), ctx.get(on)
                V3 = V.reshape(kv, hd, -1)
                O4 = O.reshape(O.shape[0], kv, rep, hd)
                v_norm = V3.norm(dim=2)
                o_norm = O4.pow(2).sum(dim=(0, 2)).sqrt()
                c = self._c(ctx, v_norm, o_norm)  # [kv, hd]
                ctx.write(vn, (V3 * c[:, :, None]).reshape(V.shape))
                ctx.write(on, (O4 / c[None, :, None, :]).reshape(O.shape))
                bn = a.layer_name(i, "v", "bias")
                if bn in s:
                    ctx.write(bn, (ctx.get(bn).reshape(kv, hd) * c).reshape(-1))


# ------------------------------------------------------------- smooth scaling
@register("smooth_scale", "transform")
class SmoothScale(Op):
    """SmoothQuant / AWQ-style activation-aware scaling (exact before quantization):
    divides each norm channel by s_j and multiplies the matching input column of
    the following linears by s_j, with s_j = rms(x_j)^alpha / max|W_:j|^(1-alpha).
    Shifts quantization difficulty from activations into weights.  Needs calibration norms."""

    def __init__(self, alpha: float = 0.5, layers=None):
        self.alpha, self.layers = alpha, layers

    def run(self, ctx: Context) -> None:
        from ..arch import parse_layers

        a, s, calib = ctx.arch, ctx.state, ctx.need_calib()
        for i in parse_layers(self.layers, a.n_layers(s)):
            for norm_role, followers in (("attn_norm", ("q", "k", "v")), ("mlp_norm", ("gate", "up"))):
                names = [a.layer_name(i, r) for r in followers if a.layer_name(i, r) in s]
                nn_ = a.layer_name(i, norm_role)
                if not names or nn_ not in s:
                    continue
                x = calib.rms(names[0]).to(ctx.device, ctx.dtype)
                w = torch.stack([ctx.get(n).abs().amax(0) for n in names]).amax(0)
                sc = (x.clamp_min(1e-5) ** self.alpha / w.clamp_min(1e-5) ** (1 - self.alpha)).clamp_min(1e-5)
                g = ctx.get(nn_) + a.norm_offset
                ctx.write(nn_, g / sc - a.norm_offset)
                for n in names:
                    ctx.write(n, ctx.get(n) * sc[None, :])


# ------------------------------------------------------------------ alignment
def assignment(sim: torch.Tensor) -> torch.Tensor:
    """perm with sim[i, perm[i]] maximised (Hungarian; greedy fallback w/o scipy)."""
    try:
        from scipy.optimize import linear_sum_assignment

        _, cols = linear_sum_assignment(-sim.cpu().double().numpy())
        return torch.as_tensor(cols, dtype=torch.long)
    except ImportError:
        n = sim.shape[0]
        order = sim.flatten().argsort(descending=True)
        used_r = torch.zeros(n, dtype=torch.bool)
        used_c = torch.zeros(n, dtype=torch.bool)
        perm = torch.full((n,), -1, dtype=torch.long)
        left = n
        for idx in order.tolist():
            r, c = divmod(idx, n)
            if used_r[r] or used_c[c]:
                continue
            perm[r] = c
            used_r[r] = used_c[c] = True
            left -= 1
            if left == 0:
                break
        return perm


@register("align", "transform")
class Align(Op):
    """Permutation-align another model's MLP neurons to the current state by
    weight matching (Git Re-Basin), *in place on that model*, so that a
    subsequent ``merge`` interpolates matching units.  Exact for ``model``.
    Useful when merging models that do not share a common fine-tuning ancestor."""

    def __init__(self, model: str, reference: str = "self", layers=None):
        self.model, self.reference, self.layers = model, reference, layers

    def run(self, ctx: Context) -> None:
        from ..arch import parse_layers

        a = ctx.arch
        src, ref = ctx.model(self.model), ctx.model(self.reference)
        if src is ref:
            raise ValueError("cannot align a model to itself")

        def feats(st, i):
            g, u, d = (ctx.read(st, a.layer_name(i, r)) for r in ("gate", "up", "down"))
            return torch.cat([g, u, d.T], dim=1)

        for i in parse_layers(self.layers, a.n_layers(ref)):
            sim = feats(ref, i) @ feats(src, i).T
            permute_mlp(src, a, i, assignment(sim))
