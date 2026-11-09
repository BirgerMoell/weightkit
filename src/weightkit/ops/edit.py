"""Direct knowledge / behaviour edits: MEMIT & ROME, abliteration, bias steering."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from ..base import Context, Op, Selection, register


def _load_tensor(x, key: Optional[str] = None):
    if isinstance(x, str):
        x = torch.load(x, map_location="cpu")
    if isinstance(x, dict):
        x = x[key] if key else next(iter(x.values()))
    return x


# ---------------------------------------------------------------------- MEMIT
def memit_update(
    W: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    lam: float = 1.0,
) -> torch.Tensor:
    """MEMIT closed form.  W:[out,in]  K:[in,n] keys  V:[out,n] target outputs
    C: E[k k^T] of typical keys.  Returns the new weight
    W + R K^T (lam*C + K K^T)^-1  with R = V - W K."""
    R = V - W @ K
    if C is None:
        C = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
    A = lam * C + K @ K.T
    return W + torch.linalg.solve(A, K @ R.T).T


def rome_update(W: torch.Tensor, k: torch.Tensor, v: torch.Tensor, C: Optional[torch.Tensor] = None):
    """ROME rank-one edit so that (W') k = v exactly, minimally changing W
    under the key covariance C."""
    if C is None:
        C = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
    u = torch.linalg.solve(C, k)
    return W + torch.outer(v - W @ k, u) / (k @ u)


@register("edit_facts", "edit")
class EditFacts(Op):
    """Insert key->value associations into one linear layer (ROME / MEMIT).

    ``edits``: a .pt file or dict holding ``keys`` [n, d_in] (the layer's input
    for each fact, e.g. MLP-down input at the subject's last token) and
    ``values`` [n, d_out] (the layer output you want for that key).
    mode: memit (batched, default) | rome (sequential exact rank-one edits).
    The key covariance comes from calibration ``xtx`` if available (strongly
    recommended - it is what keeps unrelated knowledge intact); ``lam`` scales it.
    """

    data_tags = ('supplied',)

    def __init__(self, edits: Union[str, dict], layer: int, role: str = "down", mode: str = "memit", lam: float = 1.0):
        if mode not in ("memit", "rome"):
            raise ValueError("mode must be memit or rome")
        self.edits, self.layer, self.role, self.mode, self.lam = edits, layer, role, mode, lam

    def run(self, ctx: Context) -> None:
        name = ctx.arch.layer_name(self.layer, self.role)
        W = ctx.get(name)
        K = _load_tensor(self.edits, "keys").to(ctx.device, ctx.dtype).T  # [d_in, n]
        V = _load_tensor(self.edits, "values").to(ctx.device, ctx.dtype).T  # [d_out, n]
        if K.shape[0] != W.shape[1] or V.shape[0] != W.shape[0]:
            raise ValueError(f"edit shapes K{tuple(K.shape)} V{tuple(V.shape)} do not fit {name} {tuple(W.shape)}")
        C = None
        if ctx.calib is not None and ctx.calib.has(name, "xtx"):
            C = ctx.calib.xtx(name).to(ctx.device, ctx.dtype)
            self.data_used = ("calibration",)
        else:
            ctx.log("edit_facts: no calibration covariance, using identity (edits may bleed)")
        if self.mode == "rome":
            for i in range(K.shape[1]):
                W = rome_update(W, K[:, i], V[:, i], C)
        else:
            W = memit_update(W, K, V, C, self.lam)
        ctx.write(name, W)


# ------------------------------------------------------------------ abliterate
def project_out(W: torch.Tensor, r: torch.Tensor, strength: float = 1.0, axis: int = 0) -> torch.Tensor:
    """Remove the component along unit vector ``r`` from the *output* space of
    W ([d, in], axis=0) or from each row of an embedding ([V, d], axis=1)."""
    if axis == 0:
        return W - strength * torch.outer(r, r @ W)
    return W - strength * torch.outer(W @ r, r)


@register("abliterate", "edit")
class Abliterate(Op):
    """Directional ablation: make the residual stream unable to represent a
    direction r (e.g. the 'refusal direction'), by orthogonalising every matrix
    that writes into the residual stream (embeddings, o_proj, down_proj).

    Give the direction as ``direction`` (a .pt vector / dict) or compute it from
    calibration hidden means: ``pos``/``neg`` tags and ``layer`` (decoder layer
    whose output was recorded) -> r = normalize(mean_pos - mean_neg).
    Needs untied embeddings (run ``untie_embeddings`` first).
    """

    def __init__(
        self,
        direction: Union[None, str, dict] = None,
        pos: Optional[str] = None,
        neg: Optional[str] = None,
        layer: Optional[int] = None,
        strength: float = 1.0,
        write_to: Sequence[str] = ("embed", "o", "down"),
        layers=None,
    ):
        if direction is None and not (pos and neg and layer is not None):
            raise ValueError("give `direction`, or `pos`, `neg` and `layer`")
        self.direction, self.pos, self.neg, self.layer = direction, pos, neg, layer
        self.strength, self.write_to, self.layers = strength, tuple(write_to), layers
        self.data_tags = (("supplied",) if direction is not None else ("calibration",))

    def _direction(self, ctx: Context) -> torch.Tensor:
        if self.direction is not None:
            r = _load_tensor(self.direction, "direction")
        else:
            mod = ctx.arch.layer_module(self.layer)
            c = ctx.need_calib()
            r = c.hidden_mean(self.pos, mod) - c.hidden_mean(self.neg, mod)
        r = r.to(ctx.device, torch.float32).flatten()
        return r / r.norm()

    def run(self, ctx: Context) -> None:
        a = ctx.arch
        if a.lm_head not in ctx.state and a.embed in ctx.state and "embed" in self.write_to:
            raise ValueError("tied embeddings: ablating the embedding would also change the unembedding; run untie_embeddings first")
        r = self._direction(ctx)
        sel_layers = a.select(ctx.state, roles=["o"], layers=self.layers)
        layer_ids = sorted({int(a.layer_regex().match(n).group(1)) for n in sel_layers})
        if "embed" in self.write_to:
            E = ctx.get(a.embed, torch.float32)
            ctx.write(a.embed, project_out(E, r, self.strength, axis=1))
        for i in layer_ids:
            for role in self.write_to:
                if role == "embed":
                    continue
                n = a.layer_name(i, role)
                if n not in ctx.state:
                    continue
                ctx.write(n, project_out(ctx.get(n, torch.float32), r, self.strength, axis=0))
                bn = a.layer_name(i, role, "bias")
                if bn in ctx.state:
                    b = ctx.get(bn, torch.float32)
                    ctx.write(bn, b - self.strength * (b @ r) * r)


# ---------------------------------------------------------------------- steer
@register("steer_bias", "edit")
class SteerBias(Op):
    """Bake an activation-steering vector into the weights: adds ``scale * v``
    to the residual stream after decoder layer ``layer`` for every token, via
    the layer's down_proj bias (creates zero biases on all MLPs and sets
    ``mlp_bias: true``, as HF Llama-style configs require uniform biases)."""

    data_tags = ('supplied',)

    def __init__(self, vector: Union[str, dict], layer: int, scale: float = 1.0):
        self.vector, self.layer, self.scale = vector, layer, scale

    def run(self, ctx: Context) -> None:
        a, s = ctx.arch, ctx.state
        v = _load_tensor(self.vector, "vector").to(torch.float32).flatten()
        for i in a.layer_ids(s):
            for role in ("gate", "up", "down"):
                bn = a.layer_name(i, role, "bias")
                if bn not in s:
                    out = s[a.layer_name(i, role)].shape[0]
                    ctx.write(bn, torch.zeros(out), like=s[a.layer_name(i, role)].dtype)
        s.config["mlp_bias"] = True
        bn = a.layer_name(self.layer, "down", "bias")
        b = ctx.get(bn, torch.float32)
        if v.shape != b.shape:
            raise ValueError(f"vector shape {tuple(v.shape)} != hidden size {tuple(b.shape)}")
        ctx.write(bn, b + self.scale * v.to(b.device))
