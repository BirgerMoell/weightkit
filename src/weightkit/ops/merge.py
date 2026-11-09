"""Multi-model merging: linear, slerp, task arithmetic, TIES, DARE, breadcrumbs,
model stock.  Operates on the current state ("self") as the base model."""
from __future__ import annotations

import math
from typing import List, Optional

import torch

from ..base import Context, Op, Selection, register


# ------------------------------------------------------------------ primitives
def topk_mask(x: torch.Tensor, density: float) -> torch.Tensor:
    """Boolean mask keeping the ``density`` fraction of largest-|x| entries."""
    n = x.numel()
    k = int(round(n * density))
    if k >= n:
        return torch.ones_like(x, dtype=torch.bool)
    if k <= 0:
        return torch.zeros_like(x, dtype=torch.bool)
    thresh = x.abs().flatten().kthvalue(n - k + 1).values
    return x.abs() >= thresh


def dare(x: torch.Tensor, density: float, gen: torch.Generator) -> torch.Tensor:
    """Random drop with probability 1-density, rescale survivors by 1/density."""
    if density >= 1.0:
        return x
    keep = torch.rand(x.shape, generator=gen) < density
    return x * keep.to(x.device, x.dtype) / density


def breadcrumbs(x: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    """Drop the ``beta`` smallest and ``gamma`` largest magnitude fractions."""
    n = x.numel()
    a = x.abs().flatten()
    srt = a.sort().values
    lo = srt[min(int(n * beta), n - 1)] if beta > 0 else -1.0
    hi = srt[max(n - 1 - int(n * gamma), 0)] if gamma > 0 else float("inf")
    keep = (x.abs() >= lo) & (x.abs() <= hi)
    return x * keep


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a, b = v0.flatten(), v1.flatten()
    na, nb = a.norm(), b.norm()
    if na < eps or nb < eps:
        return (1 - t) * v0 + t * v1
    cos = torch.clamp((a / na) @ (b / nb), -1.0, 1.0)
    if cos.abs() > 0.9995:
        return (1 - t) * v0 + t * v1
    theta = torch.acos(cos)
    s = torch.sin(theta)
    return (torch.sin((1 - t) * theta) / s) * v0 + (torch.sin(t * theta) / s) * v1


def ties_combine(taus: torch.Tensor, w: torch.Tensor, normalize: bool) -> torch.Tensor:
    """Sign-elect over weighted task vectors and average the agreeing ones."""
    w = w.view(-1, *([1] * (taus.dim() - 1)))
    weighted = taus * w
    sign = torch.sign(weighted.sum(0))
    mask = torch.sign(taus) == sign
    num = (weighted * mask).sum(0)
    if normalize:
        den = (w * mask).sum(0)
        den = torch.where(den.abs() < 1e-8, torch.ones_like(den), den)
        num = num / den
    return num


# method -> (preprocess, combine, normalize-default)
_TASK_METHODS = {
    "task_arithmetic": ("none", "sum", False),
    "ties": ("trim", "ties", True),
    "dare_linear": ("dare", "sum", True),
    "dare_ties": ("dare", "ties", True),
    "breadcrumbs": ("crumbs", "sum", False),
    "breadcrumbs_ties": ("crumbs", "ties", True),
}
METHODS = ("linear", "slerp", "model_stock") + tuple(_TASK_METHODS)


@register("merge", "merge")
class Merge(Op):
    """Merge other models into the current state.

    method: linear | slerp | model_stock | task_arithmetic | ties | dare_linear
            | dare_ties | breadcrumbs | breadcrumbs_ties
    sources: list of ``{model: <name>, weight: 1.0, density: <optional>}`` (or bare names).
             ``model: self`` refers to the current state.
    Task-vector methods (everything except linear/slerp) treat the current
    state as the base model.  ``scale`` multiplies the merged task vector;
    ``density`` is the per-source default for TIES trimming / DARE keep-rate;
    ``beta``/``gamma`` are the breadcrumbs small/large drop fractions; ``t`` is
    the slerp interpolation (0 = current state, 1 = sources[0]).
    """

    def __init__(
        self,
        method: str,
        sources: list,
        weights: Optional[list] = None,
        normalize: Optional[bool] = None,
        scale: float = 1.0,
        density: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        t: float = 0.5,
        **sel,
    ):
        if method not in METHODS:
            raise ValueError(f"unknown merge method {method!r}; choose from {METHODS}")
        self.method = method
        self.sources = []
        for i, s in enumerate(sources):
            s = {"model": s} if isinstance(s, str) else dict(s)
            if weights is not None:
                s["weight"] = weights[i]
            s.setdefault("weight", 1.0)
            self.sources.append(s)
        if method == "slerp" and len(self.sources) != 1:
            raise ValueError("slerp needs exactly one source (interpolates self -> source)")
        if method == "model_stock" and len(self.sources) < 2:
            raise ValueError("model_stock needs at least two fine-tuned sources")
        self.normalize = normalize
        self.scale, self.density, self.beta, self.gamma, self.t = scale, density, beta, gamma, t
        if not any(k in sel for k in ("roles", "include", "layers")):
            sel["include"] = ["*"]
        self.sel = Selection.from_kwargs(sel)

    def run(self, ctx: Context) -> None:
        stores = [ctx.model(s["model"]) for s in self.sources]
        gen = ctx.generator()
        for name in ctx.select(self.sel):
            base = ctx.get(name)
            srcs, meta = [], []
            for st, s in zip(stores, self.sources):
                if name not in st:
                    continue
                t = ctx.read(st, name)
                if t.shape != base.shape:
                    ctx.log(f"skip {name} for {s['model']}: shape {tuple(t.shape)} != {tuple(base.shape)}")
                    continue
                srcs.append(t)
                meta.append(s)
            if not srcs:
                continue
            ctx.write(name, self._merge_one(base, srcs, meta, gen))

    def _merge_one(self, base, srcs, meta, gen):
        m = self.method
        w = torch.tensor([float(s["weight"]) for s in meta], device=base.device, dtype=base.dtype)
        if m == "linear":
            if self.normalize is not False:
                w = w / w.sum()
            return sum(wi * t for wi, t in zip(w, srcs))
        if m == "slerp":
            return slerp(self.t, base, srcs[0])
        if m == "model_stock":
            deltas = [t - base for t in srcs]
            n = len(deltas)
            cos, pairs = 0.0, 0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = deltas[i].flatten(), deltas[j].flatten()
                    cos += float(a @ b / (a.norm() * b.norm() + 1e-12))
                    pairs += 1
            cos = max(cos / pairs, 0.0)
            tt = n * cos / (1 + (n - 1) * cos)
            return tt * (sum(srcs) / n) + (1 - tt) * base
        pre, comb, norm_default = _TASK_METHODS[m]
        normalize = norm_default if self.normalize is None else self.normalize
        taus = []
        for t, s in zip(srcs, meta):
            tau = t - base
            d = float(s.get("density") or self.density)
            if pre == "trim":
                tau = tau * topk_mask(tau, d)
            elif pre == "dare":
                tau = dare(tau, d, gen)
            elif pre == "crumbs":
                tau = breadcrumbs(tau, self.beta, self.gamma)
            taus.append(tau)
        stack = torch.stack(taus)
        if comb == "ties":
            merged = ties_combine(stack, w, normalize)
        else:
            merged = (stack * w.view(-1, *([1] * base.dim()))).sum(0)
            if normalize:
                merged = merged / w.sum()
        return base + self.scale * merged
