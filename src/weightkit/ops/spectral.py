"""Spectral (SVD) weight surgery: low-rank truncation and singular-value reshaping."""
from __future__ import annotations

import torch

from ..base import Context, Op, Selection, register


@register("spectral", "spectral")
class Spectral(Op):
    """Rewrite weights through their SVD.

    mode:
      truncate  keep top ``rank`` singular values (or ``rank_ratio`` of full rank)
      energy    keep the fewest values holding ``energy`` of the squared spectrum
      power     s -> s_max * (s / s_max) ** ``power`` (power<1 flattens, >1 sharpens)
      clip      clamp singular values above the ``quantile`` of the spectrum
      scale_top multiply the top ``k`` singular values by ``factor``
    """

    def __init__(
        self,
        mode: str = "truncate",
        rank: int = 0,
        rank_ratio: float = 0.0,
        energy: float = 0.99,
        power: float = 0.5,
        quantile: float = 0.99,
        k: int = 1,
        factor: float = 1.0,
        **sel,
    ):
        if mode not in ("truncate", "energy", "power", "clip", "scale_top"):
            raise ValueError(f"unknown spectral mode {mode!r}")
        self.mode, self.rank, self.rank_ratio = mode, rank, rank_ratio
        self.energy, self.power, self.quantile, self.k, self.factor = energy, power, quantile, k, factor
        self.sel = Selection.from_kwargs(sel)

    def _reshape(self, S: torch.Tensor) -> torch.Tensor:
        n = S.numel()
        if not S.max() > 0:  # all-zero matrix (e.g. an unused head): nothing to reshape, avoid 0/0
            return S
        if self.mode == "truncate":
            r = self.rank or max(1, int(n * self.rank_ratio))
            out = S.clone()
            out[r:] = 0
            return out
        if self.mode == "energy":
            e = S**2
            frac = torch.cumsum(e, 0) / e.sum()
            r = int((frac < self.energy).sum().item()) + 1
            out = S.clone()
            out[r:] = 0
            return out
        if self.mode == "power":
            return S.max() * (S / S.max()) ** self.power
        if self.mode == "clip":
            return torch.clamp(S, max=torch.quantile(S, self.quantile))
        out = S.clone()  # scale_top
        out[: self.k] *= self.factor
        return out

    def run(self, ctx: Context) -> None:
        for name in ctx.select(self.sel):
            W = ctx.get(name)
            if W.dim() != 2:
                continue
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            ctx.write(name, (U * self._reshape(S)) @ Vh)
