"""Unstructured / semi-structured pruning: magnitude, Wanda, SparseGPT."""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..base import Context, Op, Selection, register


def keep_mask(scores: torch.Tensor, sparsity: float, granularity: str, nm: Optional[Sequence[int]]):
    """Boolean mask (True = keep) removing the lowest-scoring weights."""
    out_dim, in_dim = scores.shape
    if nm:
        n, m = nm
        if in_dim % m:
            raise ValueError(f"N:M pruning needs in_features divisible by {m}, got {in_dim}")
        g = scores.reshape(out_dim, in_dim // m, m)
        idx = g.topk(n, dim=-1).indices
        mask = torch.zeros_like(g, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return mask.reshape(out_dim, in_dim)
    if granularity == "layer":
        flat = scores.flatten()
        k = int(flat.numel() * sparsity)
        mask = torch.ones_like(flat, dtype=torch.bool)
        mask[flat.argsort()[:k]] = False
        return mask.reshape(scores.shape)
    dim = 1 if granularity == "row" else 0
    k = int(scores.shape[dim] * sparsity)
    mask = torch.ones_like(scores, dtype=torch.bool)
    idx = scores.argsort(dim=dim)
    idx = idx[:, :k] if dim == 1 else idx[:k, :]
    mask.scatter_(dim, idx, False)
    return mask


def sparsegpt(
    W: torch.Tensor,
    H: torch.Tensor,
    sparsity: float,
    nm: Optional[Sequence[int]] = None,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """SparseGPT (Frantar & Alistarh 2023): prune while updating the remaining
    weights with second-order (OBS) corrections.  ``H`` = E[x x^T] of inputs."""
    W = W.clone().float()
    H = H.clone().float()
    cols = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(cols, device=H.device)
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    prunen, prunem = (nm[1] - nm[0], nm[1]) if nm else (0, 0)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if prunen == 0:
            tmp = W1**2 / (torch.diag(Hinv1).reshape(1, -1)) ** 2
            k = int(tmp.numel() * sparsity)
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
            if k > 0:
                thresh = tmp.flatten().sort().values[k - 1]
                mask1 = tmp <= thresh
        else:
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            if prunen != 0 and i % prunem == 0:
                tmp = W1[:, i : i + prunem] ** 2 / (torch.diag(Hinv1)[i : i + prunem].reshape(1, -1)) ** 2
                idx = torch.topk(tmp, prunen, dim=1, largest=False).indices
                mask1.scatter_(1, i + idx, True)
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W


@register("prune", "prune")
class Prune(Op):
    """Zero out weights.

    method: magnitude (|W|)            - no data needed
            wanda     (|W| * ||x_j||)  - needs calibration activation norms
            sparsegpt (OBS update)     - needs calibration ``xtx`` (collect with xtx=True)
    sparsity: fraction removed.  granularity (magnitude/wanda): layer | row | column | global
    (wanda default: row; global = one threshold over all selected tensors).  nm: [n, m] keeps n of every m consecutive input weights
    (e.g. [2, 4] is hardware 2:4 sparsity) and overrides ``sparsity``.
    """

    def __init__(
        self,
        method: str = "magnitude",
        sparsity: float = 0.5,
        granularity: Optional[str] = None,
        nm: Optional[Sequence[int]] = None,
        blocksize: int = 128,
        **sel,
    ):
        if method not in ("magnitude", "wanda", "sparsegpt"):
            raise ValueError(f"unknown prune method {method!r}")
        self.method, self.sparsity, self.nm, self.blocksize = method, sparsity, nm, blocksize
        self.granularity = granularity or ("row" if method == "wanda" else "layer")
        if self.granularity not in ("layer", "row", "column", "global"):
            raise ValueError("granularity must be layer, row, column or global")
        self.data_tags = () if method == "magnitude" else ("calibration",)
        sel.setdefault("exclude", None)
        self.sel = Selection.from_kwargs(sel)

    def run(self, ctx: Context) -> None:
        names = [n for n in ctx.select(self.sel) if ctx.state[n].dim() == 2]
        if self.granularity == "global":
            self._run_global(ctx, names)
            return
        total = zeros = 0
        for name in names:
            W = ctx.get(name)
            if self.method == "sparsegpt":
                new = sparsegpt(W, ctx.need_calib().xtx(name).to(ctx.device), self.sparsity, self.nm, self.blocksize)
            else:
                scores = W.abs()
                if self.method == "wanda":
                    scores = scores * ctx.need_calib().norm(name).to(ctx.device)[None, :]
                new = W * keep_mask(scores, self.sparsity, self.granularity, self.nm)
            ctx.write(name, new)
            total += new.numel()
            zeros += int((new == 0).sum())
        if total:
            ctx.log(f"prune/{self.method}: {zeros / total:.1%} of {total} weights are zero")

    def _run_global(self, ctx: Context, names) -> None:
        """One threshold across all selected tensors (needs all their scores in memory)."""
        if self.method == "sparsegpt" or self.nm:
            raise ValueError("granularity 'global' supports magnitude/wanda without nm")
        Ws = {n: ctx.get(n) for n in names}
        scores = []
        for n, W in Ws.items():
            sc = W.abs()
            if self.method == "wanda":
                sc = sc * ctx.need_calib().norm(n).to(ctx.device)[None, :]
            scores.append(sc.flatten())
        flat = torch.cat(scores)
        mask = torch.ones_like(flat, dtype=torch.bool)
        mask[flat.argsort()[: int(flat.numel() * self.sparsity)]] = False
        off = 0
        for n, W in Ws.items():
            m = mask[off : off + W.numel()].reshape(W.shape)
            off += W.numel()
            ctx.write(n, W * m)
        ctx.log(f"prune/{self.method}/global: removed {(~mask).float().mean():.1%} of {flat.numel()} weights")
