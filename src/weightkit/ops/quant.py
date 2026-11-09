"""Weight quantization applied *in the weights* (fake-quant / round-to-nearest)."""
from __future__ import annotations

import torch

from ..base import Context, Op, Selection, register


def rtn_quantize(W: torch.Tensor, bits: int = 4, group_size: int = 128, symmetric: bool = False) -> torch.Tensor:
    """Round-to-nearest quantize-dequantize along the input dim, per group."""
    out_dim, in_dim = W.shape
    g = in_dim if group_size <= 0 else min(group_size, in_dim)
    pad = (-in_dim) % g
    X = torch.nn.functional.pad(W, (0, pad)) if pad else W
    X = X.reshape(out_dim, -1, g)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = (X.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
        Q = torch.clamp(torch.round(X / scale), -qmax - 1, qmax) * scale
    else:
        qmax = 2**bits - 1
        lo, hi = X.amin(-1, keepdim=True), X.amax(-1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp(min=1e-12)
        zero = torch.round(-lo / scale)
        Q = (torch.clamp(torch.round(X / scale) + zero, 0, qmax) - zero) * scale
    Q = Q.reshape(out_dim, -1)
    return Q[:, :in_dim] if pad else Q


@register("quantize_rtn", "quant")
class QuantizeRTN(Op):
    """Round-to-nearest N-bit group quantization, written back as dequantized
    floating-point weights (so the result loads anywhere and can be re-packed by
    a real quantizer losslessly).  group_size <= 0 means per-row."""

    def __init__(self, bits: int = 4, group_size: int = 128, symmetric: bool = False, **sel):
        self.bits, self.group_size, self.symmetric = bits, group_size, symmetric
        self.sel = Selection.from_kwargs(sel)

    def run(self, ctx: Context) -> None:
        for name in ctx.select(self.sel):
            W = ctx.get(name)
            if W.dim() == 2:
                ctx.write(name, rtn_quantize(W, self.bits, self.group_size, self.symmetric))
