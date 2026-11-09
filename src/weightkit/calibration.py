"""Calibration statistics for data-aware ops (Wanda, SparseGPT, MEMIT,
SmoothQuant scaling, abliteration).

Collect once with a real torch model + data, save to a ``.pt`` file, and feed it
to any number of pipelines.  Requires only that the model's parameter names
match the checkpoint tensor names (true for Hugging Face models).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import torch


class Calibration:
    def __init__(self):
        self.n: Dict[str, int] = {}
        self.sumsq: Dict[str, torch.Tensor] = {}
        self.xtx_sum: Dict[str, torch.Tensor] = {}
        # hidden[tag][module_name] = mean activation vector
        self.hidden: Dict[str, Dict[str, torch.Tensor]] = {}

    # ---- accumulation
    def update(self, name: str, x: torch.Tensor, xtx: bool = True) -> None:
        """x: [tokens, d_in] inputs of the linear layer owning weight ``name``."""
        x = x.float()
        self.n[name] = self.n.get(name, 0) + x.shape[0]
        ss = (x * x).sum(0)
        self.sumsq[name] = self.sumsq.get(name, 0) + ss
        if xtx:
            self.xtx_sum[name] = self.xtx_sum.get(name, 0) + x.T @ x

    # ---- access
    def has(self, name: str, kind: str = "norm") -> bool:
        return name in (self.sumsq if kind == "norm" else self.xtx_sum)

    def _need(self, d, name, what):
        if name not in d:
            raise KeyError(
                f"calibration has no {what} for {name!r}; run `weightkit calibrate` "
                "(or weightkit.calibration.collect_input_stats) covering this tensor"
            )

    def norm(self, name: str) -> torch.Tensor:
        """Per-input-channel L2 norm of activations over all tokens (Wanda)."""
        self._need(self.sumsq, name, "activation norms")
        return self.sumsq[name].sqrt()

    def rms(self, name: str) -> torch.Tensor:
        self._need(self.sumsq, name, "activation norms")
        return (self.sumsq[name] / max(self.n[name], 1)).sqrt()

    def xtx(self, name: str) -> torch.Tensor:
        """Mean second-moment matrix E[x x^T] of the layer's inputs."""
        self._need(self.xtx_sum, name, "input covariance (xtx)")
        return self.xtx_sum[name] / max(self.n[name], 1)

    def hidden_mean(self, tag: str, module: str) -> torch.Tensor:
        try:
            return self.hidden[tag][module]
        except KeyError:
            raise KeyError(f"no hidden-state mean for tag={tag!r} module={module!r}") from None

    # ---- io
    def save(self, path: str) -> None:
        cpu = lambda d: {k: v.cpu() for k, v in d.items()}
        torch.save(
            dict(
                n=self.n,
                sumsq=cpu(self.sumsq),
                xtx_sum=cpu(self.xtx_sum),
                hidden={t: cpu(m) for t, m in self.hidden.items()},
            ),
            path,
        )

    @classmethod
    def load(cls, path: str) -> "Calibration":
        d = torch.load(path, map_location="cpu")
        c = cls()
        c.n, c.sumsq, c.xtx_sum, c.hidden = d["n"], d["sumsq"], d["xtx_sum"], d["hidden"]
        return c


def _forward(model, batch):
    if isinstance(batch, dict):
        return model(**batch)
    return model(batch)


def collect_input_stats(
    model: torch.nn.Module,
    batches: Iterable,
    weight_names: Sequence[str],
    xtx: bool = False,
    max_batches: Optional[int] = None,
    calib: Optional[Calibration] = None,
) -> Calibration:
    """Record input statistics of every linear layer whose ``.weight`` is named.

    ``batches`` yields token tensors or kwargs dicts (``input_ids``,
    ``attention_mask``).  Padding tokens are excluded when a mask is given.
    """
    calib = calib or Calibration()
    state = {"mask": None}
    handles = []
    for wn in weight_names:
        if not wn.endswith(".weight"):
            raise ValueError(f"{wn!r} is not a weight name")
        mod = model.get_submodule(wn[: -len(".weight")])

        def hook(m, args, wn=wn):
            x = args[0].detach()
            x = x.reshape(-1, x.shape[-1])
            mask = state["mask"]
            if mask is not None and mask.numel() == x.shape[0]:
                x = x[mask]
            calib.update(wn, x, xtx)

        handles.append(mod.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for i, b in enumerate(batches):
                if max_batches is not None and i >= max_batches:
                    break
                am = b.get("attention_mask") if isinstance(b, dict) else None
                state["mask"] = am.reshape(-1).bool() if am is not None else None
                _forward(model, b)
    finally:
        for h in handles:
            h.remove()
    return calib


def collect_hidden_mean(
    model: torch.nn.Module,
    batches: Iterable,
    module_names: Sequence[str],
    position: int = -1,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Mean output of each module at a token ``position`` (default: last token).

    Use with decoder layers (``model.layers.14``) to get residual-stream means
    for e.g. harmful vs harmless prompts; store under ``calib.hidden[tag]``.
    Assumes left-padding or no padding when ``position`` is negative.
    """
    sums: Dict[str, torch.Tensor] = {}
    count = {"n": 0}
    handles = []
    for name in module_names:
        mod = model.get_submodule(name)

        def hook(m, args, out, name=name):
            o = out[0] if isinstance(out, (tuple, list)) else out
            v = o.detach().float()[:, position, :].sum(0)
            sums[name] = sums.get(name, 0) + v

        handles.append(mod.register_forward_hook(hook))
    try:
        with torch.no_grad():
            for i, b in enumerate(batches):
                if max_batches is not None and i >= max_batches:
                    break
                ids = b["input_ids"] if isinstance(b, dict) else b
                count["n"] += ids.shape[0]
                _forward(model, b)
    finally:
        for h in handles:
            h.remove()
    return {k: v / max(count["n"], 1) for k, v in sums.items()}


def capture_inputs(model, batch, module_name: str, positions: Sequence[int]) -> torch.Tensor:
    """Inputs of ``module_name`` at the given token positions of batch row 0
    (used to build MEMIT keys: input of down_proj at the subject's last token)."""
    got = {}

    def hook(m, args):
        got["x"] = args[0].detach()[0, list(positions), :].float()

    h = model.get_submodule(module_name).register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            _forward(model, batch)
    finally:
        h.remove()
    return got["x"]
