"""Weight-delta ops: task-vector scaling/negation, LoRA merge, LoRA extraction."""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..arch import ArchSpec, detect_arch
from ..base import Context, Op, Selection, register
from ..store import TensorStore


@register("task_vector", "delta")
class TaskVector(Op):
    """Apply a task vector to the current state: new = self + scale * (finetuned - base).

    scale < 0 'unlearns' / negates a skill, scale > 1 extrapolates (ExPO),
    0 < scale < 1 interpolates (WiSE-FT; with ``base: self`` this is
    self + scale * (finetuned - self)).  ``base`` / ``finetuned`` are model
    names (``self`` = current state).
    """

    data_tags = ('other_models',)

    def __init__(self, finetuned: str, base: str = "self", scale: float = 1.0, **sel):
        self.ft, self.base, self.scale = finetuned, base, scale
        if not any(k in sel for k in ("roles", "include", "layers")):
            sel["include"] = ["*"]
        self.sel = Selection.from_kwargs(sel)

    def run(self, ctx: Context) -> None:
        ft, base = ctx.model(self.ft), ctx.model(self.base)
        for name in ctx.select(self.sel):
            if name not in ft or name not in base:
                continue
            cur, b, f = ctx.get(name), ctx.read(base, name), ctx.read(ft, name)
            if not (b.shape == f.shape == cur.shape):
                continue
            ctx.write(name, cur + self.scale * (f - b))


_LORA_KEY = re.compile(r"^(?:base_model\.model\.)?(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$")


def read_adapter(path: str):
    """-> {module_name: (A[r,in], B[out,r])}, scaling factor (alpha/r)."""
    with open(os.path.join(path, "adapter_config.json")) as f:
        cfg = json.load(f)
    r, alpha = cfg["r"], cfg.get("lora_alpha", cfg["r"])
    scaling = alpha / (r ** 0.5 if cfg.get("use_rslora") else r)
    pairs = {}
    with safe_open(os.path.join(path, "adapter_model.safetensors"), "pt", "cpu") as h:
        for k in h.keys():
            m = _LORA_KEY.match(k)
            if not m:
                continue
            pairs.setdefault(m.group(1), {})[m.group(2)] = h.get_tensor(k)
    return {k: (v["A"], v["B"]) for k, v in pairs.items() if "A" in v and "B" in v}, scaling


@register("lora_merge", "delta")
class LoraMerge(Op):
    """Merge a PEFT LoRA adapter directory into the weights: W += scale * (alpha/r) * B @ A."""

    data_tags = ('other_models',)

    def __init__(self, adapter: str, scale: float = 1.0):
        self.adapter, self.scale = adapter, scale

    def run(self, ctx: Context) -> None:
        pairs, scaling = read_adapter(self.adapter)
        merged = 0
        for module, (A, B) in pairs.items():
            name = module + ".weight"
            if name not in ctx.state:
                ctx.log(f"lora: {name} not in model, skipped")
                continue
            W = ctx.get(name)
            delta = (B.to(ctx.device, ctx.dtype) @ A.to(ctx.device, ctx.dtype)) * (scaling * self.scale)
            ctx.write(name, W + delta)
            merged += 1
        if merged == 0:
            raise ValueError("no LoRA weights matched the model's tensors")
        ctx.log(f"merged {merged} LoRA modules")


def extract_lora(
    base: TensorStore,
    finetuned: TensorStore,
    rank: int,
    out_dir: str,
    arch: Optional[ArchSpec] = None,
    roles=("all_linear",),
    device: str = "cpu",
) -> dict:
    """Extract a rank-``rank`` PEFT-compatible LoRA from ``finetuned - base``
    via truncated SVD.  Returns {name: relative reconstruction error}."""
    arch = arch or detect_arch(base)
    names = arch.select(base, roles=list(roles))
    out, errs, targets = {}, {}, set()
    for name in names:
        if name not in finetuned:
            continue
        delta = finetuned[name].to(device, torch.float32) - base[name].to(device, torch.float32)
        if delta.abs().max() == 0:
            continue
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(rank, S.numel())
        sq = S[:r].sqrt()
        B = (U[:, :r] * sq).contiguous()
        A = (sq[:, None] * Vh[:r]).contiguous()
        module = name[: -len(".weight")]
        out[f"base_model.model.{module}.lora_A.weight"] = A.cpu()
        out[f"base_model.model.{module}.lora_B.weight"] = B.cpu()
        targets.add(module.rsplit(".", 1)[-1])
        errs[name] = float((delta - B @ A).norm() / delta.norm())
    if not out:
        raise ValueError("finetuned model is identical to base; nothing to extract")
    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": rank,
                "lora_alpha": rank,
                "lora_dropout": 0.0,
                "target_modules": sorted(targets),
                "bias": "none",
            },
            f,
            indent=2,
        )
    return errs
