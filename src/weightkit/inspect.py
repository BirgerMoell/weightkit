"""Static analysis of a checkpoint: size, sparsity, spectra, outliers."""
from __future__ import annotations

import math
from typing import Dict, List

import torch

from .store import TensorStore


def tensor_stats(t: torch.Tensor, rank: bool = False, max_rank_dim: int = 4096) -> Dict:
    x = t.float()
    n = x.numel()
    nz = int(torch.count_nonzero(x))
    row = dict(
        shape=tuple(t.shape),
        dtype=str(t.dtype).replace("torch.", ""),
        params=n,
        nonzero=nz,
        sparsity=1 - nz / n if n else 0.0,
        norm=float(x.norm()),
        absmax=float(x.abs().max()) if n else 0.0,
    )
    if x.dim() == 2 and n:
        cn = x.norm(dim=0)
        med = cn.median()
        row["col_outlier"] = float(cn.max() / med) if med > 0 else float("inf")
        if rank and max(x.shape) <= max_rank_dim:
            s = torch.linalg.svdvals(x)
            if s.sum() > 0:
                p = s / s.sum()
                row["eff_rank"] = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
                row["max_rank"] = min(x.shape)
    return row


def inspect_store(store: TensorStore, rank: bool = False) -> Dict:
    tensors = {n: tensor_stats(store[n], rank) for n in store.names()}
    params = sum(r["params"] for r in tensors.values())
    nz = sum(r["nonzero"] for r in tensors.values())
    return dict(
        tensors=tensors,
        params=params,
        nonzero=nz,
        sparsity=1 - nz / params if params else 0.0,
        config={k: store.config[k] for k in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size") if k in store.config},
        manifest=store.parent_manifest,
    )


def _si(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def format_report(info: Dict, top: int = 12, rank: bool = False) -> str:
    out: List[str] = []
    out.append(f"{_si(info['params'])} parameters · {_si(info['nonzero'])} non-zero · {info['sparsity']:.1%} zeros · {len(info['tensors'])} tensors")
    if info["config"]:
        out.append("config: " + ", ".join(f"{k}={v}" for k, v in info["config"].items()))
    rows = sorted(info["tensors"].items(), key=lambda kv: -kv[1]["params"])[:top]
    hdr = f"{'tensor':<52} {'shape':<16} {'zeros':>6} {'|W|max':>8} {'outl.':>6}" + (f" {'eff.rank':>9}" if rank else "")
    out.append(hdr)
    for n, r in rows:
        line = f"{n[-52:]:<52} {str(r['shape']):<16} {r['sparsity']:>6.1%} {r['absmax']:>8.3g} {r.get('col_outlier', float('nan')):>6.1f}"
        if rank:
            line += f" {r['eff_rank']:>5.0f}/{r['max_rank']:<3}" if "eff_rank" in r else f" {'-':>9}"
        out.append(line)
    if len(info["tensors"]) > top:
        out.append(f"... {len(info['tensors']) - top} more tensors (largest {top} shown)")
    if info.get("manifest"):
        from .manifest import describe

        out.append("")
        out.append("provenance (weightkit_manifest.json):")
        out.append(describe(info["manifest"], "  "))
    return "\n".join(out)
