"""Compare two checkpoints tensor by tensor - the 'did my edit do what I think' tool."""
from __future__ import annotations

from typing import Dict, List

import torch

from .store import TensorStore


def compare(a: TensorStore, b: TensorStore) -> List[Dict]:
    rows = []
    for name in a.names():
        if name not in b:
            rows.append(dict(name=name, status="only_in_a"))
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape:
            rows.append(dict(name=name, status="shape", a=tuple(x.shape), b=tuple(y.shape)))
            continue
        xf, yf = x.float(), y.float()
        d = (xf - yf).norm().item()
        base = xf.norm().item()
        rows.append(
            dict(
                name=name,
                status="same" if d == 0 else "changed",
                rel_change=d / base if base else float("inf") if d else 0.0,
                max_abs=(xf - yf).abs().max().item() if xf.numel() else 0.0,
                zeros_a=float((xf == 0).float().mean()) if xf.numel() else 0.0,
                zeros_b=float((yf == 0).float().mean()) if xf.numel() else 0.0,
            )
        )
    rows.extend(dict(name=n, status="only_in_b") for n in b.names() if n not in a)
    return rows


def summarize(rows: List[Dict], top: int = 15) -> str:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines = ["tensors: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))]
    changed = sorted((r for r in rows if r["status"] == "changed"), key=lambda r: -r["rel_change"])
    if changed:
        lines.append(f"{'tensor':<60} {'rel.change':>11} {'max|d|':>10} {'zeros a->b':>14}")
        for r in changed[:top]:
            lines.append(
                f"{r['name'][-60:]:<60} {r['rel_change']:>11.4g} {r['max_abs']:>10.3g} "
                f"{r['zeros_a']:>6.1%}->{r['zeros_b']:<6.1%}"
            )
        if len(changed) > top:
            lines.append(f"... {len(changed) - top} more changed tensors")
    for r in rows:
        if r["status"] in ("shape", "only_in_a", "only_in_b"):
            lines.append(f"{r['status']}: {r['name']} {r.get('a', '')} {r.get('b', '')}")
    return "\n".join(lines)
