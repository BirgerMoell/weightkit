"""Provenance manifest: what was done to a checkpoint, and with what data.

``weightkit_manifest.json`` is written next to the weights whenever a store that
went through ops is saved.  It records every op with its parameters, which
outside information it consumed (``none`` = data-free), how many tensors it
touched, and the manifest of the checkpoint it started from (a chain).
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List

import torch


def jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return {"tensor": list(x.shape), "dtype": str(x.dtype).replace("torch.", "")}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def data_summary(ledger: List[dict], parent: dict = None) -> List[str]:
    tags = {t for e in ledger for t in e.get("data", []) if t != "none"}
    if parent:
        tags |= {t for t in parent.get("data_use", []) if t != "none"}
    return sorted(tags) or ["none"]


def build_manifest(store) -> Dict[str, Any]:
    from . import __version__

    ledger = [jsonable(e) for e in store.ledger]
    toks = [e["calibration_tokens"] for e in ledger if "calibration_tokens" in e]
    m: Dict[str, Any] = {
        "weightkit_version": __version__,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "base": store.source,
        "ops": ledger,
        "data_use": data_summary(ledger, store.parent_manifest),
        "calibration_tokens": max(toks) if toks else 0,
        "inputs": jsonable(store.manifest_extra),
    }
    if store.parent_manifest:
        m["parent"] = store.parent_manifest
    return m


def describe(m: dict, indent: str = "") -> str:
    lines = [f"{indent}weightkit {m.get('weightkit_version')} · {m.get('created')}",
             f"{indent}data used: {', '.join(m.get('data_use', ['none']))}"
             + (f" ({m['calibration_tokens']} calibration tokens)" if m.get("calibration_tokens") else "")]
    for e in m.get("ops", []):
        lines.append(f"{indent}  - {e['op']:<16} data={','.join(e['data']):<14} touched {e['tensors_touched']} tensors")
    if m.get("parent"):
        lines.append(f"{indent}derived from:")
        lines.append(describe(m["parent"], indent + "    "))
    return "\n".join(lines)
