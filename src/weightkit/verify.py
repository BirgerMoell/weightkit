"""Declarative post-edit verification ("the weights are the spec").

    weightkit verify spec.yaml

    model: ./out
    reference: ./base            # optional; needed by equivalence / untouched / changed / max_rel_change
    probes: {n: 8, length: 16, seed: 0}   # random token ids (or a .pt LongTensor path)
    checks:
      - {type: finite}
      - {type: equivalence, tol: 1.0e-4}
      - {type: sparsity, roles: [mlp], min: 0.45, max: 0.55}
      - {type: untouched, include: ["model.embed_tokens.weight"]}
      - {type: data_use, allowed: [none]}

Checks: finite, equivalence, sparsity, untouched, changed, max_rel_change,
direction_removed, config, data_use, ops.  ``equivalence`` runs both models, by
default through Hugging Face transformers (pass ``runner`` in Python to use
anything else).
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import yaml

from .arch import ArchSpec, detect_arch
from .store import TensorStore


@dataclass
class Result:
    check: str
    ok: bool
    detail: str


class VerificationFailed(Exception):
    def __init__(self, results: List[Result]):
        self.results = results
        super().__init__(format_results(results))


class _Ctx:
    def __init__(self, model, reference, arch, runner, probes_spec):
        self.model, self.reference, self.arch, self.runner = model, reference, arch, runner
        self.probes_spec = probes_spec

    def need_ref(self, name):
        if self.reference is None:
            raise ValueError(f"check {name!r} needs `reference:`")
        return self.reference

    def probes(self) -> torch.Tensor:
        p = self.probes_spec
        if isinstance(p, str):
            return torch.load(p, map_location="cpu").long()
        p = p or {}
        vocab = self.model.config.get("vocab_size") or int(self.model[self.arch.embed].shape[0])
        g = torch.Generator().manual_seed(int(p.get("seed", 0)))
        return torch.randint(0, vocab, (int(p.get("n", 8)), int(p.get("length", 16))), generator=g)


def hf_runner(store: TensorStore) -> Callable[[torch.Tensor], torch.Tensor]:
    """Default runner: load the saved checkpoint with transformers."""
    if not store.source:
        raise ValueError("equivalence needs a saved checkpoint (store.source); save first or pass runner=")
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(store.source, torch_dtype=torch.float32).eval()

    def f(ids):
        with torch.no_grad():
            return m(ids).logits

    return f


def _names(c: _Ctx, spec: dict) -> List[str]:
    if spec.get("include"):
        names = [n for n in c.model.names() if any(fnmatch.fnmatch(n, p) for p in spec["include"])]
    else:
        names = c.arch.select(c.model, spec.get("roles"), spec.get("layers"))
    ex = spec.get("exclude") or []
    return [n for n in names if not any(fnmatch.fnmatch(n, p) for p in ex)]


# ------------------------------------------------------------------ checks
def _finite(c, s):
    bad = [n for n in c.model.names() if not torch.isfinite(c.model[n].float()).all()]
    return (not bad), "all tensors finite" if not bad else f"{len(bad)} tensors with NaN/inf, e.g. {bad[:3]}"


def _equivalence(c, s):
    ref = c.need_ref("equivalence")
    ids = c.probes()
    a, b = c.runner(ref)(ids), c.runner(c.model)(ids)
    diff = float((a - b).abs().max())
    agree = float((a.argmax(-1) == b.argmax(-1)).float().mean())
    ok = diff <= float(s.get("tol", 1e-4)) and agree >= float(s.get("min_top1_agreement", 0.0))
    return ok, f"max |Δlogit| = {diff:.3g} (tol {s.get('tol', 1e-4)}), top-1 agreement {agree:.1%} on {tuple(ids.shape)} probes"


def _sparsity(c, s):
    names = _names(c, s)
    if not names:
        return False, "no tensors matched"
    z = sum(int((c.model[n] == 0).sum()) for n in names)
    t = sum(c.model[n].numel() for n in names)
    f = z / t
    lo, hi = float(s.get("min", 0.0)), float(s.get("max", 1.0))
    return lo <= f <= hi, f"{f:.1%} zeros over {len(names)} tensors (want {lo:.0%}–{hi:.0%})"


def _untouched(c, s):
    ref = c.need_ref("untouched")
    names = _names(c, s)
    if not names:
        return False, "no tensors matched"
    bad = [n for n in names if n not in ref or c.model[n].shape != ref[n].shape or not torch.equal(c.model[n], ref[n])]
    return (not bad), f"{len(names) - len(bad)}/{len(names)} identical" + (f"; changed: {bad[:3]}" if bad else "")


def _rel(a, b):
    a, b = a.float(), b.float()
    d = (a - b).norm().item()
    n = a.norm().item()
    return d / n if n else (0.0 if d == 0 else float("inf"))


def _changed(c, s):
    ref = c.need_ref("changed")
    names = _names(c, s)
    if not names:
        return False, "no tensors matched"
    thr = float(s.get("min_rel_change", 1e-8))
    bad = [n for n in names if n not in ref or c.model[n].shape != ref[n].shape or _rel(ref[n], c.model[n]) < thr]
    return (not bad), f"{len(names) - len(bad)}/{len(names)} changed by ≥ {thr:g}" + (f"; unchanged: {bad[:3]}" if bad else "")


def _max_rel_change(c, s):
    ref = c.need_ref("max_rel_change")
    common = [n for n in c.model.names() if n in ref and c.model[n].shape == ref[n].shape]
    num = sum((c.model[n].float() - ref[n].float()).pow(2).sum().item() for n in common)
    den = sum(ref[n].float().pow(2).sum().item() for n in common)
    r = (num / den) ** 0.5 if den else 0.0
    lo, hi = float(s.get("min", 0.0)), float(s.get("max", float("inf")))
    return lo <= r <= hi, f"global relative change {r:.4g} (want {lo:g}–{hi:g})"


def _direction_removed(c, s):
    d = s["direction"]
    r = torch.load(d, map_location="cpu") if isinstance(d, str) else torch.as_tensor(d)
    if isinstance(r, dict):
        r = next(iter(r.values()))
    r = r.float().flatten()
    r = r / r.norm()
    a, tol = c.arch, float(s.get("tol", 1e-4))
    worst, where = 0.0, ""
    comps = [(a.embed, c.model[a.embed].float() @ r)]
    for i in a.layer_ids(c.model):
        for role in ("o", "down"):
            n = a.layer_name(i, role)
            if n in c.model:
                comps.append((n, r @ c.model[n].float()))
    for n, v in comps:
        m = float(v.abs().max())
        if m > worst:
            worst, where = m, n
    return worst <= tol, f"max component along direction = {worst:.3g} at {where} (tol {tol:g})"


def _config(c, s):
    bad = {k: (c.model.config.get(k), v) for k, v in s["expect"].items() if c.model.config.get(k) != v}
    return (not bad), "config as expected" if not bad else "mismatch (got, want): " + ", ".join(f"{k}={g!r}/{w!r}" for k, (g, w) in bad.items())


def _manifest(c):
    m = c.model.parent_manifest
    if not m:
        raise ValueError("no weightkit_manifest.json next to the model (was it saved by weightkit?)")
    return m


def _data_use(c, s):
    m = _manifest(c)
    used = set(m.get("data_use", [])) - {"none"}
    allowed = set(s.get("allowed", ["none"])) - {"none"}
    toks = m.get("calibration_tokens", 0)
    mx = s.get("max_calibration_tokens")
    ok = used <= allowed and (mx is None or toks <= mx)
    return ok, f"data used: {sorted(used) or ['none']}" + (f", {toks} calibration tokens" if toks else "") + f" (allowed: {sorted(allowed) or ['none']})"


def _ops(c, s):
    got = [e["op"] for e in _manifest(c).get("ops", [])]
    if "exact" in s:
        return got == list(s["exact"]), f"ops {got} (want exactly {s['exact']})"
    want = list(s["contains"])
    it = iter(got)
    return all(w in it for w in want), f"ops {got} (want subsequence {want})"


CHECKS: Dict[str, Callable] = {
    "finite": _finite, "equivalence": _equivalence, "sparsity": _sparsity, "untouched": _untouched,
    "changed": _changed, "max_rel_change": _max_rel_change, "direction_removed": _direction_removed,
    "config": _config, "data_use": _data_use, "ops": _ops,
}


def verify(
    spec,
    runner: Optional[Callable[[TensorStore], Callable]] = None,
    model: Optional[TensorStore] = None,
    reference: Optional[TensorStore] = None,
) -> List[Result]:
    """Run the checks of ``spec`` (dict or YAML path).  Returns one Result per check."""
    if isinstance(spec, str):
        with open(spec) as f:
            spec = yaml.safe_load(f)
    model = model or TensorStore.from_path(spec["model"])
    if reference is None and spec.get("reference"):
        reference = TensorStore.from_path(spec["reference"])
    arch = detect_arch(model, strict=False)
    c = _Ctx(model, reference, arch, runner or hf_runner, spec.get("probes"))
    out: List[Result] = []
    for chk in spec.get("checks", []):
        chk = dict(chk)
        t = chk.pop("type", None)
        if t not in CHECKS:
            out.append(Result(str(t), False, f"unknown check type; available: {', '.join(CHECKS)}"))
            continue
        try:
            ok, detail = CHECKS[t](c, chk)
        except Exception as e:  # a check that cannot run is a failed check, with the reason
            ok, detail = False, f"could not run: {e}"
        out.append(Result(t, bool(ok), detail))
    return out


def format_results(results: List[Result]) -> str:
    lines = [f"{'✓' if r.ok else '✗'} {r.check:<18} {r.detail}" for r in results]
    npass = sum(r.ok for r in results)
    lines.append(f"{npass}/{len(results)} checks passed")
    return "\n".join(lines)
