"""Op abstraction, registry and execution context."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Type

import torch

from .arch import ArchSpec, detect_arch
from .calibration import Calibration
from .store import TensorStore

_REGISTRY: Dict[str, Type["Op"]] = {}


def register(name: str, category: str):
    """Class decorator: make an Op available under ``name`` in pipelines."""

    def deco(cls):
        cls.name = name
        cls.category = category
        if name in _REGISTRY:
            raise ValueError(f"duplicate op {name!r}")
        _REGISTRY[name] = cls
        return cls

    return deco


def get_op(name: str) -> Type["Op"]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown op {name!r}; available: {', '.join(sorted(_REGISTRY))}") from None


def list_ops() -> Dict[str, Type["Op"]]:
    from . import ops  # noqa: F401  (ensures registration)

    return dict(sorted(_REGISTRY.items(), key=lambda kv: (kv[1].category, kv[0])))


@dataclass
class Selection:
    """Which tensors an op touches.  ``roles`` ∈ q,k,v,o,gate,up,down,attn,mlp,
    all_linear,embed,lm_head,final_norm,attn_norm,mlp_norm; ``layers`` e.g.
    "0-7,-1"; ``include``/``exclude`` are fnmatch globs on tensor names."""

    roles: Optional[Sequence[str]] = None
    layers: Optional[object] = None
    include: Optional[Sequence[str]] = None
    exclude: Optional[Sequence[str]] = None

    @classmethod
    def from_kwargs(cls, kw: dict) -> "Selection":
        bad = set(kw) - {"roles", "layers", "include", "exclude"}
        if bad:
            raise TypeError(f"unexpected parameter(s): {', '.join(sorted(bad))}")
        return cls(**kw)


class Op:
    """Base class.  Subclasses take their parameters in ``__init__`` and mutate
    ``ctx.state`` in ``run``.  Ops that accept a tensor selection take
    ``**sel`` and store ``Selection.from_kwargs(sel)``."""

    name: str = ""
    category: str = ""
    # What outside information this op consumed (empty = data-free):
    #   "calibration"    activations / statistics from a dataset
    #   "supplied"       vectors/edits the user provided (may derive from data)
    #   "other_models"   weights of other (trained) models or adapters
    data_tags: tuple = ()

    def run(self, ctx: "Context") -> None:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def describe(cls) -> str:
        doc = inspect.getdoc(cls) or ""
        return doc.split("\n\n")[0].replace("\n", " ")

    @classmethod
    def signature(cls) -> str:
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):  # pragma: no cover
            return ""
        parts = [str(p) for n, p in sig.parameters.items() if n != "self"]
        return ", ".join(parts)


@dataclass
class Context:
    state: TensorStore
    arch: ArchSpec
    models: Dict[str, TensorStore] = field(default_factory=dict)
    calib: Optional[Calibration] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0
    log: Callable[[str], None] = lambda s: None
    _salt: int = 0

    # ---- models
    def model(self, name: str) -> TensorStore:
        if name in (None, "self"):
            return self.state
        if name not in self.models:
            raise KeyError(f"model {name!r} not defined (have: {', '.join(self.models) or 'none'})")
        return self.models[name]

    # ---- tensor IO
    def read(self, store: TensorStore, name: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return store[name].to(device=self.device, dtype=dtype or self.dtype)

    def get(self, name: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.read(self.state, name, dtype)

    def write(self, name: str, tensor: torch.Tensor, like: Optional[torch.dtype] = None) -> None:
        """Store ``tensor`` keeping the dtype of any existing tensor of that name."""
        if like is None and name in self.state:
            like = self.state[name].dtype
        self.state[name] = tensor.detach().to("cpu", like or tensor.dtype)

    def select(self, sel: Selection, kind: str = "weight") -> List[str]:
        return self.arch.select(
            self.state, sel.roles, sel.layers, sel.include, sel.exclude, kind=kind
        )

    def generator(self) -> torch.Generator:
        """Fresh deterministic CPU generator (different stream per call)."""
        self._salt += 1
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + self._salt)
        return g

    def run_op(self, op: "Op", params: Optional[dict] = None) -> None:
        """Run ``op`` and record it in the store's provenance ledger."""
        self.state.written = set()
        op.run(self)
        tags = sorted(set(op.data_tags) | set(getattr(op, "data_used", ())))
        entry = {
            "op": op.name,
            "params": params or {},
            "data": tags or ["none"],
            "tensors_touched": len(self.state.written),
        }
        if "calibration" in tags and self.calib is not None:
            entry["calibration_tokens"] = self.calib.tokens()
        self.state.ledger.append(entry)
        self.state.written = set()

    def need_calib(self) -> Calibration:
        if self.calib is None:
            raise ValueError(
                "this op needs calibration statistics: pass `calibration: file.pt` "
                "(create it with `weightkit calibrate`)"
            )
        return self.calib


def apply_op(
    store: TensorStore,
    op: str,
    *,
    calib: Optional[Calibration] = None,
    models: Optional[Dict[str, TensorStore]] = None,
    arch: Optional[ArchSpec] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    log: Callable[[str], None] = lambda s: None,
    **params,
) -> TensorStore:
    """Apply one registered op to ``store`` in place (and return it)."""
    from . import ops  # noqa: F401

    ctx = Context(
        state=store,
        arch=arch or detect_arch(store, strict=False),
        models=models or {},
        calib=calib,
        device=device,
        dtype=dtype,
        seed=seed,
        log=log,
    )
    ctx.run_op(get_op(op)(**params), params)
    return store
