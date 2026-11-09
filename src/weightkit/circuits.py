"""Bridge to hand-written networks (https://github.com/BirgerMoell/handnet).

handnet's transformers have exactly-known behaviour (100% on addition, Dyck-1
and induction) and a handful of non-zero weights, which makes them ground truth
for weight-editing ops: you can ask *task accuracy*, not just "did logits
move", and for pruning you know the right answer (keep exactly the designed
weights).  Requires ``pip install git+https://github.com/BirgerMoell/handnet``
(numpy only).

Tensor layout is handnet's own (``x @ W``, i.e. [in, out]), stacked tensors are
split into 2-D matrices (``layers.0.heads.0.Wq``), so every weight-level op
(``prune``, ``spectral``, ``quantize_rtn``, ``perturb``, ``task_vector`` ...)
works through ``include`` globs.  1-D biases are kept (those ops skip them).
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from .base import apply_op
from .store import TensorStore

_HEAD = ("Wq", "Wk", "Wv", "Wo", "bq", "bk")
_LAYER = ("W1", "b1", "W2", "b2")
_GLOBAL = ("E", "P", "U", "bu")


def _names(net) -> Dict[str, tuple]:
    m: Dict[str, tuple] = {k: (k,) for k in _GLOBAL}
    for l in range(net.L):
        for k in _LAYER:
            m[f"layers.{l}.{k}"] = (k, l)
        for h in range(net.Hn):
            for k in _HEAD:
                m[f"layers.{l}.heads.{h}.{k}"] = (k, l, h)
    return m


def store_from_net(net) -> TensorStore:
    """HandTransformer -> TensorStore (float64 copies)."""
    t = net.tensors
    return TensorStore.from_dict({n: torch.from_numpy(np.array(t[k[0]][k[1:]], dtype=np.float64)) for n, k in _names(net).items()})


def net_from_store(net, store: TensorStore):
    """A deep copy of ``net`` carrying the (edited) weights in ``store``."""
    out = copy.deepcopy(net)
    for n, k in _names(net).items():
        getattr(out, k[0])[k[1:]] = store[n].double().numpy()
    return out


def support_fraction(net, only_matrices: bool = True) -> float:
    """Fraction of (2-D) weights that are non-zero in the design."""
    s = store_from_net(net)
    ts = [s[n] for n in s.names() if not only_matrices or s[n].dim() == 2]
    return sum(int(torch.count_nonzero(t)) for t in ts) / sum(t.numel() for t in ts)


def edit_and_score(build: Callable, evaluate: Callable, ops: Iterable[dict], **eval_kw) -> float:
    """Build a fresh circuit, apply ``ops`` (list of {op: name, **params}) and
    return its task accuracy."""
    net = build()
    store = store_from_net(net)
    for step in ops:
        step = dict(step)
        seed = step.pop("seed", 0)
        apply_op(store, step.pop("op"), seed=seed, **step)
    return evaluate(net_from_store(net, store), **eval_kw)
