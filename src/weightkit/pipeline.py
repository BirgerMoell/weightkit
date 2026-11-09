"""YAML / dict pipelines: load models, run ops in order, save."""
from __future__ import annotations

import os
from typing import Callable, Union

import torch
import yaml

from . import ops  # noqa: F401  (register built-ins)
from .arch import arch_from_config, detect_arch
from .base import Context, get_op
from .calibration import Calibration
from .store import TensorStore

_KNOWN = {
    "base", "models", "calibration", "arch", "dtype", "device", "seed", "steps",
    "output", "save_dtype", "max_shard_size_gb", "spill_gb",
}


def _dtype(name):
    if name is None or isinstance(name, torch.dtype):
        return name
    return getattr(torch, name)


def run_pipeline(config: Union[str, dict], log: Callable[[str], None] = print) -> TensorStore:
    """Execute a pipeline config (path to YAML or a dict).  See README."""
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)
    unknown = set(config) - _KNOWN
    if unknown:
        raise ValueError(f"unknown top-level key(s): {', '.join(sorted(unknown))}")
    spill = int(float(config.get("spill_gb", 4)) * (1 << 30))
    state = TensorStore.from_path(config["base"], spill_bytes=spill)
    models = {k: TensorStore.from_path(v, spill_bytes=spill) for k, v in (config.get("models") or {}).items()}
    calib = Calibration.load(config["calibration"]) if config.get("calibration") else None
    arch = arch_from_config(config.get("arch")) or detect_arch(state)
    ctx = Context(
        state=state,
        arch=arch,
        models=models,
        calib=calib,
        device=config.get("device", "cpu"),
        dtype=_dtype(config.get("dtype")) or torch.float32,
        seed=int(config.get("seed", 0)),
        log=log,
    )
    state.manifest_extra = {
        "models": {k: os.path.abspath(v) if os.path.exists(str(v)) else v for k, v in (config.get("models") or {}).items()},
        "calibration_file": config.get("calibration"),
    }
    steps = config.get("steps") or []
    for n, step in enumerate(steps, 1):
        step = dict(step)
        name = step.pop("op", None)
        if name is None:
            raise ValueError(f"step {n} has no `op`")
        log(f"[{n}/{len(steps)}] {name}")
        try:
            ctx.run_op(get_op(name)(**step), step)
        except TypeError as e:
            raise ValueError(f"step {n} ({name}): bad parameters: {e}") from e
    if config.get("output"):
        state.save(
            config["output"],
            max_shard_size=int(float(config.get("max_shard_size_gb", 5)) * (1 << 30)),
            dtype=_dtype(config.get("save_dtype")),
        )
        log(f"saved -> {config['output']}")
    return state
