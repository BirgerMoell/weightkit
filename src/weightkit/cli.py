"""weightkit command line interface."""
from __future__ import annotations

import argparse
import json
import sys

import torch
import yaml

from . import __version__


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def cmd_ops(args) -> int:
    from .base import list_ops

    cat = None
    for name, cls in list_ops().items():
        if cls.category != cat:
            cat = cls.category
            print(f"\n[{cat}]")
        print(f"  {name:<18} {cls.describe()}")
        if args.verbose:
            print(f"  {'':<18} params: {cls.signature()}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import run_pipeline

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.output:
        cfg["output"] = args.output
    run_pipeline(cfg, log=_log)
    return 0


def cmd_apply(args) -> int:
    """Single-op shorthand:  weightkit apply MODEL OUT OP --params '{"sparsity": 0.5}'"""
    from .pipeline import run_pipeline

    params = json.loads(args.params) if args.params else {}
    cfg = dict(base=args.model, output=args.output, steps=[dict(op=args.op, **params)])
    if args.calibration:
        cfg["calibration"] = args.calibration
    for kv in args.model_ref or []:
        name, path = kv.split("=", 1)
        cfg.setdefault("models", {})[name] = path
    run_pipeline(cfg, log=_log)
    return 0


def cmd_diff(args) -> int:
    from .diff import compare, summarize
    from .store import TensorStore

    print(summarize(compare(TensorStore.from_path(args.a), TensorStore.from_path(args.b)), args.top))
    return 0


def cmd_extract_lora(args) -> int:
    from .ops.delta import extract_lora
    from .store import TensorStore

    errs = extract_lora(
        TensorStore.from_path(args.base), TensorStore.from_path(args.finetuned), args.rank, args.output, device=args.device
    )
    worst = max(errs.items(), key=lambda kv: kv[1])
    print(f"extracted {len(errs)} modules at rank {args.rank}; mean rel.error {sum(errs.values())/len(errs):.3f}, worst {worst[0]} {worst[1]:.3f}")
    return 0


def cmd_calibrate(args) -> int:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("calibrate needs `pip install weightkit[hf]` (transformers)", file=sys.stderr)
        return 2
    from .arch import detect_arch
    from .calibration import Calibration, collect_hidden_mean, collect_input_stats
    from .store import TensorStore

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
    arch = detect_arch(TensorStore.from_path(args.model))
    calib = Calibration()
    if args.text:
        ids = tok(open(args.text).read(), return_tensors="pt").input_ids[0]
        chunks = [ids[i : i + args.seq_len] for i in range(0, len(ids) - args.seq_len + 1, args.seq_len)][: args.max_samples]
        if not chunks:
            raise SystemExit("text is shorter than --seq-len")
        names = [n for n, p in model.named_parameters() if n.endswith(".weight") and arch.is_layer_tensor(n) and p.dim() == 2]
        batches = ({"input_ids": c[None].to(args.device)} for c in chunks)
        collect_input_stats(model, batches, names, xtx=args.xtx, calib=calib)
        _log(f"collected input statistics for {len(names)} tensors from {len(chunks)} sequences")
    for spec in args.hidden or []:
        tag, path = spec.split("=", 1)
        prompts = [l.rstrip("\n") for l in open(path) if l.strip()]
        mods = [arch.layer_module(int(i)) for i in args.hidden_layers.split(",")]
        batches = ({"input_ids": tok(p, return_tensors="pt").input_ids.to(args.device)} for p in prompts)
        calib.hidden[tag] = collect_hidden_mean(model, batches, mods)
        _log(f"collected hidden means for tag {tag!r} ({len(prompts)} prompts, layers {args.hidden_layers})")
    calib.save(args.output)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="weightkit", description="Direct weight-editing toolkit for neural networks.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ops", help="list available operations")
    s.add_argument("-v", "--verbose", action="store_true", help="show parameters")
    s.set_defaults(fn=cmd_ops)

    s = sub.add_parser("run", help="run a YAML pipeline")
    s.add_argument("config")
    s.add_argument("-o", "--output", help="override output dir")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("apply", help="apply a single op")
    s.add_argument("model")
    s.add_argument("output")
    s.add_argument("op")
    s.add_argument("--params", help="JSON object of op parameters")
    s.add_argument("--calibration")
    s.add_argument("--model-ref", action="append", metavar="NAME=PATH", help="extra named model")
    s.set_defaults(fn=cmd_apply)

    s = sub.add_parser("diff", help="compare two checkpoints tensor-by-tensor")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("extract-lora", help="SVD-extract a LoRA adapter from finetuned - base")
    s.add_argument("base")
    s.add_argument("finetuned")
    s.add_argument("output")
    s.add_argument("--rank", type=int, default=32)
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=cmd_extract_lora)

    s = sub.add_parser("calibrate", help="collect activation statistics (needs transformers)")
    s.add_argument("model")
    s.add_argument("output")
    s.add_argument("--text", help="plain-text file for input statistics (Wanda/SparseGPT/MEMIT/smooth_scale)")
    s.add_argument("--xtx", action="store_true", help="also collect full input covariance (SparseGPT/MEMIT; memory heavy)")
    s.add_argument("--hidden", action="append", metavar="TAG=FILE", help="prompts (one per line) whose last-token hidden mean to record")
    s.add_argument("--hidden-layers", default="0", help="comma-separated decoder layers for --hidden")
    s.add_argument("--seq-len", type=int, default=512)
    s.add_argument("--max-samples", type=int, default=128)
    s.add_argument("--device", default="cpu")
    s.add_argument("--dtype", default="float32")
    s.set_defaults(fn=cmd_calibrate)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
