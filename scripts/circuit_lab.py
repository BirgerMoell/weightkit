"""Measure how weightkit ops affect handnet's hand-written circuits.

    pip install git+https://github.com/BirgerMoell/handnet
    python scripts/circuit_lab.py            # writes docs/circuit_lab.json and docs/circuit_lab.js

Every number on the site's "Circuit lab" section comes from this script.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from handnet.transformer import TASKS  # noqa: E402

from weightkit.circuits import edit_and_score, support_fraction  # noqa: E402

N_EVAL = int(os.environ.get("N_EVAL", 500))
SEEDS = 5
KEPT = [1.5, 1.0, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]   # fraction of the DESIGNED weights kept
BITS = [8, 7, 6, 5, 4, 3, 2]
STDS = [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def main():
    out = {"n_eval": N_EVAL, "noise_seeds": SEEDS, "tasks": {}}
    for task, (build, ev) in TASKS.items():
        net = build()
        sf = support_fraction(net)
        score = lambda ops, **kw: float(edit_and_score(build, ev, ops, n=N_EVAL, **kw))
        prune = lambda sp, g: score([dict(op="prune", method="magnitude", sparsity=sp, granularity=g, include=["*"])])
        r = dict(params=net.n_params, nonzero=net.n_nonzero, support_pct=100 * sf, baseline=score([]))
        r["prune"] = dict(
            kept_of_design=KEPT,
            **{g: [prune(1 - min(k * sf, 1.0), g) for k in KEPT] for g in ("global", "layer")},
        )
        r["rtn"] = dict(bits=BITS, acc=[score([dict(op="quantize_rtn", bits=b, group_size=0, include=["*"])]) for b in BITS])
        noise = [[score([dict(op="perturb", std=s, seed=k, include=["*"])]) for k in range(SEEDS)] for s in STDS]
        r["noise"] = dict(std=STDS, mean=[float(np.mean(x)) for x in noise], min=[min(x) for x in noise], max=[max(x) for x in noise])
        out["tasks"][task] = r
        print(f"{task}: support {100*sf:.2f}%  global@design={r['prune']['global'][1]:.3f}  layer@design={r['prune']['layer'][1]:.3f}", file=sys.stderr)
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    with open(os.path.join(here, "circuit_lab.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(here, "circuit_lab.js"), "w") as f:
        f.write("window.CIRCUIT_LAB=" + json.dumps(out) + ";\n")


if __name__ == "__main__":
    main()
