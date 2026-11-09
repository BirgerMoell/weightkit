# weightkit

**[→ Live site & interactive playground](https://birgermoell.github.io/weightkit/)**

**One toolkit for every way of changing a network's weights directly** — no gradient descent, no training loop.

[mergekit](https://github.com/arcee-ai/mergekit) is excellent at one slice of this: *combining several models*. But "change the weights directly" is a much bigger family, and the rest of it is scattered across one-off research repos with incompatible conventions: abliteration scripts, ROME/MEMIT, Wanda/SparseGPT, QuaRot/SmoothQuant, LoRA tooling, layer surgery... `weightkit` puts them all behind a single abstraction so they **compose in one pipeline** and share one tensor store, one architecture description, one calibration format and one test harness.

```yaml
base: ./Llama-3-8B
calibration: ./calib.pt
output: ./out
steps:
  - op: merge                 # 1. combine fine-tunes (TIES + DARE)
    method: dare_ties
    density: 0.5
    sources: [{model: math, weight: 0.6}, {model: code, weight: 0.4}]
  - op: abliterate            # 2. remove a behaviour direction
    pos: harmful
    neg: harmless
    layer: 14
  - op: rotate                # 3. function-preserving outlier smoothing
  - op: prune                 # 4. 2:4 sparsity with second-order updates
    method: sparsegpt
    nm: [2, 4]
```

## What it adds over mergekit

| | mergekit | weightkit |
|---|---|---|
| Merging (linear, SLERP, task-arith, TIES, DARE, breadcrumbs, model-stock) | ✅ | ✅ (same math, as an `Op`) |
| Task-vector negation / extrapolation, LoRA merge, **LoRA extraction** | partial | ✅ |
| **Pruning**: magnitude, Wanda, SparseGPT, N:M | – | ✅ |
| **Structured** pruning: layers, MLP neurons, KV-head groups | – | ✅ |
| **Knowledge editing**: ROME / MEMIT closed-form updates | – | ✅ |
| **Abliteration** (directional ablation) and weight-baked **steering** | – | ✅ |
| **Function-preserving transforms**: norm folding, Hadamard/random rotation, permutation, rescaling, SmoothQuant scaling, Re-Basin alignment | – | ✅ |
| **Spectral surgery**: SVD truncation, energy, power, clipping | – | ✅ |
| Layer stacking / removal, KV-pooling (MHA→GQA), vocab resize, tie/untie | partial | ✅ |
| Weight-level quantization (RTN) | – | ✅ |
| Compare any two checkpoints (`weightkit diff`) | – | ✅ |

Design differences, not just more features:

* **One `Op` abstraction.** Every method is a registered class with typed parameters (`weightkit ops -v`). Adding a method is ~20 lines (see below).
* **Single-model and multi-model ops are the same thing**, so "merge, then prune, then edit" is a pipeline, not three tools.
* **Function-preserving ops are first-class** and *tested to be function-preserving* (logits equal before/after, including on a real Hugging Face `LlamaForCausalLM`).
* **Lazy, spillable tensor store**: tensors load on demand; edits live in an overlay that spills to temp safetensors past a memory budget (`spill_gb`), so whole-model ops don't need the whole model in RAM.
* **One calibration file** (`weightkit calibrate`) feeds Wanda, SparseGPT, MEMIT, SmoothQuant scaling and abliteration.

## Install

```bash
pip install -e .            # torch, safetensors, pyyaml
pip install -e ".[hf,fast]" # + transformers (calibrate CLI) + scipy (exact Hungarian matching)
```

## Operations

`weightkit ops -v` prints the live list with parameters. Currently 27 ops:

| family | ops |
|---|---|
| **merge** | `merge` — `linear`, `slerp`, `model_stock`, `task_arithmetic`, `ties`, `dare_linear`, `dare_ties`, `breadcrumbs`, `breadcrumbs_ties` |
| **delta** | `task_vector` (negate / extrapolate / interpolate), `lora_merge`; `weightkit extract-lora` |
| **spectral** | `spectral` — `truncate`, `energy`, `power`, `clip`, `scale_top` |
| **prune** | `prune` — `magnitude`, `wanda`, `sparsegpt`; granularity layer/row/column/**global**; `nm: [n, m]` |
| **edit** | `edit_facts` (MEMIT / ROME), `abliterate`, `steer_bias` |
| **grow** (exact) | `widen_mlp` (Net2Net: inert or split neurons), `add_layers` (identity decoder layers) |
| **transform** (exact) | `fold_norm`, `rotate`, `permute`, `rescale`, `smooth_scale`, `align` |
| **surgery** | `layer_stack`, `drop_layers`, `prune_mlp`, `prune_heads`, `widen_mlp`, `add_layers`, `kv_pool`, `untie_embeddings`, `tie_embeddings`, `resize_vocab`, `perturb`, `cast` |
| **quant** | `quantize_rtn` |

Most ops accept a *selection*: `roles` (`q k v o gate up down attn mlp all_linear embed lm_head ...`), `layers` (`"0-7,-1"`), `include`/`exclude` globs on tensor names.

## Usage

```bash
weightkit run pipeline.yaml                       # full pipeline
weightkit apply ./model ./out prune --params '{"method":"magnitude","sparsity":0.5}'
weightkit diff ./model ./out                      # what actually changed?
weightkit inspect ./out --rank                    # size, zeros, outliers, provenance
weightkit extract-lora ./base ./finetune ./adapter --rank 32
weightkit calibrate ./model calib.pt --text corpus.txt --xtx \
    --hidden harmful=harmful.txt --hidden harmless=harmless.txt --hidden-layers 14
```

Python:

```python
import torch
from weightkit import TensorStore, apply_op, Calibration

store = TensorStore.from_path("./Llama-3-8B")          # lazy
calib = Calibration.load("calib.pt")
apply_op(store, "prune", calib=calib, method="wanda", sparsity=0.5)
apply_op(store, "rotate", kind="hadamard")
store.save("./out", dtype=torch.bfloat16)
```

See [`examples/`](examples) for more pipelines.

### Editing facts (`edit_facts`)

`edit_facts` is the closed-form linear-algebra half of ROME/MEMIT: given keys `K` (the layer's input at the subject token) and desired outputs `V`, it solves `W' = W + (V − WK)Kᵀ(λC + KKᵀ)⁻¹`, with `C` the key covariance from calibration. Get the keys with `weightkit.calibration.capture_inputs(model, batch, "model.layers.8.mlp.down_proj", positions)`. **You supply the target values `V`.** The causal-tracing and gradient-based search for `v*` from the ROME paper is *not* implemented.

### Adding an op

```python
from weightkit import Op, register
from weightkit.base import Selection

@register("my_op", "custom")
class MyOp(Op):
    """One-line summary shown in `weightkit ops`."""
    def __init__(self, factor: float = 2.0, **sel):
        self.factor, self.sel = factor, Selection.from_kwargs(sel)

    def run(self, ctx):
        for name in ctx.select(self.sel):
            ctx.write(name, ctx.get(name) * self.factor)
```

## Provenance, inspection and data accounting

```bash
weightkit inspect ./model --rank      # params, zeros, outlier channels, effective rank, provenance
```

Every checkpoint saved after ops gets a `weightkit_manifest.json`: each op with its parameters, the number of tensors it touched, and **what outside information it used** — `none`, `calibration` (with the token count), `supplied` (vectors/edits you provided) or `other_models` (merge sources, adapters). Manifests chain through generations, so "edited with 0 data" is checkable rather than a claim. (Idea borrowed from [handnet](https://github.com/BirgerMoell/handnet)'s "labels seen" accounting.)

`inspect` is also how you catch ordering mistakes: `prune` followed by `rotate` leaves **0% zeros**, because the rotation re-densifies the weights. Rotate first, then prune.

## Verify: the weights are the spec

```yaml
base: ./Llama-3-8B
output: ./out
steps:
  - {op: widen_mlp, extra: 1024, mode: split}   # exact growth
  - {op: add_layers, after: [7, 15, 23]}        # exact depth up-scaling
  - {op: rotate}
verify:                                         # runs on the saved checkpoint; failure => exit 1
  - {type: equivalence, tol: 1.0e-4}            # logits match the base on random probes
  - {type: config, expect: {num_hidden_layers: 35}}
  - {type: data_use, allowed: [none]}           # edited with no outside data
  - {type: ops, contains: [widen_mlp, rotate]}
```

or `weightkit verify spec.yaml` on any saved model. Checks: `finite`, `equivalence` (max logit difference + top-1 agreement, via transformers by default or your own `runner`), `sparsity`, `untouched`, `changed`, `max_rel_change`, `direction_removed`, `config`, `data_use`, `ops`. A check that cannot run (e.g. no reference given) fails with the reason rather than passing silently.

`widen_mlp` and `add_layers` are the exact duals of `prune_mlp` / `drop_layers`: they add capacity (inert neurons, identity layers with `o_proj`/`down_proj` zeroed) without changing the function. Verified exact on a real HF Llama: a grown, deepened and rotated model differs from the original by ~6e-7 in logits.

## Circuit lab: ground-truth testing with hand-written networks

[handnet](https://github.com/BirgerMoell/handnet) contains transformers whose weights were *written by hand* (addition with carry, Dyck-1 brackets, induction), exactly 100% accurate with only 1–14% non-zero weights. That gives weight-editing ops something rare: a known right answer. `weightkit.circuits` bridges them into a `TensorStore`; `scripts/circuit_lab.py` measures the curves shown on the [site](https://birgermoell.github.io/weightkit/#circuits).

What this established (all asserted in `tests/test_circuits.py`):

* **Global magnitude pruning at the designed sparsity recovers the hand-written weights exactly** (identical support *and* values) on all three circuits; one percent further collapses them. Layer-wise and row-wise pruning at the same sparsity do not (8–48% accuracy), since designed weights are unevenly distributed. This is why `prune` has `granularity: global`.
* Induction survives even 2-bit per-row RTN (its rows hold ≤ 2 distinct values); Dyck survives 8-bit but not 6; addition does not survive 8-bit.
* Tiny noise (0.1% of each tensor's std) is harmless; 20% is destructive.
* It found two real bugs in weightkit that the logit-based tests had missed (`perturb` NaN on one-element tensors; `spectral` power-mode NaN on all-zero matrices). Both fixed with regression tests.

```bash
pip install git+https://github.com/BirgerMoell/handnet   # optional, numpy-only
pytest tests/test_circuits.py
```

## Testing & what is verified

`pytest` runs 129 tests (109 without the optional handnet fixtures). Beyond shape checks, they assert behaviour:

* every function-preserving op (`fold_norm`, `rotate` incl. non-power-of-two and biased models, `permute`, `rescale`, `smooth_scale`, `align`, `kv_pool` on duplicated heads, tie/untie, no-op `prune_*`) leaves **logits unchanged**, on a pure-torch tiny Llama *and* a real `transformers.LlamaForCausalLM`;
* `abliterate` makes the residual stream's component along `r` ≈ 0 at the embedding and after every layer;
* `edit_facts` hits its target outputs in a real forward pass (ROME exactly, rank-one);
* `steer_bias` shifts the residual stream by exactly `scale·v`;
* SparseGPT beats magnitude pruning on correlated inputs; N:M patterns are exact; Wanda protects high-activation channels;
* SVD truncation matches Eckart–Young; extract→merge LoRA round-trips; DARE is unbiased; TIES elects signs correctly;
* sharded save/load, memory spilling and pipelines round-trip.

## Limitations (please read)

* **Architectures:** tensor naming follows the Llama family (Llama, Mistral, Qwen2/3, Yi, SmolLM, …) with a Gemma norm-offset preset (untested). Custom naming works via `arch:` in a pipeline / `ArchSpec`. **Not supported yet:** fused QKV (Phi-3), MoE experts, Gemma-2/3 extra norms, non-RMSNorm models (rotation/folding need RMSNorm), `.bin`-only checkpoints.
* **Verified on tiny models** (and a tiny HF Llama), not yet run against multi-billion-parameter checkpoints. The store is designed to scale (lazy + spill), but whole-model ops such as `rotate` still hold the rewritten tensors until they spill.
* Heuristic ops (`prune_mlp`, `prune_heads`, `kv_pool`, `spectral`, `quantize_rtn`) change function; expect quality loss and consider brief fine-tuning. `quantize_rtn` is fake-quant (dequantized weights), not a packed GPTQ/AWQ export.
* `steer_bias` relies on `mlp_bias` support in the target architecture (HF Llama has it).
* Nothing here evaluates model quality — pair it with your own eval harness.

## Roadmap

GPTQ/AWQ-style calibrated quantization, `graft` (write a hand-designed circuit into grown capacity), automatic `v*` search for edits, MoE and fused-QKV architectures, DELLA/SCE/Arcee-fusion style merges, multi-layer MEMIT spreading, evolutionary parameter search over pipelines.

## Credits

Clean-room implementation of published methods; no mergekit code is used. Methods: Task Arithmetic (Ilharco et al.), TIES (Yadav et al.), DARE (Yu et al.), Model Stock (Jang et al.), Model Breadcrumbs (Davari & Belilovsky), Git Re-Basin (Ainsworth et al.), Wanda (Sun et al.), SparseGPT (Frantar & Alistarh), ROME (Meng et al.), MEMIT (Meng et al.), refusal-direction ablation (Arditi et al.), SmoothQuant (Xiao et al.), QuaRot (Ashkboos et al.), GQA (Ainslie et al.). Thanks to the mergekit authors for the inspiration.

License: Apache-2.0
