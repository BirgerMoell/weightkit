"""Ground-truth tests on handnet's hand-written transformers (exact 100% circuits).

Skipped unless handnet is installed:  pip install git+https://github.com/BirgerMoell/handnet
"""
import pytest
import torch

pytest.importorskip("handnet.transformer")
from handnet.transformer import TASKS  # noqa: E402

from weightkit import apply_op  # noqa: E402
from weightkit.circuits import edit_and_score, net_from_store, store_from_net, support_fraction  # noqa: E402

N = 200
TASK_NAMES = list(TASKS)


@pytest.fixture(params=TASK_NAMES)
def task(request):
    build, ev = TASKS[request.param]
    return request.param, build, ev


def test_roundtrip_is_exact(task):
    name, build, ev = task
    net = build()
    store = store_from_net(net)
    assert ev(net, n=N) == 1.0 and ev(net_from_store(net, store), n=N) == 1.0
    assert any(store[n].dim() == 2 for n in store.names())


def test_global_magnitude_prune_recovers_the_design_exactly(task):
    """Prune to the designed sparsity: the surviving weights ARE the hand-written ones."""
    name, build, ev = task
    net = build()
    orig = store_from_net(net)
    pruned = store_from_net(net)
    apply_op(pruned, "prune", method="magnitude", sparsity=1 - support_fraction(net), granularity="global", include=["*"])
    for n in orig.names():
        if orig[n].dim() == 2:
            assert torch.equal(pruned[n] != 0, orig[n] != 0), n
            assert torch.equal(pruned[n], orig[n]), n
    assert ev(net_from_store(net, pruned), n=N) == 1.0


def test_one_percent_past_the_design_collapses_the_circuit(task):
    name, build, ev = task
    sp = 1 - support_fraction(build()) + 0.01
    acc = edit_and_score(build, ev, [dict(op="prune", method="magnitude", sparsity=sp, granularity="global", include=["*"])], n=N)
    assert acc < 0.6, (name, acc)


def test_layerwise_and_rowwise_pruning_cannot_find_the_design(task):
    """Same sparsity, wrong granularity: designed weights are unevenly distributed."""
    name, build, ev = task
    sp = 1 - support_fraction(build())
    for g in ("layer", "row"):
        acc = edit_and_score(build, ev, [dict(op="prune", method="magnitude", sparsity=sp, granularity=g, include=["*"])], n=N)
        assert acc < 0.6, (name, g, acc)


def test_tiny_noise_is_harmless_large_noise_is_not(task):
    name, build, ev = task
    assert edit_and_score(build, ev, [dict(op="perturb", std=0.001, seed=1, include=["*"])], n=N) == 1.0
    assert edit_and_score(build, ev, [dict(op="perturb", std=0.2, seed=1, include=["*"])], n=N) < 0.6


def test_exact_roundtrip_ops_keep_100_percent(task):
    name, build, ev = task
    assert edit_and_score(build, ev, [dict(op="cast", dtype="float32")], n=N) == 1.0
    assert edit_and_score(build, ev, [dict(op="spectral", mode="power", power=1.0, include=["*"])], n=N) == 1.0
    assert edit_and_score(build, ev, [dict(op="task_vector", finetuned="self", base="self", scale=1.0)], n=N) == 1.0


def test_rtn_8bit_survives_dyck_and_induction_and_induction_survives_2bit():
    """Induction rows hold <=2 distinct values, so asymmetric per-row RTN is exact even at 2 bits."""
    rtn = lambda b: [dict(op="quantize_rtn", bits=b, group_size=0, include=["*"])]
    for t in ("dyck", "induction"):
        build, ev = TASKS[t]
        assert edit_and_score(build, ev, rtn(8), n=N) == 1.0
    build, ev = TASKS["induction"]
    assert edit_and_score(build, ev, rtn(2), n=N) == 1.0
    build, ev = TASKS["dyck"]
    assert edit_and_score(build, ev, rtn(6), n=N) < 0.6


def test_manifest_records_circuit_edits(tmp_path):
    build, ev = TASKS["dyck"]
    store = store_from_net(build())
    apply_op(store, "prune", method="magnitude", sparsity=0.5, granularity="global", include=["*"])
    assert store.ledger[0]["data"] == ["none"] and store.ledger[0]["tensors_touched"] > 0
