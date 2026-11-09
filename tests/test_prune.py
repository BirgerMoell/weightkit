import pytest
import torch

from weightkit import Calibration, TensorStore, apply_op, collect_input_stats
from weightkit.ops.prune import sparsegpt
from weightkit.testing import random_model, to_store


def frac_zero(t):
    return (t == 0).float().mean().item()


def test_magnitude_layer_sparsity_removes_smallest():
    s = to_store(random_model(0))
    n = "model.layers.0.mlp.up_proj.weight"
    w = s[n].clone()
    apply_op(s, "prune", method="magnitude", sparsity=0.5)
    out = s[n]
    assert abs(frac_zero(out) - 0.5) < 0.01
    thresh = w.abs().flatten().sort().values[w.numel() // 2]
    assert (out[out != 0].abs() >= thresh - 1e-8).all()


def test_row_granularity_exact_per_row():
    s = to_store(random_model(0))
    apply_op(s, "prune", method="magnitude", sparsity=0.25, granularity="row")
    w = s["model.layers.1.self_attn.q_proj.weight"]
    assert ((w == 0).sum(1) == int(w.shape[1] * 0.25)).all()


def test_nm_pattern():
    s = to_store(random_model(0))
    apply_op(s, "prune", method="magnitude", nm=[2, 4], roles=["mlp"])
    w = s["model.layers.0.mlp.gate_proj.weight"]
    groups = w.reshape(w.shape[0], -1, 4)
    assert ((groups != 0).sum(-1) == 2).all()
    # attention untouched
    assert frac_zero(s["model.layers.0.self_attn.q_proj.weight"]) == 0


def test_wanda_protects_high_activation_channels():
    torch.manual_seed(0)
    W = torch.randn(8, 16)
    store = TensorStore.from_dict({"w": W.clone()})
    c = Calibration()
    x = torch.randn(100, 16)
    x[:, 3] *= 1000  # huge activations on input channel 3
    c.update("w", x)
    apply_op(store, "prune", calib=c, method="wanda", sparsity=0.5, include=["w"])
    assert (store["w"][:, 3] != 0).all()
    assert ((store["w"] == 0).sum(1) == 8).all()


def test_wanda_needs_calibration():
    s = to_store(random_model(0))
    with pytest.raises(ValueError, match="calibration"):
        apply_op(s, "prune", method="wanda")


def test_sparsegpt_beats_magnitude_on_correlated_inputs():
    torch.manual_seed(0)
    d_in, d_out, n = 64, 32, 2000
    # strongly correlated (low-rank + noise) inputs: where OBS compensation pays off.
    # (with white inputs there is nothing to compensate and it ties magnitude pruning)
    X = torch.randn(n, 8) @ torch.randn(8, d_in) + 0.1 * torch.randn(n, d_in)
    W = torch.randn(d_out, d_in)
    H = X.T @ X / n
    ws = sparsegpt(W, H, 0.5)
    assert abs(frac_zero(ws) - 0.5) < 0.02
    wm = W * (W.abs() >= W.abs().flatten().sort().values[W.numel() // 2])
    err = lambda w: ((W - w) @ X.T).norm().item()
    assert err(ws) < 0.5 * err(wm)


def test_sparsegpt_nm_and_op():
    m = random_model(0)
    s = to_store(m)
    ids = torch.randint(0, 64, (4, 16))
    names = [n for n in s.names() if "_proj" in n]
    calib = collect_input_stats(m, [ids], names, xtx=True)
    apply_op(s, "prune", calib=calib, method="sparsegpt", nm=[2, 4], blocksize=16, roles=["down"])
    w = s["model.layers.0.mlp.down_proj.weight"]
    assert ((w.reshape(w.shape[0], -1, 4) != 0).sum(-1) <= 2).all()
