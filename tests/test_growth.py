import pytest
import torch

from weightkit import apply_op
from weightkit.testing import from_store, random_model, to_store

from conftest import logits

ATOL = 2e-4


@pytest.mark.parametrize("cfg", [dict(), dict(mlp_bias=True)])
@pytest.mark.parametrize("mode", ["zero", "split"])
def test_widen_mlp_is_exact(ids, mode, cfg):
    s = to_store(random_model(0, **cfg))
    before = logits(s, ids)
    apply_op(s, "widen_mlp", extra=20, mode=mode, seed=1)
    assert s.config["intermediate_size"] == 68
    assert s["model.layers.0.mlp.up_proj.weight"].shape == (68, 32)
    assert s["model.layers.0.mlp.down_proj.weight"].shape == (32, 68)
    assert torch.allclose(before, logits(s, ids), atol=ATOL)


def test_widen_mlp_zero_mode_inert_columns_and_ratio(ids):
    s = to_store(random_model(0))
    apply_op(s, "widen_mlp", ratio=0.5)
    assert s.config["intermediate_size"] == 72
    assert (s["model.layers.1.mlp.down_proj.weight"][:, 48:] == 0).all()
    assert s["model.layers.1.mlp.gate_proj.weight"][48:].abs().max() > 0  # random, ready to train


def test_widen_mlp_split_halves_down_columns(ids):
    s = to_store(random_model(0))
    d0 = s["model.layers.0.mlp.down_proj.weight"].clone()
    apply_op(s, "widen_mlp", extra=10, mode="split", seed=2)
    d = s["model.layers.0.mlp.down_proj.weight"]
    # every original column + its copies must sum back to the original column
    total = d[:, :48].clone()
    for j in range(48, 58):
        # find which original it duplicates
        src = (s["model.layers.0.mlp.up_proj.weight"][:48] == s["model.layers.0.mlp.up_proj.weight"][j]).all(1).nonzero()[0, 0]
        total[:, src] += d[:, j]
    assert torch.allclose(total, d0, atol=1e-6)


def test_widen_mlp_errors():
    s = to_store(random_model(0))
    with pytest.raises(ValueError):
        apply_op(s, "widen_mlp")
    with pytest.raises(ValueError, match="at most"):
        apply_op(s, "widen_mlp", extra=100, mode="split")


@pytest.mark.parametrize("kw,n_new", [(dict(after=[0]), 1), (dict(after="0,2"), 2), (dict(count=2), 2)])
def test_add_layers_is_exact_identity(ids, kw, n_new):
    s = to_store(random_model(0, attention_bias=True, mlp_bias=True))
    before = logits(s, ids)
    apply_op(s, "add_layers", **kw)
    assert s.config["num_hidden_layers"] == 3 + n_new
    assert torch.allclose(before, logits(s, ids), atol=ATOL)


def test_add_layers_new_layer_is_zero_output_copy_of_neighbour():
    s = to_store(random_model(0))
    q1 = s["model.layers.1.self_attn.q_proj.weight"].clone()
    apply_op(s, "add_layers", after=[1])
    assert torch.equal(s["model.layers.2.self_attn.q_proj.weight"], q1)          # copy of layer 1
    assert (s["model.layers.2.self_attn.o_proj.weight"] == 0).all()               # writes nothing
    assert (s["model.layers.2.mlp.down_proj.weight"] == 0).all()
    assert (s["model.layers.1.mlp.down_proj.weight"] != 0).any()                  # original untouched
    assert (s["model.layers.3.mlp.down_proj.weight"] != 0).any()                  # old layer 2 shifted, untouched


def test_add_layers_updates_layer_types_and_validates():
    s = to_store(random_model(0, layer_types=["a", "b", "c"]))
    apply_op(s, "add_layers", after=[0])
    assert s.config["layer_types"] == ["a", "a", "b", "c"]
    with pytest.raises(ValueError):
        apply_op(s, "add_layers")
    with pytest.raises(ValueError):
        apply_op(s, "add_layers", count=99)


def test_growth_then_prune_roundtrip_shapes(ids):
    """widen then prune back to the original width: still loads and runs."""
    s = to_store(random_model(0))
    apply_op(s, "widen_mlp", extra=16)
    apply_op(s, "prune_mlp", keep=48)
    assert s.config["intermediate_size"] == 48 and logits(s, ids).shape == (3, 10, 64)
