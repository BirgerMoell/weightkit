import torch

from weightkit import Calibration, apply_op, collect_input_stats
from weightkit.testing import from_store, random_model, to_store

from conftest import logits, make

ATOL = 2e-4


def check_preserved(op, ids, seed=0, cfg=None, **params):
    cfg = cfg or {}
    m = random_model(seed, **cfg)
    s = to_store(m)
    before = logits(s, ids)
    apply_op(s, op, **params)
    after = logits(s, ids)
    assert torch.allclose(before, after, atol=ATOL), (before - after).abs().max()
    return s


def test_fold_norm(ids):
    s = check_preserved("fold_norm", ids)
    assert torch.allclose(s["model.layers.0.input_layernorm.weight"], torch.ones(32))
    assert torch.allclose(s["model.norm.weight"], torch.ones(32))


def test_rotate_hadamard_and_random(ids):
    for kind in ("hadamard", "random"):
        m = random_model(1)
        s = to_store(m)
        w0 = s["model.layers.0.self_attn.q_proj.weight"].clone()
        before = logits(s, ids)
        apply_op(s, "rotate", kind=kind, seed=3)
        assert torch.allclose(before, logits(s, ids), atol=ATOL)
        assert not torch.allclose(w0, s["model.layers.0.self_attn.q_proj.weight"])


def test_rotate_non_power_of_two(ids):
    check_preserved("rotate", ids, cfg=dict(hidden_size=24, num_attention_heads=4, num_key_value_heads=2), kind="hadamard")


def test_rotate_with_biases(ids):
    check_preserved("rotate", ids, cfg=dict(mlp_bias=True, attention_bias=True), kind="random")


def test_permute_mlp_and_heads(ids):
    s = check_preserved("permute", ids, seed=2)
    m = random_model(2)
    assert not torch.equal(s["model.layers.0.mlp.up_proj.weight"], to_store(m)["model.layers.0.mlp.up_proj.weight"])


def test_permute_with_attention_bias(ids):
    check_preserved("permute", ids, cfg=dict(attention_bias=True, mlp_bias=True, num_key_value_heads=2))


def test_rescale_modes(ids):
    for mode in ("equalize", "random"):
        check_preserved("rescale", ids, mode=mode, spread=1.0)


def test_rescale_equalize_balances_norms(ids):
    m = random_model(0)
    s = to_store(m)
    apply_op(s, "rescale", mode="equalize", target=["mlp"])
    u, d = s["model.layers.0.mlp.up_proj.weight"], s["model.layers.0.mlp.down_proj.weight"]
    assert torch.allclose(u.norm(dim=1), d.norm(dim=0), rtol=1e-3)


def test_smooth_scale(ids):
    m = random_model(0)
    s = to_store(m)
    names = [n for n in s.names() if "q_proj" in n or "gate_proj" in n]
    calib = collect_input_stats(m, [ids], names)
    before = logits(s, ids)
    apply_op(s, "smooth_scale", calib=calib, alpha=0.5)
    assert torch.allclose(before, logits(s, ids), atol=ATOL)
    assert not torch.allclose(s["model.layers.0.input_layernorm.weight"], to_store(m)["model.layers.0.input_layernorm.weight"])


def test_align_recovers_permutation(ids):
    m = random_model(0)
    ref = to_store(m)
    shuffled = to_store(m)
    apply_op(shuffled, "permute", heads=False, seed=5)
    assert not torch.equal(shuffled["model.layers.1.mlp.up_proj.weight"], ref["model.layers.1.mlp.up_proj.weight"])
    apply_op(ref, "align", models={"other": shuffled}, model="other")
    for n in ref.names():
        if "mlp" in n:
            assert torch.equal(shuffled[n], ref[n]), n
    assert torch.allclose(logits(shuffled, ids), logits(ref, ids), atol=ATOL)
