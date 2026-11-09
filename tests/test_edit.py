import pytest
import torch

from weightkit import Calibration, apply_op
from weightkit.calibration import capture_inputs, collect_hidden_mean
from weightkit.ops.edit import memit_update, rome_update
from weightkit.testing import from_store, random_model, to_store

from conftest import logits


def hidden_states(model, ids):
    outs = []
    hs = [l.register_forward_hook(lambda m, a, o: outs.append(o.detach())) for l in model.model.layers]
    emb = []
    hs.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: emb.append(o.detach())))
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    return emb[0], outs


def test_abliterate_removes_direction_everywhere(ids, tmp_path):
    m = random_model(0)
    s = to_store(m)
    r = torch.randn(32)
    r = r / r.norm()
    # sanity: before the edit the direction is present in the residual stream
    emb0, outs0 = hidden_states(m, ids)
    assert (outs0[-1] @ r).abs().max() > 0.1
    f = tmp_path / "r.pt"
    torch.save(r, f)
    apply_op(s, "abliterate", direction=str(f))
    emb, outs = hidden_states(from_store(s), ids)
    assert (emb @ r).abs().max() < 1e-5
    for o in outs:
        assert (o @ r).abs().max() < 1e-4
    # but the model still computes something sensible (not destroyed)
    assert (logits(s, ids) - logits(to_store(m), ids)).abs().max() > 0  # changed
    assert torch.isfinite(logits(s, ids)).all()


def test_abliterate_direction_from_calibration(ids):
    m = random_model(0)
    s = to_store(m)
    pos, neg = torch.randint(0, 64, (6, 5)), torch.randint(0, 64, (6, 5))
    c = Calibration()
    mod = "model.layers.1"
    c.hidden["harm"] = collect_hidden_mean(m, [pos], [mod])
    c.hidden["safe"] = collect_hidden_mean(m, [neg], [mod])
    r = c.hidden["harm"][mod] - c.hidden["safe"][mod]
    r = r / r.norm()
    apply_op(s, "abliterate", calib=c, pos="harm", neg="safe", layer=1)
    _, outs = hidden_states(from_store(s), ids)
    for o in outs:
        assert (o @ r).abs().max() < 1e-4


def test_abliterate_layer_subset_and_strength():
    s = to_store(random_model(0))
    w2 = s["model.layers.2.self_attn.o_proj.weight"].clone()
    apply_op(s, "abliterate", direction={"direction": torch.eye(32)[0]}, layers=[0, 1], strength=0.5)
    assert torch.equal(s["model.layers.2.self_attn.o_proj.weight"], w2)
    w0 = to_store(random_model(0))["model.layers.0.self_attn.o_proj.weight"]
    assert torch.allclose(s["model.layers.0.self_attn.o_proj.weight"][0], 0.5 * w0[0], atol=1e-6)


def test_abliterate_rejects_tied():
    s = to_store(random_model(0, tie_word_embeddings=True))
    with pytest.raises(ValueError, match="untie"):
        apply_op(s, "abliterate", direction=torch.randn(32))


def test_rome_exact_and_memit_close():
    torch.manual_seed(0)
    W = torch.randn(16, 40)
    k, v = torch.randn(40), torch.randn(16)
    C = torch.eye(40) * 2
    assert torch.allclose(rome_update(W, k, v, C) @ k, v, atol=1e-4)
    K, V = torch.randn(40, 5), torch.randn(16, 5)
    W2 = memit_update(W, K, V, C, lam=1e-6)
    assert torch.allclose(W2 @ K, V, atol=1e-3)
    # rank-one: edit has rank 1
    assert torch.linalg.matrix_rank(rome_update(W, k, v, C) - W) == 1


def test_edit_facts_end_to_end(ids):
    """Insert a key->value mapping into a real layer and observe it in the forward pass."""
    m = random_model(0)
    s = to_store(m)
    mod = "model.layers.1.mlp.down_proj"
    keys = capture_inputs(m, ids[:1], mod, [3, 6])  # [2, 48]
    values = torch.randn(2, 32)
    apply_op(s, "edit_facts", edits={"keys": keys, "values": values}, layer=1, lam=1e-6)
    m2 = from_store(s)
    got = capture_out = {}
    h = m2.get_submodule(mod).register_forward_hook(lambda mm, a, o: got.update(o=o.detach()))
    with torch.no_grad():
        # same input prefix -> identical keys at those positions up to layer 1
        m2(ids[:1])
    h.remove()
    assert torch.allclose(got["o"][0, [3, 6]], values, atol=5e-3)


def test_edit_facts_rome_mode_with_calibration_covariance():
    m = random_model(0)
    s = to_store(m)
    ids = torch.randint(0, 64, (8, 16))
    name = "model.layers.0.mlp.down_proj.weight"
    from weightkit import collect_input_stats

    calib = collect_input_stats(m, [ids], [name], xtx=True)
    k = torch.randn(1, 48)
    v = torch.randn(1, 32)
    W0 = s[name].clone()
    apply_op(s, "edit_facts", calib=calib, edits={"keys": k, "values": v}, layer=0, mode="rome")
    assert torch.allclose(s[name] @ k[0], v[0], atol=1e-3)
    assert torch.linalg.matrix_rank(s[name] - W0, atol=1e-4) == 1


def test_edit_shape_error():
    s = to_store(random_model(0))
    with pytest.raises(ValueError, match="do not fit"):
        apply_op(s, "edit_facts", edits={"keys": torch.randn(1, 10), "values": torch.randn(1, 32)}, layer=0)


def test_steer_bias_shifts_residual_exactly(ids):
    m = random_model(0)
    s = to_store(m)
    v = torch.randn(32)
    apply_op(s, "steer_bias", vector=v, layer=1, scale=2.0)
    assert s.config["mlp_bias"] is True
    base_out, new_out = [], []
    h1 = m.model.layers[1].register_forward_hook(lambda mm, a, o: base_out.append(o.detach()))
    with torch.no_grad():
        m(ids)
    h1.remove()
    m2 = from_store(s)
    h2 = m2.model.layers[1].register_forward_hook(lambda mm, a, o: new_out.append(o.detach()))
    with torch.no_grad():
        m2(ids)
    h2.remove()
    assert torch.allclose(new_out[0] - base_out[0], (2.0 * v).expand_as(base_out[0]), atol=1e-5)
