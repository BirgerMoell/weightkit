import pytest
import torch

from weightkit import apply_op
from weightkit.testing import from_store, random_model, to_store

from conftest import logits

ATOL = 2e-4


def test_layer_stack_identity_and_duplication(ids):
    s = to_store(random_model(0))
    before = logits(s, ids)
    apply_op(s, "layer_stack", ranges=[[0, 3]])
    assert torch.allclose(before, logits(s, ids), atol=ATOL)
    apply_op(s, "layer_stack", ranges=[[0, 2], [1, 3]])
    assert s.config["num_hidden_layers"] == 4  # layers 0,1,1,2
    assert "model.layers.3.mlp.up_proj.weight" in s and "model.layers.4.mlp.up_proj.weight" not in s
    assert torch.equal(s["model.layers.1.mlp.up_proj.weight"], s["model.layers.2.mlp.up_proj.weight"])
    out = logits(s, ids)  # loads (strict) -> consistent naming
    assert out.shape == (3, 10, 64)


def test_drop_layers_matches_manual_skip(ids):
    m = random_model(0)
    s = to_store(m)
    apply_op(s, "drop_layers", layers="1")
    assert s.config["num_hidden_layers"] == 2
    with torch.no_grad():
        x = m.model.embed_tokens(ids)
        for i in (0, 2):
            x = m.model.layers[i](x)
        expect = m.lm_head(m.model.norm(x))
    assert torch.allclose(expect, logits(s, ids), atol=ATOL)


def test_layer_stack_bad_index():
    s = to_store(random_model(0))
    with pytest.raises(ValueError, match="do not exist"):
        apply_op(s, "layer_stack", indices=[0, 9])


def test_layer_stack_updates_layer_types():
    s = to_store(random_model(0, layer_types=["a", "b", "c"]))
    apply_op(s, "layer_stack", indices=[2, 0])
    assert s.config["layer_types"] == ["c", "a"]


def test_prune_mlp(ids):
    s = to_store(random_model(0))
    apply_op(s, "prune_mlp", keep_ratio=0.5)
    assert s.config["intermediate_size"] == 24
    assert s["model.layers.0.mlp.up_proj.weight"].shape == (24, 32)
    assert s["model.layers.0.mlp.down_proj.weight"].shape == (32, 24)
    assert logits(s, ids).shape == (3, 10, 64)


def test_prune_mlp_keeping_all_is_noop(ids):
    s = to_store(random_model(0))
    before = logits(s, ids)
    apply_op(s, "prune_mlp", keep=48)
    assert torch.allclose(before, logits(s, ids), atol=ATOL)


def test_prune_heads(ids):
    s = to_store(random_model(0, attention_bias=True))
    apply_op(s, "prune_heads", keep=1)
    c = s.config
    assert (c["num_key_value_heads"], c["num_attention_heads"], c["head_dim"]) == (1, 2, 8)
    assert s["model.layers.0.self_attn.q_proj.weight"].shape == (16, 32)
    assert s["model.layers.0.self_attn.k_proj.bias"].shape == (8,)
    assert logits(s, ids).shape == (3, 10, 64)


def test_prune_heads_keep_all_is_noop(ids):
    s = to_store(random_model(0))
    before = logits(s, ids)
    apply_op(s, "prune_heads", keep=2)
    assert torch.allclose(before, logits(s, ids), atol=ATOL)


def test_kv_pool_exact_when_heads_duplicated(ids):
    m = random_model(0, num_key_value_heads=4)
    with torch.no_grad():
        for layer in m.model.layers:
            for p in (layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight):
                v = p.view(4, 8, -1)
                v[1] = v[0]
                v[3] = v[2]
    s = to_store(m)
    before = logits(s, ids)
    apply_op(s, "kv_pool", kv_heads=2)
    assert s.config["num_key_value_heads"] == 2
    assert s["model.layers.0.self_attn.k_proj.weight"].shape == (16, 32)
    assert torch.allclose(before, logits(s, ids), atol=ATOL)


def test_tie_untie(ids):
    m = random_model(0, tie_word_embeddings=True)
    s = to_store(m)
    assert "lm_head.weight" not in s
    before = logits(s, ids)
    apply_op(s, "untie_embeddings")
    assert "lm_head.weight" in s and s.config["tie_word_embeddings"] is False
    assert torch.allclose(before, logits(s, ids), atol=ATOL)
    apply_op(s, "tie_embeddings")
    assert "lm_head.weight" not in s
    assert torch.allclose(before, logits(s, ids), atol=ATOL)


def test_resize_vocab():
    s = to_store(random_model(0))
    apply_op(s, "resize_vocab", size=70)
    assert s["model.embed_tokens.weight"].shape[0] == 70 and s["lm_head.weight"].shape[0] == 70
    assert torch.allclose(s["lm_head.weight"][64], s["lm_head.weight"][:64].mean(0), atol=1e-6)
    apply_op(s, "resize_vocab", size=10)
    assert s.config["vocab_size"] == 10 and s["lm_head.weight"].shape[0] == 10


def test_perturb_and_cast():
    s = to_store(random_model(0))
    w = s["lm_head.weight"].clone()
    apply_op(s, "perturb", std=0.1, roles=["lm_head"])
    assert not torch.equal(w, s["lm_head.weight"])
    assert 0.02 < ((s["lm_head.weight"] - w).std() / w.std()).item() < 0.3
    apply_op(s, "cast", dtype="bfloat16")
    assert s["lm_head.weight"].dtype == torch.bfloat16 and s.config["torch_dtype"] == "bfloat16"
