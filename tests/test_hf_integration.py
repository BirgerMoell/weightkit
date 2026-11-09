"""Validate against a real Hugging Face LlamaForCausalLM (not just our test model)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from weightkit import TensorStore, apply_op, collect_input_stats, run_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def hf(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=4, tie_word_embeddings=False,
        attn_implementation="eager",
    )
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.15)
        for n, p in m.named_parameters():
            if "layernorm" in n or n == "model.norm.weight":
                p.copy_(torch.rand_like(p) + 0.5)
    d = tmp_path_factory.mktemp("hf")
    m.save_pretrained(str(d), safe_serialization=True)
    return m, str(d)


def run(path, ids):
    from transformers import LlamaForCausalLM

    m = LlamaForCausalLM.from_pretrained(path, attn_implementation="eager", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        return m(ids).logits


def pipe(hf, tmp_path, steps, **extra):
    _, d = hf
    out = str(tmp_path / "out")
    run_pipeline(dict(base=d, output=out, steps=steps, **extra), log=lambda s: None)
    return out


@pytest.mark.parametrize(
    "step",
    [
        dict(op="fold_norm"),
        dict(op="rotate", kind="hadamard"),
        dict(op="rotate", kind="random"),
        dict(op="permute"),
        dict(op="rescale", mode="equalize"),
    ],
)
def test_function_preserving_ops_on_real_hf_model(hf, tmp_path, step):
    m, d = hf
    ids = torch.randint(0, 96, (2, 12))
    ref = run(d, ids)
    out = pipe(hf, tmp_path, [step], seed=1)
    assert torch.allclose(ref, run(out, ids), atol=1e-4), (ref - run(out, ids)).abs().max()


def test_layer_ops_load_in_hf(hf, tmp_path):
    ids = torch.randint(0, 96, (2, 12))
    out = pipe(hf, tmp_path, [dict(op="layer_stack", ranges=[[0, 3], [1, 3]]), dict(op="prune_mlp", keep_ratio=0.5),
                              dict(op="prune_heads", keep=2)])
    logits = run(out, ids)
    assert logits.shape == (2, 12, 96) and torch.isfinite(logits).all()


def test_abliterate_on_hf_and_calibration_hooks(hf, tmp_path):
    m, d = hf
    ids = torch.randint(0, 96, (2, 12))
    r = torch.randn(64)
    r /= r.norm()
    torch.save(r, tmp_path / "r.pt")
    out = pipe(hf, tmp_path, [dict(op="abliterate", direction=str(tmp_path / "r.pt"))])
    from transformers import LlamaForCausalLM

    m2 = LlamaForCausalLM.from_pretrained(out, attn_implementation="eager").eval()
    with torch.no_grad():
        hs = m2(ids, output_hidden_states=True).hidden_states
    for h in hs[:-1]:  # all but the final (post-norm) state
        assert (h @ r).abs().max() < 1e-4


def test_wanda_with_hf_calibration(hf, tmp_path):
    m, d = hf
    names = [n for n, p in m.named_parameters() if n.endswith("_proj.weight")]
    calib = collect_input_stats(m, [torch.randint(0, 96, (4, 32))], names)
    calib.save(str(tmp_path / "c.pt"))
    out = pipe(hf, tmp_path, [dict(op="prune", method="wanda", sparsity=0.5)], calibration=str(tmp_path / "c.pt"))
    w = TensorStore.from_path(out)["model.layers.1.mlp.down_proj.weight"]
    assert abs((w == 0).float().mean().item() - 0.5) < 0.01


def test_growth_ops_exact_on_real_hf_with_default_verify_runner(hf, tmp_path):
    """widen_mlp + add_layers + rotate must leave a real HF Llama's logits unchanged,
    checked by weightkit's own `verify` using the default transformers runner."""
    _, d = hf
    out = str(tmp_path / "grown")
    run_pipeline(
        dict(
            base=d, output=out, seed=1,
            steps=[dict(op="widen_mlp", extra=32, mode="split"), dict(op="add_layers", after=[0, 1]), dict(op="rotate")],
            verify=[
                dict(type="equivalence", tol=1e-4),
                dict(type="config", expect=dict(num_hidden_layers=5, intermediate_size=160)),
                dict(type="data_use", allowed=["none"]),
                dict(type="finite"),
            ],
        ),
        log=lambda s: None,
    )


def test_verify_catches_non_exact_edit_on_real_hf(hf, tmp_path):
    from weightkit.verify import VerificationFailed

    _, d = hf
    with pytest.raises(VerificationFailed) as e:
        run_pipeline(
            dict(base=d, output=str(tmp_path / "p"), steps=[dict(op="prune", method="magnitude", sparsity=0.5)],
                 verify=[dict(type="equivalence", tol=1e-4)]),
            log=lambda s: None,
        )
    assert not e.value.results[0].ok
