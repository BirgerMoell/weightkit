import json

import pytest
import torch
from safetensors.torch import save_file

from weightkit import TensorStore, apply_op
from weightkit.ops.delta import extract_lora
from weightkit.ops.quant import rtn_quantize
from weightkit.testing import random_model, to_store


def test_truncate_rank_and_error():
    torch.manual_seed(0)
    s = TensorStore.from_dict({"w": torch.randn(20, 30)})
    w = s["w"].clone()
    apply_op(s, "spectral", mode="truncate", rank=4, include=["w"])
    assert torch.linalg.matrix_rank(s["w"]) == 4
    U, S, Vh = torch.linalg.svd(w)
    assert torch.allclose((w - s["w"]).norm(), S[4:].norm(), rtol=1e-4)  # Eckart-Young


def test_rank_ratio_energy_power_clip_scaletop():
    torch.manual_seed(0)
    w = torch.randn(16, 16)
    for kw in (dict(mode="truncate", rank_ratio=0.25), dict(mode="energy", energy=0.5), dict(mode="power", power=0.5),
               dict(mode="clip", quantile=0.5), dict(mode="scale_top", k=2, factor=2.0)):
        s = TensorStore.from_dict({"w": w.clone()})
        apply_op(s, "spectral", include=["w"], **kw)
        sv = torch.linalg.svdvals(s["w"])
        ov = torch.linalg.svdvals(w)
        if kw["mode"] == "truncate":
            assert torch.linalg.matrix_rank(s["w"]) == 4
        if kw["mode"] == "energy":
            assert torch.linalg.matrix_rank(s["w"]) < 16
        if kw["mode"] == "power":
            assert sv[0] == pytest.approx(ov[0].item(), rel=1e-4) and sv[-1] > ov[-1]
        if kw["mode"] == "clip":
            assert sv[0] < ov[0]
        if kw["mode"] == "scale_top":
            assert sv[0] == pytest.approx(2 * ov[0].item(), rel=1e-4)
    s = TensorStore.from_dict({"w": w.clone()})
    apply_op(s, "spectral", mode="power", power=1.0, include=["w"])
    assert torch.allclose(s["w"], w, atol=1e-4)


def test_rtn_levels_and_error_monotone():
    torch.manual_seed(0)
    W = torch.randn(8, 256)
    e = {}
    for bits in (2, 4, 8):
        Q = rtn_quantize(W, bits, group_size=64)
        assert Q.shape == W.shape
        g = Q.reshape(8, 4, 64)
        assert max(len(torch.unique(g[i, j])) for i in range(8) for j in range(4)) <= 2**bits
        e[bits] = (Q - W).norm().item()
    assert e[8] < e[4] < e[2]
    Qs = rtn_quantize(W, 4, group_size=0, symmetric=True)
    assert max(len(torch.unique(Qs[i])) for i in range(8)) <= 16
    # non-divisible group
    assert rtn_quantize(torch.randn(4, 100), 4, 32).shape == (4, 100)


def test_quantize_op_only_touches_linears():
    s = to_store(random_model(0))
    norm = s["model.norm.weight"].clone()
    apply_op(s, "quantize_rtn", bits=4, group_size=16)
    assert torch.equal(s["model.norm.weight"], norm)


def test_extract_and_merge_lora_roundtrip(tmp_path):
    torch.manual_seed(0)
    base = to_store(random_model(0))
    ft = to_store(random_model(0))
    n = "model.layers.0.self_attn.q_proj.weight"
    low_rank = torch.randn(32, 3) @ torch.randn(3, 32) * 0.1
    ft[n] = ft[n] + low_rank
    errs = extract_lora(base, ft, rank=3, out_dir=str(tmp_path / "ad"))
    assert errs[n] < 1e-4
    fresh = to_store(random_model(0))
    apply_op(fresh, "lora_merge", adapter=str(tmp_path / "ad"))
    assert torch.allclose(fresh[n], ft[n], atol=1e-4)
    other = "model.layers.1.mlp.up_proj.weight"
    assert torch.equal(fresh[other], base[other])


def test_lora_merge_scaling_and_peft_key_variants(tmp_path):
    base = to_store(random_model(0))
    n = "model.layers.0.mlp.up_proj"
    A, B = torch.randn(2, 32), torch.randn(48, 2)
    d = tmp_path / "ad"
    d.mkdir()
    save_file(
        {f"base_model.model.{n}.lora_A.default.weight": A, f"base_model.model.{n}.lora_B.default.weight": B},
        str(d / "adapter_model.safetensors"),
    )
    json.dump({"r": 2, "lora_alpha": 8}, open(d / "adapter_config.json", "w"))
    w0 = base[n + ".weight"].clone()
    apply_op(base, "lora_merge", adapter=str(d), scale=0.5)
    assert torch.allclose(base[n + ".weight"], w0 + 0.5 * 4.0 * B @ A, atol=1e-5)


def test_lora_merge_no_match_errors(tmp_path):
    base = to_store(random_model(0))
    d = tmp_path / "ad"
    d.mkdir()
    save_file({"base_model.model.nope.lora_A.weight": torch.zeros(1, 2), "base_model.model.nope.lora_B.weight": torch.zeros(2, 1)}, str(d / "adapter_model.safetensors"))
    json.dump({"r": 1}, open(d / "adapter_config.json", "w"))
    with pytest.raises(ValueError, match="no LoRA"):
        apply_op(base, "lora_merge", adapter=str(d))
