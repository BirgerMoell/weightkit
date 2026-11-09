import torch
import pytest

from weightkit import TensorStore, apply_op
from weightkit.ops.merge import breadcrumbs, slerp, topk_mask
from weightkit.testing import random_model, to_store


def three():
    return to_store(random_model(0)), to_store(random_model(1)), to_store(random_model(2))


def test_linear_weights():
    a, b, c = three()
    exp = 0.25 * b["lm_head.weight"] + 0.75 * c["lm_head.weight"]
    apply_op(a, "merge", models={"b": b, "c": c}, method="linear", sources=[dict(model="b", weight=1), dict(model="c", weight=3)])
    assert torch.allclose(a["lm_head.weight"], exp, atol=1e-6)


def test_slerp_endpoints_and_orthogonal():
    x, y = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert torch.allclose(slerp(0.0, x, y), x) and torch.allclose(slerp(1.0, x, y), y)
    mid = slerp(0.5, x, y)
    assert torch.allclose(mid.norm(), torch.tensor(1.0)) and torch.allclose(mid, torch.tensor([0.7071, 0.7071]), atol=1e-3)


def test_slerp_op():
    a, b, _ = three()
    w0 = a["model.norm.weight"].clone()
    apply_op(a, "merge", models={"b": b}, method="slerp", sources=["b"], t=0.0)
    assert torch.allclose(a["model.norm.weight"], w0, atol=1e-5)


def test_task_arithmetic():
    base, f1, f2 = three()
    n = "model.layers.0.mlp.up_proj.weight"
    exp = base[n] + 0.5 * ((f1[n] - base[n]) + (f2[n] - base[n]))
    apply_op(base, "merge", models={"f1": f1, "f2": f2}, method="task_arithmetic", sources=["f1", "f2"], scale=0.5)
    assert torch.allclose(base[n], exp, atol=1e-6)


def test_ties_sign_election_and_trim():
    base = TensorStore.from_dict({"w": torch.zeros(4)})
    f1 = TensorStore.from_dict({"w": torch.tensor([1.0, 1.0, -0.1, 3.0])})
    f2 = TensorStore.from_dict({"w": torch.tensor([2.0, -0.5, 0.2, 0.0])})
    cfg = dict(models={"f1": f1, "f2": f2}, method="ties", sources=["f1", "f2"], include=["w"])
    # density 1: elementwise: [1.5 avg, sign + wins (1 vs -.5 -> sum .5 >0, only f1), ..]
    apply_op(base, "merge", density=1.0, **cfg)
    out = base["w"]
    assert out[0] == pytest.approx(1.5)  # both agree -> mean
    assert out[1] == pytest.approx(1.0)  # elected sign +, disagreeing -0.5 discarded
    assert out[2] == pytest.approx(0.2)  # summed sign +, so -0.1 discarded
    assert out[3] == pytest.approx(3.0)  # f2 has no vote here


def test_ties_trim_density():
    base = TensorStore.from_dict({"w": torch.zeros(10)})
    f = TensorStore.from_dict({"w": torch.arange(1.0, 11.0)})
    apply_op(base, "merge", models={"f": f}, method="ties", sources=["f"], density=0.3, include=["w"])
    assert (base["w"] != 0).sum() == 3 and base["w"][-1] == 10


def test_dare_rescales_and_density():
    base = TensorStore.from_dict({"w": torch.zeros(20000)})
    f = TensorStore.from_dict({"w": torch.ones(20000)})
    apply_op(base, "merge", models={"f": f}, method="dare_linear", sources=["f"], density=0.25, include=["w"], seed=1)
    w = base["w"]
    kept = (w != 0).float().mean().item()
    assert abs(kept - 0.25) < 0.02
    assert abs(w.mean().item() - 1.0) < 0.05  # unbiased in expectation


def test_breadcrumbs():
    x = torch.arange(1.0, 11.0)
    y = breadcrumbs(x, beta=0.2, gamma=0.2)
    assert (y[:2] == 0).all() and (y[-2:] == 0).all() and (y[2:8] != 0).all()


def test_model_stock_identical_models_returns_them():
    base, f1, _ = three()
    f2 = TensorStore.from_dict({n: f1[n].clone() for n in f1.names()}, config=f1.config)
    apply_op(base, "merge", models={"f1": f1, "f2": f2}, method="model_stock", sources=["f1", "f2"])
    for n in ("lm_head.weight", "model.layers.0.self_attn.q_proj.weight"):
        assert torch.allclose(base[n], f1[n], atol=1e-5)


def test_task_vector_negate_and_extrapolate():
    base, ft, _ = three()
    n = "lm_head.weight"
    b0, f0 = base[n].clone(), ft[n].clone()
    apply_op(base, "task_vector", models={"ft": ft}, finetuned="ft", base="self", scale=1.0)  # self-base: ft-self
    assert torch.allclose(base[n], f0, atol=1e-6)


def test_task_vector_named_base():
    base, ft, other = three()
    n = "lm_head.weight"
    exp = other[n] + (-1.0) * (ft[n] - base[n])
    apply_op(other, "task_vector", models={"ft": ft, "b": base}, finetuned="ft", base="b", scale=-1.0)
    assert torch.allclose(other[n], exp, atol=1e-6)


def test_shape_mismatch_skipped():
    base = TensorStore.from_dict({"w": torch.zeros(3), "v": torch.zeros(2)})
    f = TensorStore.from_dict({"w": torch.ones(4), "v": torch.ones(2)})
    apply_op(base, "merge", models={"f": f}, method="linear", sources=["f"], include=["*"])
    assert torch.equal(base["w"], torch.zeros(3)) and torch.equal(base["v"], torch.ones(2))
