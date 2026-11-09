import pytest
import torch
import yaml

from weightkit import TensorStore, run_pipeline
from weightkit.cli import main
from weightkit.testing import random_model, to_store


@pytest.fixture
def ckpts(tmp_path):
    paths = {}
    for name, seed in (("base", 0), ("a", 1), ("b", 2)):
        p = tmp_path / name
        to_store(random_model(seed)).save(str(p))
        paths[name] = str(p)
    return paths


def test_pipeline_end_to_end(ckpts, tmp_path):
    out = tmp_path / "out"
    cfg = dict(
        base=ckpts["base"],
        models={"a": ckpts["a"], "b": ckpts["b"]},
        seed=1,
        steps=[
            dict(op="merge", method="ties", sources=[dict(model="a", weight=0.5, density=0.5), dict(model="b", weight=0.5, density=0.5)]),
            dict(op="prune", method="magnitude", sparsity=0.3, roles=["mlp"]),
            dict(op="rotate", kind="hadamard"),
            dict(op="drop_layers", layers="-1"),
        ],
        output=str(out),
        save_dtype="float16",
    )
    logs = []
    run_pipeline(cfg, log=logs.append)
    assert any("drop_layers" in l for l in logs)
    r = TensorStore.from_path(str(out))
    assert r.config["num_hidden_layers"] == 2
    assert r["lm_head.weight"].dtype == torch.float16
    assert "model.layers.2.mlp.up_proj.weight" not in r


def test_pipeline_yaml_and_errors(ckpts, tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(dict(base=ckpts["base"], steps=[dict(op="prune", sparsity=0.5, nope=1)])))
    with pytest.raises(ValueError, match="bad parameters"):
        run_pipeline(str(f))
    f.write_text(yaml.safe_dump(dict(base=ckpts["base"], steps=[dict(op="does_not_exist")])))
    with pytest.raises(KeyError, match="unknown op"):
        run_pipeline(str(f))
    f.write_text(yaml.safe_dump(dict(base=ckpts["base"], bogus=1)))
    with pytest.raises(ValueError, match="unknown top-level"):
        run_pipeline(str(f))


def test_pipeline_calibration_file(ckpts, tmp_path):
    from weightkit import collect_input_stats

    m = random_model(0)
    ids = torch.randint(0, 64, (4, 16))
    names = [n for n, p in m.named_parameters() if "_proj" in n]
    collect_input_stats(m, [ids], names).save(str(tmp_path / "c.pt"))
    cfg = dict(base=ckpts["base"], calibration=str(tmp_path / "c.pt"), output=str(tmp_path / "o"),
               steps=[dict(op="prune", method="wanda", sparsity=0.5)])
    run_pipeline(cfg, log=lambda s: None)
    w = TensorStore.from_path(str(tmp_path / "o"))["model.layers.0.mlp.up_proj.weight"]
    assert abs((w == 0).float().mean().item() - 0.5) < 0.01


def test_cli(ckpts, tmp_path, capsys):
    assert main(["ops", "-v"]) == 0
    out = capsys.readouterr().out
    for op in ("merge", "abliterate", "edit_facts", "rotate", "prune", "spectral", "lora_merge", "layer_stack"):
        assert op in out
    o = str(tmp_path / "pruned")
    assert main(["apply", ckpts["base"], o, "prune", "--params", '{"sparsity": 0.5}']) == 0
    assert main(["diff", ckpts["base"], o, "--top", "3"]) == 0
    txt = capsys.readouterr().out
    assert "changed" in txt and "rel.change" in txt
    lo = str(tmp_path / "lora")
    assert main(["extract-lora", ckpts["base"], ckpts["a"], lo, "--rank", "4"]) == 0
    assert (tmp_path / "lora" / "adapter_model.safetensors").exists()


def test_every_op_documented_and_registered():
    from weightkit import list_ops

    ops = list_ops()
    assert len(ops) >= 25
    for name, cls in ops.items():
        assert cls.describe(), name
