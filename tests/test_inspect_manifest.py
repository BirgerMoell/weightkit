import json
import os

import torch

from weightkit import Calibration, TensorStore, apply_op, run_pipeline
from weightkit.cli import main
from weightkit.inspect import format_report, inspect_store
from weightkit.testing import random_model, to_store


def test_inspect_counts_and_rank():
    s = TensorStore.from_dict({"a": torch.eye(8), "b": torch.zeros(4, 4), "c": torch.ones(3, 5)})
    info = inspect_store(s, rank=True)
    assert info["params"] == 64 + 16 + 15 and info["nonzero"] == 8 + 15
    assert info["tensors"]["a"]["eff_rank"] > 7.9          # identity: flat spectrum
    assert abs(info["tensors"]["c"]["eff_rank"] - 1.0) < 1e-4  # rank-one
    assert info["tensors"]["b"]["sparsity"] == 1.0
    assert "parameters" in format_report(info, rank=True)


def test_inspect_outlier_ratio_flags_hot_channel():
    w = torch.randn(32, 16)
    w[:, 3] *= 50
    assert inspect_store(TensorStore.from_dict({"w": w}))["tensors"]["w"]["col_outlier"] > 20


def test_global_prune_threshold_spans_tensors():
    big, small = torch.full((4, 4), 10.0), torch.full((4, 4), 0.1)
    s = TensorStore.from_dict({"big": big.clone(), "small": small.clone()})
    apply_op(s, "prune", method="magnitude", sparsity=0.5, granularity="global", include=["*"])
    assert (s["small"] == 0).all() and (s["big"] == 10).all()       # layer-wise would give 50/50 each
    s = TensorStore.from_dict({"big": big.clone(), "small": small.clone()})
    apply_op(s, "prune", method="magnitude", sparsity=0.5, granularity="layer", include=["*"])
    assert (s["small"] == 0).float().mean() == 0.5


def test_ledger_records_ops_data_and_touched(tmp_path):
    m = random_model(0)
    s = to_store(m)
    apply_op(s, "prune", method="magnitude", sparsity=0.3, roles=["mlp"])
    apply_op(s, "rotate", kind="hadamard")
    assert [e["op"] for e in s.ledger] == ["prune", "rotate"]
    assert s.ledger[0]["data"] == ["none"] and s.ledger[0]["tensors_touched"] == 9  # 3 layers x 3 mlp mats
    assert s.ledger[1]["tensors_touched"] > 9
    s.save(str(tmp_path))
    mf = json.load(open(tmp_path / "weightkit_manifest.json"))
    assert mf["data_use"] == ["none"] and len(mf["ops"]) == 2 and mf["calibration_tokens"] == 0
    assert mf["ops"][0]["params"]["sparsity"] == 0.3


def test_data_tags_by_op_and_method():
    from weightkit import get_op, list_ops

    list_ops()
    assert get_op("prune")(method="magnitude").data_tags == ()
    assert get_op("prune")(method="wanda").data_tags == ("calibration",)
    assert get_op("merge").data_tags == ("other_models",)
    assert get_op("abliterate")(direction="r.pt").data_tags == ("supplied",)
    assert get_op("abliterate")(pos="a", neg="b", layer=1).data_tags == ("calibration",)
    assert get_op("rotate").data_tags == ()


def test_calibration_tokens_and_chain(tmp_path):
    from weightkit import collect_input_stats

    m = random_model(0)
    ids = torch.randint(0, 64, (4, 16))
    names = [n for n, p in m.named_parameters() if "_proj" in n]
    calib = collect_input_stats(m, [ids], names)
    assert calib.tokens() == 64
    s = to_store(m)
    apply_op(s, "prune", calib=calib, method="wanda", sparsity=0.5)
    s.save(str(tmp_path / "a"))
    mf = json.load(open(tmp_path / "a" / "weightkit_manifest.json"))
    assert mf["data_use"] == ["calibration"] and mf["calibration_tokens"] == 64
    # second generation keeps the chain and inherits the data tag
    s2 = TensorStore.from_path(str(tmp_path / "a"))
    apply_op(s2, "rotate")
    s2.save(str(tmp_path / "b"))
    mf2 = json.load(open(tmp_path / "b" / "weightkit_manifest.json"))
    assert mf2["data_use"] == ["calibration"] and mf2["parent"]["ops"][0]["op"] == "prune"
    assert mf2["base"] == os.path.abspath(str(tmp_path / "a"))


def test_pipeline_manifest_and_cli_inspect(tmp_path, capsys):
    to_store(random_model(0)).save(str(tmp_path / "base"))
    run_pipeline(dict(base=str(tmp_path / "base"), output=str(tmp_path / "o"),
                      steps=[dict(op="prune", method="magnitude", sparsity=0.5), dict(op="quantize_rtn", bits=8)]), log=lambda s: None)
    mf = json.load(open(tmp_path / "o" / "weightkit_manifest.json"))
    assert [e["op"] for e in mf["ops"]] == ["prune", "quantize_rtn"] and mf["data_use"] == ["none"]
    assert main(["inspect", str(tmp_path / "o"), "--rank"]) == 0
    out = capsys.readouterr().out
    assert "provenance" in out and "data used: none" in out and "zeros" in out
    assert main(["inspect", str(tmp_path / "o"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["params"] > 0


def test_manifest_not_written_for_untouched_store(tmp_path):
    to_store(random_model(0)).save(str(tmp_path))
    assert not os.path.exists(tmp_path / "weightkit_manifest.json")
