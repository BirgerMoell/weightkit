import json

import pytest
import torch
import yaml

from weightkit import TensorStore, run_pipeline
from weightkit.cli import main
from weightkit.testing import from_store, random_model, to_store
from weightkit.verify import VerificationFailed, verify


def tiny_runner(store):
    m = from_store(store)

    def f(ids):
        with torch.no_grad():
            return m(ids)

    return f


@pytest.fixture
def base(tmp_path):
    p = tmp_path / "base"
    to_store(random_model(0)).save(str(p))
    return str(p)


def build(base, tmp_path, steps, name="out", calibration=None):
    out = str(tmp_path / name)
    cfg = dict(base=base, output=out, steps=steps)
    if calibration:
        cfg["calibration"] = calibration
    run_pipeline(cfg, log=lambda s: None)
    return out


def res(spec, **kw):
    return {r.check: r for r in verify(spec, runner=tiny_runner, **kw)}


def test_exact_ops_pass_equivalence_and_prune_fails_it(base, tmp_path):
    rot = build(base, tmp_path, [dict(op="rotate")], "rot")
    assert res(dict(model=rot, reference=base, checks=[dict(type="equivalence", tol=2e-4)]))["equivalence"].ok
    pr = build(base, tmp_path, [dict(op="prune", method="magnitude", sparsity=0.5)], "pr")
    r = res(dict(model=pr, reference=base, checks=[dict(type="equivalence", tol=2e-4)]))["equivalence"]
    assert not r.ok and "max |Δlogit|" in r.detail


def test_growth_ops_pass_equivalence(base, tmp_path):
    out = build(base, tmp_path, [dict(op="widen_mlp", extra=16), dict(op="add_layers", after=[1])], "grow")
    r = verify(dict(model=out, reference=base, checks=[dict(type="equivalence", tol=2e-4), dict(type="config", expect=dict(num_hidden_layers=4, intermediate_size=64))]), runner=tiny_runner)
    assert all(x.ok for x in r), r


def test_sparsity_untouched_changed(base, tmp_path):
    out = build(base, tmp_path, [dict(op="prune", method="magnitude", sparsity=0.5, roles=["mlp"])])
    r = res(dict(model=out, reference=base, checks=[
        dict(type="sparsity", roles=["mlp"], min=0.45, max=0.55),
        dict(type="untouched", include=["model.embed_tokens.weight", "lm_head.weight"]),
        dict(type="changed", roles=["mlp"]),
    ]))
    assert r["sparsity"].ok and r["untouched"].ok and r["changed"].ok
    bad = res(dict(model=out, reference=base, checks=[dict(type="sparsity", roles=["attn"], min=0.4)]))["sparsity"]
    assert not bad.ok
    bad = res(dict(model=out, reference=base, checks=[dict(type="untouched", roles=["mlp"])]))["untouched"]
    assert not bad.ok and "changed" in bad.detail


def test_untouched_fails_after_rotate(base, tmp_path):
    out = build(base, tmp_path, [dict(op="rotate")])
    assert not res(dict(model=out, reference=base, checks=[dict(type="untouched", include=["model.embed_tokens.weight"])]))["untouched"].ok


def test_direction_removed(base, tmp_path):
    r = torch.randn(32)
    torch.save(r, tmp_path / "r.pt")
    out = build(base, tmp_path, [dict(op="abliterate", direction=str(tmp_path / "r.pt"))])
    chk = dict(type="direction_removed", direction=str(tmp_path / "r.pt"))
    assert res(dict(model=out, checks=[chk]))["direction_removed"].ok
    assert not res(dict(model=base, checks=[chk]))["direction_removed"].ok


def test_data_use_and_ops_manifest_checks(base, tmp_path):
    from weightkit import collect_input_stats

    m = random_model(0)
    names = [n for n, _ in m.named_parameters() if "_proj" in n]
    collect_input_stats(m, [torch.randint(0, 64, (4, 16))], names).save(str(tmp_path / "c.pt"))
    free = build(base, tmp_path, [dict(op="prune", method="magnitude", sparsity=0.3), dict(op="rotate")], "free")
    used = build(base, tmp_path, [dict(op="prune", method="wanda", sparsity=0.3)], "used", calibration=str(tmp_path / "c.pt"))
    assert res(dict(model=free, checks=[dict(type="data_use", allowed=["none"])]))["data_use"].ok
    r = res(dict(model=used, checks=[dict(type="data_use", allowed=["none"])]))["data_use"]
    assert not r.ok and "calibration" in r.detail
    assert res(dict(model=used, checks=[dict(type="data_use", allowed=["calibration"], max_calibration_tokens=64)]))["data_use"].ok
    assert not res(dict(model=used, checks=[dict(type="data_use", allowed=["calibration"], max_calibration_tokens=10)]))["data_use"].ok
    assert res(dict(model=free, checks=[dict(type="ops", exact=["prune", "rotate"])]))["ops"].ok
    assert res(dict(model=free, checks=[dict(type="ops", contains=["rotate"])]))["ops"].ok
    assert not res(dict(model=free, checks=[dict(type="ops", contains=["rotate", "prune"])]))["ops"].ok  # order matters


def test_manifest_required_and_unknown_check_and_finite(base, tmp_path):
    r = res(dict(model=base, checks=[dict(type="data_use")]))["data_use"]
    assert not r.ok and "manifest" in r.detail
    assert not verify(dict(model=base, checks=[dict(type="nope")]))[0].ok
    assert res(dict(model=base, checks=[dict(type="finite")]))["finite"].ok
    s = TensorStore.from_path(base)
    s["lm_head.weight"] = torch.full((64, 32), float("nan"))
    assert not verify(dict(model=base, checks=[dict(type="finite")]), model=s)[0].ok


def test_missing_reference_is_a_failed_check_not_a_crash(base):
    r = verify(dict(model=base, checks=[dict(type="equivalence")]), runner=tiny_runner)[0]
    assert not r.ok and "reference" in r.detail


def test_pipeline_verify_block_passes_and_raises(base, tmp_path):
    ok = dict(base=base, output=str(tmp_path / "a"), steps=[dict(op="rotate")], verify=[dict(type="equivalence", tol=2e-4), dict(type="finite")])
    run_pipeline(ok, log=lambda s: None, runner=tiny_runner)
    bad = dict(base=base, output=str(tmp_path / "b"), steps=[dict(op="prune", sparsity=0.5)], verify=dict(checks=[dict(type="equivalence", tol=2e-4)]))
    with pytest.raises(VerificationFailed, match="1 checks|0/1"):
        run_pipeline(bad, log=lambda s: None, runner=tiny_runner)
    with pytest.raises(ValueError, match="needs `output`"):
        run_pipeline(dict(base=base, steps=[], verify=[dict(type="finite")]), log=lambda s: None)


def test_cli_verify_exit_codes(base, tmp_path, capsys):
    out = build(base, tmp_path, [dict(op="prune", method="magnitude", sparsity=0.5)])
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(dict(model=out, reference=base, checks=[dict(type="sparsity", min=0.4), dict(type="data_use", allowed=["none"])])))
    assert main(["verify", str(good)]) == 0
    assert "2/2 checks passed" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(dict(model=out, reference=base, checks=[dict(type="sparsity", max=0.1)])))
    assert main(["verify", str(bad), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)[0]["ok"] is False
