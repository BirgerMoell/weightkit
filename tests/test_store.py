import json
import os

import torch

from weightkit import TensorStore
from weightkit.testing import random_model, to_store


def test_roundtrip_sharded(tmp_path):
    m = random_model(0)
    s = to_store(m)
    s.save(str(tmp_path), max_shard_size=4000)
    files = sorted(os.listdir(tmp_path))
    assert "model.safetensors.index.json" in files and "config.json" in files
    assert sum(f.endswith(".safetensors") for f in files) > 1
    r = TensorStore.from_path(str(tmp_path))
    assert r.names() and set(r.names()) == set(s.names())
    for n in s.names():
        assert torch.equal(r[n], s[n])
    assert r.config["hidden_size"] == 32


def test_single_shard_and_dtype(tmp_path):
    s = to_store(random_model(0))
    s.save(str(tmp_path), dtype=torch.bfloat16)
    assert os.path.exists(tmp_path / "model.safetensors")
    r = TensorStore.from_path(str(tmp_path))
    assert r["lm_head.weight"].dtype == torch.bfloat16
    assert json.load(open(tmp_path / "config.json"))["torch_dtype"] == "bfloat16"


def test_spill_preserves_content(tmp_path):
    s = TensorStore(spill_bytes=1000)
    ref = {}
    for i in range(20):
        t = torch.randn(10, 10)
        ref[f"w{i}"] = t
        s[f"w{i}"] = t
    assert s._layers, "overlay should have spilled"
    s["w3"] = torch.zeros(2)  # overwrite after spill
    del s["w4"]
    assert "w4" not in s and torch.equal(s["w3"], torch.zeros(2))
    for i in (0, 7, 19):
        assert torch.equal(s[f"w{i}"], ref[f"w{i}"])
    s.save(str(tmp_path))
    r = TensorStore.from_path(str(tmp_path))
    assert len(r) == 19 and "w4" not in r


def test_shared_tensors_save(tmp_path):
    s = TensorStore()
    t = torch.randn(4, 4)
    s["a"], s["b"] = t, t  # same storage
    s.save(str(tmp_path))
    r = TensorStore.from_path(str(tmp_path))
    assert torch.equal(r["a"], r["b"])
