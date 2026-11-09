"""Lazy, spill-capable tensor store.

A ``TensorStore`` presents a checkpoint (one or many safetensors files) as a
mutable ``name -> tensor`` mapping.  Reads are lazy; writes go to an in-memory
overlay which is transparently *spilled* to temporary safetensors files when it
grows past ``spill_bytes``, so whole-model operations do not need the whole
model in RAM.

Convention: tensors returned by the store are read-only.  Ops must clone before
mutating in place (or, better, just build new tensors).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import weakref
from typing import Dict, Iterator, List, Mapping, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5")
_NO_COPY = {"weightkit_manifest.json", "config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"}


class _FileSource:
    """Read-only name -> tensor source backed by safetensors files."""

    def __init__(self, files: Mapping[str, str]):
        self.files = dict(files)
        self._handles: Dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.files

    def get(self, name: str) -> torch.Tensor:
        path = self.files[name]
        h = self._handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = h
        return h.get_tensor(name)


def _download(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise FileNotFoundError(
            f"{repo_id!r} is not a local path and huggingface_hub is not installed"
        ) from e
    return snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"],
    )


class TensorStore:
    def __init__(
        self,
        config: Optional[dict] = None,
        extras_dir: Optional[str] = None,
        spill_bytes: int = 4 << 30,
    ):
        self.config: dict = dict(config or {})
        self.extras_dir = extras_dir
        self.spill_bytes = spill_bytes
        self._names: Dict[str, None] = {}
        self._layers: List[_FileSource] = []
        self._overlay: Dict[str, torch.Tensor] = {}
        self._overlay_bytes = 0
        self._spill_dir: Optional[str] = None
        self._spill_n = 0
        self._parent = None  # keeps a snapshot's source (and its spill files) alive
        # provenance (see manifest.py)
        self.source: Optional[str] = None
        self.parent_manifest: Optional[dict] = None
        self.ledger: List[dict] = []
        self.written: set = set()  # tensor names set/deleted since last reset (per-op accounting)
        self.manifest_extra: dict = {}

    # ------------------------------------------------------------ construction
    @classmethod
    def from_path(cls, path: str, **kw) -> "TensorStore":
        """Load a directory / single .safetensors file / HF hub repo id."""
        path = str(path)
        if not os.path.exists(path):
            path = _download(path)
        mapping: Dict[str, str] = {}
        if os.path.isfile(path):
            root, files = os.path.dirname(path) or ".", [path]
        else:
            root = path
            index = os.path.join(root, "model.safetensors.index.json")
            files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
            if os.path.exists(index):
                with open(index) as f:
                    wm = json.load(f)["weight_map"]
                mapping = {k: os.path.join(root, v) for k, v in wm.items()}
        if not mapping:
            for fpath in files:
                with safe_open(fpath, framework="pt", device="cpu") as h:
                    for k in h.keys():
                        mapping[k] = fpath
        if not mapping:
            raise FileNotFoundError(f"no .safetensors weights found in {path}")
        config = {}
        cfg = os.path.join(root, "config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                config = json.load(f)
        store = cls(config=config, extras_dir=root, **kw)
        store.source = os.path.abspath(path)
        mf = os.path.join(root, "weightkit_manifest.json")
        if os.path.exists(mf):
            try:
                with open(mf) as f:
                    store.parent_manifest = json.load(f)
            except (OSError, ValueError):
                pass
        store._layers.append(_FileSource(mapping))
        store._names = {k: None for k in mapping}
        return store

    @classmethod
    def from_dict(cls, tensors: Mapping[str, torch.Tensor], config: Optional[dict] = None, **kw):
        store = cls(config=config, **kw)
        for k, v in tensors.items():
            store[k] = v
        return store

    def snapshot(self) -> "TensorStore":
        """Cheap read-only-ish copy (shares tensors/files).  Used by ops that
        rename or reorder tensors."""
        s = TensorStore(self.config.copy(), self.extras_dir, self.spill_bytes)
        s._layers = list(self._layers)
        s._overlay = dict(self._overlay)
        s._overlay_bytes = self._overlay_bytes
        s._names = dict(self._names)
        s._parent = self
        return s

    # ------------------------------------------------------------ mapping API
    def names(self) -> List[str]:
        return list(self._names)

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._names))

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, name: object) -> bool:
        return name in self._names

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._names:
            raise KeyError(name)
        t = self._overlay.get(name)
        if t is not None:
            return t
        for layer in reversed(self._layers):
            if layer.has(name):
                return layer.get(name)
        raise KeyError(name)  # pragma: no cover

    def get(self, name: str, default=None):
        return self[name] if name in self else default

    def __setitem__(self, name: str, tensor: torch.Tensor) -> None:
        old = self._overlay.get(name)
        if old is not None:
            self._overlay_bytes -= old.numel() * old.element_size()
        tensor = tensor.detach().cpu()
        self._overlay[name] = tensor
        self._overlay_bytes += tensor.numel() * tensor.element_size()
        self._names[name] = None
        self.written.add(name)
        if self._overlay_bytes > self.spill_bytes:
            self._spill()

    def __delitem__(self, name: str) -> None:
        del self._names[name]
        self.written.add(name)
        old = self._overlay.pop(name, None)
        if old is not None:
            self._overlay_bytes -= old.numel() * old.element_size()

    # ------------------------------------------------------------ spilling
    def _spill(self) -> None:
        if not self._overlay:
            return
        if self._spill_dir is None:
            self._spill_dir = tempfile.mkdtemp(prefix="weightkit-spill-")
            weakref.finalize(self, shutil.rmtree, self._spill_dir, True)
        path = os.path.join(self._spill_dir, f"spill-{self._spill_n:05d}.safetensors")
        self._spill_n += 1
        save_file(_dedupe(self._overlay), path, metadata={"format": "pt"})
        self._layers.append(_FileSource({k: path for k in self._overlay}))
        self._overlay = {}
        self._overlay_bytes = 0

    # ------------------------------------------------------------ saving
    def save(
        self,
        out_dir: str,
        max_shard_size: int = 5 * 1024**3,
        dtype: Optional[torch.dtype] = None,
        copy_extras: bool = True,
    ) -> None:
        os.makedirs(out_dir, exist_ok=True)
        shards: List[Dict[str, torch.Tensor]] = []
        cur: Dict[str, torch.Tensor] = {}
        cur_bytes = 0
        for name in list(self._names):
            t = self[name]
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            nbytes = t.numel() * t.element_size()
            if cur and cur_bytes + nbytes > max_shard_size:
                shards.append(self._write_shard(out_dir, len(shards), cur))
                cur, cur_bytes = {}, 0
            cur[name] = t
            cur_bytes += nbytes
        if cur:
            shards.append(self._write_shard(out_dir, len(shards), cur))
        n = len(shards)
        weight_map, total = {}, 0
        for i, (tmp, names, size) in enumerate(shards):
            final = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            os.replace(os.path.join(out_dir, tmp), os.path.join(out_dir, final))
            weight_map.update({k: final for k in names})
            total += size
        if n > 1:
            with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
                json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
        cfg = dict(self.config)
        if dtype is not None:
            cfg["torch_dtype"] = str(dtype).replace("torch.", "")
        if cfg:
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
        if self.ledger or self.parent_manifest:
            from .manifest import build_manifest

            with open(os.path.join(out_dir, "weightkit_manifest.json"), "w") as f:
                json.dump(build_manifest(self), f, indent=2)
        if copy_extras and self.extras_dir and os.path.isdir(self.extras_dir):
            if os.path.abspath(self.extras_dir) != os.path.abspath(out_dir):
                for fn in os.listdir(self.extras_dir):
                    src = os.path.join(self.extras_dir, fn)
                    if fn in _NO_COPY or fn.endswith(_WEIGHT_EXT) or not os.path.isfile(src):
                        continue
                    shutil.copy2(src, os.path.join(out_dir, fn))

    @staticmethod
    def _write_shard(out_dir, idx, tensors):
        tmp = f".shard-{idx:05d}.tmp.safetensors"
        save_file(_dedupe(tensors), os.path.join(out_dir, tmp), metadata={"format": "pt"})
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        return tmp, list(tensors), size


def _dedupe(tensors: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """safetensors refuses tensors that share storage; clone those (and views)."""
    seen = set()
    out = {}
    for k, v in tensors.items():
        v = v.detach().contiguous()
        st = v.untyped_storage()
        ptr = st.data_ptr()
        if ptr in seen or st.nbytes() != v.numel() * v.element_size():
            v = v.clone()
        else:
            seen.add(ptr)
        out[k] = v
    return out
