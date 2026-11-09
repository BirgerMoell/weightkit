"""Architecture specs: where the interesting tensors live in a checkpoint."""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Union

from .store import TensorStore

ROLE_GROUPS = {
    "attn": ("q", "k", "v", "o"),
    "mlp": ("gate", "up", "down"),
    "all_linear": ("q", "k", "v", "o", "gate", "up", "down"),
}
SINGLETON_ROLES = ("embed", "lm_head", "final_norm")


@dataclass
class ArchSpec:
    """Name templates for a decoder-only transformer.

    The default spec covers the Llama family (Llama, Mistral, Qwen2/3, Yi,
    SmolLM, ...).  Pass a custom spec for anything else.
    """

    name: str = "llama"
    embed: str = "model.embed_tokens.weight"
    lm_head: str = "lm_head.weight"
    final_norm: str = "model.norm.weight"
    layer_prefix: str = "model.layers.{i}."
    modules: Dict[str, str] = field(
        default_factory=lambda: dict(
            q="self_attn.q_proj",
            k="self_attn.k_proj",
            v="self_attn.v_proj",
            o="self_attn.o_proj",
            gate="mlp.gate_proj",
            up="mlp.up_proj",
            down="mlp.down_proj",
            attn_norm="input_layernorm",
            mlp_norm="post_attention_layernorm",
        )
    )
    norm_offset: float = 0.0  # Gemma stores (scale - 1)

    # ---- naming
    def layer_module(self, i: int) -> str:
        return self.layer_prefix.format(i=i).rstrip(".")

    def layer_name(self, i: int, role: str, kind: str = "weight") -> str:
        return f"{self.layer_prefix.format(i=i)}{self.modules[role]}.{kind}"

    def role_name(self, role: str, i: Optional[int] = None, kind: str = "weight") -> str:
        if role == "embed":
            return self.embed
        if role == "lm_head":
            return self.lm_head
        if role == "final_norm":
            return self.final_norm
        if i is None:
            raise ValueError(f"role {role!r} needs a layer index")
        return self.layer_name(i, role, kind)

    def layer_regex(self):
        pre = re.escape(self.layer_prefix).replace(re.escape("{i}"), r"(\d+)")
        pre = pre.replace(r"\{i\}", r"(\d+)")
        return re.compile("^" + pre)

    def layer_ids(self, store: TensorStore) -> List[int]:
        rx = self.layer_regex()
        ids = set()
        for n in store.names():
            m = rx.match(n)
            if m:
                ids.add(int(m.group(1)))
        return sorted(ids)

    def n_layers(self, store: TensorStore) -> int:
        ids = self.layer_ids(store)
        return (max(ids) + 1) if ids else 0

    def is_layer_tensor(self, name: str) -> bool:
        return self.layer_regex().match(name) is not None

    # ---- shapes from config
    def dims(self, store: TensorStore) -> Dict[str, int]:
        c = store.config
        h = c.get("num_attention_heads")
        d = c.get("hidden_size")
        if d is None and self.embed in store:
            d = store[self.embed].shape[1]
        if h is None:
            raise ValueError("config.json with num_attention_heads is required for this op")
        kv = c.get("num_key_value_heads") or h
        hd = c.get("head_dim") or d // h
        return dict(hidden=d, heads=h, kv_heads=kv, head_dim=hd, rep=h // kv)

    # ---- selection
    def select(
        self,
        store: TensorStore,
        roles: Optional[Sequence[str]] = None,
        layers: Union[None, str, Sequence[Union[int, str]]] = None,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        kind: str = "weight",
    ) -> List[str]:
        """Resolve a selection to existing tensor names (stable order)."""
        names: List[str] = []
        if include:
            names = [n for n in store.names() if any(fnmatch.fnmatch(n, p) for p in include)]
        else:
            if isinstance(roles, str):
                roles = [roles]
            expanded: List[str] = []
            for r in roles or ["all_linear"]:
                expanded.extend(ROLE_GROUPS.get(r, (r,)))
            ids = parse_layers(layers, self.n_layers(store))
            for r in expanded:
                if r in SINGLETON_ROLES:
                    n = self.role_name(r, kind=kind)
                    if n in store:
                        names.append(n)
                else:
                    if r not in self.modules:
                        raise ValueError(f"unknown role {r!r}")
                    for i in ids:
                        n = self.layer_name(i, r, kind)
                        if n in store:
                            names.append(n)
        if exclude:
            names = [n for n in names if not any(fnmatch.fnmatch(n, p) for p in exclude)]
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out


def parse_layers(spec, n_layers: int) -> List[int]:
    """None -> all; "0-3,7,-1" or [0, "2-4", -1] -> explicit (end-inclusive)."""
    if spec is None:
        return list(range(n_layers))
    if isinstance(spec, (int, str)):
        spec = [spec]
    out: List[int] = []
    for item in spec:
        if isinstance(item, int):
            out.append(item % n_layers if item < 0 else item)
            continue
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            m = re.fullmatch(r"(-?\d+)(?:-(\d+))?", part)
            if not m:
                raise ValueError(f"bad layer spec {part!r}")
            a = int(m.group(1))
            if m.group(2) is None:
                out.append(a % n_layers if a < 0 else a)
            else:
                out.extend(range(a, int(m.group(2)) + 1))
    return [i for i in dict.fromkeys(out) if 0 <= i < n_layers]


LLAMA = ArchSpec()
GEMMA = ArchSpec(name="gemma", norm_offset=1.0)
PRESETS = {"llama": LLAMA, "gemma": GEMMA}


def detect_arch(store: TensorStore, strict: bool = True) -> ArchSpec:
    mt = str(store.config.get("model_type", "")).lower()
    arch = GEMMA if mt.startswith("gemma") else LLAMA
    if strict and arch.embed not in store:
        raise ValueError(
            f"Could not find {arch.embed!r}; this checkpoint does not follow the "
            "default naming. Pass a custom ArchSpec (pipeline: `arch:` mapping)."
        )
    return arch


def arch_from_config(spec) -> Optional[ArchSpec]:
    if spec is None:
        return None
    if isinstance(spec, ArchSpec):
        return spec
    if isinstance(spec, str):
        return PRESETS[spec]
    spec = dict(spec)
    base = PRESETS[spec.pop("preset", "llama")]
    d = {**base.__dict__, **spec}
    if "modules" in spec:
        d["modules"] = {**base.modules, **spec["modules"]}
    return ArchSpec(**d)
