"""A tiny pure-torch Llama (RMSNorm, RoPE, GQA, SwiGLU) with Hugging Face
parameter names - used to verify ops numerically without downloading anything."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn

from .store import TensorStore

DEFAULT_CONFIG = dict(
    model_type="llama",
    vocab_size=64,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    mlp_bias=False,
    attention_bias=False,
    tie_word_embeddings=False,
)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


def _rope(x, theta):
    b, h, t, d = x.shape
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = torch.arange(t, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.kv = c["num_attention_heads"], c["num_key_value_heads"]
        self.hd = c.get("head_dim") or c["hidden_size"] // self.h
        d, b = c["hidden_size"], c["attention_bias"]
        self.q_proj = nn.Linear(d, self.h * self.hd, bias=b)
        self.k_proj = nn.Linear(d, self.kv * self.hd, bias=b)
        self.v_proj = nn.Linear(d, self.kv * self.hd, bias=b)
        self.o_proj = nn.Linear(self.h * self.hd, d, bias=b)
        self.theta = c["rope_theta"]

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = _rope(q, self.theta), _rope(k, self.theta)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        out = att.softmax(-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f, b = c["hidden_size"], c["intermediate_size"], c["mlp_bias"]
        self.gate_proj = nn.Linear(d, f, bias=b)
        self.up_proj = nn.Linear(d, f, bias=b)
        self.down_proj = nn.Linear(f, d, bias=b)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, e = c["hidden_size"], c["rms_norm_eps"]
        self.self_attn, self.mlp = Attn(c), MLP(c)
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(d, e), RMSNorm(d, e)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Inner(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList(Layer(c) for _ in range(c["num_hidden_layers"]))
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])


class TinyLlama(nn.Module):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        c = {**DEFAULT_CONFIG, **(config or {})}
        self.config = c
        self.model = Inner(c)
        self.lm_head = nn.Linear(c["hidden_size"], c["vocab_size"], bias=False)
        if c["tie_word_embeddings"]:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, ids):
        x = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


def random_model(seed: int = 0, **config) -> TinyLlama:
    """Randomly initialised model with *non-trivial* norm scales and (optionally) biases."""
    torch.manual_seed(seed)
    m = TinyLlama(config)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
        for mod in m.modules():
            if isinstance(mod, RMSNorm):
                mod.weight.copy_(torch.rand_like(mod.weight) + 0.5)
    return m.eval()


def to_store(model: TinyLlama) -> TensorStore:
    sd: Dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if model.config["tie_word_embeddings"]:
        sd.pop("lm_head.weight", None)
    return TensorStore.from_dict(sd, config=dict(model.config))


def from_store(store: TensorStore) -> TinyLlama:
    """Instantiate a TinyLlama from a (possibly edited) store, using store.config."""
    m = TinyLlama(store.config)
    sd = {n: store[n].float() for n in store.names()}
    if store.config.get("tie_word_embeddings"):
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    m.load_state_dict(sd, strict=True)
    return m.eval()
