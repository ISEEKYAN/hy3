from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from megatron.lite.primitive.quantization.qat import apply_qat_to_chunks

from mlite_hy3.lite.qat import normalize_hy3_qat_spec


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 32))


def _toy_hy3_chunk() -> nn.Module:
    chunk = nn.Module()
    chunk.embed = nn.Module()
    chunk.embed.embedding = _Linear()
    chunk.layers = nn.ModuleList([nn.Module(), nn.Module()])

    dense = chunk.layers[0]
    dense.attn = nn.Module()
    dense.attn.qkv = _Linear()
    dense.attn.proj = _Linear()
    dense.mlp = nn.Module()
    dense.mlp.gate_up = _Linear()
    dense.mlp.down = _Linear()

    sparse = chunk.layers[1]
    sparse.attn = nn.Module()
    sparse.attn.qkv = _Linear()
    sparse.attn.proj = _Linear()
    sparse.moe = nn.Module()
    sparse.moe.router = nn.Module()
    sparse.moe.router.gate = _Linear()
    sparse.moe.shared_mlp = nn.Module()
    sparse.moe.shared_mlp.gate_up = _Linear()
    sparse.moe.shared_mlp.down = _Linear()
    sparse.moe.experts = nn.Module()
    sparse.moe.experts.fc1 = _Linear()
    sparse.moe.experts.fc2 = _Linear()

    chunk.head = _Linear()
    return chunk


def test_mxfp4_qat_only_parametrizes_routed_expert_linears():
    chunk = _toy_hy3_chunk()
    spec = normalize_hy3_qat_spec(
        {"enabled": True, "format": "mxfp4", "ignore_patterns": ()}
    )

    stats = apply_qat_to_chunks([chunk], spec)
    parametrized = {
        name
        for name, module in chunk.named_modules()
        if parametrize.is_parametrized(module, "weight")
    }

    assert parametrized == {
        "layers.1.moe.experts.fc1",
        "layers.1.moe.experts.fc2",
    }
    assert stats["quantized_modules"] == 2


def test_disabled_qat_is_inert():
    chunk = _toy_hy3_chunk()

    stats = apply_qat_to_chunks([chunk], normalize_hy3_qat_spec(None))

    assert not any(
        parametrize.is_parametrized(module, "weight") for module in chunk.modules()
    )
    assert stats["quantized_modules"] == 0
