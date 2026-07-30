from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from mlite_hy3.lite.qat import apply_hy3_qat_to_chunks


_ROUTED_EXPERT_WEIGHT = re.compile(r"^layers\.\d+\.moe\.experts\.fc[12]\.weight\d+$")


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 32))


class _SplitGroupedLinear(nn.Module):
    """CPU stand-in for Hy3's deployed TE GroupedLinear parameter surface."""

    def __init__(self, num_experts: int):
        super().__init__()
        for expert in range(num_experts):
            self.register_parameter(
                f"weight{expert}",
                nn.Parameter(torch.randn(2, 32)),
            )
        self.register_parameter(
            "weight_scale0",
            nn.Parameter(torch.ones(2, 1), requires_grad=False),
        )


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
    sparse.moe.experts.fc1 = _SplitGroupedLinear(2)
    sparse.moe.experts.fc2 = _SplitGroupedLinear(2)

    chunk.head = _Linear()
    return chunk


def test_mxfp4_qat_only_parametrizes_routed_expert_linears():
    chunk = _toy_hy3_chunk()
    routed_expert_weights = {
        name
        for name, _ in chunk.named_parameters()
        if _ROUTED_EXPERT_WEIGHT.fullmatch(name)
    }
    expected_masters = {
        f"{module}.parametrizations.{parameter}.original"
        for name in routed_expert_weights
        for module, parameter in (name.rsplit(".", 1),)
    }

    stats = apply_hy3_qat_to_chunks(
        [chunk], {"enabled": True, "format": "mxfp4", "ignore_patterns": ()}
    )

    masters = {
        name
        for name, _ in chunk.named_parameters()
        if ".parametrizations.weight" in name and name.endswith(".original")
    }

    assert masters == expected_masters
    assert len(masters) == len(routed_expert_weights)
    assert stats["quantized_modules"] == len(routed_expert_weights)
    assert not any("weight_scale" in name for name in masters)
    assert not any("attn" in name for name in masters)
    assert not any("shared_mlp" in name for name in masters)
    assert not any(".mlp." in name for name in masters)


def test_disabled_qat_is_inert():
    chunk = _toy_hy3_chunk()

    stats = apply_hy3_qat_to_chunks([chunk], None)

    assert not any(parametrize.is_parametrized(module) for module in chunk.modules())
    assert stats["quantized_modules"] == 0
