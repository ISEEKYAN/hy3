# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small Hy3-specific compositions over Megatron Lite primitives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hy3Router(nn.Module):
    """Hy3 sigmoid router with persistent fp32 expert-selection bias."""

    def __init__(self, config) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.scaling_factor = config.router_scaling_factor
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def _apply(self, fn):
        result = super()._apply(fn)
        self.expert_bias.data = self.expert_bias.data.float()
        return result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x.float(), self.gate.weight.float())
        routing_scores = torch.sigmoid(logits)
        selection_scores = routing_scores + self.expert_bias.to(routing_scores.dtype)
        _selection_values, topk_indices = torch.topk(
            selection_scores,
            k=self.topk,
            dim=-1,
            sorted=False,
        )
        topk_scores = routing_scores.gather(-1, topk_indices)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        topk_scores = topk_scores * self.scaling_factor
        return topk_scores.to(x.dtype), topk_indices


class SwiGLUMLP(nn.Module):
    """Bias-free tensor-parallel SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, ps) -> None:
        super().__init__()
        from megatron.lite.primitive.parallel import ColumnParallelLinear, RowParallelLinear

        self.gate_up = ColumnParallelLinear(
            hidden_size,
            intermediate_size * 2,
            ps,
            bias=False,
        )
        self.down = RowParallelLinear(
            intermediate_size,
            hidden_size,
            ps,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Hy3MTPDecoderLayer(nn.Module):
    """MTP decoder layer using Hy3's non-zero-centered RMSNorm weights."""

    def __init__(
        self,
        *,
        hidden_size: int,
        rms_norm_eps: float,
        ps,
        embedding: nn.Module,
        transformer_layer: nn.Module,
        detach_encoder: bool,
    ) -> None:
        super().__init__()
        import transformer_engine.pytorch as te
        from megatron.lite.primitive.parallel import VanillaColumnParallelLinear

        self.ps = ps
        self.embedding = embedding
        self.detach_encoder = detach_encoder
        self.enorm = te.RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            zero_centered_gamma=False,
        )
        self.hnorm = te.RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            zero_centered_gamma=False,
        )
        self.eh_proj = VanillaColumnParallelLinear(
            hidden_size * 2,
            hidden_size,
            ps,
            sp=ps.tp_size > 1,
            gather_output=True,
        )
        self.transformer_layer = transformer_layer
        self.final_layernorm = te.RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            zero_centered_gamma=False,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
        rotary_position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from megatron.lite.primitive.parallel import (
            roll_packed_thd_left,
            scatter_to_sequence_parallel,
        )

        attention_position_ids = (
            rotary_position_ids if rotary_position_ids is not None else position_ids
        )
        input_ids, _ = roll_packed_thd_left(
            input_ids,
            packed_seq_params=packed_seq_params,
            dims=-1,
        )
        if position_ids is not None:
            position_ids, _ = roll_packed_thd_left(
                position_ids,
                packed_seq_params=packed_seq_params,
                dims=-1,
            )
        decoder_input = scatter_to_sequence_parallel(self.embedding(input_ids), self.ps)
        if self.detach_encoder:
            decoder_input = decoder_input.detach()
            hidden_states = hidden_states.detach()
        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = torch.cat((decoder_input, hidden_states), dim=-1)
        hidden_states = scatter_to_sequence_parallel(self.eh_proj(hidden_states), self.ps)
        hidden_states = self.transformer_layer(
            hidden_states,
            position_ids=attention_position_ids,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, input_ids, position_ids


__all__ = ["Hy3MTPDecoderLayer", "Hy3Router", "SwiGLUMLP"]
