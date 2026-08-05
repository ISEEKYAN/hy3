from __future__ import annotations

import torch
import torch.nn as nn

from mlite_hy3.config import Hy3Config
from mlite_hy3.lite.checkpoint import Hy3WeightSpec, iter_checkpoint_tensors


def _config() -> Hy3Config:
    return Hy3Config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=2,
        vocab_size=16,
        intermediate_size=12,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        num_nextn_predict_layers=1,
    )


def test_weight_spec_maps_dense_sparse_shared_bias_and_mtp_names():
    weight_map = Hy3WeightSpec(_config()).weight_map()

    assert weight_map["layers.0.mlp.gate_up.linear.weight"] == [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ]
    assert weight_map["layers.1.moe.router.expert_bias"] == [
        "model.layers.1.mlp.expert_bias"
    ]
    assert weight_map["layers.1.moe.shared_mlp.down.linear.weight"] == [
        "model.layers.1.mlp.shared_mlp.down_proj.weight"
    ]
    assert weight_map["mtp.layers.0.enorm.weight"] == ["model.layers.2.enorm.weight"]
    assert weight_map["mtp.layers.0.final_layernorm.weight"] == [
        "model.layers.2.final_layernorm.weight"
    ]


def test_weight_spec_round_trips_qkv_dense_and_shared_swiglu():
    config = _config()
    spec = Hy3WeightSpec(config)
    hidden = config.hidden_size

    query = torch.arange(8 * hidden).reshape(8, hidden)
    key = torch.arange(4 * hidden).reshape(4, hidden) + 1000
    value = torch.arange(4 * hidden).reshape(4, hidden) + 2000
    packed = spec.hf_to_native("layers.0.attn.qkv.linear.weight", [query, key, value])
    qkv = dict(spec.native_to_hf("layers.0.attn.qkv.linear.weight", packed))
    assert torch.equal(qkv["model.layers.0.self_attn.q_proj.weight"], query)
    assert torch.equal(qkv["model.layers.0.self_attn.k_proj.weight"], key)
    assert torch.equal(qkv["model.layers.0.self_attn.v_proj.weight"], value)

    gate = torch.arange(12 * hidden).reshape(12, hidden)
    up = gate + 1000
    for native_name in (
        "layers.0.mlp.gate_up.linear.weight",
        "layers.1.moe.shared_mlp.gate_up.linear.weight",
    ):
        packed = spec.hf_to_native(native_name, [gate, up])
        exported = dict(spec.native_to_hf(native_name, packed))
        assert any(
            torch.equal(tensor, gate)
            for name, tensor in exported.items()
            if "gate_proj" in name
        )
        assert any(
            torch.equal(tensor, up)
            for name, tensor in exported.items()
            if "up_proj" in name
        )


def test_weight_spec_declares_gqa_layout_for_all_fused_qkv_weights():
    config = _config()
    spec = Hy3WeightSpec(config)

    expected = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    assert spec.qkv_spec("layers.0.attn.qkv.linear.weight") == expected
    assert (
        spec.qkv_spec("mtp.layers.0.transformer_layer.attn.qkv.linear.weight")
        == expected
    )
    assert spec.qkv_spec("layers.0.attn.qkv.linear.layer_norm_weight") is None
    assert spec.qkv_spec("layers.0.attn.proj.linear.weight") is None


def test_weight_spec_exports_runtime_expert_parameter_names():
    spec = Hy3WeightSpec(_config())
    hidden = spec.config.hidden_size
    gate = torch.arange(4 * hidden).reshape(4, hidden)
    up = gate + 1000
    packed = torch.cat([gate, up])

    fc1 = dict(spec.native_to_hf("layers.1.moe.experts.fc1.weight0", packed))
    fc2 = dict(
        spec.native_to_hf(
            "layers.1.moe.experts.fc2.weight0",
            torch.arange(hidden * 4).reshape(hidden, 4),
        )
    )

    assert torch.equal(fc1["model.layers.1.mlp.experts.0.gate_proj.weight"], gate)
    assert torch.equal(fc1["model.layers.1.mlp.experts.0.up_proj.weight"], up)
    assert "model.layers.1.mlp.experts.0.down_proj.weight" in fc2


def test_checkpoint_tensor_iterator_includes_only_mapped_persistent_buffers():
    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("expert_bias", torch.zeros(1), persistent=True)
            self.register_buffer("scratch", torch.zeros(1), persistent=False)

    weight_map = {"weight": ["weight"], "expert_bias": ["expert_bias"]}
    tensors = dict(iter_checkpoint_tensors(Module(), weight_map))
    assert set(tensors) == {"weight", "expert_bias"}
