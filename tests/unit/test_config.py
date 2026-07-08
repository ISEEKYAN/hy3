from __future__ import annotations

import pytest

from mlite_hy3.config import Hy3Config


def _tiny_hf_dict() -> dict:
    return {
        "model_type": "hy_v3",
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_hidden_layers": 3,
        "vocab_size": 64,
        "intermediate_size": 24,
        "moe_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "router_scaling_factor": 2.5,
        "qk_norm": True,
        "moe_router_use_sigmoid": True,
        "moe_router_enable_expert_bias": True,
        "route_norm": True,
        "hidden_act": "silu",
        "rope_parameters": {"rope_theta": 12345.0, "rope_type": "default"},
        "num_nextn_predict_layers": 1,
    }


def test_config_preserves_hy3_architecture_contract():
    config = Hy3Config._from_hf_dict(_tiny_hf_dict())

    assert config.layer_types == ["dense", "sparse", "sparse"]
    assert config.rope_theta == 12345.0
    assert config.shared_expert_intermediate_size == 8
    assert config.num_nextn_predict_layers == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qk_norm", False),
        ("moe_router_use_sigmoid", False),
        ("moe_router_enable_expert_bias", False),
        ("route_norm", False),
        ("num_shared_experts", 2),
        ("hidden_act", "gelu"),
    ],
)
def test_config_rejects_unsupported_architecture_drift(field, value):
    source = _tiny_hf_dict()
    source[field] = value

    with pytest.raises(ValueError, match=field):
        Hy3Config._from_hf_dict(source)


def test_config_rejects_wrong_hf_model_type():
    source = _tiny_hf_dict()
    source["model_type"] = "qwen3_moe"

    with pytest.raises(ValueError, match="model_type"):
        Hy3Config._from_hf_dict(source)
