from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
]


def _configs():
    from transformers import HYV3Config

    from mlite_hy3.config import Hy3Config

    common = dict(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        intermediate_size=48,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=16,
        router_scaling_factor=2.826,
        rms_norm_eps=1e-5,
        max_position_embeddings=32,
        enable_moe_fp32_combine=False,
    )
    hf = HYV3Config(
        **common,
        architectures=["HYV3ForCausalLM"],
        first_k_dense_replace=1,
        mlp_layer_types=["dense", "sparse"],
        num_nextn_predict_layers=0,
        use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 11_158_840.0},
    )
    native = Hy3Config(
        **common,
        first_k_dense_replace=1,
        num_nextn_predict_layers=0,
    )
    return hf, native


def _tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    assert actual_f.shape == expected_f.shape
    absolute = (actual_f - expected_f).abs()
    reference_scale = expected_f.abs().max().clamp_min(torch.finfo(torch.float32).eps)
    return {
        "max_abs": float(absolute.max()),
        "max_rel": float(absolute.max() / reference_scale),
    }


def _compare_named_tensors(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, object]:
    assert actual.keys() == expected.keys(), {
        "missing": sorted(expected.keys() - actual.keys()),
        "unexpected": sorted(actual.keys() - expected.keys()),
    }
    per_tensor = {
        name: _tensor_metrics(actual[name], expected[name]) for name in expected
    }
    worst_abs = max(per_tensor, key=lambda name: per_tensor[name]["max_abs"])
    worst_rel = max(per_tensor, key=lambda name: per_tensor[name]["max_rel"])
    return {
        "tensor_count": len(per_tensor),
        "max_abs": per_tensor[worst_abs]["max_abs"],
        "max_abs_tensor": worst_abs,
        "max_rel": per_tensor[worst_rel]["max_rel"],
        "max_rel_tensor": worst_rel,
    }


def _read_safetensors(root: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            tensors.update(
                {name: handle.get_tensor(name).clone() for name in handle.keys()}
            )
    assert tensors
    return tensors


def _init_cuda_dist() -> bool:
    assert torch.cuda.is_available(), "CUDA is required for three-way parity."
    torch.cuda.set_device(0)
    if dist.is_initialized():
        return False
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29633")
    dist.init_process_group("nccl", init_method="env://")
    return True


def _batch_first(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    assert tensor.shape[1] == batch_size
    return tensor.transpose(0, 1).contiguous()


def test_megatron_bridge_hy3_config_contract():
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    hf_config, _ = _configs()
    provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(
        load_weights=False
    )

    assert provider.num_layers == 2
    assert provider.hidden_size == 32
    assert provider.num_attention_heads == 4
    assert provider.num_query_groups == 2
    assert provider.kv_channels == 8
    assert provider.num_moe_experts == 4
    assert provider.moe_router_topk == 2
    assert provider.moe_router_score_function == "sigmoid"
    assert provider.moe_router_enable_expert_bias
    assert provider.moe_router_dtype == "fp32"
    assert provider.moe_router_topk_scaling_factor == 2.826
    assert provider.moe_layer_freq == [0, 1]
    assert provider.qk_layernorm
    assert provider.rotary_base == 11_158_840.0


@pytest.mark.gpu
@pytest.mark.distributed
def test_hf_mlite_megatron_bridge_weight_and_logits_parity(tmp_path: Path):
    pytest.importorskip("transformer_engine.pytorch")
    pytest.importorskip("megatron.bridge")
    from transformers import HYV3ForCausalLM

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.lite.runtime.contracts.config import ParallelConfig

    from mlite_hy3.lite import protocol
    from mlite_hy3.lite.checkpoint import export_hf_weights, load_hf_weights

    created_dist = _init_cuda_dist()
    hf_config, native_config = _configs()
    hf_config._attn_implementation = "eager"
    checkpoint = tmp_path / "hf"

    try:
        torch.manual_seed(711)
        hf_model = HYV3ForCausalLM(hf_config).to(device="cuda", dtype=torch.bfloat16)
        hf_model.save_pretrained(checkpoint, safe_serialization=True)
        hf_weights = _read_safetensors(checkpoint)

        impl = protocol.ImplConfig(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1),
            optimizer=None,
            use_deepep=False,
            deterministic=True,
        )
        native_bundle = protocol.build_model(native_config, impl_cfg=impl)
        native_model = native_bundle.chunks[0]
        load_hf_weights(
            native_model,
            str(checkpoint),
            native_config,
            native_bundle.parallel_state,
        )
        mlite_weights = {
            name: tensor.detach().cpu()
            for name, tensor in export_hf_weights(
                native_model,
                native_config,
                native_bundle.parallel_state,
            )
        }

        bridge = AutoBridge.from_hf_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = 1
        provider.expert_tensor_parallel_size = 1
        provider.sequence_parallel = False
        provider.seq_length = 6
        provider.pipeline_dtype = torch.bfloat16
        provider.params_dtype = torch.bfloat16
        provider.finalize()
        bridge_model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            mixed_precision_wrapper=None,
        )[0]
        bridge_weights = {
            name: tensor.detach().cpu()
            for name, tensor in bridge.export_hf_weights(
                [bridge_model],
                cpu=True,
                show_progress=False,
            )
        }

        weight_metrics = {
            "mlite_vs_hf": _compare_named_tensors(mlite_weights, hf_weights),
            "bridge_vs_hf": _compare_named_tensors(bridge_weights, hf_weights),
            "mlite_vs_bridge": _compare_named_tensors(mlite_weights, bridge_weights),
        }

        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            device="cuda",
        )
        position_ids = torch.arange(6, device="cuda").unsqueeze(0).expand(2, -1)
        hf_model.eval()
        native_model.eval()
        bridge_model.eval()
        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
            mlite_logits = _batch_first(
                native_model(input_ids=input_ids)["logits"], input_ids.shape[0]
            )
            bridge_logits = _batch_first(
                bridge_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                ),
                input_ids.shape[0],
            )

        logits_metrics = {
            "mlite_vs_hf": _tensor_metrics(mlite_logits, hf_logits),
            "bridge_vs_hf": _tensor_metrics(bridge_logits, hf_logits),
            "mlite_vs_bridge": _tensor_metrics(mlite_logits, bridge_logits),
        }
        metrics = {"weights": weight_metrics, "logits": logits_metrics}
        print("HY3_THREE_WAY_PARITY " + json.dumps(metrics, sort_keys=True))

        for comparison in weight_metrics.values():
            assert comparison["max_abs"] == 0.0
        for comparison in logits_metrics.values():
            assert comparison["max_abs"] <= 3e-2
            assert comparison["max_rel"] <= 3e-2
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if created_dist and dist.is_initialized():
            dist.destroy_process_group()
