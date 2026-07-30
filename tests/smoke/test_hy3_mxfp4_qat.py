"""Scheduler-only Hy3 routed-expert MXFP4 QAT acceptance."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]


def _init_cuda_dist() -> bool:
    assert torch.cuda.is_available(), "CUDA is required for Hy3 MXFP4 QAT."
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if dist.is_initialized():
        return False
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29634")
    dist.init_process_group("nccl", init_method="env://")
    return True


def _tiny_config():
    from mlite_hy3.config import Hy3Config

    return Hy3Config(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=128,
        intermediate_size=64,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        moe_intermediate_size=32,
        first_k_dense_replace=1,
        num_nextn_predict_layers=0,
        max_position_embeddings=32,
    )


def test_qat_checkpoint_forward_and_mxfp4_export(tmp_path: Path):
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.runtime.contracts.config import ParallelConfig

    from mlite_hy3.lite import protocol
    from mlite_hy3.lite.checkpoint import export_hf_weights, load_hf_weights

    created_dist = _init_cuda_dist()
    config = _tiny_config()
    parallel = ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1)
    checkpoint = tmp_path / "hf"

    try:
        torch.manual_seed(20260729)
        baseline = protocol.build_model(
            config,
            impl_cfg=protocol.ImplConfig(parallel=parallel, optimizer=None),
        )
        routed_parameter_names = sorted(
            name
            for name, _ in baseline.chunks[0].named_parameters()
            if ".moe.experts.fc" in name
        )
        print(
            "HY3_ROUTED_PARAMETER_NAMES=" + json.dumps(routed_parameter_names),
            flush=True,
        )
        assert routed_parameter_names == [
            "layers.1.moe.experts.fc1.weight0",
            "layers.1.moe.experts.fc1.weight1",
            "layers.1.moe.experts.fc2.weight0",
            "layers.1.moe.experts.fc2.weight1",
        ]
        baseline_weights = dict(
            export_hf_weights(
                baseline.chunks[0],
                config,
                baseline.parallel_state,
                cpu=True,
            )
        )
        save_safetensors(baseline_weights, str(checkpoint))
        del baseline
        gc.collect()
        torch.cuda.empty_cache()

        qat = protocol.build_model(
            config,
            impl_cfg=protocol.ImplConfig(
                parallel=parallel,
                optimizer=None,
                qat={"enabled": True, "format": "mxfp4"},
            ),
        )
        model = qat.chunks[0]
        load_hf_weights(model, str(checkpoint), config, qat.parallel_state)
        reloaded = dict(export_hf_weights(model, config, qat.parallel_state, cpu=True))
        assert baseline_weights.keys() == reloaded.keys()
        checkpoint_differences = {
            name: float(
                (baseline_weights[name].float() - reloaded[name].float()).abs().max()
            )
            for name in baseline_weights
            if not torch.equal(baseline_weights[name], reloaded[name])
        }
        assert not checkpoint_differences, checkpoint_differences

        masters = {
            name
            for name, _ in model.named_parameters()
            if ".parametrizations.weight" in name and name.endswith(".original")
        }
        assert len(masters) == len(routed_parameter_names), {
            "masters": sorted(masters),
            "routed_parameter_names": routed_parameter_names,
        }
        assert all(".moe.experts." in name for name in masters), sorted(masters)
        assert not any(".shared_mlp." in name for name in masters)
        assert qat.extras["qat"]["quantized_modules"] == len(masters)

        exported = dict(
            export_hf_weights(
                model,
                config,
                qat.parallel_state,
                target="mxfp4",
                cpu=True,
            )
        )
        routed = {
            name
            for name in exported
            if ".mlp.experts." in name and name.endswith(".weight")
        }
        assert routed
        assert all(exported[name].dtype == torch.uint8 for name in routed)
        assert all(f"{name[:-7]}.weight_scale" in exported for name in routed)
        assert exported["model.embed_tokens.weight"].dtype == torch.bfloat16
        assert exported["lm_head.weight"].dtype == torch.bfloat16

        input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
        labels = torch.tensor([[2, 3, 4, 5]], device="cuda")
        output = model(input_ids=input_ids, labels=labels)
        output["loss"].backward()
        assert torch.isfinite(output["loss"])
        assert all(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name in masters
        )
        print(
            "HY3_QAT_SMOKE="
            + json.dumps(
                {
                    "checkpoint_tensors": len(reloaded),
                    "mxfp4_routed_weights": len(routed),
                    "qat_modules": len(masters),
                    "loss": float(output["loss"].detach()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if created_dist and dist.is_initialized():
            dist.destroy_process_group()


def test_real_hy3_weights_match_modelopt_bitwise():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real-weight ModelOpt parity.")
    checkpoint = Path(os.environ["HY3_REAL_CHECKPOINT"])
    minimum_elements = int(os.environ.get("HY3_QAT_MIN_REAL_ELEMENTS", "99090432"))

    from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor

    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader
    from megatron.lite.primitive.quantization.mxfp4 import (
        MXFP4_BLOCK_SIZE,
        dequantize_mxfp4,
        quantize_mxfp4,
    )
    from megatron.lite.primitive.quantization.qat import (
        QATSpec,
        fake_quantize_weight,
    )

    from mlite_hy3.config import Hy3Config
    from mlite_hy3.lite.checkpoint import Hy3WeightSpec

    config = Hy3Config.from_hf(str(checkpoint))
    spec = Hy3WeightSpec(config)
    expected = sorted(
        hf_name
        for native_name, hf_names in spec.weight_map().items()
        if spec.is_expert(native_name)
        for hf_name in hf_names
    )
    assert expected
    qat_spec = QATSpec(enabled=True, format="mxfp4")
    compared = 0
    tensors = 0

    with SafeTensorReader(str(checkpoint)) as reader:
        for name in expected:
            weight = reader.get_tensor(name).cuda()
            assert weight.ndim == 2
            assert weight.shape[-1] % MXFP4_BLOCK_SIZE == 0

            modelopt, scale = MXFP4QTensor.quantize(weight.clone(), MXFP4_BLOCK_SIZE)
            reference = modelopt.dequantize(
                dtype=torch.float32,
                scale=scale,
                block_sizes={-1: MXFP4_BLOCK_SIZE},
            ).reshape(weight.shape)
            fake = fake_quantize_weight(weight, qat_spec).float()
            packed, encoded_scale = quantize_mxfp4(weight)
            serialized = dequantize_mxfp4(packed, encoded_scale).reshape(weight.shape)
            assert torch.equal(fake, reference), name
            assert torch.equal(serialized, reference), name

            compared += weight.numel()
            tensors += 1
            del weight, modelopt, scale, reference, fake, packed, encoded_scale
            if compared >= minimum_elements:
                break

    assert compared >= minimum_elements
    print(
        "HY3_REAL_MODELOPT_PARITY="
        + json.dumps(
            {
                "elements": compared,
                "expected_routed_tensors": len(expected),
                "tensors_compared": tensors,
            },
            sort_keys=True,
        ),
        flush=True,
    )
