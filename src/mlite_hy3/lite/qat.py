"""Hy3 declarations for Megatron Lite's model-agnostic QAT primitive."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

    from megatron.lite.primitive.quantization.qat import QATSpec


# Hy3 quantizes routed experts only. The generic primitive already excludes
# router gates, embeddings, and output heads. These model-local components
# additionally exclude attention, dense MLPs, and the sparse block's shared MLP.
_HY3_MXFP4_QAT_IGNORES = ("attn", "mlp", "shared_mlp")

# Hy3 uses TE GroupedLinear for routed experts. The deployed TE exposes one
# parameter per local expert (weight0, weight1, ...) instead of K3's ordinary
# Linear.weight leaves or newer TE's stacked GroupedLinear.weight. Keep that
# model-specific name map declarative; fake quantization remains in MLite.
_HY3_QAT_TARGETS = (
    (
        re.compile(r"(?:^|\.)moe\.experts\.fc[12]$"),
        re.compile(r"weight\d+$"),
    ),
)


def normalize_hy3_qat_spec(
    config: QATSpec | dict[str, Any] | None,
) -> QATSpec:
    from megatron.lite.primitive.quantization.qat import (
        QATSpec,
        normalize_qat_spec,
    )

    spec = normalize_qat_spec(config)
    if spec.enabled and spec.format == "mxfp4":
        generic_ignores = QATSpec().ignore_patterns
        spec = replace(
            spec,
            ignore_patterns=tuple(
                dict.fromkeys(
                    (
                        *generic_ignores,
                        *spec.ignore_patterns,
                        *_HY3_MXFP4_QAT_IGNORES,
                    )
                )
            ),
        )
    return spec


def apply_hy3_qat_to_chunks(
    chunks: list[nn.Module],
    config: QATSpec | dict[str, Any] | None,
) -> dict[str, int]:
    """Apply generic QAT plus Hy3's split GroupedLinear parameter name map."""
    import torch.nn.utils.parametrize as parametrize

    from megatron.lite.primitive.quantization.qat import (
        WeightFakeQuant,
        apply_qat_to_chunks,
    )

    spec = normalize_hy3_qat_spec(config)
    stats = apply_qat_to_chunks(chunks, spec)
    if not spec.enabled:
        return stats

    for chunk in chunks:
        for module_name, module in chunk.named_modules():
            parameter_pattern = next(
                (
                    parameters
                    for modules, parameters in _HY3_QAT_TARGETS
                    if modules.search(module_name)
                ),
                None,
            )
            if parameter_pattern is None or not spec.targets_module(module_name):
                continue
            for parameter_name, parameter in tuple(
                module.named_parameters(recurse=False)
            ):
                if not parameter_pattern.fullmatch(parameter_name):
                    continue
                if parametrize.is_parametrized(module, parameter_name):
                    continue
                parametrize.register_parametrization(
                    module,
                    parameter_name,
                    WeightFakeQuant(spec, parameter.shape),
                    unsafe=True,
                )
                stats["quantized_modules"] += 1
    return stats


__all__ = ["apply_hy3_qat_to_chunks", "normalize_hy3_qat_spec"]
