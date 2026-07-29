"""Hy3 declarations for Megatron Lite's model-agnostic QAT primitive."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from megatron.lite.primitive.quantization.qat import QATSpec


# Hy3 quantizes routed experts only. The generic primitive already excludes
# router gates, embeddings, and output heads. These model-local components
# additionally exclude attention, dense MLPs, and the sparse block's shared MLP.
_HY3_MXFP4_QAT_IGNORES = ("attn", "mlp", "shared_mlp")


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


__all__ = ["normalize_hy3_qat_spec"]
