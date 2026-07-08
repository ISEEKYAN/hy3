"""Explicit Megatron Lite registration entry point."""

from __future__ import annotations


def register_model() -> None:
    """Register Hy3 without modifying Megatron Lite's built-in registry."""
    from megatron.lite.model.registry import register_model as register

    register(
        "hy3",
        package="mlite_hy3",
        hf_model_types=["hy_v3"],
        impls={"lite": "mlite_hy3.lite.protocol"},
    )


__all__ = ["register_model"]
