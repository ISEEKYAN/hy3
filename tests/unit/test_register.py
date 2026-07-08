from __future__ import annotations

import importlib
import sys
from types import ModuleType


def test_register_model_is_explicit_and_uses_external_package_paths(monkeypatch):
    calls = []
    registry = ModuleType("megatron.lite.model.registry")
    registry.register_model = lambda *args, **kwargs: calls.append((args, kwargs))

    modules = {
        "megatron": ModuleType("megatron"),
        "megatron.lite": ModuleType("megatron.lite"),
        "megatron.lite.model": ModuleType("megatron.lite.model"),
        "megatron.lite.model.registry": registry,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("mlite_hy3.register", None)
    register = importlib.import_module("mlite_hy3.register")
    assert calls == []

    register.register_model()

    assert calls == [
        (
            ("hy3",),
            {
                "package": "mlite_hy3",
                "hf_model_types": ["hy_v3"],
                "impls": {"lite": "mlite_hy3.lite.protocol"},
            },
        )
    ]
