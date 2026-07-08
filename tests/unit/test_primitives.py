from __future__ import annotations

from types import SimpleNamespace

import torch

from mlite_hy3.primitives import Hy3Router


def test_router_uses_bias_only_for_selection_and_persists_it():
    config = SimpleNamespace(
        hidden_size=2,
        num_experts=3,
        num_experts_per_tok=2,
        router_scaling_factor=2.5,
    )
    router = Hy3Router(config)
    with torch.no_grad():
        router.gate.weight.copy_(
            torch.tensor([[2.0, 0.0], [1.0, 0.0], [-2.0, 0.0]])
        )
        router.expert_bias.copy_(torch.tensor([-10.0, 0.0, 10.0]))

    scores, indices = router(torch.tensor([[1.0, 0.0]]))

    assert set(indices[0].tolist()) == {1, 2}
    raw = torch.sigmoid(torch.tensor([1.0, -2.0]))
    expected = raw / raw.sum() * 2.5
    by_index = {index.item(): score for index, score in zip(indices[0], scores[0])}
    torch.testing.assert_close(by_index[1], expected[0])
    torch.testing.assert_close(by_index[2], expected[1])
    assert "expert_bias" in router.state_dict()
    assert dict(router.named_parameters())["expert_bias"].requires_grad is False
    router.to(dtype=torch.bfloat16)
    assert router.gate.weight.dtype == torch.bfloat16
    assert router.expert_bias.dtype == torch.float32
