"""Contract checks for M3 model outputs and architecture variants."""
from __future__ import annotations

import pytest
import torch

from src.models.dl.multitask import make_model, multitask_loss


@pytest.mark.parametrize("input_dim", [520, 1040])
def test_multitask_output_contract_and_embedding(input_dim: int) -> None:
    model = make_model(input_dim=input_dim, seed=17)
    outputs = model(torch.rand(4, input_dim))
    assert outputs["building_logits"].shape == (4, 3)
    assert outputs["floor_logits"].shape == (4, 5)
    assert outputs["coordinates"].shape == (4, 2)
    assert model.encode(torch.rand(4, input_dim)).shape == (4, 128)


def test_factory_is_seeded_and_a9_shapes_are_configurable() -> None:
    left = make_model(seed=17)
    right = make_model(seed=17)
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a, b)
    for depth in (2, 3, 4):
        for width in (256, 512, 1024):
            model = make_model(depth=depth, width=width, seed=2)
            assert model.encode(torch.rand(2, 520)).shape == (2, width)


def test_hierarchical_floor_uses_predicted_building_probabilities() -> None:
    model = make_model(floor_mode="hierarchical", seed=4)
    features = torch.rand(6, 520)
    outputs = model(features)
    outputs["floor_logits"].sum().backward()
    assert model.building_head.weight.grad is not None
    assert model.building_head.weight.grad.abs().sum() > 0


def test_uncertainty_loss_has_trainable_task_scales() -> None:
    model = make_model(loss_weighting="uncertainty", seed=8)
    outputs = model(torch.rand(5, 520))
    batch = {
        "building": torch.tensor([0, 1, 2, 0, 1]),
        "floor": torch.tensor([0, 1, 2, 3, 4]),
        "coordinates": torch.rand(5, 2),
    }
    loss, components = multitask_loss(outputs, batch, model)
    loss.backward()
    assert set(components) == {"building", "floor", "coordinates"}
    assert model.log_task_vars is not None
    assert model.log_task_vars.grad is not None
    assert torch.isfinite(loss)
