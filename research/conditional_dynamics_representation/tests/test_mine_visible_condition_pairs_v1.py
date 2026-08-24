from __future__ import annotations

import torch
import pytest

from research.conditional_dynamics_representation.scripts import (
    canonical_response_only_v1 as objective,
)
from research.conditional_dynamics_representation.scripts import (
    mine_visible_condition_pairs_v1 as mining,
)


def _visible_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.zeros(4, 4, 3, 224, 224, dtype=torch.uint8)
    actions = torch.zeros(4, 4, 5, 2, dtype=torch.float32)
    pixels[0, 2].fill_(10)
    pixels[1, 2].fill_(10)
    pixels[2, 2].fill_(20)
    pixels[3, 2].fill_(20)
    actions[2:].fill_(0.25)
    for row in range(4):
        pixels[row, 0].fill_(row + 1)
        pixels[row, 1].fill_(row + 5)
        pixels[row, 3].fill_(row + 30)
    return pixels, actions


def test_visible_query_action_recovers_binary_groups() -> None:
    pixels, actions = _visible_fixture()
    groups, keys = mining.mine_visible_condition_groups(pixels, actions)
    assert groups.tolist() == [[0, 1], [2, 3]]
    assert len(keys) == 2


def test_history_and_future_are_not_grouping_inputs() -> None:
    pixels, actions = _visible_fixture()
    original, _ = mining.mine_visible_condition_groups(pixels, actions)
    pixels[:, :2].random_(0, 255)
    pixels[:, 3].random_(0, 255)
    changed, _ = mining.mine_visible_condition_groups(pixels, actions)
    assert torch.equal(original, changed)


def test_nonbinary_visible_collision_fails_closed() -> None:
    pixels, actions = _visible_fixture()
    pixels[2, 2].copy_(pixels[0, 2])
    actions[2].copy_(actions[0])
    with pytest.raises(RuntimeError, match="not uniquely binary"):
        mining.mine_visible_condition_groups(pixels, actions)


def test_canonical_pair_loss_is_group_and_orientation_invariant() -> None:
    generator = torch.Generator().manual_seed(7)
    prediction = torch.randn(4, 2, 8, generator=generator)
    target = torch.randn(4, 2, 8, generator=generator)
    forward = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    reversed_and_permuted = torch.tensor([[3, 2], [1, 0]], dtype=torch.long)
    left = objective.canonical_response_only(
        prediction,
        target,
        forward,
    )["loss"]
    right = objective.canonical_response_only(
        prediction,
        target,
        reversed_and_permuted,
    )["loss"]
    torch.testing.assert_close(left, right, rtol=0.0, atol=1e-7)
