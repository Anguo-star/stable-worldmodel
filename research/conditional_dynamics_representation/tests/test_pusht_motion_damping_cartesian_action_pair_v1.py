import pytest
import torch

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_cartesian_action_pair_v1 as runner,
)


def test_cartesian_stream_keeps_complete_forward_reverse_twins() -> None:
    stream = runner.CartesianTwinBatchStream(
        8,
        batch_size=16,
        seed=7,
    )
    indices = next(iter(stream))
    assert indices.shape == (16,)
    assert sorted(indices.tolist()) == list(range(16))
    for start in range(0, 16, 4):
        assert indices[start : start + 4].tolist() == list(
            range(int(indices[start]), int(indices[start]) + 4)
        )


def test_cartesian_grid_validates_real_four_tuple_contract() -> None:
    pixels = torch.zeros(4, 4, 3, 224, 224, dtype=torch.uint8)
    actions = torch.zeros(4, 4, 5, 2)
    pixels[0, 1, 0, 0, 0] = 1
    pixels[2, 1, 0, 0, 0] = 1
    pixels[1, 1, 0, 0, 0] = 2
    pixels[3, 1, 0, 0, 0] = 2
    pixels[0, 3, 0, 0, 0] = 3
    pixels[1, 3, 0, 0, 0] = 4
    pixels[2, 3, 0, 0, 0] = 5
    pixels[3, 3, 0, 0, 0] = 6
    actions[2:, 2] = 1.0
    checks = runner.validate_cartesian_grid(
        pixels,
        actions,
        template_count=1,
    )
    assert all(checks.values())


def test_cartesian_stream_and_grid_fail_closed() -> None:
    with pytest.raises(ValueError):
        runner.CartesianTwinBatchStream(2, batch_size=16, seed=0)
    with pytest.raises(ValueError):
        runner.validate_cartesian_grid(
            torch.zeros(4, 4, 3, 32, 32, dtype=torch.uint8),
            torch.zeros(4, 4, 5, 2),
            template_count=1,
        )


def test_private_overlay_argument_is_not_forwarded() -> None:
    assert runner._without_overlay_argument(
        ["--seed", "14321", "--cartesian-overlay", "/tmp/grid.pt"]
    ) == ["--seed", "14321"]
