from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.conditional_dynamics_representation.scripts import (
    build_pusht_motion_damping_replay_cartesian_action_overlay_v1 as subject,
)


def _ramp_actions(total: int) -> np.ndarray:
    values = np.arange(total, dtype=np.float64) / float(total)
    return np.stack([values, -values], axis=1)


def _template(index: int):
    direction = "forward" if index % 2 == 0 else "reverse"
    return SimpleNamespace(template_id=f"tpl_{index // 2:03d}_{direction}")


# ---------------------------------------------------------------- eligibility


def test_blocks_are_aligned_and_never_cross_episodes():
    actions = _ramp_actions(30)
    # two episodes: 12 steps then 13 steps, with a two-step gap in between.
    ep_len = np.asarray([12, 13], dtype=np.int64)
    ep_offset = np.asarray([0, 14], dtype=np.int64)
    starts = subject.eligible_block_starts(actions, ep_len, ep_offset)
    assert starts.tolist() == [0, 5, 14, 19]
    for start in starts.tolist():
        episode = 0 if start < 14 else 1
        relative = start - ep_offset[episode]
        assert relative % subject.BLOCK_LENGTH == 0
        assert relative + subject.BLOCK_LENGTH <= ep_len[episode]


def test_short_episodes_yield_no_blocks():
    actions = _ramp_actions(8)
    starts = subject.eligible_block_starts(
        actions, np.asarray([4, 4]), np.asarray([0, 4])
    )
    assert starts.size == 0


def test_illegal_and_nonfinite_blocks_are_dropped():
    actions = np.zeros((15, 2), dtype=np.float64)
    actions[2, 0] = 1.5  # first block out of the action box
    actions[7, 1] = np.nan  # second block not finite
    starts = subject.eligible_block_starts(
        actions, np.asarray([15]), np.asarray([0])
    )
    assert starts.tolist() == [10]


def test_boundary_magnitude_of_exactly_one_stays_eligible():
    actions = np.full((5, 2), -1.0, dtype=np.float64)
    starts = subject.eligible_block_starts(
        actions, np.asarray([5]), np.asarray([0])
    )
    assert starts.tolist() == [0]


# ------------------------------------------------------------------ selection


def test_selection_is_deterministic_and_without_replacement():
    actions = _ramp_actions(200)
    starts = subject.eligible_block_starts(
        actions, np.asarray([200]), np.asarray([0])
    )
    first, blocks = subject.select_replay_blocks(actions, starts, block_count=8)
    second, _ = subject.select_replay_blocks(actions, starts, block_count=8)
    assert np.array_equal(first, second)
    assert np.unique(first).size == 8
    assert blocks.shape == (8, 5, 2)
    for row, start in enumerate(first.tolist()):
        assert np.array_equal(blocks[row], actions[start : start + 5])


def test_selection_changes_with_a_different_seed():
    actions = _ramp_actions(500)
    starts = subject.eligible_block_starts(
        actions, np.asarray([500]), np.asarray([0])
    )
    frozen, _ = subject.select_replay_blocks(actions, starts, block_count=16)
    other, _ = subject.select_replay_blocks(
        actions, starts, block_count=16, seed=subject.REPLAY_SEED + 1
    )
    assert not np.array_equal(frozen, other)


def test_selection_requires_enough_eligible_blocks():
    actions = _ramp_actions(10)
    starts = subject.eligible_block_starts(
        actions, np.asarray([10]), np.asarray([0])
    )
    with pytest.raises(RuntimeError, match="eligible replay blocks"):
        subject.select_replay_blocks(actions, starts, block_count=5)


# ----------------------------------------------------------------- assignment


def test_twin_templates_share_the_same_block():
    blocks = np.arange(3 * 5 * 2, dtype=np.float64).reshape(3, 5, 2) / 100.0
    assigner = subject.ReplayAssigner(blocks=blocks)
    handed = [assigner(_template(index)) for index in range(6)]
    for pair in range(3):
        assert np.array_equal(handed[2 * pair], handed[2 * pair + 1])
        assert np.array_equal(handed[2 * pair], blocks[pair])
    assert assigner.calls == 6
    assert assigner.reuse_counts() == {0: 2, 1: 2, 2: 2}
    assert assigner.twin_sharing_is_exact()


def test_assigner_rejects_broken_twin_order():
    blocks = np.zeros((2, 5, 2), dtype=np.float64)
    assigner = subject.ReplayAssigner(blocks=blocks)
    with pytest.raises(RuntimeError, match="twin order"):
        assigner(SimpleNamespace(template_id="tpl_000_reverse"))


def test_assigner_rejects_more_templates_than_blocks():
    assigner = subject.ReplayAssigner(blocks=np.zeros((1, 5, 2)))
    assigner(_template(0))
    assigner(_template(1))
    with pytest.raises(RuntimeError, match="more templates"):
        assigner(_template(2))


def test_partial_twin_assignment_is_not_exact():
    assigner = subject.ReplayAssigner(blocks=np.zeros((2, 5, 2)))
    assigner(_template(0))
    assert not assigner.twin_sharing_is_exact()


# ------------------------------------------------------------------ statistics


def _sector_spanning_blocks() -> np.ndarray:
    """One constant block per pi/4 sector, aimed at each sector midpoint."""

    angles = np.arange(8, dtype=np.float64) * (np.pi / 4.0) + (np.pi / 8.0)
    directions = 0.5 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return np.repeat(directions[:, None, :], 5, axis=1)


def test_statistics_cover_all_eight_sectors_evenly():
    stats = subject.replay_action_statistics(_sector_spanning_blocks())
    assert stats["angular_sector_counts"] == [5, 5, 5, 5, 5, 5, 5, 5]
    assert stats["selected_block_count"] == 8
    assert stats["zero_norm_step_count"] == 0
    # eight evenly spread directions almost cancel out.
    assert stats["circular_resultant_length"] < 1.0e-9
    # constant blocks never turn.
    assert stats["turn_sample_count"] == 32
    assert stats["mean_absolute_wrapped_turn_angle"] == pytest.approx(0.0)
    assert stats["fraction_turns_above_quarter_pi"] == 0.0
    assert stats["action_component_maximum"] <= 1.0
    assert stats["action_component_minimum"] >= -1.0
    assert set(stats["per_step_action_norm_quantiles"]) == {
        "q000",
        "q005",
        "q025",
        "q050",
        "q075",
        "q095",
        "q100",
    }
    assert set(stats["per_sequence_action_rms_quantiles"]) == set(
        stats["per_step_action_norm_quantiles"]
    )


def test_turn_statistics_react_to_a_reversal_inside_a_block():
    blocks = _sector_spanning_blocks()
    blocks[0, 3:] = -blocks[0, 0]
    stats = subject.replay_action_statistics(blocks)
    assert sum(stats["angular_sector_counts"]) == 40
    # exactly one of the 32 consecutive step pairs turns by pi.
    assert stats["turn_sample_count"] == 32
    assert stats["fraction_turns_above_quarter_pi"] == pytest.approx(1.0 / 32.0)
    assert stats["mean_absolute_wrapped_turn_angle"] == pytest.approx(
        np.pi / 32.0
    )


def test_per_step_norm_quantiles_track_amplitude():
    blocks = np.zeros((2, 5, 2), dtype=np.float64)
    blocks[:, :, 0] = 0.25
    stats = subject.replay_action_statistics(blocks)
    quantiles = stats["per_step_action_norm_quantiles"]
    assert quantiles["q000"] == pytest.approx(0.25)
    assert quantiles["q100"] == pytest.approx(0.25)
    rms = stats["per_sequence_action_rms_quantiles"]
    assert rms["q050"] == pytest.approx(0.25)


def test_zero_norm_steps_are_excluded_from_angular_statistics():
    blocks = np.zeros((2, 5, 2), dtype=np.float64)
    blocks[0, 0] = [0.5, 0.0]
    stats = subject.replay_action_statistics(blocks)
    assert stats["zero_norm_step_count"] == 9
    assert sum(stats["angular_sector_counts"]) == 1
    assert stats["turn_sample_count"] == 0
    assert stats["mean_absolute_wrapped_turn_angle"] == 0.0


def test_playfield_bounds_are_strict():
    inside = {"agent": [5.0, 5.0, 20.0, 20.0], "block": [6, 6, 506, 506]}
    outside = {"agent": [5.0, 5.0, 20.0, 20.0], "block": [6, 6, 506.1, 30]}
    assert subject._bounds_inside_playfield(inside)
    assert not subject._bounds_inside_playfield(outside)


# ----------------------------------------------------------------- hard checks


def _receipt(**overrides):
    receipt = {
        "selected_unique_block_count": 32,
        "selected_block_count": 32,
        "every_replay_block_used_by_exactly_two_twin_templates": True,
        "replay_block_assignment_count": 64,
        "all_replay_action_components_finite_and_legal": True,
        "angular_sector_counts": [3] * 8,
        "maximum_history_or_query_pixel_difference_across_actions": 0,
    }
    receipt.update(overrides)
    return receipt


def test_hard_checks_pass_on_a_clean_receipt():
    checks = subject.replay_hard_checks(_receipt(), template_count=64)
    assert all(checks.values())


@pytest.mark.parametrize(
    "override, failing",
    [
        (
            {"selected_unique_block_count": 31},
            "selection_without_replacement_exact",
        ),
        (
            {"replay_block_assignment_count": 63},
            "forward_reverse_twin_reuse_exact",
        ),
        (
            {"all_replay_action_components_finite_and_legal": False},
            "all_action_components_finite_and_legal",
        ),
        (
            {"angular_sector_counts": [3, 3, 3, 3, 3, 3, 3, 0]},
            "all_eight_angular_sectors_nonempty",
        ),
        (
            {"maximum_history_or_query_pixel_difference_across_actions": 1},
            "exact_history_and_query_prefix_equality",
        ),
    ],
)
def test_each_invariant_has_its_own_gate(override, failing):
    checks = subject.replay_hard_checks(_receipt(**override), template_count=64)
    assert checks[failing] is False
    assert all(value for key, value in checks.items() if key != failing)


def test_sector_coverage_is_not_required_below_64_templates():
    receipt = _receipt(
        selected_unique_block_count=8,
        selected_block_count=8,
        replay_block_assignment_count=16,
        angular_sector_counts=[16, 0, 0, 0, 0, 0, 0, 0],
    )
    checks = subject.replay_hard_checks(receipt, template_count=16)
    assert checks["all_eight_angular_sectors_nonempty"] is True


def test_hard_checks_ignore_contact_and_pixel_gap_outcomes():
    checks = subject.replay_hard_checks(_receipt(), template_count=64)
    joined = " ".join(checks)
    assert "contact" not in joined
    assert "gap" not in joined
    assert "bounds" not in joined
    assert "interaction" not in joined


# --------------------------------------------------------------------- CLI


def test_replay_argument_is_removed_before_delegating():
    replay, remaining = subject.strip_replay_argument(
        [
            "--output",
            "/tmp/out.pt",
            "--replay-h5",
            "/data/replay.h5",
            "--template-count",
            "64",
        ]
    )
    assert str(replay) == "/data/replay.h5"
    assert remaining == [
        "--output",
        "/tmp/out.pt",
        "--template-count",
        "64",
    ]


def test_replay_argument_supports_equals_form_and_default():
    replay, remaining = subject.strip_replay_argument(["--replay-h5=/x/y.h5"])
    assert str(replay) == "/x/y.h5"
    assert remaining == []

    default, rest = subject.strip_replay_argument(["--output", "/tmp/out.pt"])
    assert default == subject.DEFAULT_REPLAY_H5
    assert rest == ["--output", "/tmp/out.pt"]


def test_replay_argument_without_value_is_rejected():
    with pytest.raises(SystemExit):
        subject.strip_replay_argument(["--replay-h5"])


def test_template_count_peek_reads_remaining_argv():
    assert subject.peek_template_count(["--template-count", "16"]) == 16
    assert subject.peek_template_count([]) == 256


def test_frozen_branch_contract():
    assert subject.ACTION_BRANCHES == ("observed_zero", "empirical_replay_5step")
    assert len(subject.ACTION_BRANCHES) == 2
    assert subject.REPLAY_SEED == 20260824
    assert subject.BLOCK_LENGTH == 5
    assert "not the exact planner/CEM" in subject.NOT_A_PLANNER_NOTE
