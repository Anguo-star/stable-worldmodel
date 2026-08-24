#!/usr/bin/env python3
"""Build a teacher-free empirical-replay History x Action Motion overlay.

The observed branch stays the frozen zero query action.  The counterfactual
branch replays one real five-step action block taken verbatim from the original
PushT expert HDF5 action column, so the alternate action marginal is the
*empirical replay marginal* of that dataset.  It is deliberately NOT the exact
planner/CEM proposal distribution the deployed policy would sample from: no
teacher is queried, nothing is rescaled, amplitude-matched, tuned, or filtered
by outcome, contact, or Development.  Blocks are drawn only at episode-relative
multiples of five, never crossing an episode boundary, and each drawn block is
shared by the adjacent forward/reverse twin templates.

Both damping modes are simulated for both branches, so all four futures are
real, the 2x2 grid is unchanged, and the deployed LeWM boundary still sees only
pixels and actions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    build_pusht_motion_damping_cartesian_action_overlay_v1 as base,
)


DEFAULT_REPLAY_H5 = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "quentinll/lewm-pusht/pusht_expert_train.h5"
)
ACTION_BRANCHES = ("observed_zero", "empirical_replay_5step")
REPLAY_SEED = 20260824
BLOCK_LENGTH = 5
ACTION_LIMIT = 1.0
ZERO_NORM_TOLERANCE = 1.0e-12
PLAYFIELD_MIN = 5.0
PLAYFIELD_MAX = 506.0
QUANTILE_LEVELS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
ACTION_SOURCE_DESCRIPTION = (
    "empirical_replay_marginal_of_original_pusht_expert_train_action_column"
)
NOT_A_PLANNER_NOTE = (
    "verbatim dataset action blocks; this is the empirical replay marginal, "
    "explicitly not the exact planner/CEM proposal distribution"
)


# --------------------------------------------------------------------------
# replay source
# --------------------------------------------------------------------------


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_replay_columns(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only ``action``/``ep_len``/``ep_offset`` from the original HDF5.

    Pixels, state, and proprio columns are never touched; the action column is
    a few tens of megabytes and is read whole so its hash covers the exact
    source the blocks come from.
    """

    import h5py

    with h5py.File(path, "r") as handle:
        action_key = "action" if "action" in handle else "actions"
        if action_key not in handle:
            raise RuntimeError(f"replay HDF5 has no action column: {path}")
        for key in ("ep_len", "ep_offset"):
            if key not in handle:
                raise RuntimeError(f"replay HDF5 lacks {key}: {path}")
        actions = np.asarray(handle[action_key][:])
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
    return actions, lengths, offsets


def eligible_block_starts(
    actions: np.ndarray,
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
) -> np.ndarray:
    """Global start indices of legal aligned five-step blocks.

    A block is eligible when it starts at an episode-relative multiple of five,
    ends at or before ``ep_len``, and every component is finite with absolute
    value at most one.  No other filter is applied.
    """

    actions = np.asarray(actions, dtype=np.float64)
    lengths = np.asarray(ep_len, dtype=np.int64)
    offsets = np.asarray(ep_offset, dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("replay action column must be (T, 2)")
    if lengths.shape != offsets.shape or lengths.ndim != 1:
        raise ValueError("ep_len and ep_offset must be matching 1-D arrays")

    starts: list[int] = []
    for length, offset in zip(lengths.tolist(), offsets.tolist()):
        if length < BLOCK_LENGTH or offset < 0:
            continue
        if offset + length > actions.shape[0]:
            raise ValueError("episode extends past the replay action column")
        for relative in range(0, length - BLOCK_LENGTH + 1, BLOCK_LENGTH):
            start = offset + relative
            block = actions[start : start + BLOCK_LENGTH]
            if not np.all(np.isfinite(block)):
                continue
            if float(np.max(np.abs(block))) > ACTION_LIMIT:
                continue
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def select_replay_blocks(
    actions: np.ndarray,
    starts: np.ndarray,
    *,
    block_count: int,
    seed: int = REPLAY_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``block_count`` eligible blocks without replacement."""

    starts = np.asarray(starts, dtype=np.int64)
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if starts.size < block_count:
        raise RuntimeError(
            f"only {starts.size} eligible replay blocks for {block_count} draws"
        )
    generator = np.random.default_rng(seed)
    chosen = generator.choice(starts.size, size=block_count, replace=False)
    selected = starts[chosen].astype(np.int64)
    if np.unique(selected).size != block_count:
        raise RuntimeError("replay selection repeated a block start")
    actions = np.asarray(actions, dtype=np.float64)
    blocks = np.stack(
        [actions[start : start + BLOCK_LENGTH] for start in selected.tolist()]
    )
    if blocks.shape != (block_count, BLOCK_LENGTH, 2):
        raise RuntimeError(f"unexpected replay block shape {blocks.shape}")
    if not np.all(np.isfinite(blocks)):
        raise RuntimeError("selected replay block is not finite")
    if float(np.max(np.abs(blocks))) > ACTION_LIMIT:
        raise RuntimeError("selected replay block left the legal action box")
    return selected, blocks


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------


@dataclass
class ReplayAssigner:
    """Hand the same drawn block to each adjacent forward/reverse twin pair."""

    blocks: np.ndarray
    calls: int = 0

    def __post_init__(self) -> None:
        self.assigned_block_index: list[int] = []
        self.assigned_template_ids: list[str] = []

    def __call__(self, template: Any) -> np.ndarray:
        index = self.calls
        expected_direction = "forward" if index % 2 == 0 else "reverse"
        template_id = str(template.template_id)
        if not template_id.endswith(expected_direction):
            raise RuntimeError("frozen forward/reverse twin order changed")
        block_index = index // 2
        if block_index >= self.blocks.shape[0]:
            raise RuntimeError("more templates than drawn replay blocks")
        block = np.asarray(self.blocks[block_index], dtype=np.float64)
        if (
            block.shape != (BLOCK_LENGTH, 2)
            or not np.all(np.isfinite(block))
            or float(np.max(np.abs(block))) > ACTION_LIMIT
        ):
            raise RuntimeError("replay block left the frozen action contract")
        self.calls = index + 1
        self.assigned_block_index.append(block_index)
        self.assigned_template_ids.append(template_id)
        return block

    def reuse_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for block_index in self.assigned_block_index:
            counts[block_index] = counts.get(block_index, 0) + 1
        return counts

    def twin_sharing_is_exact(self) -> bool:
        counts = self.reuse_counts()
        if not counts or any(value != 2 for value in counts.values()):
            return False
        return sorted(counts) == list(range(self.blocks.shape[0]))


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        f"q{int(round(level * 100)):03d}": float(np.quantile(values, level))
        for level in QUANTILE_LEVELS
    }


def replay_action_statistics(blocks: np.ndarray) -> dict[str, Any]:
    """Summarize the drawn empirical action blocks (no gating on outcomes)."""

    blocks = np.asarray(blocks, dtype=np.float64)
    steps = blocks.reshape(-1, 2)
    norms = np.linalg.norm(steps, axis=1)
    sequence_rms = np.sqrt(np.mean(np.sum(blocks**2, axis=2), axis=1))

    moving = norms > ZERO_NORM_TOLERANCE
    angles = np.arctan2(steps[moving, 1], steps[moving, 0])
    sector_index = np.floor((angles + np.pi) / (np.pi / 4.0)).astype(np.int64)
    sector_index = np.clip(sector_index, 0, 7)
    sector_counts = [int(np.sum(sector_index == value)) for value in range(8)]
    if angles.size:
        resultant = float(np.abs(np.mean(np.exp(1j * angles))))
    else:
        resultant = 0.0

    turns: list[float] = []
    block_angles = np.arctan2(blocks[:, :, 1], blocks[:, :, 0])
    block_moving = np.linalg.norm(blocks, axis=2) > ZERO_NORM_TOLERANCE
    for row in range(blocks.shape[0]):
        for step in range(BLOCK_LENGTH - 1):
            if not (block_moving[row, step] and block_moving[row, step + 1]):
                continue
            delta = block_angles[row, step + 1] - block_angles[row, step]
            turns.append(float(np.abs(np.arctan2(np.sin(delta), np.cos(delta)))))
    turn_array = np.asarray(turns, dtype=np.float64)

    return {
        "selected_block_count": int(blocks.shape[0]),
        "action_component_minimum": float(np.min(steps)),
        "action_component_maximum": float(np.max(steps)),
        "per_step_action_norm_quantiles": _quantiles(norms),
        "per_sequence_action_rms_quantiles": _quantiles(sequence_rms),
        "angular_sector_counts": sector_counts,
        "angular_sector_definition": (
            "atan2(ay, ax) split into eight pi/4 sectors from -pi; "
            "steps with norm <= 1e-12 excluded"
        ),
        "zero_norm_step_count": int(np.sum(~moving)),
        "circular_resultant_length": resultant,
        "mean_absolute_wrapped_turn_angle": (
            float(np.mean(turn_array)) if turn_array.size else 0.0
        ),
        "fraction_turns_above_quarter_pi": (
            float(np.mean(turn_array > np.pi / 4.0)) if turn_array.size else 0.0
        ),
        "turn_sample_count": int(turn_array.size),
    }


def _bounds_inside_playfield(bounds: dict[str, list[float]]) -> bool:
    return bool(
        all(
            box[0] >= PLAYFIELD_MIN
            and box[1] >= PLAYFIELD_MIN
            and box[2] <= PLAYFIELD_MAX
            and box[3] <= PLAYFIELD_MAX
            for box in bounds.values()
        )
    )


def replay_hard_checks(receipt: dict[str, Any], *, template_count: int) -> dict[str, bool]:
    """Data-identity gates only; physical outcomes remain reported covariates.

    Whether an action contacts the block, changes the hidden-by-action
    interaction, or moves a rendered body partly outside the playfield is an
    outcome of applying an unconditional replay action to a new query state.
    The native Cartesian builder does not select on those outcomes, so neither
    does this empirical-marginal arm.
    """

    sectors = list(receipt["angular_sector_counts"])
    return {
        "selection_without_replacement_exact": (
            int(receipt["selected_unique_block_count"]) == template_count // 2
            and int(receipt["selected_block_count"]) == template_count // 2
        ),
        "forward_reverse_twin_reuse_exact": bool(
            receipt["every_replay_block_used_by_exactly_two_twin_templates"]
        )
        and int(receipt["replay_block_assignment_count"]) == template_count,
        "all_action_components_finite_and_legal": bool(
            receipt["all_replay_action_components_finite_and_legal"]
        ),
        "all_eight_angular_sectors_nonempty": (
            all(count > 0 for count in sectors) if template_count >= 64 else True
        ),
        "exact_history_and_query_prefix_equality": (
            int(
                receipt[
                    "maximum_history_or_query_pixel_difference_across_actions"
                ]
            )
            == 0
        ),
    }


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------


def strip_replay_argument(argv: list[str]) -> tuple[Path, list[str]]:
    """Pull ``--replay-h5`` out of the argv handed to the base builder."""

    remaining: list[str] = []
    replay = DEFAULT_REPLAY_H5
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--replay-h5":
            if index + 1 >= len(argv):
                raise SystemExit("--replay-h5 requires a value")
            replay = Path(argv[index + 1])
            index += 2
            continue
        if token.startswith("--replay-h5="):
            replay = Path(token.split("=", 1)[1])
            index += 1
            continue
        remaining.append(token)
        index += 1
    return replay, remaining


def peek_template_count(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--template-count", type=int, default=base.DEFAULT_TEMPLATE_COUNT
    )
    known, _ = parser.parse_known_args(argv)
    return int(known.template_count)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    replay_path, remaining = strip_replay_argument(arguments)
    template_count = peek_template_count(remaining)
    if template_count <= 0 or template_count % 2:
        raise ValueError("template_count must be positive and even")
    replay_path = replay_path.expanduser()
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)

    actions, lengths, offsets = load_replay_columns(replay_path)
    action_column_sha = _array_sha256(actions)
    starts = eligible_block_starts(actions, lengths, offsets)
    selected_starts, blocks = select_replay_blocks(
        actions,
        starts,
        block_count=template_count // 2,
        seed=REPLAY_SEED,
    )
    assigner = ReplayAssigner(blocks=blocks)
    statistics = replay_action_statistics(blocks)

    native_simulate = base.simulate_motion_damping_clip
    native_build = base._build
    observations: dict[tuple[str, str, str], np.ndarray] = {}
    branch_calls: dict[str, int] = {}
    contact_steps: list[int] = []
    interaction_norms: list[float] = []
    model_bounds_inside: list[bool] = []

    def tracked_simulate(template: Any, *, mode: str, resolution: int) -> Any:
        template_id = str(template.template_id)
        call_index = branch_calls.get(template_id, 0)
        branch_calls[template_id] = call_index + 1
        branch = "zero" if call_index < len(base.ENDPOINT_MODES) else "replay"
        query = np.asarray(template.query_actions, dtype=np.float64)
        position = assigner.assigned_template_ids.index(template_id)
        expected = (
            np.zeros((BLOCK_LENGTH, 2))
            if branch == "zero"
            else np.asarray(
                blocks[assigner.assigned_block_index[position]], dtype=np.float64
            )
        )
        if branch == "replay" and not np.allclose(query, expected):
            raise RuntimeError("replay branch did not receive its drawn block")

        result = native_simulate(template, mode=mode, resolution=resolution)
        key = (template_id, branch, str(mode))
        if key in observations:
            raise RuntimeError(f"duplicate simulator branch {key}")
        observations[key] = np.asarray(
            result["natural_future_snapshot"], dtype=np.float64
        )
        if branch == "replay":
            contact_steps.append(int(result["query_contact_steps"]))
            model_bounds_inside.extend(
                _bounds_inside_playfield(bounds)
                for bounds in result["body_bounds"]
            )

        keys = [
            (template_id, action_branch, hidden_mode)
            for action_branch in ("zero", "replay")
            for hidden_mode in base.ENDPOINT_MODES
        ]
        if all(candidate in observations for candidate in keys):
            zero_delta = observations[keys[0]] - observations[keys[1]]
            replay_delta = observations[keys[2]] - observations[keys[3]]
            interaction_norms.append(
                float(np.linalg.norm(replay_delta - zero_delta))
            )
            for candidate in keys:
                del observations[candidate]
        return result

    def tracked_build(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, receipt = native_build(**kwargs)
        count = int(kwargs["template_count"])
        if count != template_count:
            raise RuntimeError("template_count changed after replay sampling")
        if observations:
            raise RuntimeError("incomplete replay Cartesian simulator grid")
        if (
            len(contact_steps) != 2 * count
            or len(interaction_norms) != count
            or len(model_bounds_inside) != 8 * count
        ):
            raise RuntimeError("replay-support accounting changed")

        receipt.update(statistics)
        receipt.update(
            {
                "action_rule": "verbatim_5step_action_block_from_original_h5",
                "action_source": ACTION_SOURCE_DESCRIPTION,
                "action_distribution_note": NOT_A_PLANNER_NOTE,
                "teacher_free": True,
                "replay_h5": str(replay_path),
                "replay_h5_bytes": int(replay_path.stat().st_size),
                "replay_action_column_sha256": action_column_sha,
                "replay_action_column_shape": list(actions.shape),
                "replay_sampling_seed": REPLAY_SEED,
                "replay_block_length": BLOCK_LENGTH,
                "replay_block_alignment": "episode_relative_multiple_of_5",
                "eligible_block_count": int(starts.size),
                "selected_unique_block_count": int(
                    np.unique(selected_starts).size
                ),
                "selected_block_start_indices_sha256": _array_sha256(
                    np.asarray(selected_starts, dtype=np.int64)
                ),
                "replay_block_assignment_count": int(assigner.calls),
                "every_replay_block_used_by_exactly_two_twin_templates": (
                    assigner.twin_sharing_is_exact()
                ),
                "all_replay_action_components_finite_and_legal": bool(
                    np.all(np.isfinite(blocks))
                    and float(np.max(np.abs(blocks))) <= 1.0
                ),
                "minimum_replay_query_contact_steps": int(min(contact_steps)),
                "mean_replay_query_contact_steps": float(np.mean(contact_steps)),
                "maximum_replay_query_contact_steps": int(max(contact_steps)),
                "all_replay_model_bounds_inside_playfield": bool(
                    all(model_bounds_inside)
                ),
                "replay_model_bounds_inside_count": int(
                    sum(model_bounds_inside)
                ),
                "replay_model_bounds_total_count": int(
                    len(model_bounds_inside)
                ),
                "replay_model_bounds_inside_fraction": float(
                    np.mean(model_bounds_inside)
                ),
                "minimum_hidden_action_interaction_norm": float(
                    min(interaction_norms)
                ),
                "median_hidden_action_interaction_norm": float(
                    np.median(interaction_norms)
                ),
                "interaction_definition": (
                    "norm((future_low_a1-future_high_a1)"
                    "-(future_low_a0-future_high_a0))"
                ),
                "physical_outcomes_used_for_selection": False,
            }
        )
        hard_checks = replay_hard_checks(receipt, template_count=count)
        receipt["replay_support_hard_checks"] = hard_checks
        if not all(hard_checks.values()):
            raise RuntimeError(f"replay-support gate failed: {hard_checks}")
        return payload, receipt

    base.THIS_SOURCE = THIS_SOURCE
    base.ACTION_BRANCHES = ACTION_BRANCHES
    base.alternate_query_actions = assigner
    base.simulate_motion_damping_clip = tracked_simulate
    base._build = tracked_build
    sys.argv = [sys.argv[0], *remaining]
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
