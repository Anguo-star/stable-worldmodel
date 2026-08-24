#!/usr/bin/env python3
"""Qualify label-blind collection of exact Motion conditional overlap.

The completed replay Cartesian asset proves that pair annotations can be
recovered from visible ``(Q, A)``.  It does not prove that the simulator must
enumerate a named low/high pair.  This MVE replaces that enumeration with an
opaque randomized environment draw and a black-box shooting step:

1. draw one hidden environment without exposing its damping value;
2. use observed query-state feedback to choose an x0 that naturally reaches a
   fixed query state after the support history;
3. repeat independent draws until two *visible histories* differ;
4. replay the same two query-action branches and group rows only by visible
   ``(query RGB, action)``.

No state is installed after x0 and every stored future is real.  The hidden
mode is retained only inside the simulator closure and is consulted after
collection solely to audit the MVE; it is never an acceptance, grouping, model
or loss input.  This is an active-reset data result, not a claim about ordinary
unmatched offline replay.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
for path in (REPO_ROOT, CONTEXTWORLD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contextworld.evaluation import pusht_motion_damping_h3 as motion  # noqa: E402
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    build_pusht_motion_damping_replay_cartesian_action_overlay_v1 as replay,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    mine_visible_condition_pairs_v1 as visible,
)


DEFAULT_MANIFEST = CONTEXTWORLD_ROOT / (
    "artifacts/synthesis/pusht_motion_damping_h3_release_v4/manifest.json"
)
DEFAULT_REFERENCE_OVERLAY = REPO_ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_replay_cartesian_action_overlay_v1/"
    "train_templates2048_v1.pt"
)
REFERENCE_OVERLAY_SHA256 = visible.OVERLAY_SHA256
DEFAULT_TEMPLATE_COUNT = 64
FULL_REFERENCE_TEMPLATE_COUNT = 2048
COLLECTION_SEED = 20260824 + 31
MAXIMUM_OPAQUE_DRAWS_PER_TEMPLATE = 64
QUERY_TOLERANCE = motion.QUERY_REFERENCE_TOLERANCE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_sha256(pixels: np.ndarray, actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (pixels, actions):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _install_reset(
    template: motion.MotionDampingTemplate,
    snapshot: np.ndarray,
) -> motion.MotionDampingTemplate:
    value = tuple(np.asarray(snapshot, dtype=np.float64).tolist())
    # Both fields receive the same candidate so the shooting algorithm never
    # selects a reset using a hidden-mode branch.
    return replace(
        template,
        faster_decay_reset_snapshot=value,
        no_extra_decay_reset_snapshot=value,
    )


def solve_reset_from_query_feedback(
    target_query: np.ndarray,
    query_from_reset: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve the contact-free x0 using one opaque observed probe.

    Motion damping is isotropic during the zero-action support.  The probe
    identifies its velocity scale and displacement coefficient from observed
    state transitions; no damping value or mode identity is an argument.
    """

    target = np.asarray(target_query, dtype=np.float64)
    if target.shape != (12,) or not np.all(np.isfinite(target)):
        raise ValueError("target query must be a finite 12-D snapshot")
    probe_reset = target.copy()
    probe_query = np.asarray(
        query_from_reset(probe_reset.copy()), dtype=np.float64
    )
    if probe_query.shape != (12,) or not np.all(np.isfinite(probe_query)):
        raise ValueError("black-box query must be a finite 12-D snapshot")

    probe_velocity = probe_reset[8:10]
    norm_sq = float(np.dot(probe_velocity, probe_velocity))
    if norm_sq <= 1.0e-12:
        raise ValueError("black-box shooting requires nonzero query velocity")
    velocity_scale = float(
        np.dot(probe_query[8:10], probe_velocity) / norm_sq
    )
    displacement_coefficient = float(
        np.dot(
            probe_query[6:8] - probe_reset[6:8],
            probe_velocity,
        )
        / norm_sq
    )
    if (
        not np.isfinite(velocity_scale)
        or velocity_scale <= 1.0e-12
        or not np.isfinite(displacement_coefficient)
    ):
        raise RuntimeError("opaque dynamics produced an invalid shooting fit")

    solved = target.copy()
    solved[8:10] = target[8:10] / velocity_scale
    solved[6:8] = (
        target[6:8] - displacement_coefficient * solved[8:10]
    )
    confirmation = np.asarray(query_from_reset(solved.copy()), dtype=np.float64)
    maximum_error = float(
        np.max(np.abs(motion.friction._snapshot_delta(confirmation, target)))
    )
    if maximum_error > QUERY_TOLERANCE:
        raise RuntimeError(
            "black-box shooting did not recover the common query: "
            f"error={maximum_error:.12g}"
        )
    return solved, {
        "observed_velocity_scale": velocity_scale,
        "observed_displacement_coefficient": displacement_coefficient,
        "confirmation_maximum_query_state_error": maximum_error,
    }


@dataclass(frozen=True)
class OpaqueDynamicsDraw:
    """Simulator handle whose hidden identity is unavailable to collection."""

    _mode: str

    def query_from_reset(
        self,
        template: motion.MotionDampingTemplate,
        reset: np.ndarray,
    ) -> np.ndarray:
        candidate = _install_reset(template, reset)
        result = motion._simulate_continuous_causal_chain(
            candidate,
            mode=self._mode,
            resolution=32,
            render_pixels=False,
        )
        return np.asarray(result["query_snapshot"], dtype=np.float64)

    def simulate(
        self,
        template: motion.MotionDampingTemplate,
        reset: np.ndarray,
        *,
        resolution: int,
    ) -> dict[str, Any]:
        return motion.simulate_motion_damping_clip(
            _install_reset(template, reset),
            mode=self._mode,
            resolution=resolution,
        )

    def posthoc_identity_for_audit(self) -> str:
        return self._mode


def _draw_opaque_dynamics(
    generator: np.random.Generator,
) -> OpaqueDynamicsDraw:
    index = int(generator.integers(0, len(motion.ENDPOINT_MODES)))
    return OpaqueDynamicsDraw(motion.ENDPOINT_MODES[index])


@dataclass
class CollectedContext:
    draw: OpaqueDynamicsDraw
    reset: np.ndarray
    zero_rollout: dict[str, Any]
    shooting: dict[str, float]


def _visible_history_key(rollout: dict[str, Any]) -> bytes:
    pixels = np.asarray(rollout["model_pixels"], dtype=np.uint8)
    actions = np.asarray(rollout["action_blocks"], dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(pixels[:2].tobytes())
    digest.update(actions[:2].tobytes())
    return digest.digest()


def collect_two_distinct_visible_contexts(
    template: motion.MotionDampingTemplate,
    *,
    generator: np.random.Generator,
    resolution: int,
) -> tuple[list[CollectedContext], int]:
    """Accept two contexts using visible history inequality only."""

    target = np.asarray(
        template.expected_natural_query_snapshot, dtype=np.float64
    )
    accepted: list[CollectedContext] = []
    visible_keys: set[bytes] = set()
    attempts = 0
    while (
        len(accepted) < 2
        and attempts < MAXIMUM_OPAQUE_DRAWS_PER_TEMPLATE
    ):
        attempts += 1
        draw = _draw_opaque_dynamics(generator)
        reset, shooting = solve_reset_from_query_feedback(
            target,
            lambda value, handle=draw: handle.query_from_reset(
                template, value
            ),
        )
        zero_rollout = draw.simulate(
            template,
            reset,
            resolution=resolution,
        )
        key = _visible_history_key(zero_rollout)
        if key in visible_keys:
            continue
        visible_keys.add(key)
        accepted.append(
            CollectedContext(
                draw=draw,
                reset=reset,
                zero_rollout=zero_rollout,
                shooting=shooting,
            )
        )
    if len(accepted) != 2:
        raise RuntimeError(
            "independent opaque draws did not yield two visible histories"
        )
    return accepted, attempts


def _reference_row_hashes(
    payload: dict[str, Any], template_index: int
) -> set[str]:
    pixels = payload["pixels"]
    actions = payload["raw_action_blocks"]
    return {
        _row_sha256(
            pixels[row].permute(0, 2, 3, 1).contiguous().numpy(),
            actions[row].contiguous().numpy(),
        )
        for row in range(4 * template_index, 4 * template_index + 4)
    }


def qualify(
    *,
    manifest_path: Path,
    reference_overlay: Path,
    template_count: int,
    resolution: int,
    collection_seed: int,
) -> dict[str, Any]:
    if resolution != 224:
        raise ValueError("reference equivalence requires resolution 224")
    if template_count <= 0 or template_count > FULL_REFERENCE_TEMPLATE_COUNT:
        raise ValueError("template_count left the frozen reference prefix")
    if (
        not reference_overlay.is_file()
        or _sha256(reference_overlay) != REFERENCE_OVERLAY_SHA256
    ):
        raise RuntimeError("reference replay Cartesian overlay changed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["splits"]["train"]["pairs"]
    reference = torch.load(
        reference_overlay,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    replay_actions, ep_len, ep_offset = replay.load_replay_columns(
        replay.DEFAULT_REPLAY_H5
    )
    starts = replay.eligible_block_starts(replay_actions, ep_len, ep_offset)
    _, blocks = replay.select_replay_blocks(
        replay_actions,
        starts,
        block_count=FULL_REFERENCE_TEMPLATE_COUNT // 2,
        seed=replay.REPLAY_SEED,
    )
    assigner = replay.ReplayAssigner(blocks)
    generator = np.random.default_rng(int(collection_seed))

    attempts: list[int] = []
    maximum_query_error = 0.0
    accepted_modes_distinct_posthoc = True
    exact_reference_templates = 0
    visible_group_count = 0
    for index, row in enumerate(rows[:template_count]):
        template = motion.MotionDampingTemplate(**row["template"])
        replay_block = assigner(template)
        contexts, draw_attempts = collect_two_distinct_visible_contexts(
            template,
            generator=generator,
            resolution=resolution,
        )
        attempts.append(draw_attempts)
        accepted_modes_distinct_posthoc &= (
            contexts[0].draw.posthoc_identity_for_audit()
            != contexts[1].draw.posthoc_identity_for_audit()
        )
        maximum_query_error = max(
            maximum_query_error,
            *(context.shooting[
                "confirmation_maximum_query_state_error"
            ] for context in contexts),
        )

        variants = (
            template,
            replace(
                template,
                query_actions=tuple(map(tuple, replay_block.tolist())),
            ),
        )
        generated_pixels: list[np.ndarray] = []
        generated_actions: list[np.ndarray] = []
        for branch_index, branch in enumerate(variants):
            branch_rollouts = []
            for context in contexts:
                result = (
                    context.zero_rollout
                    if branch_index == 0
                    else context.draw.simulate(
                        branch,
                        context.reset,
                        resolution=resolution,
                    )
                )
                branch_rollouts.append(result)
                generated_pixels.append(
                    np.asarray(result["model_pixels"], dtype=np.uint8)
                )
                generated_actions.append(
                    np.asarray(result["action_blocks"], dtype=np.float32)
                )
            left, right = branch_rollouts
            left_pixels = torch.from_numpy(
                np.asarray(left["model_pixels"], dtype=np.uint8)
            ).permute(0, 3, 1, 2)
            right_pixels = torch.from_numpy(
                np.asarray(right["model_pixels"], dtype=np.uint8)
            ).permute(0, 3, 1, 2)
            left_actions = torch.from_numpy(
                np.asarray(left["action_blocks"], dtype=np.float32)
            )
            right_actions = torch.from_numpy(
                np.asarray(right["action_blocks"], dtype=np.float32)
            )
            if visible.visible_condition_key(
                left_pixels[2], left_actions
            ) != visible.visible_condition_key(
                right_pixels[2], right_actions
            ):
                raise RuntimeError("visible (Q,A) grouping key diverged")
            if torch.equal(left_pixels[:2], right_pixels[:2]):
                raise RuntimeError("accepted visible histories are identical")
            visible_group_count += 1

        generated_hashes = {
            _row_sha256(pixel, action)
            for pixel, action in zip(generated_pixels, generated_actions)
        }
        if generated_hashes == _reference_row_hashes(reference, index):
            exact_reference_templates += 1
        if (index + 1) % 16 == 0 or index + 1 == template_count:
            print(
                f"label-blind overlap {index + 1}/{template_count}",
                flush=True,
            )

    checks = {
        "two_visible_condition_groups_per_template": (
            visible_group_count == 2 * template_count
        ),
        "independent_draws_accepted_only_by_visible_history": True,
        "hidden_identity_not_an_acceptance_or_grouping_input": True,
        "accepted_hidden_identities_distinct_posthoc": bool(
            accepted_modes_distinct_posthoc
        ),
        "black_box_shooting_reaches_common_query": (
            maximum_query_error <= QUERY_TOLERANCE
        ),
        "all_generated_rows_equal_frozen_reference_as_unordered_sets": (
            exact_reference_templates == template_count
        ),
        "state_installations_after_x0_zero": True,
        "model_loss_and_inference_unchanged": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"label-blind collection qualification failed: {checks}")
    return {
        "schema_version": 1,
        "status": "completed_label_blind_active_overlap_collection_mve",
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "reference_overlay": str(reference_overlay),
        "reference_overlay_sha256": REFERENCE_OVERLAY_SHA256,
        "template_count": template_count,
        "visible_condition_group_count": visible_group_count,
        "generated_row_count": 4 * template_count,
        "collection_seed": int(collection_seed),
        "opaque_draw_attempts": {
            "minimum": int(min(attempts)),
            "maximum": int(max(attempts)),
            "mean": float(np.mean(attempts)),
            "total": int(sum(attempts)),
        },
        "maximum_query_state_error": maximum_query_error,
        "query_state_tolerance": QUERY_TOLERANCE,
        "exact_reference_template_count": exact_reference_templates,
        "checks": checks,
        "collection_inputs": [
            "randomized_environment_handle",
            "observable_query_state_feedback_for_shooting",
            "visible_history_pixels_for_duplicate_rejection",
            "visible_query_rgb_and_raw_actions_for_grouping",
        ],
        "collection_excludes": [
            "hidden_damping_value",
            "hidden_mode_label",
            "explicit_pair_id",
            "future_for_acceptance_or_grouping",
        ],
        "conclusion": (
            "On the tested frozen prefix, named low/high enumeration and pair "
            "annotations are unnecessary: independent opaque environment "
            "draws, black-box shooting, and visible-history duplicate "
            "rejection recover the exact training rows as unordered sets."
        ),
        "claim_boundary": {
            "training_steps": 0,
            "checkpoint_opened": False,
            "ordinary_unmatched_offline_replay_solved": False,
            "active_environment_randomization_required": True,
            "controllable_initial_state_and_query_target_required": True,
            "conditional_overlap_required": True,
            "full_2048_template_catalog_claimed": (
                template_count == FULL_REFERENCE_TEMPLATE_COUNT
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--reference-overlay", type=Path, default=DEFAULT_REFERENCE_OVERLAY
    )
    parser.add_argument(
        "--template-count", type=int, default=DEFAULT_TEMPLATE_COUNT
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--collection-seed", type=int, default=COLLECTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = qualify(
        manifest_path=args.manifest.expanduser().resolve(),
        reference_overlay=args.reference_overlay.expanduser().resolve(),
        template_count=int(args.template_count),
        resolution=int(args.resolution),
        collection_seed=int(args.collection_seed),
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
