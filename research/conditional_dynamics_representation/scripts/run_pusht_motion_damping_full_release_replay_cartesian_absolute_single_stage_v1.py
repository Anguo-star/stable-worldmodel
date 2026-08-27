#!/usr/bin/env python3
"""Full-query empirical History x Action training in unchanged absolute LeWM.

This is the missing factorial cell identified by the Motion hidden-planning
analysis: all frozen training queries, one simulator-real empirical-replay
action counterfactual per forward/reverse twin, original absolute LeWM
coordinates, and the established center-free conditional response relation.
It adds no parameter, module, teacher, or inference computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_v1
    as absolute,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_replay_cartesian_action_pair_v1 as replay,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_canonical_response_only_freeze_v1 as joint,
)


ALLOWED_TEMPLATE_COUNTS = (4096, 8192)
ALLOWED_OPTIMIZER_STEPS = (4096, 8192)
CANDIDATE_PREFIX = (
    "pusht_motion_damping_full_release_replay_cartesian_"
    "absolute_single_stage"
)
ALLOWED_AUXILIARY_WEIGHTS = (0.0, 0.09)
MINIMUM_TARGET_CENTERED_ENERGY = 1.0e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cartesian-overlay", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--input-basis")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--auxiliary-weight", type=float, default=0.09)
    args, _ = parser.parse_known_args(list(argv))
    if args.optimizer_steps not in ALLOWED_OPTIMIZER_STEPS:
        raise ValueError(
            f"optimizer steps must be one of {ALLOWED_OPTIMIZER_STEPS}"
        )
    if args.model != "lewm" or args.seed != 14321:
        raise ValueError("full-query discovery is fixed to LeWM seed 14321")
    if args.input_basis != "absolute" or args.output is None:
        raise ValueError("absolute input basis and explicit output are required")
    if args.auxiliary_weight not in ALLOWED_AUXILIARY_WEIGHTS:
        raise ValueError(
            "auxiliary weight must be exactly 0.0 (matched native control) "
            "or 0.09 (conditional-relation candidate)"
        )
    return args


def _without_auxiliary_weight(argv: Sequence[str]) -> list[str]:
    """Remove the wrapper-only two-arm selector before the frozen CLI."""

    values = list(argv)
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] == "--auxiliary-weight":
            if index + 1 >= len(values):
                raise ValueError("--auxiliary-weight requires a value")
            index += 2
            continue
        cleaned.append(values[index])
        index += 1
    return cleaned


def _rewrite_objective_identity(
    output: Path,
    *,
    candidate: str,
    auxiliary_weight: float,
    relation_state: dict[str, int],
) -> Path:
    """Bind the report to the sole candidate/control intervention.

    Both arms consume the identical Cartesian overlay.  The control still
    computes the paired statistic for diagnostics, but its optimization
    coefficient is exactly zero.
    """

    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    contract = result["motion_cartesian_action_pair_contract"]
    first = contract["first_loss_components"]
    last = contract["last_loss_components"]
    is_control = auxiliary_weight == 0.0
    checks = {
        "auxiliary_weight_in_registered_two_arm_set": (
            auxiliary_weight in ALLOWED_AUXILIARY_WEIGHTS
        ),
        "matched_native_total_equals_native_first": (
            (not is_control)
            or first["total_prediction_loss"]
            == first["native_prediction_loss"]
        ),
        "matched_native_total_equals_native_last": (
            (not is_control)
            or last["total_prediction_loss"]
            == last["native_prediction_loss"]
        ),
        "same_50_50_data_contract": (
            result["batch"]["total"] == 128
            and result["batch"]["original"] == 64
            and result["batch"]["hidden"] == 64
        ),
        "one_relation_call_per_optimizer_step": (
            relation_state["calls"] == int(result["optimizer_steps"])
        ),
        "relation_group_accounting_exact": (
            relation_state["total_groups"]
            == relation_state["valid_groups"]
            + relation_state["native_only_groups"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"matched objective identity failed: {checks}")
    objective = (
        "native_MSE_on_real_2x2_history_action_grid + "
        "0.09*SIGReg(all_128_rows_all_latent_times)"
    )
    if not is_control:
        objective += " + 0.09*center_free_conditional_response_relation"
    contract.update(
        {
            "candidate": candidate,
            "objective": objective,
            "matched_native_no_aux_control": is_control,
            "joint_auxiliary": {
                "computed_each_step": True,
                "optimization_weight": auxiliary_weight,
                "diagnostic_only": is_control,
                "relation_masking": {
                    **relation_state,
                    "minimum_target_centered_energy": (
                        MINIMUM_TARGET_CENTERED_ENERGY
                    ),
                    "zero_response_groups_receive_native_MSE_only": True,
                },
            },
            "one_factor_comparison": {
                "fixed": (
                    "initialization, full-query Cartesian overlay, 64/64 "
                    "batch, optimizer, schedule, seed, and budget"
                ),
                "changed": "joint auxiliary coefficient 0.0 vs 0.09",
            },
            "checks": {**contract["checks"], **checks},
        }
    )
    payload["provenance"]["method"] = {
        "candidate": candidate,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = output / absolute.EXPECTED_SIDE_CAR
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload.update(
        {
            "candidate": candidate,
            "source": str(THIS_SOURCE),
            "source_sha256": _sha256(THIS_SOURCE),
            "training_report": str(report),
            "training_report_sha256": _sha256(report),
            "joint_auxiliary_optimization_weight": auxiliary_weight,
            "matched_native_no_aux_control": is_control,
        }
    )
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _bind_overlay(overlay: Path) -> dict[str, Any]:
    overlay = overlay.expanduser().resolve()
    receipt_path = overlay.with_suffix(overlay.suffix + ".json")
    if not overlay.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing overlay or receipt: {overlay}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    template_count = int(receipt.get("template_count", -1))
    condition_pair_count = 2 * template_count
    selected_replay_block_count = template_count // 2
    checks = {
        "overlay_hash_self_consistent": (
            receipt.get("overlay_sha256") == _sha256(overlay)
        ),
        "template_count_registered": template_count in ALLOWED_TEMPLATE_COUNTS,
        "condition_pair_count_exact": (
            int(receipt.get("condition_pair_count", -1))
            == condition_pair_count
        ),
        "model_row_count_exact": (
            int(receipt.get("model_row_count", -1)) == 4 * template_count
        ),
        "selected_blocks_exact": (
            int(receipt.get("selected_block_count", -1))
            == selected_replay_block_count
            and int(receipt.get("selected_unique_block_count", -1))
            == selected_replay_block_count
        ),
        "teacher_free": receipt.get("teacher_free") is True,
        "model_boundary_is_pixels_actions_only": (
            receipt.get("hidden_labels_stored") is False
            and receipt.get("pair_metadata_at_model_boundary") is False
        ),
        "public_test_closed": receipt.get("public_test_opened") is False,
        "degenerate_relation_accounting_present": (
            int(receipt.get("zero_action_future_gap_count", -1)) >= 0
            and int(receipt.get("zero_hidden_future_gap_count", -1)) >= 0
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"full-query overlay contract failed: {checks}")

    replay.OVERLAY_SHA256 = str(receipt["overlay_sha256"])
    replay.OVERLAY_RECEIPT_SHA256 = _sha256(receipt_path)
    replay.OVERLAY_TEMPLATE_COUNT = template_count
    replay.OVERLAY_CONDITION_PAIR_COUNT = condition_pair_count
    replay.SELECTED_BLOCK_COUNT = selected_replay_block_count
    replay.validate_receipt(receipt)
    return {
        "overlay": str(overlay),
        "overlay_sha256": replay.OVERLAY_SHA256,
        "receipt": str(receipt_path),
        "receipt_sha256": replay.OVERLAY_RECEIPT_SHA256,
        "checks": checks,
        "template_count": template_count,
        "condition_pair_count": condition_pair_count,
        "selected_replay_block_count": selected_replay_block_count,
        "zero_action_future_gap_count": int(
            receipt["zero_action_future_gap_count"]
        ),
        "zero_hidden_future_gap_count": int(
            receipt["zero_hidden_future_gap_count"]
        ),
    }


def _masked_conditional_relation(
    native_relation: Any,
    state: dict[str, int],
    prediction: Any,
    target: Any,
    groups: Any,
) -> dict[str, Any]:
    """Apply the normalized relation only where its target axis exists."""

    detached = target[groups, -1].detach().float()
    centered = detached - detached.mean(dim=1, keepdim=True)
    energy = centered.square().mean(dim=(1, 2))
    valid = energy > MINIMUM_TARGET_CENTERED_ENERGY
    valid_count = int(valid.sum().item())
    total_count = int(valid.numel())
    state["calls"] += 1
    state["total_groups"] += total_count
    state["valid_groups"] += valid_count
    state["native_only_groups"] += total_count - valid_count
    if valid_count:
        return native_relation(prediction, target, groups[valid])
    zero = prediction.sum() * 0.0
    return {
        "loss": zero,
        "response_loss": zero,
        "canonical_margin_loss": zero,
        "normalized_common_center_mse_by_group": zero.reshape(1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = _arguments(effective)
    overlay = _bind_overlay(args.cartesian_overlay)
    relation_state = {
        "calls": 0,
        "total_groups": 0,
        "valid_groups": 0,
        "native_only_groups": 0,
    }
    base_prefix = (
        CANDIDATE_PREFIX
        if overlay["template_count"] == 8192
        else CANDIDATE_PREFIX.replace("full_release", "half_release")
    )
    prefix = (
        base_prefix + "_native_control"
        if args.auxiliary_weight == 0.0
        else base_prefix
    )
    candidate = f"{prefix}_step{args.optimizer_steps}_v1"
    original = {
        "absolute_candidate": absolute.CANDIDATE,
        "absolute_steps": absolute.OPTIMIZER_STEPS,
        "absolute_source": absolute.THIS_SOURCE,
        "replay_candidate": replay.CANDIDATE,
        "replay_source": replay.THIS_SOURCE,
        "joint_auxiliary_weight": joint.AUXILIARY_WEIGHT,
        "joint_relation": joint.objective.canonical_response_only,
        "allow_action_future_degeneracy": (
            absolute.cartesian.ALLOW_ACTION_FUTURE_DEGENERACY
        ),
        "expected_degenerate_action_future_count": (
            absolute.cartesian.EXPECTED_DEGENERATE_ACTION_FUTURE_COUNT
        ),
    }
    absolute.CANDIDATE = candidate
    absolute.OPTIMIZER_STEPS = int(args.optimizer_steps)
    absolute.THIS_SOURCE = THIS_SOURCE
    replay.CANDIDATE = candidate
    replay.THIS_SOURCE = THIS_SOURCE
    joint.AUXILIARY_WEIGHT = float(args.auxiliary_weight)
    absolute.cartesian.ALLOW_ACTION_FUTURE_DEGENERACY = bool(
        overlay["zero_action_future_gap_count"]
    )
    absolute.cartesian.EXPECTED_DEGENERATE_ACTION_FUTURE_COUNT = int(
        overlay["zero_action_future_gap_count"]
    )
    native_relation = joint.objective.canonical_response_only

    def masked_relation(prediction: Any, target: Any, groups: Any):
        return _masked_conditional_relation(
            native_relation,
            relation_state,
            prediction,
            target,
            groups,
        )

    joint.objective.canonical_response_only = masked_relation
    try:
        status = absolute.main(_without_auxiliary_weight(effective))
    finally:
        absolute.CANDIDATE = original["absolute_candidate"]
        absolute.OPTIMIZER_STEPS = original["absolute_steps"]
        absolute.THIS_SOURCE = original["absolute_source"]
        replay.CANDIDATE = original["replay_candidate"]
        replay.THIS_SOURCE = original["replay_source"]
        joint.AUXILIARY_WEIGHT = original["joint_auxiliary_weight"]
        joint.objective.canonical_response_only = original["joint_relation"]
        absolute.cartesian.ALLOW_ACTION_FUTURE_DEGENERACY = original[
            "allow_action_future_degeneracy"
        ]
        absolute.cartesian.EXPECTED_DEGENERATE_ACTION_FUTURE_COUNT = original[
            "expected_degenerate_action_future_count"
        ]
    if status == 0 and "--dry-run" not in effective:
        output = args.output.expanduser().resolve()
        report = _rewrite_objective_identity(
            output,
            candidate=candidate,
            auxiliary_weight=float(args.auxiliary_weight),
            relation_state=relation_state,
        )
        is_control = args.auxiliary_weight == 0.0
        addendum = {
            "schema_version": 1,
            "status": "completed",
            "candidate": candidate,
            "source": str(THIS_SOURCE),
            "source_sha256": _sha256(THIS_SOURCE),
            "factorial_cell": {
                "query_coverage": (
                    f"{overlay['template_count']}_frozen_training_queries"
                ),
                "action_coverage": "empirical_replay_2x2_history_action_grid",
                "temporal_coordinates": "absolute",
                "training_mix": "64_original_plus_64_ContextWorld_rows",
                "objective": (
                    "native_MSE + 0.09*SIGReg"
                    + (
                        ""
                        if is_control
                        else " + 0.09*center_free_conditional_response_relation"
                    )
                ),
            },
            "overlay": overlay,
            "optimizer_steps": int(args.optimizer_steps),
            "auxiliary_weight": float(args.auxiliary_weight),
            "matched_native_no_aux_control": is_control,
            "relation_masking": {
                **relation_state,
                "minimum_target_centered_energy": (
                    MINIMUM_TARGET_CENTERED_ENERGY
                ),
                "zero_response_groups_receive_native_MSE_only": True,
            },
            "training_report": str(report),
            "training_report_sha256": _sha256(report),
            "learned_parameters_added_to_saved_model": 0,
            "model_modules_added_to_saved_model": 0,
            "inference_compute_added": 0,
            "training_only_frozen_teacher": False,
            "explicit_simulator_matched_pairs_retained": True,
            "public_test_opened": False,
        }
        path = output / "full_release_replay_cartesian_method_v1.json"
        path.write_text(
            json.dumps(addendum, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
