#!/usr/bin/env python3
"""Test rollout-consistent COJA without changing the deployed LeWM.

Both arms start from the same completed four-action planner-curve checkpoint,
use the same 50/50 original/ContextWorld mixture, optimizer, batch stream, and
1,024 fresh steps.  The placebo retains the one-step objective.  The candidate
keeps the total native-MSE and COJA weights fixed but moves a registered share
of each hidden one-step weight to the true second autoregressive rollout step.  No parameter,
module, input, inference path, teacher, or hidden label is added.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterator, Sequence

import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    canonical_response_only_v1 as objective,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_planner_curve_cartesian_absolute_single_stage_v1
    as planner,
)


SOURCE_CHECKPOINT_SHA256 = (
    "0cece445c08f3edabb04fe48716d7b0ce94e2a6d137cf992533aecb15aa14789"
)
OPTIMIZER_STEPS = 1024
HIDDEN_NATIVE_TOTAL_WEIGHT = 0.5
AUXILIARY_TOTAL_WEIGHT = 0.09
ROLLOUT_SHARE = 0.5
ALLOWED_ROLLOUT_SHARES = (0.0, 0.25, 0.5)
RC_BATCH_SIZE = 64
RC_SEED = 20260826
MINIMUM_TARGET_CENTERED_ENERGY = 1.0e-8
MODES = ("placebo_one_step", "rollout2")
# The parent runner temporarily replaces the module attribute with its own
# masking/accounting wrapper.  Keep the mathematical primitive itself here so
# the extra horizons do not masquerade as extra parent-relation calls.
NATIVE_RELATION = objective.canonical_response_only


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rollout2-targets", type=Path, required=True)
    parser.add_argument("--rollout-consistency-mode", choices=MODES, required=True)
    parser.add_argument(
        "--native-rollout-share", type=float, default=ROLLOUT_SHARE
    )
    parser.add_argument(
        "--relation-rollout-share", type=float, default=ROLLOUT_SHARE
    )
    parser.add_argument("--cartesian-overlay", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--input-basis")
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args(list(argv))
    checkpoint = args.checkpoint.expanduser().resolve()
    checks = {
        "source_checkpoint_exact": checkpoint.is_file()
        and _sha256(checkpoint) == SOURCE_CHECKPOINT_SHA256,
        "optimizer_steps_exact": args.optimizer_steps == OPTIMIZER_STEPS,
        "model_exact": args.model == "lewm",
        "seed_exact": args.seed == 14321,
        "input_basis_exact": args.input_basis == "absolute",
        "native_rollout_share_registered": args.native_rollout_share
        in ALLOWED_ROLLOUT_SHARES,
        "relation_rollout_share_registered": args.relation_rollout_share
        in ALLOWED_ROLLOUT_SHARES,
        "placebo_shares_zero": args.rollout_consistency_mode != "placebo_one_step"
        or (
            args.native_rollout_share == 0.0
            and args.relation_rollout_share == 0.0
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"rollout-consistency arguments failed: {checks}")
    return args


def _strip_wrapper_arguments(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    values = list(argv)
    index = 0
    while index < len(values):
        if values[index] in {
            "--rollout2-targets",
            "--rollout-consistency-mode",
            "--native-rollout-share",
            "--relation-rollout-share",
        }:
            if index + 1 >= len(values):
                raise ValueError(f"{values[index]} requires a value")
            index += 2
            continue
        result.append(values[index])
        index += 1
    return result


def _load_rollout2_contract(
    path: Path, *, source_overlay: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    receipt_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("rollout2 targets or receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_overlay = source_overlay.expanduser().resolve()
    checks = {
        "targets_self_hash": receipt.get("rollout2_targets_sha256") == _sha256(path),
        "source_overlay_exact": receipt.get("source_overlay_sha256")
        == _sha256(source_overlay),
        "status_completed": receipt.get("status") == "completed_rollout2_targets",
        "resolution_exact": int(receipt.get("resolution", -1)) == 224,
        "template_count_positive_even": int(receipt.get("template_count", -1)) > 0
        and int(receipt["template_count"]) % 2 == 0,
        "row_count_exact": int(receipt.get("model_row_count", -1))
        == 8 * int(receipt.get("template_count", -1)),
        "row_order_exact": receipt.get("row_order")
        == "template_then_action_branch_then_damping_mode",
        "actions_exact": receipt.get("action_branches") == planner.ACTION_BRANCHES,
        "second_action_zero_hold": receipt.get("second_action_block")
        == "five_zero_actions",
        "teacher_label_outcome_free": receipt.get("teacher_free") is True
        and receipt.get("hidden_label_used") is False
        and receipt.get("future_outcome_used_for_selection") is False
        and receipt.get("contact_used_for_selection") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"rollout2 target contract failed: {checks}")
    return (
        {
            "path": str(path),
            "sha256": _sha256(path),
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "template_count": int(receipt["template_count"]),
            "row_count": int(receipt["model_row_count"]),
            "condition_pair_count": int(receipt["condition_pair_count"]),
            "zero_hidden_future2_gap_count": int(
                receipt["zero_hidden_future2_gap_count"]
            ),
            "checks": checks,
        },
        receipt,
    )


def _masked_relation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    paired_target = target[groups, -1].detach().float()
    centered = paired_target - paired_target.mean(dim=1, keepdim=True)
    energy = centered.square().mean(dim=(1, 2))
    valid = energy > MINIMUM_TARGET_CENTERED_ENERGY
    valid_count = int(valid.sum().item())
    if not valid_count:
        return prediction.sum() * 0.0, 0, int(valid.numel())
    result = NATIVE_RELATION(
        prediction, target, groups[valid]
    )
    return result["loss"], valid_count, int(valid.numel())


def _install_rollout_data(
    trainer: Any,
    *,
    native_install: Callable[..., dict[str, Any]],
    overlay: Path,
    rollout2: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    installed = native_install(trainer, overlay=overlay)
    native_split = trainer.trainer._training_split

    def training_split(*args: Any, **kwargs: Any):
        split = native_split(*args, **kwargs)
        payload = torch.load(
            Path(rollout2["path"]), map_location="cpu", weights_only=True
        )
        row_count = int(rollout2["row_count"])
        future2 = payload.get("future2_pixels")
        checks = {
            "payload_template_count": int(payload.get("template_count", -1))
            == int(rollout2["template_count"]),
            "payload_condition_pair_count": int(
                payload.get("condition_pair_count", -1)
            )
            == int(rollout2["condition_pair_count"]),
            "payload_row_order": payload.get("row_order")
            == "template_then_action_branch_then_damping_mode",
            "future2_shape": isinstance(future2, torch.Tensor)
            and future2.dtype == torch.uint8
            and tuple(future2.shape) == (row_count, 3, 224, 224),
            "source_prefix_available": split.pixels.size(0) >= row_count,
            "source_pixels_shape": tuple(split.pixels.shape[1:])
            == (4, 3, 224, 224),
            "source_actions_shape": tuple(split.action.shape[1:]) == (4, 10),
        }
        if not all(checks.values()):
            raise RuntimeError(f"rollout2 materialization failed: {checks}")
        state["pixels4"] = split.pixels[:row_count]
        state["actions4"] = split.action[:row_count]
        state["future2"] = future2
        state["stream"] = iter(
            planner.PlannerCurveTwinBatchStream(
                int(rollout2["condition_pair_count"]),
                batch_size=RC_BATCH_SIZE,
                seed=RC_SEED,
            )
        )
        state["materialization_checks"] = checks
        return split

    trainer.trainer._training_split = training_split
    return installed


def _install_rollout_objective(
    trainer: Any,
    *,
    freeze_state: dict[str, Any],
    rollout_state: dict[str, Any],
    mode: str,
    native_rollout_share: float,
    relation_rollout_share: float,
) -> None:
    mixed = trainer.trainer.mixed
    base_loss = mixed.mixed_prediction_loss
    rollout_state.update(
        {
            "calls": 0,
            "valid_h1_groups": 0,
            "valid_h2_groups": 0,
            "total_groups": 0,
            "first_components": None,
            "last_components": None,
        }
    )

    def augmented_loss(**kwargs: Any) -> torch.Tensor:
        prediction = kwargs["prediction"]
        if rollout_state.get("stream") is None:
            raise RuntimeError("rollout2 data was not materialized")
        model_state = freeze_state.get("model")
        if not model_state:
            raise RuntimeError("rollout2 model was not loaded")
        model = model_state["model"]
        indices = next(rollout_state["stream"])
        raw_pixels = torch.cat(
            [
                rollout_state["pixels4"][indices],
                rollout_state["future2"][indices].unsqueeze(1),
            ],
            dim=1,
        )
        pixels = mixed.pilot.preprocess_pixels(raw_pixels, prediction.device)
        actions = rollout_state["actions4"][indices].to(
            device=prediction.device, non_blocking=True
        )
        with mixed.temporary_eval_modules(model.predictor, model.pred_proj):
            encoded = model.encode({"pixels": pixels, "action": actions})
            targets = encoded["emb"].detach()
            action_embeddings = encoded["act_emb"].detach()
            prediction1 = model_state["native_predict"](
                targets[:, :3], action_embeddings[:, :3]
            )
            history2 = torch.cat(
                [targets[:, 1:3], prediction1[:, -1:]], dim=1
            )
            prediction2 = model_state["native_predict"](
                history2, action_embeddings[:, 1:4]
            )
        target1 = targets[:, 3:4]
        target2 = targets[:, 4:5]
        predicted1 = prediction1[:, -1:]
        predicted2 = prediction2[:, -1:]
        mse1 = torch.square(predicted1 - target1).mean()
        mse2 = torch.square(predicted2 - target2).mean()
        groups = torch.arange(
            RC_BATCH_SIZE, device=prediction.device, dtype=torch.long
        ).reshape(-1, 2)
        relation1, valid1, total_groups = _masked_relation(
            predicted1, target1, groups
        )
        relation2, valid2, total_groups2 = _masked_relation(
            predicted2, target2, groups
        )
        if total_groups2 != total_groups:
            raise RuntimeError("rollout horizon group counts differ")

        base_value = base_loss(**kwargs)
        native_shift = (
            HIDDEN_NATIVE_TOTAL_WEIGHT
            * native_rollout_share
            * (mse2 - mse1)
        )
        relation_shift = (
            AUXILIARY_TOTAL_WEIGHT
            * relation_rollout_share
            * (relation2 - relation1)
        )
        total = base_value + native_shift + relation_shift
        components = {
            "base_one_step_objective": float(base_value.detach().float()),
            "rollout1_mse": float(mse1.detach().float()),
            "rollout2_mse": float(mse2.detach().float()),
            "rollout1_relation": float(relation1.detach().float()),
            "rollout2_relation": float(relation2.detach().float()),
            "native_weight_shift": float(native_shift.detach().float()),
            "relation_weight_shift": float(relation_shift.detach().float()),
            "total_prediction_loss": float(total.detach().float()),
            "valid_rollout1_groups": valid1,
            "valid_rollout2_groups": valid2,
            "total_groups": total_groups,
        }
        rollout_state["calls"] += 1
        rollout_state["valid_h1_groups"] += valid1
        rollout_state["valid_h2_groups"] += valid2
        rollout_state["total_groups"] += total_groups
        if rollout_state["first_components"] is None:
            rollout_state["first_components"] = copy.deepcopy(components)
        rollout_state["last_components"] = components
        return total

    mixed.mixed_prediction_loss = augmented_loss


def _rewrite_report(
    output: Path,
    *,
    args: argparse.Namespace,
    rollout2: dict[str, Any],
    rollout_state: dict[str, Any],
) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    checks = {
        "source_checkpoint_exact": result["source_checkpoint"]["sha256"]
        == SOURCE_CHECKPOINT_SHA256,
        "fresh_steps_exact": int(result["optimizer_steps"]) == OPTIMIZER_STEPS,
        "one_rollout_call_per_step": int(rollout_state["calls"])
        == OPTIMIZER_STEPS,
        "rollout_data_materialized": bool(
            rollout_state.get("materialization_checks")
            and all(rollout_state["materialization_checks"].values())
        ),
        "same_total_hidden_native_weight": HIDDEN_NATIVE_TOTAL_WEIGHT == 0.5,
        "same_total_auxiliary_weight": AUXILIARY_TOTAL_WEIGHT == 0.09,
        "saved_model_parameter_count_unchanged": result[
            "motion_cartesian_action_pair_contract"
        ]["checks"]["saved_model_parameter_count_unchanged"]
        is True,
    }
    if args.rollout_consistency_mode == "placebo_one_step":
        first = rollout_state["first_components"]
        last = rollout_state["last_components"]
        checks["placebo_shift_exact_zero"] = all(
            value == 0.0
            for row in (first, last)
            for value in (
                row["native_weight_shift"], row["relation_weight_shift"]
            )
        )
    if not all(checks.values()):
        raise RuntimeError(f"rollout-consistency terminal checks failed: {checks}")

    initialization = result[
        "absolute_single_stage_joint_training_contract"
    ]["initialization_source"]
    initialization.update(
        {
            "role": "four_action_planner_curve_step4096_checkpoint",
            "sha256": SOURCE_CHECKPOINT_SHA256,
            "source_optimizer_steps": 4096,
            "fresh_optimizer_state": True,
        }
    )
    initialization.pop("motion_adaptation_steps_before_joint_training", None)
    result["rollout_consistent_coja_contract"] = {
        "mode": args.rollout_consistency_mode,
        "checks": checks,
        "source_checkpoint": initialization,
        "rollout2_targets": rollout2,
        "objective": {
            "original_rows": "unchanged full-horizon native MSE",
            "hidden_native_total_weight": HIDDEN_NATIVE_TOTAL_WEIGHT,
            "hidden_horizon_weights": {
                "rollout1": HIDDEN_NATIVE_TOTAL_WEIGHT
                * (1.0 - args.native_rollout_share),
                "rollout2": HIDDEN_NATIVE_TOTAL_WEIGHT
                * args.native_rollout_share,
            },
            "conditional_relation_total_weight": AUXILIARY_TOTAL_WEIGHT,
            "conditional_relation_horizon_weights": {
                "rollout1": AUXILIARY_TOTAL_WEIGHT
                * (1.0 - args.relation_rollout_share),
                "rollout2": AUXILIARY_TOTAL_WEIGHT
                * args.relation_rollout_share,
            },
            "second_rollout_is_autoregressive": True,
            "first_prediction_detached_before_second": False,
        },
        "first_loss_components": rollout_state["first_components"],
        "last_loss_components": rollout_state["last_components"],
        "valid_group_totals": {
            "rollout1": rollout_state["valid_h1_groups"],
            "rollout2": rollout_state["valid_h2_groups"],
            "all": rollout_state["total_groups"],
        },
        "learned_parameters_added_to_saved_model": 0,
        "model_modules_added_to_saved_model": 0,
        "inference_compute_added": 0,
        "hidden_label_at_model_or_loss_boundary": False,
        "training_only_frozen_teacher": False,
        "single_seed_discovery": True,
        "public_test_opened": False,
    }
    payload["provenance"]["method"] = {
        "candidate": (
            "pusht_motion_damping_planner_curve_rollout_consistent_"
            f"{args.rollout_consistency_mode}_"
            f"mse{args.native_rollout_share:.3f}_"
            f"rel{args.relation_rollout_share:.3f}_continuation_v1"
        ),
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar = output / "rollout_consistent_coja_v1.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "mode": args.rollout_consistency_mode,
                "native_rollout_share": args.native_rollout_share,
                "relation_rollout_share": args.relation_rollout_share,
                "source": str(THIS_SOURCE),
                "source_sha256": _sha256(THIS_SOURCE),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "fresh_optimizer_steps": OPTIMIZER_STEPS,
                "training_report": str(report),
                "training_report_sha256": _sha256(report),
                "rollout2_targets": rollout2,
                "learned_parameters_added_to_saved_model": 0,
                "model_modules_added_to_saved_model": 0,
                "inference_compute_added": 0,
                "single_seed_discovery": True,
                "public_test_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = _arguments(effective)
    rollout2, _ = _load_rollout2_contract(
        args.rollout2_targets,
        source_overlay=args.cartesian_overlay,
    )
    clean = _strip_wrapper_arguments(effective)
    rollout_state: dict[str, Any] = {}

    full = planner.full
    absolute = full.absolute
    cartesian = full.replay.base
    original = {
        "allowed_steps": full.ALLOWED_OPTIMIZER_STEPS,
        "source_sha": absolute.PUSHT_BASELINE_SHA256,
        "planner_prefix": planner.CANDIDATE_PREFIX,
        "install_data": cartesian._install_cartesian_data,
        "install_absolute": absolute._install_absolute_freeze_only,
    }

    def install_data(trainer: Any, *, overlay: Path) -> dict[str, Any]:
        return _install_rollout_data(
            trainer,
            native_install=original["install_data"],
            overlay=overlay,
            rollout2=rollout2,
            state=rollout_state,
        )

    def install_absolute(
        trainer: Any, *, native_install: Callable[..., dict[str, Any]]
    ) -> dict[str, Any]:
        state = original["install_absolute"](
            trainer, native_install=native_install
        )
        _install_rollout_objective(
            trainer,
            freeze_state=state,
            rollout_state=rollout_state,
            mode=args.rollout_consistency_mode,
            native_rollout_share=float(args.native_rollout_share),
            relation_rollout_share=float(args.relation_rollout_share),
        )
        return state

    full.ALLOWED_OPTIMIZER_STEPS = tuple(
        sorted(set(full.ALLOWED_OPTIMIZER_STEPS) | {OPTIMIZER_STEPS})
    )
    absolute.PUSHT_BASELINE_SHA256 = SOURCE_CHECKPOINT_SHA256
    planner.CANDIDATE_PREFIX = (
        "pusht_motion_damping_planner_curve_rollout_consistent_"
        f"{args.rollout_consistency_mode}_"
        f"mse{args.native_rollout_share:.3f}_"
        f"rel{args.relation_rollout_share:.3f}_continuation"
    )
    cartesian._install_cartesian_data = install_data
    absolute._install_absolute_freeze_only = install_absolute
    try:
        status = planner.main(clean)
    finally:
        full.ALLOWED_OPTIMIZER_STEPS = original["allowed_steps"]
        absolute.PUSHT_BASELINE_SHA256 = original["source_sha"]
        planner.CANDIDATE_PREFIX = original["planner_prefix"]
        cartesian._install_cartesian_data = original["install_data"]
        absolute._install_absolute_freeze_only = original["install_absolute"]
    if status == 0 and "--dry-run" not in clean:
        _rewrite_report(
            args.output.expanduser().resolve(),
            args=args,
            rollout2=rollout2,
            rollout_state=rollout_state,
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
