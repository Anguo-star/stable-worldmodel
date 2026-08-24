#!/usr/bin/env python3
"""Transfer the simplest Motion joint relation to Contact Friction.

This discovery arm starts directly from the published PushT LeWM checkpoint,
keeps its absolute temporal coordinates and every parameter, and trains once
on the frozen Contact population.  The only auxiliary is the same center-free
conditional response plus target-stationary assignment relation used by the
teacher-free Motion Cartesian candidate.  There is no Motion warm start,
teacher, function anchor, residual reset, new module, new parameter, or added
inference computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_contact_friction_residual_transition_source_v1 as contact,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_canonical_response_only_freeze_v1 as joint,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_v1
    as motion_absolute,
)


CANDIDATE = "pusht_contact_friction_visible_joint_absolute_single_stage_v1"
CHECKPOINT_SHA256 = contact.INITIAL_CHECKPOINT_SHA256
OPTIMIZER_STEPS = 2048
SEED = contact.SEED
INPUT_BASIS = "absolute"
OUTPUT_BASIS = "absolute"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args(list(argv))
    checkpoint = args.checkpoint.expanduser().resolve()
    checks = {
        "checkpoint_exists": checkpoint.is_file(),
        "published_pusht_checkpoint_exact": (
            checkpoint.is_file() and _sha256(checkpoint) == CHECKPOINT_SHA256
        ),
        "model_exact": args.model == "lewm",
        "variant_exact": args.variant in {None, contact.NATIVE_VARIANT},
        "seed_exact": args.seed == SEED,
        "optimizer_steps_exact": args.optimizer_steps == OPTIMIZER_STEPS,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Contact visible-joint absolute contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return args


def _rewrite_report(
    output: Path,
    *,
    state: dict[str, Any],
) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    model_state = state.get("model") or {}
    model = model_state["model"]
    basis = state.get("absolute_basis") or {}
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    trainable_roots = tuple(
        result["representation_freeze"]["trainable_top_level_modules"]
    )
    checks = {
        "published_pusht_checkpoint_exact": (
            result["source_checkpoint"]["sha256"] == CHECKPOINT_SHA256
        ),
        "single_stage_contact_adaptation": True,
        "optimizer_steps_exact": (
            int(result["optimizer_steps"]) == OPTIMIZER_STEPS
        ),
        "seed_exact": int(result["seed"]) == SEED,
        "native_mse_sigreg_0p09_retained": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "absolute_input_basis": (
            basis.get("input_basis") == INPUT_BASIS
            and config.get("temporal_input_basis") == INPUT_BASIS
        ),
        "absolute_output_basis": (
            basis.get("output_basis") == OUTPUT_BASIS
            and config.get("temporal_output_basis") == OUTPUT_BASIS
        ),
        "no_parameter_reinitialization": (
            basis.get("output_projection_reinitialized") is False
            and "residual_output_initialization"
            not in result["source_checkpoint"]
        ),
        "one_joint_auxiliary_call_per_step": (
            int(state["loss_calls"]) == OPTIMIZER_STEPS
        ),
        "centered_response_and_assignment_recorded": all(
            name in (state.get("last_components") or {})
            for name in (
                "centered_response_loss",
                "canonical_margin_loss",
                "excluded_direct_common_center_mse",
            )
        ),
        "batch_partition_exact": (
            result["batch"]["original"] == 64
            and result["batch"]["hidden"] == 64
            and result["batch"]["hidden_pairs"] == 32
        ),
        "terminal_only_hidden_supervision": (
            result["prediction_supervision"][
                "hidden_rows_transition_indices"
            ]
            == [2]
        ),
        "only_existing_predictor_and_head_trainable": (
            trainable_roots == ("pred_proj", "predictor")
        ),
        "action_encoder_frozen_and_unchanged": (
            model_state["action_encoder_state_sha256_before"]
            == joint.hashing._state_sha256(model.action_encoder)
            and all(
                not parameter.requires_grad
                for parameter in model.action_encoder.parameters()
            )
        ),
        "pred_proj_buffers_frozen": (
            model_state["pred_proj_buffer_state_sha256_before"]
            == joint.anchor._buffer_state_sha256(model.pred_proj)
            and model.pred_proj.training is False
        ),
        "saved_model_parameter_count_unchanged": (
            model_state["parameter_count"]
            == sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise RuntimeError(
            f"Contact visible-joint terminal contract failed: {failed}"
        )
    result["contact_visible_joint_absolute_single_stage_contract"] = {
        "checks": checks,
        "objective": (
            "native_mse + 0.09*SIGReg + "
            "0.09*(centered_response + canonical_assignment_0p5)"
        ),
        "initialization": {
            "role": "published_standard_pusht_baseline",
            "sha256": CHECKPOINT_SHA256,
            "contact_adaptation_steps_before_joint_training": 0,
            "random_initialization": False,
        },
        "first_loss_components": state["first_components"],
        "last_loss_components": state["last_components"],
        "fresh_optimizer_steps": OPTIMIZER_STEPS,
        "temporal_input_basis": INPUT_BASIS,
        "temporal_output_basis": OUTPUT_BASIS,
        "parameter_reinitialization": False,
        "training_only_frozen_teacher": False,
        "source_function_anchor": False,
        "learned_parameters_added_to_saved_model": 0,
        "model_modules_added_to_saved_model": 0,
        "inference_compute_added": 0,
        "single_seed_discovery": True,
        "public_test_opened": False,
    }
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = _args(effective)
    trainer = contact._load_trainer()
    contact._install_runtime(
        trainer,
        candidate=CANDIDATE,
        fixed_checkpoint_step=OPTIMIZER_STEPS,
    )
    trainer.trainer = types.SimpleNamespace(mixed=trainer.mixed)
    original = {
        "source_checkpoint": joint.SOURCE_CHECKPOINT_SHA256,
        "optimizer_steps": joint.OPTIMIZER_STEPS,
    }
    joint.SOURCE_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    joint.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    try:
        state = motion_absolute._install_absolute_freeze_only(
            trainer,
            native_install=joint._install_freeze_only,
        )
        previous = list(sys.argv)
        try:
            sys.argv = [str(THIS_SOURCE), *effective]
            trainer.main()
        finally:
            sys.argv = previous
        if not args.dry_run:
            output = args.output.expanduser().resolve()
            report = _rewrite_report(output, state=state)
            sidecar = output / (
                "contact_visible_joint_absolute_single_stage_method_v1.json"
            )
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "candidate": CANDIDATE,
                        "source": str(THIS_SOURCE),
                        "source_sha256": _sha256(THIS_SOURCE),
                        "source_checkpoint_sha256": CHECKPOINT_SHA256,
                        "fresh_optimizer_steps": OPTIMIZER_STEPS,
                        "training_report": str(report),
                        "training_report_sha256": _sha256(report),
                        "single_seed_discovery": True,
                        "public_test_opened": False,
                        "learned_parameters_added_to_saved_model": 0,
                        "model_modules_added_to_saved_model": 0,
                        "inference_compute_added": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return 0
    finally:
        joint.SOURCE_CHECKPOINT_SHA256 = original["source_checkpoint"]
        joint.OPTIMIZER_STEPS = original["optimizer_steps"]


if __name__ == "__main__":
    raise SystemExit(main())
