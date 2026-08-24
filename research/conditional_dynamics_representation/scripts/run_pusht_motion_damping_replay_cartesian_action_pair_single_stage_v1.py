#!/usr/bin/env python3
"""Run one-stage Motion adaptation with the teacher-free replay 2x2 data.

The completed replay-pair candidate first trains a 2,048-step native Motion
source and then continues it for 1,024 paired steps.  This bounded MVE removes
that *Motion-specific* warm-start stage: it starts from the unchanged published
PushT LeWM baseline, switches the existing Predictor to the same causal-
transition/residual coordinates, and runs the same replay-pair objective for
3,072 consecutive steps.

This is not training LeWM from random initialization, and it does not remove
the explicit simulator-matched 2x2 data.  It tests exactly one simplification:
whether a separate native Motion adaptation stage is required.  No learned
parameter, module, loss family, or inference path is added.
"""

from __future__ import annotations

import argparse
import copy
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
    run_pusht_motion_damping_canonical_response_only_freeze_v1 as canonical,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_cartesian_action_pair_v1 as cartesian,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_replay_cartesian_action_pair_v1 as parent,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_residual_output_basis_v1 as residual,
)


CANDIDATE = "pusht_motion_damping_replay_cartesian_action_pair_single_stage_v1"
PUSHT_BASELINE_SHA256 = (
    "9f13b2c28bb909338047e08d62bf6eb16ea3616cb53e57cfa9b5072b43db1c59"
)
OPTIMIZER_STEPS = 3072
SEED = 14321
INPUT_BASIS = "causal_transition"
EXPECTED_SIDE_CAR = "motion_cartesian_action_pair_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--input-basis")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args(list(argv))
    checkpoint = args.checkpoint.expanduser().resolve()
    checks = {
        "checkpoint_exists": checkpoint.is_file(),
        "published_pusht_baseline_sha_exact": (
            checkpoint.is_file()
            and _sha256(checkpoint) == PUSHT_BASELINE_SHA256
        ),
        "model_exact": args.model == "lewm",
        "variant_exact": args.variant in {None, canonical.causal.NATIVE_VARIANT},
        "seed_exact": args.seed == SEED,
        "optimizer_steps_exact": args.optimizer_steps == OPTIMIZER_STEPS,
        "input_basis_exact": args.input_basis == INPUT_BASIS,
        "output_present": args.output is not None,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "single-stage replay-pair contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return args


def _rewrite_single_stage_report(
    output: Path,
    *,
    residual_state: dict[str, Any],
) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    pair_contract = result["motion_cartesian_action_pair_contract"]
    freeze_contract = copy.deepcopy(pair_contract)
    initialization = freeze_contract.pop("continuation_source")
    initialization.pop("source_optimizer_steps", None)
    initialization.update(
        {
            "role": "published_standard_pusht_baseline",
            "sha256": PUSHT_BASELINE_SHA256,
            "motion_adaptation_steps_before_joint_training": 0,
            "random_initialization": False,
        }
    )
    reset = (residual_state.get("model") or {}).get("initialization") or {}
    checks = {
        "published_pusht_baseline_sha_exact": (
            result["source_checkpoint"]["sha256"] == PUSHT_BASELINE_SHA256
        ),
        "motion_adaptation_steps_before_joint_training_zero": True,
        "joint_optimizer_steps_exact": (
            int(result["optimizer_steps"]) == OPTIMIZER_STEPS
        ),
        "existing_displacement_projection_zero_initialized": (
            reset.get("weight_exactly_zero") is True
            and reset.get("bias_exactly_zero") is True
        ),
        "parameter_count_unchanged_by_residual_basis": (
            (residual_state.get("model") or {}).get("parameter_count_before")
            == (residual_state.get("model") or {}).get("parameter_count_after")
        ),
        "causal_transition_input_basis": (
            (residual_state.get("model") or {}).get("input_basis")
            == INPUT_BASIS
        ),
        "residual_output_basis": (
            (residual_state.get("model") or {}).get("output_basis")
            == "residual"
        ),
        "teacher_free": pair_contract["training_only_frozen_teacher"] is False,
        "saved_model_parameter_count_unchanged": (
            pair_contract["checks"]["saved_model_parameter_count_unchanged"]
            is True
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"single-stage terminal contract failed: {checks}")
    freeze_contract["initialization_source"] = initialization
    freeze_contract["fresh_optimizer_steps"] = OPTIMIZER_STEPS
    freeze_contract["checks"].update(checks)
    result["motion_cartesian_action_pair_contract"] = freeze_contract
    result["single_stage_joint_training_contract"] = {
        "checks": checks,
        "removed_stage": "separate_2048_step_native_motion_adaptation",
        "initialization_source": initialization,
        "joint_optimizer_steps": OPTIMIZER_STEPS,
        "temporal_input_basis": INPUT_BASIS,
        "temporal_output_basis": "residual",
        "residual_output_initialization": reset,
        "explicit_simulator_matched_pairs_retained": True,
        "training_only_frozen_teacher": False,
        "learned_parameters_added_to_saved_model": 0,
        "model_modules_added_to_saved_model": 0,
        "inference_compute_added": 0,
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
    sidecar = output / EXPECTED_SIDE_CAR
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload.update(
        {
            "candidate": CANDIDATE,
            "source": str(THIS_SOURCE),
            "source_sha256": _sha256(THIS_SOURCE),
            "source_checkpoint_sha256": PUSHT_BASELINE_SHA256,
            "motion_adaptation_steps_before_joint_training": 0,
            "fresh_optimizer_steps": OPTIMIZER_STEPS,
            "training_report": str(report),
            "training_report_sha256": _sha256(report),
            "explicit_simulator_matched_pairs_retained": True,
        }
    )
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = _validate_args(effective)
    state: dict[str, Any] = {}

    native_install = canonical._install_freeze_only
    original_values = {
        "canonical_source_sha": canonical.SOURCE_CHECKPOINT_SHA256,
        "canonical_steps": canonical.OPTIMIZER_STEPS,
        "canonical_install": canonical._install_freeze_only,
        "cartesian_steps": cartesian.OPTIMIZER_STEPS,
        "parent_candidate": parent.CANDIDATE,
        "parent_source": parent.THIS_SOURCE,
    }

    def install_single_stage(trainer: Any) -> dict[str, Any]:
        state["residual"] = residual._install_residual_output(
            trainer,
            input_basis=INPUT_BASIS,
        )
        return native_install(trainer)

    canonical.SOURCE_CHECKPOINT_SHA256 = PUSHT_BASELINE_SHA256
    canonical.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    canonical._install_freeze_only = install_single_stage
    cartesian.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    parent.CANDIDATE = CANDIDATE
    parent.THIS_SOURCE = THIS_SOURCE
    try:
        status = parent.main(effective)
        if not args.dry_run:
            _rewrite_single_stage_report(
                args.output.expanduser().resolve(),
                residual_state=state["residual"],
            )
        return status
    finally:
        canonical.SOURCE_CHECKPOINT_SHA256 = original_values[
            "canonical_source_sha"
        ]
        canonical.OPTIMIZER_STEPS = original_values["canonical_steps"]
        canonical._install_freeze_only = original_values["canonical_install"]
        cartesian.OPTIMIZER_STEPS = original_values["cartesian_steps"]
        parent.CANDIDATE = original_values["parent_candidate"]
        parent.THIS_SOURCE = original_values["parent_source"]


if __name__ == "__main__":
    raise SystemExit(main())
