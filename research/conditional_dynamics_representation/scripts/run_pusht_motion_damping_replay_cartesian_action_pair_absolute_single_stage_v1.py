#!/usr/bin/env python3
"""Teacher-free replay pairing in the original LeWM temporal coordinates.

The matched-budget residual single-stage arm learned Motion ICL but lost CEM
after its existing output projection was reset to introduce residual dynamics.
This one-factor MVE keeps the published PushT LeWM's original absolute-history
and absolute-future interpretation and all of its weights.  It adds only the
same 1,024-step teacher-free replay 2x2 training relation already validated in
the residual arm.  No parameter, module, initialization reset, or inference
path changes.
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
    run_pusht_motion_damping_replay_cartesian_action_pair_single_stage_v1 as single_stage,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_replay_cartesian_action_pair_v1 as parent,
)


CANDIDATE = (
    "pusht_motion_damping_replay_cartesian_action_pair_"
    "absolute_single_stage_v1"
)
PUSHT_BASELINE_SHA256 = single_stage.PUSHT_BASELINE_SHA256
OPTIMIZER_STEPS = 1024
SEED = 14321
INPUT_BASIS = "absolute"
OUTPUT_BASIS = "absolute"
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
            "absolute single-stage replay-pair contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return args


def _install_absolute_freeze_only(
    trainer: Any,
    *,
    native_install: Any,
) -> dict[str, Any]:
    state = native_install(trainer)
    mixed = trainer.trainer.mixed
    residual_loader = mixed.load_model_for_variant
    residual_model_config = mixed.model_config
    state["absolute_basis"] = None

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = residual_loader(*args, **kwargs)
        model.temporal_input_basis = INPUT_BASIS
        model.temporal_output_basis = OUTPUT_BASIS
        state["absolute_basis"] = {
            "input_basis": getattr(model, "temporal_input_basis", None),
            "output_basis": getattr(model, "temporal_output_basis", None),
            "output_projection_reinitialized": False,
        }
        updated = dict(receipt)
        updated.update(
            {
                "temporal_input_basis": INPUT_BASIS,
                "temporal_output_basis": OUTPUT_BASIS,
                "output_projection_reinitialized": False,
            }
        )
        updated.pop("residual_output_initialization", None)
        return model, updated

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(residual_model_config(*args, **kwargs))
        config["temporal_input_basis"] = INPUT_BASIS
        config["temporal_output_basis"] = OUTPUT_BASIS
        return config

    mixed.load_model_for_variant = load_model
    mixed.model_config = model_config
    return state


def _rewrite_absolute_report(
    output: Path,
    *,
    freeze_state: dict[str, Any],
) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    pair_contract = result["motion_cartesian_action_pair_contract"]
    initialization = pair_contract.pop("continuation_source")
    initialization.pop("source_optimizer_steps", None)
    initialization.update(
        {
            "role": "published_standard_pusht_baseline",
            "sha256": PUSHT_BASELINE_SHA256,
            "motion_adaptation_steps_before_joint_training": 0,
            "random_initialization": False,
        }
    )
    basis = freeze_state.get("absolute_basis") or {}
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    checks = {
        "published_pusht_baseline_sha_exact": (
            result["source_checkpoint"]["sha256"] == PUSHT_BASELINE_SHA256
        ),
        "motion_adaptation_steps_before_joint_training_zero": True,
        "joint_optimizer_steps_exact": (
            int(result["optimizer_steps"]) == OPTIMIZER_STEPS
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
        ),
        "teacher_free": pair_contract["training_only_frozen_teacher"] is False,
        "saved_model_parameter_count_unchanged": (
            pair_contract["checks"]["saved_model_parameter_count_unchanged"]
            is True
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"absolute single-stage terminal failure: {checks}")
    pair_contract["initialization_source"] = initialization
    pair_contract["fresh_optimizer_steps"] = OPTIMIZER_STEPS
    pair_contract["checks"].update(checks)
    result["absolute_single_stage_joint_training_contract"] = {
        "checks": checks,
        "removed_stage": "separate_2048_step_native_motion_adaptation",
        "initialization_source": initialization,
        "joint_optimizer_steps": OPTIMIZER_STEPS,
        "temporal_input_basis": INPUT_BASIS,
        "temporal_output_basis": OUTPUT_BASIS,
        "parameter_reinitialization": False,
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
            "temporal_input_basis": INPUT_BASIS,
            "temporal_output_basis": OUTPUT_BASIS,
            "parameter_reinitialization": False,
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
    native_validate = canonical._validate_args
    native_install = canonical._install_freeze_only
    original_values = {
        "canonical_source_sha": canonical.SOURCE_CHECKPOINT_SHA256,
        "canonical_steps": canonical.OPTIMIZER_STEPS,
        "canonical_validate": canonical._validate_args,
        "canonical_install": canonical._install_freeze_only,
        "cartesian_steps": cartesian.OPTIMIZER_STEPS,
        "parent_candidate": parent.CANDIDATE,
        "parent_source": parent.THIS_SOURCE,
    }

    def validate_absolute(runtime_argv: Sequence[str]) -> argparse.Namespace:
        return _validate_args(runtime_argv)

    def install_absolute(trainer: Any) -> dict[str, Any]:
        state["freeze"] = _install_absolute_freeze_only(
            trainer,
            native_install=native_install,
        )
        return state["freeze"]

    canonical.SOURCE_CHECKPOINT_SHA256 = PUSHT_BASELINE_SHA256
    canonical.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    canonical._validate_args = validate_absolute
    canonical._install_freeze_only = install_absolute
    cartesian.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    parent.CANDIDATE = CANDIDATE
    parent.THIS_SOURCE = THIS_SOURCE
    try:
        status = parent.main(effective)
        if not args.dry_run:
            _rewrite_absolute_report(
                args.output.expanduser().resolve(),
                freeze_state=state["freeze"],
            )
        return status
    finally:
        canonical.SOURCE_CHECKPOINT_SHA256 = original_values[
            "canonical_source_sha"
        ]
        canonical.OPTIMIZER_STEPS = original_values["canonical_steps"]
        canonical._validate_args = original_values["canonical_validate"]
        canonical._install_freeze_only = original_values["canonical_install"]
        cartesian.OPTIMIZER_STEPS = original_values["cartesian_steps"]
        parent.CANDIDATE = original_values["parent_candidate"]
        parent.THIS_SOURCE = original_values["parent_source"]


if __name__ == "__main__":
    raise SystemExit(main())
