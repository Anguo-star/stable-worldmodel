#!/usr/bin/env python3
"""Exact native-loss control for the Contact visible-joint candidate.

This control reuses the candidate's complete 4,096-step runtime path, including
the published initialization, current Contact release, 64/64 mixed batch,
module freeze, optimizer, scheduler, seed, and deterministic paired forward.
The sole optimization change is that the joint auxiliary is multiplied by
zero.  Its value is still recorded as a diagnostic, but it contributes no
gradient and no value to the optimized objective.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_contact_friction_visible_joint_absolute_single_stage_v1 as parent,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_canonical_response_only_freeze_v1 as joint,
)


CANDIDATE = "pusht_contact_friction_visible_joint_native_control_step4096_v1"
OPTIMIZER_STEPS = 4096
AUXILIARY_WEIGHT = 0.0
PARENT_AUXILIARY_WEIGHT = 0.09
PARENT_SIDECAR = "contact_visible_joint_absolute_single_stage_method_v1.json"
SIDECAR = "contact_visible_joint_native_control_step4096_method_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_control_receipts(output: Path) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    contract = result["contact_visible_joint_absolute_single_stage_contract"]
    first = contract["first_loss_components"]
    last = contract["last_loss_components"]
    checks = {
        "published_pusht_checkpoint_exact": contract["checks"][
            "published_pusht_checkpoint_exact"
        ],
        "optimizer_steps_exact": int(result["optimizer_steps"])
        == OPTIMIZER_STEPS,
        "seed_exact": int(result["seed"]) == 13313,
        "batch_rows_exactly_50_50": result["batch"] == {
            "total": 128,
            "original": 64,
            "hidden": 64,
            "hidden_pairs": 32,
            "ordering": "original_then_adjacent_hidden_pairs",
        },
        "original_full_horizon_supervision": result["prediction_supervision"][
            "standard_rows_transition_indices"
        ]
        == [0, 1, 2],
        "contextworld_terminal_only_supervision": result[
            "prediction_supervision"
        ]["hidden_rows_transition_indices"]
        == [2],
        "native_sigreg_0p09_retained": result["regularizer"] == "native"
        and float(result["regularizer_weight"]) == 0.09,
        "only_existing_predictor_and_head_trainable": result[
            "representation_freeze"
        ]["trainable_top_level_modules"]
        == ["pred_proj", "predictor"],
        "auxiliary_computed_each_step": int(
            contract["checks"]["one_joint_auxiliary_call_per_step"]
        )
        == 1,
        "auxiliary_optimization_weight_exactly_zero": AUXILIARY_WEIGHT == 0.0,
        "first_total_equals_native_prediction": first[
            "total_prediction_loss"
        ]
        == first["native_prediction_loss"],
        "last_total_equals_native_prediction": last["total_prediction_loss"]
        == last["native_prediction_loss"],
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise RuntimeError(f"Contact native-control contract failed: {failed}")

    contract.update(
        {
            "candidate": CANDIDATE,
            "checks": {**contract["checks"], **checks},
            "objective": (
                "0.5*original_full_horizon_MSE + "
                "0.5*ContextWorld_terminal_MSE + "
                "0.09*SIGReg(all_128_rows_all_latent_times)"
            ),
            "joint_auxiliary": {
                "computed_on": "64_ContextWorld_rows_32_binary_pairs_terminal_only",
                "diagnostic_only": True,
                "optimization_weight": AUXILIARY_WEIGHT,
                "parent_optimization_weight": PARENT_AUXILIARY_WEIGHT,
            },
            "one_factor_parent": (
                "pusht_contact_friction_visible_joint_absolute_single_stage_"
                "step4096_v1"
            ),
            "one_factor_change": "joint auxiliary weight 0.09 -> 0.0",
            "matched_native_no_aux_control": True,
            "data_routing": {
                "batch_rows": {
                    "original_replay": 64,
                    "ContextWorld_contact": 64,
                },
                "native_prediction": {
                    "original_replay": "weight_0.5_full_horizon_transitions_0_1_2",
                    "ContextWorld_contact": "weight_0.5_terminal_transition_2",
                },
                "native_SIGReg": "all_128_rows_all_latent_times_jointly",
                "joint_auxiliary": "ContextWorld_64_rows_only_weight_0",
            },
        }
    )
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    old_sidecar = output / PARENT_SIDECAR
    sidecar_payload = json.loads(old_sidecar.read_text(encoding="utf-8"))
    sidecar_payload.update(
        {
            "candidate": CANDIDATE,
            "source": str(THIS_SOURCE),
            "source_sha256": _sha256(THIS_SOURCE),
            "training_report": str(report),
            "training_report_sha256": _sha256(report),
            "one_factor_parent": (
                "pusht_contact_friction_visible_joint_absolute_single_stage_"
                "step4096_v1"
            ),
            "one_factor_change": "joint auxiliary weight 0.09 -> 0.0",
            "joint_auxiliary_optimization_weight": AUXILIARY_WEIGHT,
            "matched_native_no_aux_control": True,
        }
    )
    new_sidecar = output / SIDECAR
    new_sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    old_sidecar.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    original = {
        "candidate": parent.CANDIDATE,
        "steps": parent.OPTIMIZER_STEPS,
        "auxiliary_weight": joint.AUXILIARY_WEIGHT,
    }
    parent.CANDIDATE = CANDIDATE
    parent.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    joint.AUXILIARY_WEIGHT = AUXILIARY_WEIGHT
    try:
        parsed = parent._args(effective)
        status = parent.main(effective)
    finally:
        parent.CANDIDATE = original["candidate"]
        parent.OPTIMIZER_STEPS = original["steps"]
        joint.AUXILIARY_WEIGHT = original["auxiliary_weight"]
    if status == 0 and not parsed.dry_run:
        _rewrite_control_receipts(parsed.output.expanduser().resolve())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
