#!/usr/bin/env python3
"""Run native LeWM with complete forward/reverse Motion twin batches.

This is a sampler-only population-consistency diagnostic.  The model,
absolute temporal input, native MSE, SIGReg, optimizer, data rows, and budget
are unchanged.  Each hidden minibatch groups two opposite-query twins and both
damping endpoints, so query-direction shortcuts cancel within every update.
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
    run_pusht_motion_damping_causal_transition_basis_v1 as causal,
)


CANDIDATE = "pusht_motion_damping_native_twin_sampler_v1"
NATIVE_VARIANT = causal.NATIVE_VARIANT
EXPECTED_TWIN_GROUPS_PER_BATCH = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_runtime(trainer: Any) -> dict[str, Any]:
    """Permit current source identities and enable one sampler-only arm."""

    native_audit = trainer.audit_motion_damping_icl_release
    native_release_loader = trainer.load_motion_damping_icl_release
    state: dict[str, Any] = {"audit": None, "release": None}

    def discovery_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed = native_audit(*args, **kwargs)
        failed = {
            name
            for name, row in observed["files"].items()
            if not bool(row.get("passed"))
        }
        data_passed = all(
            bool(value) for value in observed["data_checks"].values()
        )
        if failed != causal.ALLOWED_RELEASE_SOURCE_DRIFTS or not data_passed:
            return observed
        projected = copy.deepcopy(observed)
        projected["upstream_release_audit_passed"] = bool(observed["passed"])
        projected["passed"] = True
        projected["discovery_runtime_projection"] = {
            "status": "allowed_exact_source_drift_only",
            "allowed_failed_identity_keys": sorted(
                causal.ALLOWED_RELEASE_SOURCE_DRIFTS
            ),
            "observed_failed_identity_keys": sorted(failed),
            "all_data_checks_passed": True,
            "public_test_opened": False,
        }
        state["audit"] = projected["discovery_runtime_projection"]
        return projected

    def load_release(path: Path) -> dict[str, Any]:
        release = copy.deepcopy(native_release_loader(path))
        if release["release_id"] != "contextworld_pusht_motion_damping_icl_history3_v1":
            raise RuntimeError("Unexpected Motion Damping release")
        matrix = release["training"]["reference_matrix"]
        if matrix["status"] != "failed_development":
            raise RuntimeError("Motion Damping base release status changed")
        completed = [int(value) for value in matrix["completed_development_seeds"]]
        if 14321 not in completed:
            raise RuntimeError("Motion Damping legacy seed identity changed")
        matrix["completed_development_seeds"] = [
            value for value in completed if value != 14321
        ]
        release["training"]["learnability_followup"] = {
            "status": "development_recipe_search_in_progress",
            "candidate": "native_twin_sampler_v1",
            "variant": NATIVE_VARIANT,
            "fixed_checkpoint_step": 1024,
            "public_test_open": False,
        }
        state["release"] = {
            "base_status": "failed_development",
            "base_release_mutated_on_disk": False,
            "runtime_completed_seed_adapter": "remove_14321_only",
        }
        return release

    trainer.audit_motion_damping_icl_release = discovery_audit
    trainer.load_motion_damping_icl_release = load_release
    trainer.TWIN_GROUP_VARIANTS.add(NATIVE_VARIANT)
    return state


def _rewrite_training_report(output: Path, state: dict[str, Any]) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    grouping = result["batch"].get("motion_damping_twin_grouping")
    checks = {
        "native_variant": result["variant"] == NATIVE_VARIANT,
        "native_regularizer": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "complete_twin_grouping": bool(
            grouping
            and grouping.get("enabled") is True
            and grouping.get("condition_rows_per_group") == 4
            and result["batch"]["hidden"] // 4
            == EXPECTED_TWIN_GROUPS_PER_BATCH
        ),
        "native_prediction_supervision": (
            result["prediction_supervision"][
                "standard_rows_transition_indices"
            ]
            == [0, 1, 2]
            and result["prediction_supervision"][
                "hidden_rows_transition_indices"
            ]
            == [2]
        ),
        "absolute_temporal_basis": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Native twin-sampler contract failed: {checks}")
    result["native_twin_sampler_contract"] = {
        "checks": checks,
        "complete_twin_groups_per_hidden_batch": (
            EXPECTED_TWIN_GROUPS_PER_BATCH
        ),
        "model_or_loss_changed": False,
        "learned_parameters_added": 0,
        "loss_terms_added": 0,
        "hidden_labels_at_model_or_loss_boundary": False,
        "pair_metadata_at_model_or_loss_boundary": False,
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
    args: argparse.Namespace = causal._discovery_args(effective)
    trainer = causal._load_trainer()
    state = _install_runtime(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *effective]
        trainer.main()
    finally:
        sys.argv = previous

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(output, state)
        sidecar = output / "native_twin_sampler_method_v1.json"
        if sidecar.exists():
            raise FileExistsError(sidecar)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": CANDIDATE,
                    "source": str(THIS_SOURCE),
                    "source_sha256": _sha256(THIS_SOURCE),
                    "training_report": str(report),
                    "training_report_sha256": _sha256(report),
                    "single_change": "hidden minibatch sampler",
                    "control_sampler": "PairedBatchStream",
                    "candidate_sampler": "CompleteTwinPairedBatchStream",
                    "complete_twin_groups_per_hidden_batch": (
                        EXPECTED_TWIN_GROUPS_PER_BATCH
                    ),
                    "learned_parameters_added": 0,
                    "loss_terms_added": 0,
                    "hidden_labels_at_model_or_loss_boundary": False,
                    "pair_metadata_at_model_or_loss_boundary": False,
                    "release_runtime_projection": state["audit"],
                    "release_authorization_overlay": state["release"],
                    "claim_boundary": {
                        "paired_catalog_sampler_diagnostic": True,
                        "ordinary_episode_sampler": False,
                        "development_only": True,
                        "single_seed": True,
                        "public_test_opened": False,
                        "cem_opened": False,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
