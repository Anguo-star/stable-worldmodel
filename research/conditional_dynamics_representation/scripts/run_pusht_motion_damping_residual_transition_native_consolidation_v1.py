#!/usr/bin/env python3
"""Test one native-only consolidation after successful paired ICL learning.

This is a bounded mechanism continuation, not a new loss candidate.  It starts
from the fixed residual-transition + weight-0.09 exact-future checkpoint that
passed all six Motion mechanism gates, preserves its ordinary causal/residual
inference function exactly at continuation step zero, and applies only native
LeWM MSE + 0.09 SIGReg for 1,024 further optimizer steps.  No parameter is
reset, added, frozen, or routed differently.
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
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_native_twin_sampler_v1 as native_twin,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_residual_output_basis_v1 as residual,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_temporal_homotopy_exact_future_v1 as hashing,
)


CANDIDATE = "pusht_motion_damping_residual_transition_native_consolidation_v1"
SOURCE_CHECKPOINT_SHA256 = (
    "0387490bcd1a3b33758e58bd7be954c458e8ea51343d03ace01b71dbbaac19a0"
)
CONSOLIDATION_STEPS = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_checkpoint(argv: Sequence[str]) -> Path:
    """Reject a wrong continuation source before dry-run or data loading."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args, _ = parser.parse_known_args(list(argv))
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed = _sha256(checkpoint)
    if observed != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Native consolidation source checkpoint changed: "
            f"expected {SOURCE_CHECKPOINT_SHA256}, observed {observed}"
        )
    return checkpoint


def _install_native_consolidation(trainer: Any) -> dict[str, Any]:
    mixed = trainer.trainer.mixed
    native_loader = mixed.load_model_for_variant
    native_model_config = mixed.model_config
    native_loss = mixed.mixed_prediction_loss
    state: dict[str, Any] = {
        "model": None,
        "_native_loss_object": native_loss,
        "native_loss_unchanged": None,
    }

    def load_model(checkpoint: Path, *args: Any, **kwargs: Any):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if _sha256(checkpoint) != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError("Native consolidation source checkpoint changed")
        model, receipt = native_loader(checkpoint, *args, **kwargs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        state_before = hashing._state_sha256(model)
        model.temporal_input_basis = "causal_transition"
        model.temporal_output_basis = "residual"
        state_after = hashing._state_sha256(model)
        observed_count = sum(parameter.numel() for parameter in model.parameters())
        if observed_count != parameter_count or state_after != state_before:
            raise RuntimeError("Native consolidation mutated source model at load")
        state["model"] = {
            "parameter_count_before": parameter_count,
            "parameter_count_after": observed_count,
            "source_state_sha256": state_before,
            "installed_state_sha256": state_after,
            "parameter_reset": False,
        }
        updated = dict(receipt)
        updated["native_consolidation"] = {
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "temporal_input_basis": "causal_transition",
            "temporal_output_basis": "residual",
            "parameter_reset": False,
            "learned_parameters_added": 0,
            "auxiliary_loss_terms": 0,
        }
        return model, updated

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(native_model_config(*args, **kwargs))
        config["temporal_input_basis"] = "causal_transition"
        config["temporal_output_basis"] = "residual"
        return config

    mixed.load_model_for_variant = load_model
    mixed.model_config = model_config
    return state


def _rewrite_training_report(
    output: Path,
    *,
    method_state: dict[str, Any],
) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    model_state = method_state.get("model") or {}
    grouping = result["batch"].get("motion_damping_twin_grouping")
    source = result["source_checkpoint"]
    checks = {
        "source_checkpoint_exact": source["sha256"] == SOURCE_CHECKPOINT_SHA256,
        "native_mse_sigreg_0p09_only": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "optimizer_steps_exact": int(result["optimizer_steps"]) == CONSOLIDATION_STEPS,
        "complete_twin_grouping": bool(
            grouping
            and grouping.get("enabled") is True
            and grouping.get("condition_rows_per_group") == 4
        ),
        "parameter_count_unchanged_at_install": (
            model_state.get("parameter_count_before")
            == model_state.get("parameter_count_after")
        ),
        "source_state_unchanged_at_install": (
            model_state.get("source_state_sha256")
            == model_state.get("installed_state_sha256")
        ),
        "no_parameter_reset": model_state.get("parameter_reset") is False,
        "native_loss_function_unchanged": bool(
            method_state.get("native_loss_unchanged")
        ),
        "no_auxiliary_during_consolidation": (
            "residual_transition_exact_future_contract" not in result
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Native consolidation contract failed: {checks}")
    result["residual_transition_native_consolidation_contract"] = {
        "checks": checks,
        "continuation_source": {
            "checkpoint": source["path"],
            "sha256": SOURCE_CHECKPOINT_SHA256,
            "source_mechanism_screen_passed": True,
            "source_cem_screen_successes": "3/20",
            "source_optimizer_steps": 2048,
        },
        "objective": "native_mse + 0.09*SIGReg",
        "optimizer_semantics": {
            "kind": "weight_only_restart",
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "fresh_optimizer_steps": CONSOLIDATION_STEPS,
            "total_parameter_update_exposure": 2048 + CONSOLIDATION_STEPS,
            "matched_native_1024_comparison": False,
        },
        "temporal_input_basis": "causal_transition",
        "temporal_output_basis": "residual",
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "auxiliary_terms_added": 0,
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
    _validate_source_checkpoint(effective)
    args = residual._discovery_args(effective)
    if (
        args.input_basis != "causal_transition"
        or args.optimizer_steps != CONSOLIDATION_STEPS
    ):
        raise RuntimeError("Native consolidation is fixed to 1024 steps")
    trainer = causal._load_trainer()
    native_twin._install_runtime(trainer)
    method_state = _install_native_consolidation(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *residual._trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous
    method_state["native_loss_unchanged"] = (
        trainer.trainer.mixed.mixed_prediction_loss
        is method_state.pop("_native_loss_object")
    )

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(output, method_state=method_state)
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("temporal_input_basis") != "causal_transition"
            or config.get("temporal_output_basis") != "residual"
        ):
            raise RuntimeError("Native consolidation saved the wrong model config")
        sidecar = output / "residual_transition_native_consolidation_method_v1.json"
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
                    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                    "objective": "native_mse + 0.09*SIGReg",
                    "consolidation_optimizer_steps": CONSOLIDATION_STEPS,
                    "optimizer_semantics": "fresh_AdamW_and_schedule",
                    "optimizer_state_restored": False,
                    "total_parameter_update_exposure": (
                        2048 + CONSOLIDATION_STEPS
                    ),
                    "learned_parameters_added": 0,
                    "model_modules_added": 0,
                    "auxiliary_terms_added": 0,
                    "claim_boundary": {
                        "development_mechanism_screen_only": True,
                        "single_seed_discovery": True,
                        "public_test_opened": False,
                        "additional_seeds_opened": False,
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
