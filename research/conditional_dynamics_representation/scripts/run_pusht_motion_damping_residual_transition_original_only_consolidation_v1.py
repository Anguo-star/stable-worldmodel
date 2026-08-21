#!/usr/bin/env python3
"""Consolidate paired Motion ICL using only ordinary PushT prediction rows.

This single-factor arm starts from the fixed residual-transition checkpoint
that passed all six Motion mechanism gates.  It preserves every model tensor
and the causal/residual inference function, removes the paired auxiliary, and
uses the native LeWM recipe for 1,024 fresh optimizer steps with a full batch
of ordinary PushT rows.  It adds no parameter, module, head, or loss term.
"""

from __future__ import annotations

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
    run_pusht_motion_damping_residual_transition_native_consolidation_v1 as parent,
)


CANDIDATE = (
    "pusht_motion_damping_residual_transition_"
    "original_only_consolidation_v1"
)
CONSOLIDATION_STEPS = 1024


def _install_original_only_training(trainer: Any) -> dict[str, Any]:
    """Use the unchanged native objective on a full ordinary-data batch."""

    mixed = trainer.trainer.mixed
    native_train_variant = mixed.train_variant
    state: dict[str, Any] = {"calls": 0, "batch": None}

    def train_variant(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("variant") != causal.NATIVE_VARIANT:
            raise RuntimeError("Unexpected original-only consolidation variant")
        updated = dict(kwargs)
        total = int(updated["batch_size"])
        requested_original = int(updated["original_batch_size"])
        updated["original_batch_size"] = total
        result = native_train_variant(*args, **updated)
        observed = result["batch"]
        state["calls"] += 1
        state["batch"] = {
            "total": total,
            "requested_original": requested_original,
            "actual_original": int(observed["original"]),
            "actual_hidden": int(observed["hidden"]),
        }
        return result

    mixed.train_variant = train_variant
    return state


def _rewrite_training_report(
    output: Path,
    *,
    model_state: dict[str, Any],
    data_state: dict[str, Any],
) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    loaded = model_state.get("model") or {}
    batch = result["batch"]
    checks = {
        "source_checkpoint_exact": (
            result["source_checkpoint"]["sha256"]
            == parent.SOURCE_CHECKPOINT_SHA256
        ),
        "native_mse_sigreg_0p09_recipe": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "optimizer_steps_exact": (
            int(result["optimizer_steps"]) == CONSOLIDATION_STEPS
        ),
        "ordinary_rows_fill_global_batch": (
            int(batch["original"]) == int(batch["total"])
            and int(batch["hidden"]) == 0
        ),
        "training_wrapper_called_once": data_state.get("calls") == 1,
        "parameter_count_unchanged_at_install": (
            loaded.get("parameter_count_before")
            == loaded.get("parameter_count_after")
        ),
        "source_state_unchanged_at_install": (
            loaded.get("source_state_sha256")
            == loaded.get("installed_state_sha256")
        ),
        "no_parameter_reset": loaded.get("parameter_reset") is False,
        "native_loss_function_unchanged": bool(
            model_state.get("native_loss_unchanged")
        ),
        "no_auxiliary_during_consolidation": (
            "residual_transition_exact_future_contract" not in result
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Original-only consolidation contract failed: {checks}")
    result["residual_transition_original_only_consolidation_contract"] = {
        "checks": checks,
        "continuation_source": {
            "checkpoint": result["source_checkpoint"]["path"],
            "sha256": parent.SOURCE_CHECKPOINT_SHA256,
            "source_optimizer_steps": 2048,
            "source_mechanism_screen_passed": True,
            "source_cem_screen_successes": "3/20",
        },
        "objective": "native_mse + 0.09*SIGReg",
        "prediction_population": "ordinary_pusht_only",
        "hidden_rows_in_consolidation_batch": 0,
        "temporal_input_basis": "causal_transition",
        "temporal_output_basis": "residual",
        "optimizer_semantics": {
            "kind": "weight_only_restart",
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "fresh_optimizer_steps": CONSOLIDATION_STEPS,
            "total_parameter_update_exposure": 2048 + CONSOLIDATION_STEPS,
            "matched_mixed_consolidation_control": True,
        },
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "auxiliary_terms_added": 0,
    }
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": parent._sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    parent._validate_source_checkpoint(effective)
    args = residual._discovery_args(effective)
    if (
        args.input_basis != "causal_transition"
        or args.optimizer_steps != CONSOLIDATION_STEPS
    ):
        raise RuntimeError("Original-only consolidation is fixed to 1024 steps")

    trainer = causal._load_trainer()
    native_twin._install_runtime(trainer)
    trainer.TWIN_GROUP_VARIANTS.discard(causal.NATIVE_VARIANT)
    model_state = parent._install_native_consolidation(trainer)
    data_state = _install_original_only_training(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *residual._trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous
    model_state["native_loss_unchanged"] = (
        trainer.trainer.mixed.mixed_prediction_loss
        is model_state.pop("_native_loss_object")
    )

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(
            output,
            model_state=model_state,
            data_state=data_state,
        )
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("temporal_input_basis") != "causal_transition"
            or config.get("temporal_output_basis") != "residual"
        ):
            raise RuntimeError("Original-only consolidation saved wrong basis")
        sidecar = output / (
            "residual_transition_original_only_consolidation_method_v1.json"
        )
        if sidecar.exists():
            raise FileExistsError(sidecar)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": CANDIDATE,
                    "source": str(THIS_SOURCE),
                    "source_sha256": parent._sha256(THIS_SOURCE),
                    "training_report": str(report),
                    "training_report_sha256": parent._sha256(report),
                    "source_checkpoint_sha256": (
                        parent.SOURCE_CHECKPOINT_SHA256
                    ),
                    "prediction_population": "ordinary_pusht_only",
                    "consolidation_optimizer_steps": CONSOLIDATION_STEPS,
                    "optimizer_state_restored": False,
                    "total_parameter_update_exposure": 3072,
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
