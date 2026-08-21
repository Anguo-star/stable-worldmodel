#!/usr/bin/env python3
"""Run the final zero-parameter transition-context transfer Motion MVE.

This composes two already isolated mechanisms without adding a network:

* leakage-free prefix-aligned transition inputs expose the support as deltas;
* complete twins transfer those deltas across queries at fixed damping mode.

The terminal Predictor input for a transferred row is exactly
``[dz1_source, dz2_source, z2_destination]`` and its actions are
``[a0_source, a1_source, a2_destination]``.  This removes the unsupported
assumption in the absolute-latent transfer diagnostic that image latents can
be translated additively.  Native LeWM parameters, native MSE, SIGReg 0.09,
optimizer, and deployment parameter count remain unchanged.
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
    run_pusht_motion_damping_anchored_context_transfer_v1 as transfer,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_causal_transition_basis_v1 as causal,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_native_twin_sampler_v1 as native_twin,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_prefix_aligned_transition_v2 as prefix,
)


CANDIDATE = "pusht_motion_damping_transition_context_transfer_v1"
NATIVE_VARIANT = causal.NATIVE_VARIANT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_prefix_loader(trainer: Any) -> dict[str, Any]:
    mixed = trainer.trainer.mixed
    native_loader = mixed.load_model_for_variant
    state: dict[str, Any] = {"method": None}

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = native_loader(*args, **kwargs)
        method = prefix._install_model_predict(model)
        state["method"] = method
        receipt = dict(receipt)
        receipt["temporal_input_basis"] = "prefix_aligned_transition"
        receipt["prefix_aligned_transition"] = method
        return model, receipt

    mixed.load_model_for_variant = load_model
    return state


def _rewrite_training_report(
    output: Path,
    prefix_state: dict[str, Any],
    transfer_state: dict[str, Any],
) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    grouping = result["batch"].get("motion_damping_twin_grouping")
    model_state = transfer_state.get("model") or {}
    checks = {
        "native_variant": result["variant"] == NATIVE_VARIANT,
        "native_sigreg_0p09": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "complete_twin_grouping": bool(
            grouping
            and grouping.get("enabled") is True
            and grouping.get("condition_rows_per_group") == 4
        ),
        "prefix_transition_installed": prefix_state.get("method") is not None,
        "one_loss_call_per_optimizer_step": (
            int(transfer_state["loss_calls"])
            == int(result["optimizer_steps"])
        ),
        "parameter_count_unchanged": (
            model_state.get("parameter_count_before")
            == model_state.get("parameter_count_after")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Transition-transfer contract failed: {checks}")
    result["transition_context_transfer_contract"] = {
        "checks": checks,
        "terminal_input_formula": "[dz1_source,dz2_source,z2_destination]",
        "terminal_action_formula": "[a0_source,a1_source,a2_destination]",
        "source_mapping_within_twin": [2, 3, 0, 1],
        "same_damping_mode_preserved_by_row_position": True,
        "native_prediction_supervision_retained": True,
        "standard_rows_transition_indices": [0, 1, 2],
        "hidden_rows_transition_indices": [2],
        "population_weights": {
            "original_full_horizon": 0.5,
            "hidden_native_terminal": 0.25,
            "hidden_transfer_terminal": 0.25,
        },
        "first_loss_components": copy.deepcopy(
            transfer_state["first_components"]
        ),
        "last_loss_components": copy.deepcopy(
            transfer_state["last_components"]
        ),
        "prefix_implementation": copy.deepcopy(prefix_state["method"]),
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "inference_path_parameter_count_changed": False,
        "loss_family": "native_mse_on_real_and_transferred_examples",
        "tunable_auxiliary_weight_added": False,
        "hidden_value_entered_model": False,
        "paired_catalog_row_order_used_by_trainer": True,
        "ordinary_episode_sampler": False,
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
    release_state = native_twin._install_runtime(trainer)
    prefix_state = _install_prefix_loader(trainer)
    transfer_state = transfer._install_transfer_overlay(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *effective]
        trainer.main()
    finally:
        sys.argv = previous

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(
            output, prefix_state, transfer_state
        )
        sidecar = output / "transition_context_transfer_method_v1.json"
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
                    "release_runtime_projection": release_state["audit"],
                    "release_authorization_overlay": release_state["release"],
                    "learned_parameters_added": 0,
                    "model_modules_added": 0,
                    "tunable_auxiliary_weight_added": False,
                    "claim_boundary": {
                        "development_only": True,
                        "single_seed_discovery": True,
                        "paired_catalog_diagnostic": True,
                        "ordinary_episode_sampler": False,
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
