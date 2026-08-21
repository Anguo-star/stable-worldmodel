#!/usr/bin/env python3
"""Test zero-parameter residual dynamics with two temporal input bases.

The existing LeWM Predictor is interpreted as a latent displacement and the
current latent is added back at its output.  Its final projection is reset to
zero after loading the common PushT initialization, so step zero is the exact
persistence predictor.  No module, parameter, loss term, or inference branch
is added.  ``--input-basis`` completes the two untested cells of the original
absolute/transition-input x absolute/residual-output matrix.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from torch import nn


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


CANDIDATE = "pusht_motion_damping_residual_output_basis_v1"
NATIVE_VARIANT = causal.NATIVE_VARIANT
INPUT_BASES = ("absolute", "causal_transition")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discovery_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-basis", choices=INPUT_BASES, required=True)
    args, _ = parser.parse_known_args(argv)
    checks = {
        "model": args.model == "lewm",
        "variant": args.variant in {None, NATIVE_VARIANT},
        "seed": args.seed == 14321,
        "optimizer_steps": args.optimizer_steps in {1024, 2048},
        "output": args.output is not None,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Motion residual-output discovery contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return args


def _trainer_argv(argv: Sequence[str]) -> list[str]:
    """Remove the method-only flag before calling the frozen trainer CLI."""

    values = list(argv)
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] == "--input-basis":
            if index + 1 >= len(values):
                raise ValueError("--input-basis requires a value")
            index += 2
            continue
        cleaned.append(values[index])
        index += 1
    return cleaned


def zero_initialize_displacement_projection(model: nn.Module) -> dict[str, Any]:
    """Reset only the existing final pred-projection linear map."""

    linear_layers = [
        module for module in model.pred_proj.modules()
        if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
        raise RuntimeError("LeWM pred_proj contains no linear output layer")
    output = linear_layers[-1]
    with torch.no_grad():
        output.weight.zero_()
        if output.bias is not None:
            output.bias.zero_()
    zeroed = int(output.weight.numel())
    if output.bias is not None:
        zeroed += int(output.bias.numel())
    if torch.count_nonzero(output.weight).item() != 0:
        raise RuntimeError("Residual output weight did not reset to zero")
    if output.bias is not None and torch.count_nonzero(output.bias).item() != 0:
        raise RuntimeError("Residual output bias did not reset to zero")
    return {
        "module": "pred_proj final Linear",
        "parameter_count_reinitialized": zeroed,
        "weight_exactly_zero": True,
        "bias_exactly_zero": output.bias is None
        or torch.count_nonzero(output.bias).item() == 0,
    }


def _install_residual_output(
    trainer: Any,
    *,
    input_basis: str,
) -> dict[str, Any]:
    mixed = trainer.trainer.mixed
    native_loader = mixed.load_model_for_variant
    native_model_config = mixed.model_config
    state: dict[str, Any] = {"model": None}

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = native_loader(*args, **kwargs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model.temporal_input_basis = input_basis
        model.temporal_output_basis = "residual"
        reset = zero_initialize_displacement_projection(model)
        observed_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if observed_count != parameter_count:
            raise RuntimeError("Residual output changed LeWM parameter count")
        state["model"] = {
            "parameter_count_before": parameter_count,
            "parameter_count_after": observed_count,
            "input_basis": input_basis,
            "output_basis": "residual",
            "initialization": reset,
        }
        updated = dict(receipt)
        updated["temporal_input_basis"] = input_basis
        updated["temporal_output_basis"] = "residual"
        updated["residual_output_initialization"] = copy.deepcopy(reset)
        updated["learned_parameters_added"] = 0
        return model, updated

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(native_model_config(*args, **kwargs))
        config["temporal_input_basis"] = input_basis
        config["temporal_output_basis"] = "residual"
        return config

    mixed.load_model_for_variant = load_model
    mixed.model_config = model_config
    return state


def _rewrite_training_report(
    output: Path,
    *,
    input_basis: str,
    release_state: dict[str, Any],
    method_state: dict[str, Any],
) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    grouping = result["batch"].get("motion_damping_twin_grouping")
    model_state = method_state.get("model") or {}
    checks = {
        "native_variant": result["variant"] == NATIVE_VARIANT,
        "native_mse_sigreg_0p09": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "complete_twin_grouping": bool(
            grouping
            and grouping.get("enabled") is True
            and grouping.get("condition_rows_per_group") == 4
        ),
        "parameter_count_unchanged": (
            model_state.get("parameter_count_before")
            == model_state.get("parameter_count_after")
        ),
        "existing_output_projection_zero_initialized": bool(
            model_state.get("initialization", {}).get(
                "weight_exactly_zero", False
            )
        ),
        "input_basis_exact": model_state.get("input_basis") == input_basis,
        "output_basis_residual": (
            model_state.get("output_basis") == "residual"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Residual-output contract failed: {checks}")
    result["residual_output_basis_contract"] = {
        "checks": checks,
        "input_basis": input_basis,
        "output_basis": "residual",
        "prediction_formula": "z_hat_next = z_current + F_theta(history,action)",
        "step0_prediction": "z_current_exact_persistence",
        "initialization": copy.deepcopy(model_state["initialization"]),
        "native_prediction_supervision_retained": True,
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "loss_terms_added": 0,
        "hidden_labels_at_model_or_loss_boundary": False,
        "pair_metadata_at_model_or_loss_boundary": False,
    }
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
        "input_basis": input_basis,
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    discovery_args = _discovery_args(effective)
    method_args = discovery_args
    trainer = causal._load_trainer()
    release_state = native_twin._install_runtime(trainer)
    method_state = _install_residual_output(
        trainer,
        input_basis=method_args.input_basis,
    )
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *_trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous

    if not discovery_args.dry_run:
        output = discovery_args.output.expanduser().resolve()
        report = _rewrite_training_report(
            output,
            input_basis=method_args.input_basis,
            release_state=release_state,
            method_state=method_state,
        )
        sidecar = output / "residual_output_basis_method_v1.json"
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
                    "input_basis": method_args.input_basis,
                    "output_basis": "residual",
                    "learned_parameters_added": 0,
                    "model_modules_added": 0,
                    "loss_terms_added": 0,
                    "release_runtime_projection": release_state["audit"],
                    "release_authorization_overlay": release_state["release"],
                    "claim_boundary": {
                        "development_only": True,
                        "single_seed_discovery": True,
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
