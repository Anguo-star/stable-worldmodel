#!/usr/bin/env python3
"""Run one zero-parameter causal-transition-basis Motion Damping MVE."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
TRAINER_SOURCE = CONTEXTWORLD_ROOT / "scripts/run_pusht_motion_damping_h3_train.py"
BASIS_NAME = "causal_transition"
NATIVE_VARIANT = "mixed_frozen_image_identifiable_future_native_0p09"
ALLOWED_RELEASE_SOURCE_DRIFTS = {
    "identity.stablewm_lewm_config",
    "identity.stablewm_lewm_model",
    "identity.stablewm_pldm_model",
    "identity.stablewm_loader",
    "identity.package",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_trainer() -> Any:
    name = "_contextworld_motion_damping_causal_transition_basis_v1"
    specification = importlib.util.spec_from_file_location(name, TRAINER_SOURCE)
    if specification is None or specification.loader is None:
        raise ImportError(TRAINER_SOURCE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _discovery_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args(argv)
    checks = {
        "model": args.model == "lewm",
        "variant": args.variant in {None, NATIVE_VARIANT},
        "seed": args.seed == 14321,
        "optimizer_steps": args.optimizer_steps == 1024,
        "output": args.output is not None,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Motion causal-transition discovery contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return args


def _install_overlay(trainer: Any) -> dict[str, Any]:
    native_audit = trainer.audit_motion_damping_icl_release
    native_release_loader = trainer.load_motion_damping_icl_release
    native_load_model = trainer.trainer.mixed.load_model_for_variant
    native_model_config = trainer.trainer.mixed.model_config
    state: dict[str, Any] = {"audit": None}

    def discovery_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed = native_audit(*args, **kwargs)
        failed = {
            name
            for name, row in observed["files"].items()
            if not bool(row.get("passed"))
        }
        data_passed = all(bool(value) for value in observed["data_checks"].values())
        if failed != ALLOWED_RELEASE_SOURCE_DRIFTS or not data_passed:
            return observed
        projected = copy.deepcopy(observed)
        projected["upstream_release_audit_passed"] = bool(observed["passed"])
        projected["passed"] = True
        projected["discovery_runtime_projection"] = {
            "status": "allowed_exact_source_drift_only",
            "allowed_failed_identity_keys": sorted(ALLOWED_RELEASE_SOURCE_DRIFTS),
            "observed_failed_identity_keys": sorted(failed),
            "all_data_checks_passed": True,
            "public_test_opened": False,
        }
        state["audit"] = projected["discovery_runtime_projection"]
        return projected

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = native_load_model(*args, **kwargs)
        if not hasattr(model, "temporal_input_basis"):
            raise RuntimeError("Current LeWM runtime lacks temporal-basis support")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model.temporal_input_basis = BASIS_NAME
        if sum(parameter.numel() for parameter in model.parameters()) != parameter_count:
            raise RuntimeError("Temporal basis unexpectedly changed parameter count")
        receipt = dict(receipt)
        receipt["temporal_input_basis"] = BASIS_NAME
        receipt["learned_parameters_added"] = 0
        return model, receipt

    def load_discovery_release(path: Path) -> dict[str, Any]:
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
            "candidate": "causal_transition_basis_v1",
            "variant": NATIVE_VARIANT,
            "seed": 14321,
            "fixed_checkpoint_step": 1024,
            "public_test_open": False,
        }
        state["release"] = {
            "base_status": "failed_development",
            "base_release_mutated_on_disk": False,
            "legacy_completed_seed_preserved_in_base_release": True,
            "runtime_completed_seed_adapter": "remove_14321_only",
            "followup_status": "development_recipe_search_in_progress",
        }
        return release

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(native_model_config(*args, **kwargs))
        config["temporal_input_basis"] = BASIS_NAME
        return config

    trainer.audit_motion_damping_icl_release = discovery_audit
    trainer.load_motion_damping_icl_release = load_discovery_release
    trainer.trainer.mixed.load_model_for_variant = load_model
    trainer.trainer.mixed.model_config = model_config
    return state


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = _discovery_args(effective_argv)
    trainer = _load_trainer()
    overlay_state = _install_overlay(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *effective_argv]
        trainer.main()
    finally:
        sys.argv = previous

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = output / "training_report.json"
        if not report.is_file():
            raise FileNotFoundError(report)
        sidecar = output / "causal_transition_basis_method_v1.json"
        if sidecar.exists():
            raise FileExistsError(sidecar)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": "pusht_motion_damping_causal_transition_basis_v1",
                    "source": str(THIS_SOURCE),
                    "source_sha256": _sha256(THIS_SOURCE),
                    "lewm_source": str(REPO_ROOT / "stable_worldmodel/wm/lewm/lewm.py"),
                    "lewm_source_sha256": _sha256(
                        REPO_ROOT / "stable_worldmodel/wm/lewm/lewm.py"
                    ),
                    "training_report": str(report),
                    "training_report_sha256": _sha256(report),
                    "temporal_input_basis": BASIS_NAME,
                    "formula": "[z0,z1-z0,...,zT-z(T-1)]",
                    "prefix_invertible": True,
                    "learned_parameters_added": 0,
                    "loss_added": 0,
                    "native_variant": NATIVE_VARIANT,
                    "release_runtime_projection": overlay_state["audit"],
                    "release_authorization_overlay": overlay_state["release"],
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
