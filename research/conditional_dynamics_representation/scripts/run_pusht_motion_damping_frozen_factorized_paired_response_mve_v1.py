#!/usr/bin/env python3
"""Run the frozen-base factorized paired-response Motion-Damping MVE.

All pretrained LeWM parameters and buffers are fixed.  Only the zero-start
rank-64 B(q)c response branch is optimized by hidden terminal MSE and the one
paired normalized-response term.  This closes the moving-target confound in
the prior jointly trained experiment and structurally preserves the standard
PushT/CEM path because standard rows have c=0.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Mapping, Sequence

import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONTEXTWORLD_ROOT = ROOT.parent / "ContextWorld"
CONTEXTWORLD_SCRIPTS = CONTEXTWORLD_ROOT / "scripts"
for source_root in (ROOT, CONTEXTWORLD_ROOT, CONTEXTWORLD_SCRIPTS):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_factorized_paired_response_mve_v1 as paired,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_oracle_context_predictor_mve_v1 as additive,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_routed_factorized_response_mve_v1 as routed,
)


PREREG = (
    ROOT
    / "research/conditional_dynamics_representation/configs/"
    "pusht_motion_damping_frozen_factorized_paired_response_mve_v1.yaml"
)
TRAJECTORY_ADDENDUM = (
    ROOT
    / "research/conditional_dynamics_representation/configs/"
    "pusht_motion_damping_frozen_factorized_paired_response_trajectory1024_v1.yaml"
)
RELEASE_CONFIG = additive.RELEASE_CONFIG
ARTIFACT_ROOT = (
    ROOT
    / "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_frozen_factorized_paired_response_mve_v1"
)
EXPECTED_RELEASE_ID = additive.EXPECTED_RELEASE_ID
EXPECTED_RELEASE_SHA256 = additive.EXPECTED_RELEASE_SHA256
FROZEN_SEED = additive.FROZEN_SEED
DEFAULT_STEPS = additive.DEFAULT_STEPS
ARMS = {
    "oracle": "mixed_frozen_base_factorized_paired_response_oracle",
    "constant": "mixed_frozen_base_factorized_paired_response_constant",
}
BASE_MODULE_NAMES = (
    "encoder",
    "projector",
    "predictor",
    "action_encoder",
    "pred_proj",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def freeze_native_lewm_before_response_attachment(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Freeze all current parameters and keep native stochastic modules in eval."""

    model.requires_grad_(False)
    native_train = model.train
    base_modules = tuple(getattr(model, name) for name in BASE_MODULE_NAMES)

    def train_with_frozen_base(self: torch.nn.Module, mode: bool = True):
        result = native_train(mode)
        for module in base_modules:
            module.eval()
        response = getattr(self.predictor, "episode_context_response", None)
        if response is not None:
            response.train(mode)
        return result

    model.train = MethodType(train_with_frozen_base, model)
    model.train(True)
    return {
        "frozen_parameter_count_before_response_attachment": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "native_trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "native_modules_forced_eval_during_training": list(BASE_MODULE_NAMES),
    }


def install_training_overlay(motion: Any, *, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(arm)
    paired.ARMS = dict(ARMS)
    variant = paired.install_training_overlay(motion, arm=arm)
    mixed = motion.trainer.mixed
    mixed.FROZEN_IMAGE_VARIANTS.add(variant)
    native_loader = mixed.load_model_for_variant
    freeze_receipt: dict[str, Any] = {}

    def load_frozen_base(*args: Any, **kwargs: Any):
        model, receipt = native_loader(*args, **kwargs)
        freeze_receipt.update(
            freeze_native_lewm_before_response_attachment(model)
        )
        receipt = dict(receipt)
        receipt["frozen_native_lewm_overlay"] = dict(freeze_receipt)
        return model, receipt

    mixed.load_model_for_variant = load_frozen_base
    paired_train_variant = mixed.train_variant

    def train_variant_with_freeze_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = paired_train_variant(*args, **kwargs)
        if kwargs.get("variant") != variant:
            return result
        contract = result["routed_factorized_response_contract"]
        contract.update(
            {
                "method": "frozen_base_factorized_paired_response_v1",
                "native_lewm_parameters_frozen": True,
                "native_lewm_buffers_frozen_by_eval_mode": True,
                "native_modules_receive_original_mse_and_original_sigreg": False,
                "native_mse_and_sigreg_computed_without_trainable_destination": True,
                "only_trainable_module": "predictor.episode_context_response",
                "only_trainable_parameter_count": routed.ADDED_PARAMETER_COUNT,
                "standard_context_zero_makes_response_structurally_zero": True,
                "standard_cem_function_preserved_exactly": True,
                "freeze_receipt": dict(freeze_receipt),
            }
        )
        result["objective"] = (
            "frozen native LeWM + trainable B(q)c; hidden terminal MSE + "
            "1.0*paired normalized response; original SIGReg computed but "
            "cannot update frozen base; no other trainable parameters"
        )
        return result

    mixed.train_variant = train_variant_with_freeze_receipt
    return variant


def install_release_authorization_overlay(
    motion: Any, *, arm: str, steps: int, authority: Path = PREREG
) -> dict[str, Any]:
    native_loader = motion.load_motion_damping_icl_release
    audit: dict[str, Any] = {}

    def load_with_followup(path: Path) -> dict[str, Any]:
        release = copy.deepcopy(native_loader(path))
        _require(release["release_id"] == EXPECTED_RELEASE_ID, "release id changed")
        matrix = release["training"]["reference_matrix"]
        _require(matrix["status"] == "failed_development", "base status changed")
        completed = [int(value) for value in matrix["completed_development_seeds"]]
        _require(FROZEN_SEED in completed, "legacy seed identity changed")
        matrix["completed_development_seeds"] = [
            value for value in completed if value != FROZEN_SEED
        ]
        release["training"]["learnability_followup"] = {
            "status": "development_recipe_search_in_progress",
            "diagnostic": "authorized_frozen_factorized_paired_response_mve",
            "authority": str(authority),
            "arm": arm,
            "seed": FROZEN_SEED,
            "optimizer_steps": steps,
            "development_only": True,
            "public_test_open": False,
            "cem_open": False,
        }
        audit.update(
            {
                "base_status": "failed_development",
                "removed_completed_seed_in_memory_only": FROZEN_SEED,
                "all_other_completed_seeds_preserved": True,
                "base_release_mutated_on_disk": False,
                "public_test_open": False,
                "cem_open": False,
            }
        )
        return release

    motion.load_motion_damping_icl_release = load_with_followup
    return audit


def validate_static_contract() -> dict[str, Any]:
    config = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    checks = {
        "release_sha_matches": (
            additive.file_sha256(RELEASE_CONFIG) == EXPECTED_RELEASE_SHA256
        ),
        "seed_matches": int(config["training"]["seed"]) == FROZEN_SEED,
        "steps_match": int(config["training"]["optimizer_steps"]) == DEFAULT_STEPS,
        "rank_matches": int(config["architecture"]["rank"]) == routed.RANK,
        "parameter_count_matches": (
            int(config["architecture"]["only_trainable_parameters"])
            == routed.ADDED_PARAMETER_COUNT
        ),
        "base_frozen": config["gradient_routing"]["native_lewm"] == "frozen",
        "public_and_cem_closed": (
            config["evidence_boundary"]["public_test_open"] is False
            and config["evidence_boundary"]["cem_open"] is False
        ),
    }
    _require(all(checks.values()), f"static frozen contract failed: {checks}")
    return checks


def validate_trajectory_addendum(
    path: Path, *, arm: str, steps: int
) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    observed = config["observed_256_step_basis"]
    oracle_report = ROOT / observed["oracle_training_report"]["path"]
    constant_report = ROOT / observed["constant_training_report"]["path"]
    checks = {
        "path_is_registered_addendum": path.resolve() == TRAJECTORY_ADDENDUM.resolve(),
        "parent_sha_matches": (
            config["parent_preregistration"]["sha256"]
            == additive.file_sha256(PREREG)
        ),
        "oracle_only": arm == config["authorization"]["arm"] == "oracle",
        "steps_match": int(config["authorization"]["optimizer_steps"]) == steps == 1024,
        "single_seed": int(config["authorization"]["seed"]) == FROZEN_SEED,
        "oracle_report_sha_matches": (
            additive.file_sha256(oracle_report)
            == observed["oracle_training_report"]["sha256"]
        ),
        "constant_report_sha_matches": (
            additive.file_sha256(constant_report)
            == observed["constant_training_report"]["sha256"]
        ),
        "no_public_or_cem": (
            config["authorization"]["public_test_open"] is False
            and config["authorization"]["cem_open"] is False
            and config["authorization"]["additional_seeds_open"] is False
        ),
    }
    _require(all(checks.values()), f"trajectory addendum failed: {checks}")
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--optimizer-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--original-h5", type=Path, default=None)
    parser.add_argument("--original-lance", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--trajectory-addendum", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require(args.seed == FROZEN_SEED, f"first MVE requires seed {FROZEN_SEED}")
    _require(args.optimizer_steps > 0, "optimizer steps must be positive")
    output = args.output.expanduser().resolve()
    if not args.dry_run and output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    static_checks = validate_static_contract()
    trajectory_checks = None
    authority = PREREG
    if args.optimizer_steps != DEFAULT_STEPS:
        _require(
            args.trajectory_addendum is not None,
            "non-default budget requires the registered trajectory addendum",
        )
        authority = args.trajectory_addendum.expanduser().resolve()
        trajectory_checks = validate_trajectory_addendum(
            authority, arm=args.arm, steps=args.optimizer_steps
        )
    else:
        _require(
            args.trajectory_addendum is None,
            "the 256-step MVE must not claim a trajectory addendum",
        )

    import run_pusht_motion_damping_h3_train as motion

    variant = install_training_overlay(motion, arm=args.arm)
    release_overlay = install_release_authorization_overlay(
        motion,
        arm=args.arm,
        steps=args.optimizer_steps,
        authority=authority,
    )
    native_argv = sys.argv
    forwarded = [
        str(motion.__file__),
        "--release-config",
        str(RELEASE_CONFIG),
        "--model",
        "lewm",
        "--variant",
        variant,
        "--seed",
        str(args.seed),
        "--optimizer-steps",
        str(args.optimizer_steps),
        "--output",
        str(output),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    for name in ("original_h5", "original_lance", "checkpoint"):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend(["--" + name.replace("_", "-"), str(value.resolve())])
    if args.dry_run:
        forwarded.append("--dry-run")
    sys.argv = forwarded
    try:
        motion.main()
    finally:
        sys.argv = native_argv
    if args.dry_run:
        return

    report_path = output / "training_report.json"
    provenance_path = output / "training_provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    disclosure = {
        "experiment_id": "pusht_motion_damping_frozen_factorized_paired_response_mve_v1",
        "arm": args.arm,
        "variant": variant,
        "preregistration": {
            "path": str(PREREG),
            "sha256": additive.file_sha256(PREREG),
        },
        "release": {
            "path": str(RELEASE_CONFIG),
            "sha256": additive.file_sha256(RELEASE_CONFIG),
        },
        "release_authorization_overlay": release_overlay,
        "static_checks": static_checks,
        "trajectory_addendum": (
            None
            if trajectory_checks is None
            else {
                "path": str(authority),
                "sha256": additive.file_sha256(authority),
                "checks": trajectory_checks,
            }
        ),
        "oracle_privileged_context_opened": args.arm == "oracle",
        "public_test_opened": False,
        "cem_opened": False,
    }
    report["research_overlay"] = disclosure
    provenance["research_overlay"] = disclosure
    if args.arm == "oracle":
        provenance["visible_fields"] = [
            "pixels",
            "action",
            "oracle_motion_damping_identity_diagnostic_only",
        ]
    _atomic_json(report_path, report)
    _atomic_json(provenance_path, provenance)


if __name__ == "__main__":
    main()
