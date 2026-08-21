#!/usr/bin/env python3
"""Run the factorized paired-response Motion-Damping MVE.

This is the first experiment that combines the two established facts: paired
response gradients are locally corrective, and sending hidden terminal MSE
through the shared Predictor trunk is harmful.  Native LeWM retains its
original-data MSE and original SIGReg; the rank-64 B(q)c branch receives
hidden terminal MSE plus one target-energy-normalized paired response term.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
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
    run_pusht_motion_damping_oracle_context_predictor_mve_v1 as additive,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_routed_factorized_response_mve_v1 as routed,
)


PREREG = (
    ROOT
    / "research/conditional_dynamics_representation/configs/"
    "pusht_motion_damping_factorized_paired_response_mve_v1.yaml"
)
RELEASE_CONFIG = additive.RELEASE_CONFIG
ARTIFACT_ROOT = (
    ROOT
    / "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_factorized_paired_response_mve_v1"
)
EXPECTED_RELEASE_ID = additive.EXPECTED_RELEASE_ID
EXPECTED_RELEASE_SHA256 = additive.EXPECTED_RELEASE_SHA256
FROZEN_SEED = additive.FROZEN_SEED
DEFAULT_STEPS = additive.DEFAULT_STEPS
RESPONSE_WEIGHT = 1.0
ARMS = {
    "oracle": "mixed_factorized_paired_response_oracle_native_0p09",
    "constant": "mixed_factorized_paired_response_constant_native_0p09",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def paired_response_loss(
    prediction: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    original_batch_size: int = additive.ORIGINAL_BATCH_SIZE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match high-minus-low response, normalized per real target pair."""

    _require(
        prediction.ndim == 3
        and embeddings.ndim == 3
        and prediction.shape[:2]
        == (embeddings.shape[0], embeddings.shape[1] - 1),
        "prediction/embedding boundary changed",
    )
    hidden_prediction = prediction[original_batch_size:, -1]
    hidden_target = embeddings[original_batch_size:, -1].detach()
    _require(
        hidden_prediction.shape[0] == 64,
        "paired response requires 64 adjacent hidden rows",
    )
    predicted_pairs = hidden_prediction.reshape(-1, 2, hidden_prediction.shape[-1])
    target_pairs = hidden_target.reshape(-1, 2, hidden_target.shape[-1])
    predicted_response = predicted_pairs[:, 1] - predicted_pairs[:, 0]
    target_response = target_pairs[:, 1] - target_pairs[:, 0]
    target_energy = target_response.square().mean(dim=-1).detach().clamp_min(1e-8)
    error_energy = (predicted_response - target_response).square().mean(dim=-1)
    loss = (error_energy / target_energy).mean()
    dot = (predicted_response * target_response).sum(dim=-1)
    return loss, {
        "target_energy_minimum": target_energy.min(),
        "target_energy_mean": target_energy.mean(),
        "predicted_response_energy_mean": predicted_response.square().mean(dim=-1).mean(),
        "positive_alignment_rate": (dot > 0).float().mean(),
    }


def make_augmented_loss(state: dict[str, Any]):
    native_routed_loss = routed.routed_prediction_loss

    def augmented_loss(**kwargs: Any) -> torch.Tensor:
        native_value = native_routed_loss(**kwargs)
        response_value, components = paired_response_loss(
            kwargs["prediction"],
            kwargs["embeddings"],
            original_batch_size=int(kwargs["original_batch_size"]),
        )
        step = int(state["calls"]) + 1
        state["calls"] = step
        state["records"][step] = {
            "native_routed_prediction_loss": float(native_value.detach()),
            "paired_response_loss": float(response_value.detach()),
            "paired_response_weight": RESPONSE_WEIGHT,
            **{
                name: float(value.detach())
                for name, value in components.items()
            },
        }
        return native_value + RESPONSE_WEIGHT * response_value

    return augmented_loss


def install_training_overlay(motion: Any, *, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(arm)
    # The routed implementation is reused verbatim; only its registered names
    # and prediction-loss callable are replaced inside this process.
    routed.ARMS = dict(ARMS)
    state: dict[str, Any] = {"calls": 0, "records": {}}
    routed.routed_prediction_loss = make_augmented_loss(state)
    variant = routed.install_training_overlay(motion, arm=arm)
    mixed = motion.trainer.mixed
    routed_train_variant = mixed.train_variant

    def train_variant_with_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = routed_train_variant(*args, **kwargs)
        if kwargs.get("variant") != variant:
            return result
        max_steps = int(kwargs["max_steps"])
        _require(state["calls"] == max_steps, "paired response call count changed")
        for trace in result["loss_trace"]:
            step = int(trace["optimizer_step"])
            trace.update(state["records"][step])
        result["objective"] = (
            "native original-data MSE + routed hidden terminal MSE + "
            "0.09*original SIGReg + 1.0*paired normalized response; "
            "paired terms update B(q)c only"
        )
        contract = result["routed_factorized_response_contract"]
        contract.update(
            {
                "method": "factorized_paired_response_v1",
                "additional_loss": "paired_target_energy_normalized_response",
                "additional_loss_weight": RESPONSE_WEIGHT,
                "paired_response_calls": state["calls"],
                "paired_response_orientation": "no_extra_decay_minus_faster_decay",
                "paired_response_target_detached": True,
                "paired_response_gradient_destination": "factorized_response_branch_only",
                "pair_identity_source": "adjacent complete-pair order",
                "hidden_numeric_label_read_by_loss": False,
                "first_step_paired_response": state["records"][1],
                "terminal_step_paired_response": state["records"][max_steps],
            }
        )
        return result

    mixed.train_variant = train_variant_with_receipt
    return variant


def install_release_authorization_overlay(
    motion: Any, *, arm: str, steps: int
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
            "diagnostic": "authorized_factorized_paired_response_mve",
            "authority": str(PREREG),
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
            int(config["architecture"]["added_trainable_parameters"])
            == routed.ADDED_PARAMETER_COUNT
        ),
        "response_weight_matches": (
            float(config["objective"]["paired_response_weight"])
            == RESPONSE_WEIGHT
        ),
        "public_and_cem_closed": (
            config["evidence_boundary"]["public_test_open"] is False
            and config["evidence_boundary"]["cem_open"] is False
        ),
    }
    _require(all(checks.values()), f"static paired-response contract failed: {checks}")
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

    import run_pusht_motion_damping_h3_train as motion

    variant = install_training_overlay(motion, arm=args.arm)
    release_overlay = install_release_authorization_overlay(
        motion, arm=args.arm, steps=args.optimizer_steps
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
        "experiment_id": "pusht_motion_damping_factorized_paired_response_mve_v1",
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
