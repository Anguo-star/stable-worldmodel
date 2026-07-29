#!/usr/bin/env python3
"""Audit and summarize the conditional-SIGReg single-seed screen.

The primary mechanism metric is the frozen Door two-target rule-use score.
The primary utility metric is real-environment CEM success on paired original
TwoRoom queries.  Rollout error and the final training loss are diagnostics;
candidate ranking and Spearman metrics are intentionally excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SEEDS = (42, 43, 44, 45, 46, 47)
ROLLOUT_HORIZONS = (1, 2, 3, 5)
EXPECTED_PLANNING_PROTOCOL = {
    "action_block": 5,
    "cem_samples": 300,
    "cem_steps": 30,
    "cem_topk": 30,
    "eval_budget": 50,
    "evaluations": 50,
    "history_size": 3,
    "horizon": 5,
    "receding_horizon": 5,
}
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260728
NONINFERIORITY_MARGIN = 0.05


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected a JSON mapping: {path}")
    return payload


def _planning_result(root: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    records: dict[str, dict[str, Any]] = {}
    files: list[dict[str, str]] = []
    identity: dict[str, Any] | None = None
    by_seed: dict[str, dict[str, Any]] = {}
    for seed in EVAL_SEEDS:
        path = root / f"seed{seed}.json"
        payload = _load_json(path)
        _require(payload.get("status") == "passed", f"Incomplete CEM result: {path}")
        protocol = dict(payload["protocol"])
        _require(
            int(protocol.pop("eval_seed")) == seed
            and protocol == EXPECTED_PLANNING_PROTOCOL,
            f"Unexpected CEM protocol: {path}",
        )
        _require(
            payload["frozen_weight_audit"]["passed"] is True,
            f"Checkpoint changed during CEM: {path}",
        )
        rows = list(payload["raw_records"])
        _require(len(rows) == 50, f"Expected 50 CEM records: {path}")
        successes = 0
        for row in rows:
            key = str(row["evaluation_id"])
            _require(key not in records, f"Duplicate CEM query: {key}")
            _require(
                int(row["eval_seed"]) == seed,
                f"CEM record seed mismatch: {key}",
            )
            records[key] = row
            successes += int(bool(row["success"]))
        aggregate = payload["aggregate"]
        _require(
            int(aggregate["evaluations"]) == 50
            and int(aggregate["successes"]) == successes,
            f"CEM aggregate mismatch: {path}",
        )
        by_seed[str(seed)] = {
            "queries": 50,
            "successes": successes,
            "success_rate": successes / 50,
            "mean_final_distance_px": float(aggregate["mean_final_distance"]),
        }
        current_identity = {
            "checkpoint": dict(payload["checkpoint"]),
            "catalog": dict(payload["catalog"]),
            "normalizer": dict(payload["normalizer"]),
            "stable_worldmodel": dict(payload["stable_worldmodel"]),
        }
        if identity is None:
            identity = current_identity
        else:
            _require(
                current_identity == identity,
                f"CEM input identity changed across seeds: {path}",
            )
        files.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    _require(len(records) == 300, "Expected 300 combined CEM records")
    ordered = list(records.values())
    successes = sum(int(bool(row["success"])) for row in ordered)
    metrics = {
        "queries": len(ordered),
        "successes": successes,
        "success_rate": successes / len(ordered),
        "mean_final_distance_px": float(
            fmean(float(row["final_distance"]) for row in ordered)
        ),
        "by_eval_seed": by_seed,
    }
    _require(identity is not None, "Missing CEM identity")
    return records, metrics, {"inputs": identity, "files": files}


def _rollout_result(path: Path) -> tuple[
    dict[str, Any],
    frozenset[str],
    dict[str, Any],
]:
    payload = _load_json(path)
    _require(payload.get("status") == "passed", f"Incomplete rollout: {path}")
    _require(
        payload["frozen_weight_audit"]["passed"] is True,
        f"Checkpoint changed during rollout: {path}",
    )
    rows = [
        row
        for row in payload["raw_records"]
        if row["domain"] == "original_heldout"
    ]
    ids = frozenset(str(row["evaluation_id"]) for row in rows)
    _require(len(rows) == len(ids) == 300, f"Invalid rollout records: {path}")
    aggregates = {
        int(row["horizon_action_blocks"]): row
        for row in payload["aggregates"]
        if row["domain"] == "original_heldout"
    }
    _require(
        set(aggregates) == set(ROLLOUT_HORIZONS),
        f"Unexpected rollout horizons: {path}",
    )
    metrics = {
        str(horizon): {
            "queries": int(aggregates[horizon]["evaluations"]),
            "mean_native_latent_mse": float(
                aggregates[horizon]["mean_latent_mse"]
            ),
            "mean_native_latent_rmse": float(
                aggregates[horizon]["mean_latent_rmse"]
            ),
        }
        for horizon in ROLLOUT_HORIZONS
    }
    provenance = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "inputs": {
            "checkpoint": dict(payload["checkpoint"]),
            "catalog": dict(payload["catalog"]),
            "normalizer": dict(payload["normalizer"]),
            "stable_worldmodel": dict(payload["stable_worldmodel"]),
        },
    }
    return metrics, ids, provenance


def _door_result(path: Path, *, method_id: str) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    payload = _load_json(path)
    _require(payload["method_id"] == method_id, f"Door method mismatch: {path}")
    screen = payload["registered_screen"]
    metrics = {
        "passed": bool(screen["passed"]),
        "correct_target_selection_rate": float(
            screen["metrics"]["correct_target_selection_rate"]
        ),
        "correct_history_win_rate": float(
            screen["metrics"]["correct_history_win_rate"]
        ),
        "worst_seed_direction_rule_accuracy": float(
            screen["metrics"]["worst_stratum_correct_rate"]
        ),
        "frozen_rule_switch_v2_passed": bool(
            screen["checks"]["frozen_rule_switch_v2_passed"]
        ),
        "matching_history_true_target_native_latent_mse": float(
            payload["summary"]["overall"]["native_latent_mse"][
                "same_rule_history"
            ]
        ),
    }
    provenance = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "checkpoint_sha256": payload["training_provenance"][
            "checkpoint_sha256"
        ],
        "training_report_sha256": payload["training_provenance"][
            "training_report_sha256"
        ],
        "stable_worldmodel_commit": payload["training_provenance"][
            "stable_worldmodel_commit"
        ],
    }
    return metrics, provenance


def _training_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path)
    _require(payload.get("passed") is True, f"Training report failed: {path}")
    training = payload["training"]
    _require(
        training["training_complete"] is True
        and int(training["global_step"]) == 1024
        and int(training["world_size"]) == 8,
        f"Formal training contract failed: {path}",
    )
    trace = Path(training["loss_trace"]["path"]).resolve()
    _require(
        _sha256(trace) == training["loss_trace"]["sha256"],
        f"Loss trace hash mismatch: {trace}",
    )
    lines = [line for line in trace.read_text(encoding="utf-8").splitlines() if line]
    last = json.loads(lines[-1])
    _require(
        int(last["optimizer_step"]) == 1024,
        f"Loss trace does not end at step 1024: {trace}",
    )
    metrics = {
        "objective": dict(payload["model"]["training_objective"]),
        "final_optimizer_step": 1024,
        "final_losses": dict(last["losses"]),
        "checkpoint_sha256": payload["artifacts"]["pretrained_sha256"],
        "training_seed": int(training["seed_before_model_initialization"]),
        "world_size": int(training["world_size"]),
        "global_batch": int(training["batch_size_per_device"])
        * int(training["world_size"]),
        "swanlab_initialized": bool(payload["logger"]["initialized"]),
    }
    provenance = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "loss_trace": str(trace),
        "loss_trace_sha256": _sha256(trace),
    }
    return metrics, provenance


def paired_stratified_bootstrap(
    control: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    margin: float = NONINFERIORITY_MARGIN,
) -> dict[str, Any]:
    """Bootstrap paired binary differences within each evaluation seed."""
    _require(set(control) == set(candidate), "CEM query IDs are not paired")
    strata: dict[int, np.ndarray] = {}
    candidate_only = 0
    control_only = 0
    for eval_seed in EVAL_SEEDS:
        keys = sorted(
            key
            for key, row in control.items()
            if int(row["eval_seed"]) == eval_seed
        )
        _require(len(keys) == 50, f"Expected 50 pairs for seed {eval_seed}")
        values = np.asarray(
            [
                int(bool(candidate[key]["success"]))
                - int(bool(control[key]["success"]))
                for key in keys
            ],
            dtype=np.float64,
        )
        candidate_only += int(np.sum(values == 1))
        control_only += int(np.sum(values == -1))
        strata[eval_seed] = values
    point = float(np.mean(np.concatenate(list(strata.values()))))
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = [
            values[generator.integers(0, len(values), len(values))]
            for values in strata.values()
        ]
        draws[index] = np.mean(np.concatenate(sample))
    lower = float(np.quantile(draws, 0.05))
    return {
        "comparison": "candidate_minus_control",
        "point_difference": point,
        "one_sided_95_percent_lower_bound": lower,
        "noninferiority_margin_absolute": margin,
        "noninferior": bool(lower > -margin),
        "method": "eval_seed_stratified_paired_percentile_bootstrap",
        "resamples": resamples,
        "bootstrap_seed": seed,
        "discordant_pairs": {
            "candidate_only_success": candidate_only,
            "control_only_success": control_only,
        },
    }


def analyze(*, artifact_root: Path) -> dict[str, Any]:
    evaluation_root = (
        artifact_root
        / "evaluation/history3/conditional_sigreg_screen_v1"
    )
    ability_root = evaluation_root / "ability"
    door_root = evaluation_root / "door"
    reports_root = artifact_root / "training/reports"
    reference_path = (
        REPO_ROOT
        / "research/conditional_dynamics_representation/results/"
        "sigreg_replay50_falsification_summary.json"
    )
    reference = _load_json(reference_path)
    reference_models = reference["models"]

    specifications = {
        "native_sigreg_0p09": {
            "ability": ability_root / "native_sigreg_0p09_reference",
            "reference": reference_models["H3_PassageReplay50_SIGReg0p09"],
        },
        "paired_native_0p09": {
            "ability": ability_root / "paired_native",
            "door": door_root / "paired_native.json",
            "method_id": "paired_native",
            "training": reports_root
            / "h3_passage_replay50_paired_native_sigreg0p09_"
            "passage_formal_s3072.json",
        },
        "conditional_sigreg_0p09": {
            "ability": ability_root / "conditional_sigreg_0p09",
            "door": door_root / "conditional_sigreg_0p09.json",
            "method_id": "conditional_sigreg_0p09",
            "training": reports_root
            / "h3_passage_replay50_conditional_sigreg0p09_"
            "passage_formal_s3072.json",
        },
    }
    methods: dict[str, Any] = {}
    cem_records: dict[str, dict[str, dict[str, Any]]] = {}
    rollout_ids: dict[str, frozenset[str]] = {}
    provenance: dict[str, Any] = {}
    for name, specification in specifications.items():
        ability = specification["ability"]
        records, cem, cem_provenance = _planning_result(
            ability / "planning_original_heldout"
        )
        rollout, ids, rollout_provenance = _rollout_result(
            ability / "rollout_error.json"
        )
        cem_records[name] = records
        rollout_ids[name] = ids
        methods[name] = {
            "original_domain_real_environment_cem": cem,
            "original_domain_rollout": rollout,
        }
        provenance[name] = {
            "cem": cem_provenance,
            "rollout": rollout_provenance,
        }
        if "training" in specification:
            training, training_provenance = _training_result(
                specification["training"]
            )
            door, door_provenance = _door_result(
                specification["door"],
                method_id=specification["method_id"],
            )
            checkpoint_sha256 = training["checkpoint_sha256"]
            _require(
                checkpoint_sha256
                == door_provenance["checkpoint_sha256"]
                == cem_provenance["inputs"]["checkpoint"]["sha256"]
                == rollout_provenance["inputs"]["checkpoint"]["sha256"],
                f"Checkpoint identity mismatch for {name}",
            )
            methods[name]["training"] = training
            methods[name]["door_rule_use"] = door
            provenance[name]["training"] = training_provenance
            provenance[name]["door"] = door_provenance
        else:
            row = specification["reference"]
            _require(
                row["checkpoint_sha256"]
                == cem_provenance["inputs"]["checkpoint"]["sha256"]
                == rollout_provenance["inputs"]["checkpoint"]["sha256"],
                "Native 0.09 checkpoint identity mismatch",
            )
            methods[name]["training"] = {
                "objective": {
                    "representation_regularizer": "sigreg",
                    "regularizer_weight": 0.09,
                },
                "final_losses": {
                    "loss": row["final_training_loss"],
                    "pred_loss": row["final_prediction_loss"],
                    "sigreg_loss": row["final_sigreg_loss"],
                },
                "checkpoint_sha256": row["checkpoint_sha256"],
                "training_seed": 3072,
            }
            methods[name]["door_rule_use"] = dict(row["door"])

    reference_ids = set(cem_records["native_sigreg_0p09"])
    _require(
        all(set(rows) == reference_ids for rows in cem_records.values()),
        "CEM query identities differ across methods",
    )
    reference_rollout_ids = rollout_ids["native_sigreg_0p09"]
    _require(
        all(ids == reference_rollout_ids for ids in rollout_ids.values()),
        "Rollout query identities differ across methods",
    )
    reference_cem_inputs = provenance["native_sigreg_0p09"]["cem"]["inputs"]
    reference_rollout_inputs = provenance["native_sigreg_0p09"]["rollout"][
        "inputs"
    ]
    for name in specifications:
        cem_inputs = provenance[name]["cem"]["inputs"]
        rollout_inputs = provenance[name]["rollout"]["inputs"]
        _require(
            cem_inputs["catalog"] == reference_cem_inputs["catalog"]
            and cem_inputs["normalizer"] == reference_cem_inputs["normalizer"]
            and cem_inputs["stable_worldmodel"]
            == reference_cem_inputs["stable_worldmodel"],
            f"CEM inputs are not matched for {name}",
        )
        _require(
            rollout_inputs["catalog"] == reference_rollout_inputs["catalog"]
            and rollout_inputs["normalizer"]
            == reference_rollout_inputs["normalizer"]
            and rollout_inputs["stable_worldmodel"]
            == reference_rollout_inputs["stable_worldmodel"],
            f"Rollout inputs are not matched for {name}",
        )

    comparisons = {
        name: paired_stratified_bootstrap(
            cem_records["native_sigreg_0p09"],
            cem_records[name],
        )
        for name in ("paired_native_0p09", "conditional_sigreg_0p09")
    }
    comparisons["conditional_minus_paired_native"] = (
        paired_stratified_bootstrap(
            cem_records["paired_native_0p09"],
            cem_records["conditional_sigreg_0p09"],
        )
    )
    conditional_rollout = methods["conditional_sigreg_0p09"][
        "original_domain_rollout"
    ]
    paired_rollout = methods["paired_native_0p09"][
        "original_domain_rollout"
    ]
    lower_rollout_mse_at_every_horizon = all(
        conditional_rollout[str(horizon)]["mean_native_latent_mse"]
        < paired_rollout[str(horizon)]["mean_native_latent_mse"]
        for horizon in ROLLOUT_HORIZONS
    )
    conditional_viable = bool(
        methods["conditional_sigreg_0p09"]["door_rule_use"]["passed"]
        and comparisons["conditional_sigreg_0p09"]["noninferior"]
    )
    tuned_reference = reference_models["H3_PassageReplay50_SIGReg0p90"]
    return {
        "schema_version": 1,
        "benchmark": "tworoom_conditional_sigreg_screen_v1",
        "status": "completed_single_training_seed_screen",
        "date": "2026-07-28",
        "protocol": {
            "training_seed": 3072,
            "formal_training": "8 GPUs, global batch 1024, 1024 optimizer steps",
            "door": "50 queries x 6 eval seeds; frozen rule-switch-v2",
            "utility": "50 queries x 6 eval seeds; real-environment CEM",
            "eval_seeds": list(EVAL_SEEDS),
            "noninferiority_margin_absolute": NONINFERIORITY_MARGIN,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "excluded_metrics": [
                "spearman_candidate_ranking",
                "generic_candidate_order_correlation",
            ],
        },
        "methods": methods,
        "tuned_native_sigreg_0p90_reference": {
            "role": "prior_sufficiently_tuned_native_SIGReg_reference",
            "door_rule_use": dict(tuned_reference["door"]),
            "original_domain_rollout": dict(
                tuned_reference["original_rollout_native_latent_mse"]
            ),
            "original_domain_real_environment_cem": dict(
                tuned_reference["original_cem"]
            ),
            "checkpoint_sha256": tuned_reference["checkpoint_sha256"],
        },
        "paired_cem_comparisons": comparisons,
        "decision": {
            "paired_order_explanation_rejected": bool(
                not methods["paired_native_0p09"]["door_rule_use"]["passed"]
                and methods["conditional_sigreg_0p09"]["door_rule_use"]["passed"]
            ),
            "conditional_candidate_passes_current_screen": conditional_viable,
            "conditional_has_lower_rollout_mse_than_paired_native_at_every_horizon": (
                lower_rollout_mse_at_every_horizon
            ),
            "higher_final_training_pred_loss_is_not_observed_as_eval_degradation": (
                lower_rollout_mse_at_every_horizon
                and methods["conditional_sigreg_0p09"]["training"][
                    "final_losses"
                ]["pred_loss"]
                > methods["paired_native_0p09"]["training"]["final_losses"][
                    "pred_loss"
                ]
            ),
            "method_claim_allowed": False,
            "claim_boundary": (
                "This one-training-seed, one-hidden-dynamics-task screen "
                "calibrates the mechanism and candidate. It does not establish "
                "cross-seed or cross-task superiority over tuned native "
                "SIGReg 0.90."
            ),
        },
        "provenance": {
            "reference_summary": {
                "path": str(reference_path.resolve()),
                "sha256": _sha256(reference_path),
            },
            "methods": provenance,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="ContextWorld artifact root containing training/ and evaluation/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "research/conditional_dynamics_representation/results/"
            "conditional_sigreg_screen_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(artifact_root=args.artifact_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_passed": result["decision"][
                    "conditional_candidate_passes_current_screen"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
