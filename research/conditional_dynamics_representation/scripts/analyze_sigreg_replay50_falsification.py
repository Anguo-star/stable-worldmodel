#!/usr/bin/env python3
"""Analyze the replay-matched SIGReg falsification experiment.

The primary mechanism gate is the frozen Door two-target test.  The primary
utility gate is paired real-environment CEM success on the original domain.
Native latent rollout MSE is retained as a secondary diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_CONFIG = (
    REPO_ROOT
    / "research/conditional_dynamics_representation/configs/"
    "tworoom_hidden_passage_h3_sigreg_replay50_analysis_v1.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected a JSON mapping: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected a YAML mapping: {path}")
    return payload


def _same_path(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _resolve_stable_path(logical: str) -> Path:
    path = Path(logical)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_context_path(
    logical: str,
    *,
    contextworld_root: Path,
    artifact_root: Path,
) -> Path:
    path = Path(logical)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "artifacts":
        return artifact_root.joinpath(*parts[1:])
    return contextworld_root / path


def _verify_frozen_inputs(
    config: dict[str, Any],
    *,
    contextworld_root: Path,
    artifact_root: Path,
) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for key in ("training_config", "door_validation_config"):
        specification = config["inputs"][key]
        path = _resolve_stable_path(specification["path"]).resolve()
        digest = _file_sha256(path)
        _require(
            digest == specification["sha256"],
            f"Frozen {key} hash mismatch: {digest}",
        )
        observed[key] = {"path": str(path), "sha256": digest}
    for key in (
        "original_ability_protocol",
        "original_planning_catalog",
        "original_rollout_catalog",
        "normalizer",
    ):
        specification = config["inputs"][key]
        path = _resolve_context_path(
            specification["path"],
            contextworld_root=contextworld_root,
            artifact_root=artifact_root,
        ).resolve()
        digest = _file_sha256(path)
        _require(
            digest == specification["sha256"],
            f"Frozen {key} hash mismatch: {digest}",
        )
        observed[key] = {"path": str(path), "sha256": digest}
    return observed


def _mean_boolean(rows: Iterable[bool]) -> float:
    values = [bool(value) for value in rows]
    _require(bool(values), "Cannot average an empty boolean sequence")
    return float(fmean(values))


def _door_metrics(
    path: Path,
    *,
    model_id: str,
    expected_queries: int,
    expected_seeds: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path)
    _require(payload.get("status") == "completed", f"Incomplete Door result: {path}")
    _require(payload.get("model_id") == model_id, f"Door model mismatch: {path}")
    records = list(payload.get("records", []))
    query_ids = {str(row["query_id"]) for row in records}
    _require(
        len(query_ids) == expected_queries,
        f"{model_id} Door query count is {len(query_ids)}, not {expected_queries}",
    )
    expected_records = expected_queries * 2 * 3
    _require(
        len(records) == expected_records,
        f"{model_id} Door record count is {len(records)}, not {expected_records}",
    )
    matching = [
        row
        for row in records
        if row["history_condition"] == f"observed_{row['true_rule']}"
    ]
    _require(
        len(matching) == expected_queries * 2,
        f"{model_id} matching-history row count is invalid",
    )
    cells: dict[tuple[int, str, str], list[bool]] = defaultdict(list)
    for row in matching:
        cells[
            (
                int(row["eval_seed"]),
                str(row["direction"]),
                str(row["true_rule"]),
            )
        ].append(bool(row["true_target_closer"]))
    expected_cells = {
        (seed, direction, rule)
        for seed in expected_seeds
        for direction in ("left_to_right", "right_to_left")
        for rule in ("passable", "blocked")
    }
    _require(set(cells) == expected_cells, f"{model_id} Door strata changed")
    summary = payload["summary"]
    by_rule = summary["by_true_rule"]
    all_history_strict_win_rate = fmean(
        float(by_rule[rule]["overall"]["strict_win_rate"])
        for rule in ("passable", "blocked")
    )
    matching_vs_opposite_win_rate = fmean(
        float(
            by_rule[rule]["overall"][
                "matching_vs_opposite_history_win_rate"
            ]
        )
        for rule in ("passable", "blocked")
    )
    training = payload["training_provenance"]
    checkpoint_sha256 = str(training["checkpoint_sha256"])
    metrics = {
        "passed": bool(summary["decision"]["passed"]),
        "failed_checks": list(summary["decision"].get("failed_checks", [])),
        "queries": len(query_ids),
        "matching_history_target_accuracy": _mean_boolean(
            row["true_target_closer"] for row in matching
        ),
        "matching_vs_opposite_history_win_rate": float(
            matching_vs_opposite_win_rate
        ),
        "all_history_strict_win_rate": float(
            all_history_strict_win_rate
        ),
        "minimum_seed_direction_rule_target_accuracy": min(
            _mean_boolean(values) for values in cells.values()
        ),
        "matching_history_true_target_native_latent_mse": float(
            fmean(float(row["true_next_frame_latent_mse"]) for row in matching)
        ),
        "matching_history_two_target_margin": float(
            fmean(float(row["two_target_margin"]) for row in matching)
        ),
        "target_pair_native_latent_mse": dict(
            summary["target_latent_separation"]
        ),
    }
    provenance = {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "checkpoint": str(payload["identity"]["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha256,
        "training_report": str(payload["identity"]["training_report"]),
        "training_report_sha256": str(
            payload["identity"]["training_report_sha256"]
        ),
        "validation_config_sha256": str(payload["identity"]["config_sha256"]),
        "catalog_sha256": str(payload["identity"]["catalog_sha256"]),
        "normalizer_sha256": str(payload["identity"]["normalizer_sha256"]),
    }
    return metrics, provenance


def _audit_common_ability_result(
    payload: dict[str, Any],
    *,
    path: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    catalog_sha256: str,
    normalizer_sha256: str,
    stable_worldmodel_commit: str,
) -> None:
    checks = {
        "status": payload.get("status") == "passed",
        "checkpoint_path": _same_path(
            payload.get("checkpoint", {}).get("path"), checkpoint
        ),
        "checkpoint_sha256": payload.get("checkpoint", {}).get("sha256")
        == checkpoint_sha256,
        "catalog_sha256": payload.get("catalog", {}).get("sha256")
        == catalog_sha256,
        "normalizer_sha256": payload.get("normalizer", {}).get("sha256")
        == normalizer_sha256,
        "stable_worldmodel_commit": payload.get("stable_worldmodel", {}).get(
            "commit"
        )
        == stable_worldmodel_commit,
        "weights_frozen": bool(
            payload.get("frozen_weight_audit", {}).get("passed")
        ),
        "weight_hash_unchanged": payload.get("frozen_weight_audit", {}).get(
            "state_dict_sha256_before"
        )
        == payload.get("frozen_weight_audit", {}).get(
            "state_dict_sha256_after"
        ),
    }
    _require(all(checks.values()), f"Ability result audit failed at {path}: {checks}")


def _planning_records(
    root: Path,
    *,
    model_id: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    expected_seeds: tuple[int, ...],
    queries_per_seed: int,
    planning_catalog_sha256: str,
    normalizer_sha256: str,
    stable_worldmodel_commit: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    combined: dict[str, dict[str, Any]] = {}
    provenance = []
    for seed in expected_seeds:
        path = root / model_id / "planning_original_heldout" / f"seed{seed}.json"
        payload = _load_json(path)
        _audit_common_ability_result(
            payload,
            path=path,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            catalog_sha256=planning_catalog_sha256,
            normalizer_sha256=normalizer_sha256,
            stable_worldmodel_commit=stable_worldmodel_commit,
        )
        protocol = payload["protocol"]
        _require(int(protocol["eval_seed"]) == seed, f"Planning seed mismatch: {path}")
        rows = list(payload.get("raw_records", []))
        _require(
            len(rows) == queries_per_seed,
            f"Planning count mismatch for {model_id}/seed{seed}",
        )
        for row in rows:
            key = str(row["evaluation_id"])
            _require(key not in combined, f"Duplicate planning ID: {key}")
            _require(int(row["eval_seed"]) == seed, f"Record seed mismatch: {key}")
            combined[key] = row
        provenance.append({"path": str(path.resolve()), "sha256": _file_sha256(path)})
    _require(
        len(combined) == len(expected_seeds) * queries_per_seed,
        f"Combined planning count mismatch for {model_id}",
    )
    return combined, provenance


def _rollout_metrics(
    root: Path,
    *,
    model_id: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    rollout_catalog_sha256: str,
    normalizer_sha256: str,
    stable_worldmodel_commit: str,
    horizons: tuple[int, ...],
    expected_original_records: int,
) -> tuple[dict[str, Any], dict[str, str], frozenset[str]]:
    path = root / model_id / "rollout_error.json"
    payload = _load_json(path)
    _audit_common_ability_result(
        payload,
        path=path,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        catalog_sha256=rollout_catalog_sha256,
        normalizer_sha256=normalizer_sha256,
        stable_worldmodel_commit=stable_worldmodel_commit,
    )
    rows = [
        row
        for row in payload.get("raw_records", [])
        if row.get("domain") == "original_heldout"
    ]
    _require(
        len(rows) == expected_original_records,
        f"Original rollout count mismatch for {model_id}",
    )
    ids = frozenset(str(row["evaluation_id"]) for row in rows)
    _require(len(ids) == len(rows), f"Duplicate rollout ID for {model_id}")
    aggregates = {
        int(row["horizon_action_blocks"]): row
        for row in payload.get("aggregates", [])
        if row.get("domain") == "original_heldout"
    }
    _require(set(aggregates) == set(horizons), f"Rollout horizons changed for {model_id}")
    metrics = {
        str(horizon): {
            "evaluations": int(aggregates[horizon]["evaluations"]),
            "mean_native_latent_mse": float(
                aggregates[horizon]["mean_latent_mse"]
            ),
        }
        for horizon in horizons
    }
    return (
        metrics,
        {"path": str(path.resolve()), "sha256": _file_sha256(path)},
        ids,
    )


def _planning_summary(
    rows: dict[str, dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    ordered = list(rows.values())
    by_seed = {
        str(seed): {
            "queries": sum(int(row["eval_seed"]) == seed for row in ordered),
            "success_rate": _mean_boolean(
                row["success"]
                for row in ordered
                if int(row["eval_seed"]) == seed
            ),
        }
        for seed in expected_seeds
    }
    return {
        "queries": len(ordered),
        "successes": sum(bool(row["success"]) for row in ordered),
        "success_rate": _mean_boolean(row["success"] for row in ordered),
        "mean_final_distance_px": float(
            fmean(float(row["final_distance"]) for row in ordered)
        ),
        "by_eval_seed": by_seed,
    }


def _paired_stratified_bootstrap(
    control: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    resamples: int,
    confidence: float,
    seed: int,
    margin: float,
) -> dict[str, Any]:
    _require(set(control) == set(candidate), "Planning query IDs are not paired")
    strata: dict[int, np.ndarray] = {}
    discordance = {"candidate_only_success": 0, "control_only_success": 0}
    for eval_seed in expected_seeds:
        keys = sorted(
            key
            for key, row in control.items()
            if int(row["eval_seed"]) == eval_seed
        )
        _require(bool(keys), f"Empty paired planning stratum: seed {eval_seed}")
        differences = np.asarray(
            [
                float(bool(candidate[key]["success"]))
                - float(bool(control[key]["success"]))
                for key in keys
            ],
            dtype=np.float64,
        )
        strata[eval_seed] = differences
        discordance["candidate_only_success"] += int(np.sum(differences == 1.0))
        discordance["control_only_success"] += int(np.sum(differences == -1.0))
    point_difference = float(
        np.mean(np.concatenate([strata[value] for value in expected_seeds]))
    )
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = [
            values[generator.integers(0, len(values), size=len(values))]
            for values in strata.values()
        ]
        draws[index] = np.mean(np.concatenate(sampled))
    lower = float(np.quantile(draws, 1.0 - confidence))
    upper = float(np.quantile(draws, confidence))
    return {
        "comparison": "candidate_minus_control",
        "point_difference": point_difference,
        "one_sided_lower_confidence_bound": lower,
        "one_sided_upper_confidence_bound": upper,
        "confidence": confidence,
        "method": "eval_seed_stratified_paired_percentile_bootstrap",
        "resamples": resamples,
        "seed": seed,
        "noninferiority_margin_absolute": margin,
        "noninferior": bool(lower > -margin),
        "discordant_pairs": discordance,
    }


def analyze(
    *,
    analysis_config_path: Path,
    contextworld_root: Path,
    artifact_root: Path,
    door_root: Path,
    ability_root: Path,
) -> dict[str, Any]:
    config = _load_yaml(analysis_config_path)
    _require(
        config.get("status")
        in {
            "frozen_before_checkpoint_scoring",
            "corrective_alignment_to_preexisting_rule_switch_v2",
        },
        "Analysis protocol is neither frozen nor a declared correction",
    )
    frozen = _verify_frozen_inputs(
        config,
        contextworld_root=contextworld_root,
        artifact_root=artifact_root,
    )
    mechanism = config["door_rule_inference"]
    planning = config["original_domain_planning"]
    expected_seeds = tuple(map(int, mechanism["eval_seeds"]))
    _require(
        expected_seeds == tuple(map(int, planning["eval_seeds"])),
        "Door and CEM evaluation seeds differ",
    )
    expected_queries = int(mechanism["queries"])
    queries_per_seed = int(planning["queries_per_seed"])
    _require(
        expected_queries == len(expected_seeds) * queries_per_seed,
        "Frozen query count is inconsistent",
    )
    control = str(config["inputs"]["models"]["control"])
    candidates = tuple(map(str, config["inputs"]["models"]["candidates"]))
    model_ids = (control, *candidates)
    stable_commit = str(
        config["inputs"]["original_ability_protocol"][
            "stable_worldmodel_commit"
        ]
    )
    door = {}
    provenance = {}
    planning_rows = {}
    planning_metrics = {}
    rollout_metrics = {}
    rollout_id_sets = {}
    for model_id in model_ids:
        door_path = door_root / f"{model_id}.json"
        door[model_id], door_provenance = _door_metrics(
            door_path,
            model_id=model_id,
            expected_queries=expected_queries,
            expected_seeds=expected_seeds,
        )
        _require(
            door_provenance["validation_config_sha256"]
            == frozen["door_validation_config"]["sha256"],
            f"Door result used a different validation config: {model_id}",
        )
        checkpoint = Path(door_provenance["checkpoint"]).resolve()
        checkpoint_sha256 = door_provenance["checkpoint_sha256"]
        _require(
            _file_sha256(checkpoint) == checkpoint_sha256,
            f"Checkpoint changed after Door scoring: {model_id}",
        )
        rows, planning_provenance = _planning_records(
            ability_root,
            model_id=model_id,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            expected_seeds=expected_seeds,
            queries_per_seed=queries_per_seed,
            planning_catalog_sha256=frozen["original_planning_catalog"]["sha256"],
            normalizer_sha256=frozen["normalizer"]["sha256"],
            stable_worldmodel_commit=stable_commit,
        )
        planning_rows[model_id] = rows
        planning_metrics[model_id] = _planning_summary(
            rows, expected_seeds=expected_seeds
        )
        (
            rollout_metrics[model_id],
            rollout_provenance,
            rollout_id_sets[model_id],
        ) = _rollout_metrics(
            ability_root,
            model_id=model_id,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            rollout_catalog_sha256=frozen["original_rollout_catalog"]["sha256"],
            normalizer_sha256=frozen["normalizer"]["sha256"],
            stable_worldmodel_commit=stable_commit,
            horizons=tuple(
                map(
                    int,
                    config["secondary_metrics"][
                        "original_rollout_horizons_action_blocks"
                    ],
                )
            ),
            expected_original_records=expected_queries,
        )
        provenance[model_id] = {
            "door": door_provenance,
            "planning": planning_provenance,
            "rollout": rollout_provenance,
        }
    reference_planning_ids = set(planning_rows[control])
    _require(
        all(set(planning_rows[row]) == reference_planning_ids for row in model_ids),
        "CEM query IDs differ across checkpoints",
    )
    _require(
        all(rollout_id_sets[row] == rollout_id_sets[control] for row in model_ids),
        "Rollout query IDs differ across checkpoints",
    )
    interval = planning["interval"]
    comparisons = {
        model_id: _paired_stratified_bootstrap(
            planning_rows[control],
            planning_rows[model_id],
            expected_seeds=expected_seeds,
            resamples=int(interval["resamples"]),
            confidence=float(planning["confidence"]),
            seed=int(interval["seed"]),
            margin=float(planning["noninferiority_margin_absolute"]),
        )
        for model_id in candidates
    }
    qualifying = [
        model_id
        for model_id in candidates
        if door[model_id]["passed"] and comparisons[model_id]["noninferior"]
    ]
    door_passing = [
        model_id for model_id in candidates if door[model_id]["passed"]
    ]
    cem_noninferior = [
        model_id
        for model_id in candidates
        if comparisons[model_id]["noninferior"]
    ]
    stop_route = bool(qualifying)
    if stop_route:
        interpretation = (
            "At least one SIGReg increase learns the Door rule and is "
            "noninferior on paired original-domain CEM; the current "
            "experiment therefore does not justify a new objective."
        )
        unresolved_reason = None
    elif door_passing:
        interpretation = (
            "At least one increased weight learns the Door rule, but none "
            "is paired-CEM-noninferior to the control. A replay-matched "
            "Door-versus-planning tradeoff remains at this gate."
        )
        unresolved_reason = "door_passes_but_cem_noninferiority_fails"
    else:
        interpretation = (
            "No tested increased weight passes every Door gate. The tested "
            "range is therefore insufficient to decide whether a stronger "
            "plain SIGReg weight would solve Door without a CEM cost."
        )
        unresolved_reason = "no_candidate_passes_door"
    return {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "analysis_protocol": {
            "path": str(analysis_config_path.resolve()),
            "sha256": _file_sha256(analysis_config_path),
            "status": config["status"],
        },
        "analysis_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "frozen_inputs": frozen,
        "model_order": list(model_ids),
        "door_rule_inference": door,
        "original_domain_cem": {
            "control": control,
            "per_model": planning_metrics,
            "paired_noninferiority": comparisons,
        },
        "original_domain_rollout_mse_secondary": rollout_metrics,
        "decision": {
            "qualifying_candidates": qualifying,
            "door_passing_candidates": door_passing,
            "cem_noninferior_candidates": cem_noninferior,
            "stop_structural_defect_and_new_objective_route": stop_route,
            "structural_claim_status": (
                "falsified_at_the_replay50_seed3072_gate"
                if stop_route
                else "not_falsified_at_the_replay50_seed3072_gate"
            ),
            "unresolved_reason": unresolved_reason,
            "interpretation": interpretation,
            "scope_limit": config["falsification_decision"]["scope_limit"],
        },
        "artifact_provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG
    )
    parser.add_argument("--contextworld-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--door-root", type=Path, required=True)
    parser.add_argument("--ability-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"Refusing to overwrite {args.output}")
    result = analyze(
        analysis_config_path=args.analysis_config.resolve(),
        contextworld_root=args.contextworld_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        door_root=args.door_root.resolve(),
        ability_root=args.ability_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
