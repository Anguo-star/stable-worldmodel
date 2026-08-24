#!/usr/bin/env python3
"""Decide the paired PushT CEM non-inferiority question from frozen aggregates.

The analyzer is deterministic and read-only: it loads the frozen configuration,
the fresh paired aggregate and the prior seed-42 screen aggregate, fails closed
on every declared identity, and emits the preregistered paired-bootstrap
decision. No training, model loading or network access happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ANALYSIS_ID = "pusht_paired_cem_noninferiority_v1"
MODEL_ORDER = ("source", "candidate")
CATALOG_FIELDS = ("episode_indices", "row_indices", "start_steps")
PROTOCOL_FIELDS = (
    "action_block",
    "cem_iterations",
    "cem_samples",
    "cem_topk",
    "eval_budget",
    "goal_offset_steps",
    "history_len",
    "horizon",
    "receding_horizon",
    "videos_written",
)


class ValidationError(RuntimeError):
    """Raised when a fail-closed identity or shape check does not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _model_map(aggregate: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    models = aggregate.get("models")
    _require(isinstance(models, list), f"{label}: models is not a list")
    names = tuple(model.get("model") for model in models)
    _require(
        names == MODEL_ORDER,
        f"{label}: model order/names {names} != {MODEL_ORDER}",
    )
    return {model["model"]: model for model in models}


def _outcomes(
    model: dict[str, Any],
    eval_seeds: Sequence[int],
    queries_per_seed: int,
    label: str,
) -> dict[int, np.ndarray]:
    seeds = model.get("seeds")
    _require(isinstance(seeds, list), f"{label}: seeds is not a list")
    observed = tuple(int(entry["eval_seed"]) for entry in seeds)
    _require(
        observed == tuple(int(seed) for seed in eval_seeds),
        f"{label}: eval seeds {observed} != {tuple(eval_seeds)}",
    )
    outcomes: dict[int, np.ndarray] = {}
    for entry in seeds:
        seed = int(entry["eval_seed"])
        successes = entry.get("episode_successes")
        _require(
            isinstance(successes, list),
            f"{label}: seed {seed} episode_successes is not a list",
        )
        _require(
            all(isinstance(value, bool) for value in successes),
            f"{label}: seed {seed} episode_successes are not all booleans",
        )
        _require(
            len(successes) == queries_per_seed,
            f"{label}: seed {seed} has {len(successes)} outcomes,"
            f" expected {queries_per_seed}",
        )
        _require(
            int(entry["query_count"]) == queries_per_seed,
            f"{label}: seed {seed} query_count != {queries_per_seed}",
        )
        _require(
            int(entry["success_count"]) == sum(successes),
            f"{label}: seed {seed} success_count disagrees with outcomes",
        )
        outcomes[seed] = np.asarray(successes, dtype=np.float64)
    return outcomes


def _contingency(source: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    both_success = int(np.sum((source == 1.0) & (candidate == 1.0)))
    candidate_only = int(np.sum((source == 0.0) & (candidate == 1.0)))
    source_only = int(np.sum((source == 1.0) & (candidate == 0.0)))
    both_failure = int(np.sum((source == 0.0) & (candidate == 0.0)))
    return {
        "both_success": both_success,
        "candidate_only_success": candidate_only,
        "source_only_success": source_only,
        "both_failure": both_failure,
        "discordant": candidate_only + source_only,
        "paired_episodes": int(source.size),
    }


def _exact_mcnemar_two_sided_p(candidate_only: int, source_only: int) -> float:
    """Exact two-sided binomial McNemar p-value for a zero paired difference."""
    total = candidate_only + source_only
    if total == 0:
        return 1.0
    smaller = min(candidate_only, source_only)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2.0**total)
    return float(min(1.0, 2.0 * tail))


def _slice_summary(
    differences: dict[int, np.ndarray],
    source: dict[int, np.ndarray],
    candidate: dict[int, np.ndarray],
    seeds: Sequence[int],
) -> dict[str, Any]:
    pooled_source = np.concatenate([source[seed] for seed in seeds])
    pooled_candidate = np.concatenate([candidate[seed] for seed in seeds])
    pooled_difference = np.concatenate([differences[seed] for seed in seeds])
    contingency = _contingency(pooled_source, pooled_candidate)
    return {
        "eval_seeds": [int(seed) for seed in seeds],
        "paired_episodes": int(pooled_difference.size),
        "source_successes": int(pooled_source.sum()),
        "candidate_successes": int(pooled_candidate.sum()),
        "source_success_rate": float(pooled_source.mean()),
        "candidate_success_rate": float(pooled_candidate.mean()),
        "difference": float(pooled_difference.mean()),
        "contingency": contingency,
        "exact_two_sided_mcnemar_p": _exact_mcnemar_two_sided_p(
            contingency["candidate_only_success"],
            contingency["source_only_success"],
        ),
    }


def _stratified_paired_bootstrap(
    differences: dict[int, np.ndarray],
    seeds: Sequence[int],
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample paired episodes with replacement within each eval-seed stratum."""
    rng = np.random.default_rng(seed)
    total = sum(differences[key].size for key in seeds)
    pooled = np.zeros(replicates, dtype=np.float64)
    for key in seeds:
        stratum = differences[key]
        size = stratum.size
        indices = rng.integers(0, size, size=(replicates, size))
        pooled += stratum[indices].sum(axis=1)
    return pooled / float(total)


def _bootstrap_report(
    replicate_means: np.ndarray,
    inference: dict[str, Any],
) -> dict[str, Any]:
    lower_q = float(inference["lower_one_sided_95_quantile"])
    upper_q = float(inference["upper_one_sided_95_quantile"])
    two_sided = [float(value) for value in inference["two_sided_95_quantiles"]]
    lower = float(np.quantile(replicate_means, lower_q, method="linear"))
    upper = float(np.quantile(replicate_means, upper_q, method="linear"))
    interval = [
        float(np.quantile(replicate_means, two_sided[0], method="linear")),
        float(np.quantile(replicate_means, two_sided[1], method="linear")),
    ]
    return {
        "replicates": int(replicate_means.size),
        "seed": int(inference["bootstrap_seed"]),
        "quantile_method": "numpy_linear",
        "resampling_unit": str(inference["resampling_unit"]),
        "replicate_mean": float(replicate_means.mean()),
        "lower_one_sided_95": lower,
        "upper_one_sided_95": upper,
        "two_sided_95_interval": interval,
    }


def _decide(lower: float, upper: float, margin: float) -> str:
    if lower > margin:
        return "pass"
    if upper < margin:
        return "fail"
    return "inconclusive"


def _catalog_rows(catalog: dict[str, Any], seed: int, label: str) -> dict[str, list[int]]:
    key = str(seed)
    _require(key in catalog, f"{label}: no seed {seed} rows")
    rows = catalog[key]
    _require(isinstance(rows, dict), f"{label}: seed {seed} rows are not a mapping")
    extracted: dict[str, list[int]] = {}
    for field in CATALOG_FIELDS:
        _require(field in rows, f"{label}: seed {seed} missing {field}")
        extracted[field] = [int(value) for value in rows[field]]
    return extracted


def _catalog_path(aggregate: dict[str, Any], base: Path) -> Path | None:
    catalog = aggregate.get("query_catalog")
    if not isinstance(catalog, dict) or "path" not in catalog:
        return None
    path = Path(str(catalog["path"]))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path if path.is_file() else None


def _check_finite(payload: Any, trail: str = "report") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            _check_finite(value, f"{trail}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _check_finite(value, f"{trail}[{index}]")
    elif isinstance(payload, float):
        _require(math.isfinite(payload), f"{trail} is not finite")


def analyze(
    config_path: Path,
    aggregate_path: Path,
    prior_aggregate_path: Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    aggregate = _load_json(aggregate_path)
    prior_aggregate = _load_json(prior_aggregate_path)

    execution = config["execution"]
    prior_screen = config["prior_screen"]
    inference = config["primary_inference"]
    eval_seeds = [int(seed) for seed in execution["eval_seeds"]]
    queries_per_seed = int(execution["queries_per_seed"])
    prior_seed = int(prior_screen["eval_seed"])
    margin = float(inference["noninferiority_margin"])

    _require(
        prior_seed in eval_seeds,
        f"prior screen seed {prior_seed} is not among the protocol eval seeds",
    )
    _require(config.get("no_training") is True, "config does not declare no_training")

    prior_sha256 = _sha256(prior_aggregate_path)
    _require(
        prior_sha256 == str(prior_screen["aggregate_sha256"]),
        f"prior aggregate sha256 {prior_sha256} != declared"
        f" {prior_screen['aggregate_sha256']}",
    )

    fresh_models = _model_map(aggregate, "aggregate")
    prior_models = _model_map(prior_aggregate, "prior aggregate")
    for name in MODEL_ORDER:
        declared = str(config[name]["checkpoint_sha256"])
        for label, model in (
            ("aggregate", fresh_models[name]),
            ("prior aggregate", prior_models[name]),
        ):
            observed = str(model["checkpoint_sha256"])
            _require(
                observed == declared,
                f"{label}: {name} checkpoint sha256 {observed} != {declared}",
            )

    protocol = aggregate.get("protocol", {})
    observed_seeds = [int(seed) for seed in protocol.get("eval_seeds", [])]
    _require(
        observed_seeds == eval_seeds,
        f"aggregate protocol eval seeds {observed_seeds} != {eval_seeds}",
    )
    _require(
        int(protocol.get("num_eval_per_seed", -1)) == queries_per_seed,
        "aggregate protocol num_eval_per_seed != config queries_per_seed",
    )
    for field in PROTOCOL_FIELDS:
        _require(
            protocol.get(field) == config["fixed_protocol"][field],
            f"aggregate protocol {field} != frozen fixed_protocol value",
        )

    fresh = {
        name: _outcomes(fresh_models[name], eval_seeds, queries_per_seed, f"aggregate {name}")
        for name in MODEL_ORDER
    }
    prior = {
        name: _outcomes(
            prior_models[name], [prior_seed], queries_per_seed, f"prior aggregate {name}"
        )
        for name in MODEL_ORDER
    }
    for seed in eval_seeds:
        _require(
            fresh["source"][seed].size == fresh["candidate"][seed].size,
            f"aggregate: seed {seed} paired outcome lengths differ",
        )
    total_paired = sum(fresh["source"][seed].size for seed in eval_seeds)
    _require(
        total_paired == int(execution["paired_episode_count"]),
        f"aggregate: {total_paired} paired episodes !="
        f" {execution['paired_episode_count']}",
    )
    for name, declared_key in (
        ("source", "source_successes"),
        ("candidate", "candidate_successes"),
    ):
        observed = int(prior[name][prior_seed].sum())
        declared = int(prior_screen[declared_key])
        _require(
            observed == declared,
            f"prior aggregate: {name} successes {observed} != declared {declared}",
        )
    _require(
        int(prior_screen["paired_episodes"]) == queries_per_seed,
        "prior screen paired_episodes != config queries_per_seed",
    )

    fresh_catalog_path = _catalog_path(aggregate, aggregate_path.parent)
    prior_catalog_path = _catalog_path(prior_aggregate, prior_aggregate_path.parent)
    catalogs_available = fresh_catalog_path is not None and prior_catalog_path is not None
    catalog_report: dict[str, Any] = {
        "catalogs_available_through_aggregate_paths": bool(catalogs_available),
        "is_validity_gate": True,
    }
    if catalogs_available:
        prior_catalog_sha256 = _sha256(prior_catalog_path)
        _require(
            prior_catalog_sha256 == str(prior_screen["query_catalog_sha256"]),
            f"prior query catalog sha256 {prior_catalog_sha256} != declared"
            f" {prior_screen['query_catalog_sha256']}",
        )
        fresh_rows = _catalog_rows(
            _load_json(fresh_catalog_path), prior_seed, "aggregate query catalog"
        )
        prior_rows = _catalog_rows(
            _load_json(prior_catalog_path), prior_seed, "prior query catalog"
        )
        _require(
            fresh_rows == prior_rows,
            f"seed {prior_seed} query-catalog rows do not reproduce the prior catalog",
        )
        catalog_report.update(
            {
                "aggregate_query_catalog": _identity(fresh_catalog_path),
                "prior_query_catalog": _identity(prior_catalog_path),
                "rows_compared": list(CATALOG_FIELDS),
                "row_count": len(prior_rows["episode_indices"]),
                "seed42_rows_reproduce_prior_catalog": True,
            }
        )

    outcome_reproduction: dict[str, Any] = {
        "is_validity_gate": False,
        "note": (
            "GPU CEM numerics already changed the fresh source seed-42 count;"
            " the fresh same-run source/candidate pairing is primary."
        ),
    }
    for name in MODEL_ORDER:
        fresh_seed_outcomes = fresh[name][prior_seed]
        prior_seed_outcomes = prior[name][prior_seed]
        differing = int(np.sum(fresh_seed_outcomes != prior_seed_outcomes))
        outcome_reproduction[name] = {
            "prior_successes": int(prior_seed_outcomes.sum()),
            "fresh_successes": int(fresh_seed_outcomes.sum()),
            "differing_episodes": differing,
            "outcomes_reproduce_prior_aggregate": differing == 0,
        }

    differences = {
        seed: fresh["candidate"][seed] - fresh["source"][seed] for seed in eval_seeds
    }
    per_seed = {
        str(seed): _slice_summary(differences, fresh["source"], fresh["candidate"], [seed])
        for seed in eval_seeds
    }
    pooled = _slice_summary(differences, fresh["source"], fresh["candidate"], eval_seeds)

    replicates = int(inference["bootstrap_replicates"])
    bootstrap_seed = int(inference["bootstrap_seed"])
    _require(
        str(inference["quantile_method"]) == "numpy_linear",
        "config quantile_method is not numpy_linear",
    )
    primary_means = _stratified_paired_bootstrap(
        differences, eval_seeds, replicates, bootstrap_seed
    )
    primary_bootstrap = _bootstrap_report(primary_means, inference)
    decision = _decide(
        primary_bootstrap["lower_one_sided_95"],
        primary_bootstrap["upper_one_sided_95"],
        margin,
    )

    prospective_seeds = [seed for seed in eval_seeds if seed != prior_seed]
    secondary: dict[str, Any] = {
        "role": "secondary_prospective_slice_only",
        "is_primary_decision": False,
    }
    if prospective_seeds:
        secondary.update(
            _slice_summary(
                differences, fresh["source"], fresh["candidate"], prospective_seeds
            )
        )
        secondary_means = _stratified_paired_bootstrap(
            differences, prospective_seeds, replicates, bootstrap_seed
        )
        secondary_bootstrap = _bootstrap_report(secondary_means, inference)
        secondary["bootstrap"] = secondary_bootstrap
        secondary["decision_if_applied"] = _decide(
            secondary_bootstrap["lower_one_sided_95"],
            secondary_bootstrap["upper_one_sided_95"],
            margin,
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "completed_analysis_only_no_training",
        "inputs": {
            "config": _identity(config_path),
            "aggregate": _identity(aggregate_path),
            "prior_aggregate": _identity(prior_aggregate_path),
        },
        "identities": {
            "source_checkpoint_sha256": str(config["source"]["checkpoint_sha256"]),
            "candidate_checkpoint_sha256": str(config["candidate"]["checkpoint_sha256"]),
            "prior_aggregate_sha256": prior_sha256,
            "declared_prior_query_catalog_sha256": str(
                prior_screen["query_catalog_sha256"]
            ),
            "eval_seeds": eval_seeds,
            "queries_per_seed": queries_per_seed,
            "paired_episode_count": total_paired,
            "aggregate_model_order": list(MODEL_ORDER),
        },
        "seed42_query_catalog_reproduction": catalog_report,
        "seed42_outcome_reproduction": outcome_reproduction,
        "per_seed": per_seed,
        "pooled": pooled,
        "primary_bootstrap": primary_bootstrap,
        "decision": {
            "outcome": decision,
            "noninferiority_margin": margin,
            "rule": {
                "pass": str(inference["pass"]),
                "fail": str(inference["fail"]),
                "inconclusive": str(inference["inconclusive"]),
            },
            "lower_one_sided_95": primary_bootstrap["lower_one_sided_95"],
            "upper_one_sided_95": primary_bootstrap["upper_one_sided_95"],
            "two_sided_95_interval": primary_bootstrap["two_sided_95_interval"],
            "pooled_difference": pooled["difference"],
        },
        "secondary_prospective_seeds": secondary,
        "claim_boundary": config["claim_boundary"],
    }
    _check_finite(report)
    return report


def _write(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--prior-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = analyze(
        args.config.expanduser().resolve(),
        args.aggregate.expanduser().resolve(),
        args.prior_aggregate.expanduser().resolve(),
    )
    _write(report, output)
    print(
        json.dumps(
            {
                "decision": report["decision"]["outcome"],
                "pooled_difference": report["pooled"]["difference"],
                "lower_one_sided_95": report["decision"]["lower_one_sided_95"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


if __name__ == "__main__":
    main()
