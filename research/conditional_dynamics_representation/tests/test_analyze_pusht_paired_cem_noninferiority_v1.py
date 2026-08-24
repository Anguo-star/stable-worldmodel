from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from research.conditional_dynamics_representation.scripts import (
    analyze_pusht_paired_cem_noninferiority_v1 as analyzer,
)


SOURCE_SHA = "a" * 64
CANDIDATE_SHA = "b" * 64
EVAL_SEEDS = (42, 43, 44)
QUERIES_PER_SEED = 4
FIXED_PROTOCOL = {
    "action_block": 5,
    "cem_iterations": 3,
    "cem_samples": 30,
    "cem_topk": 3,
    "eval_budget": 50,
    "goal_offset_steps": 25,
    "history_len": 3,
    "horizon": 5,
    "receding_horizon": 5,
    "videos_written": False,
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump(path: Path, payload: Any) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return _sha256_text(text)


def _catalog(seeds=EVAL_SEEDS, shift: int = 0) -> dict[str, Any]:
    return {
        str(seed): {
            "episode_indices": [seed * 10 + i + shift for i in range(QUERIES_PER_SEED)],
            "row_indices": [seed * 100 + i for i in range(QUERIES_PER_SEED)],
            "start_steps": [i for i in range(QUERIES_PER_SEED)],
        }
        for seed in seeds
    }


def _model(name: str, sha: str, outcomes: dict[int, list[bool]]) -> dict[str, Any]:
    seeds = [
        {
            "episode_successes": list(values),
            "eval_seed": seed,
            "query_count": len(values),
            "success_count": sum(values),
            "success_rate": sum(values) / len(values),
        }
        for seed, values in outcomes.items()
    ]
    total = sum(entry["query_count"] for entry in seeds)
    successes = sum(entry["success_count"] for entry in seeds)
    return {
        "aggregate": {
            "evaluation_count": total,
            "success_count": successes,
            "success_rate": successes / total,
        },
        "checkpoint": f"/fixture/{name}.pt",
        "checkpoint_sha256": sha,
        "model": name,
        "seeds": seeds,
    }


def _aggregate(
    source: dict[int, list[bool]],
    candidate: dict[int, list[bool]],
    catalog_path: Path,
) -> dict[str, Any]:
    protocol = dict(FIXED_PROTOCOL)
    protocol.update(
        {
            "dataset": "/fixture/dataset.h5",
            "eval_seeds": sorted(source),
            "num_eval_per_seed": QUERIES_PER_SEED,
        }
    )
    return {
        "models": [
            _model("source", SOURCE_SHA, source),
            _model("candidate", CANDIDATE_SHA, candidate),
        ],
        "protocol": protocol,
        "query_catalog": {"path": str(catalog_path), "sha256": "unused"},
        "schema_version": 1,
        "status": "standard_pusht_real_environment_cem",
    }


def _config(prior_sha: str, prior_catalog_sha: str, prior: dict[str, Any]) -> dict[str, Any]:
    source_successes = sum(prior["source"][42])
    candidate_successes = sum(prior["candidate"][42])
    return {
        "schema_version": 1,
        "no_training": True,
        "source": {"checkpoint": "/fixture/source.pt", "checkpoint_sha256": SOURCE_SHA},
        "candidate": {
            "checkpoint": "/fixture/candidate.pt",
            "checkpoint_sha256": CANDIDATE_SHA,
        },
        "prior_screen": {
            "eval_seed": 42,
            "paired_episodes": QUERIES_PER_SEED,
            "source_successes": source_successes,
            "candidate_successes": candidate_successes,
            "aggregate_sha256": prior_sha,
            "query_catalog_sha256": prior_catalog_sha,
        },
        "execution": {
            "eval_seeds": list(EVAL_SEEDS),
            "queries_per_seed": QUERIES_PER_SEED,
            "paired_episode_count": QUERIES_PER_SEED * len(EVAL_SEEDS),
        },
        "fixed_protocol": dict(FIXED_PROTOCOL),
        "primary_inference": {
            "estimand": "mean over paired episodes of candidate_success minus source_success",
            "noninferiority_margin": -0.05,
            "resampling_unit": "paired episode within eval-seed stratum",
            "bootstrap_replicates": 200,
            "bootstrap_seed": 20260824,
            "quantile_method": "numpy_linear",
            "lower_one_sided_95_quantile": 0.05,
            "upper_one_sided_95_quantile": 0.95,
            "two_sided_95_quantiles": [0.025, 0.975],
            "pass": "lower_one_sided_95 > -0.05",
            "fail": "upper_one_sided_95 < -0.05",
            "inconclusive": "otherwise",
        },
        "claim_boundary": {"single_training_seed": True},
    }


def _write_case(
    tmp_path: Path,
    fresh_source: dict[int, list[bool]],
    fresh_candidate: dict[int, list[bool]],
    prior_source: list[bool] | None = None,
    prior_candidate: list[bool] | None = None,
    fresh_catalog_shift: int = 0,
    config_mutator=None,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prior_source = fresh_source[42] if prior_source is None else prior_source
    prior_candidate = fresh_candidate[42] if prior_candidate is None else prior_candidate

    prior_catalog_path = tmp_path / "prior_query_catalog.json"
    prior_catalog_sha = _dump(prior_catalog_path, _catalog(seeds=(42,)))
    fresh_catalog_path = tmp_path / "query_catalog.json"
    _dump(fresh_catalog_path, _catalog(shift=fresh_catalog_shift))

    prior_aggregate_path = tmp_path / "prior_aggregate.json"
    prior_sha = _dump(
        prior_aggregate_path,
        _aggregate({42: prior_source}, {42: prior_candidate}, prior_catalog_path),
    )
    aggregate_path = tmp_path / "aggregate.json"
    _dump(aggregate_path, _aggregate(fresh_source, fresh_candidate, fresh_catalog_path))

    prior = {"source": {42: prior_source}, "candidate": {42: prior_candidate}}
    config = _config(prior_sha, prior_catalog_sha, prior)
    if config_mutator is not None:
        config_mutator(config)
    config_path = tmp_path / "config.json"
    _dump(config_path, config)
    return {
        "config": config_path,
        "aggregate": aggregate_path,
        "prior_aggregate": prior_aggregate_path,
        "output": tmp_path / "result.json",
    }


def _run(paths: dict[str, Path]) -> dict[str, Any]:
    return analyzer.main(
        [
            "--config",
            str(paths["config"]),
            "--aggregate",
            str(paths["aggregate"]),
            "--prior-aggregate",
            str(paths["prior_aggregate"]),
            "--output",
            str(paths["output"]),
        ]
    )


T = True
F = False


def test_pass_decision_and_stable_output(tmp_path: Path) -> None:
    source = {42: [T, F, F, F], 43: [T, F, F, F], 44: [T, F, F, F]}
    candidate = {42: [T, T, T, T], 43: [T, T, T, T], 44: [T, T, T, T]}
    paths = _write_case(tmp_path, source, candidate)
    report = _run(paths)

    assert report["decision"]["outcome"] == "pass"
    assert report["decision"]["lower_one_sided_95"] > -0.05
    assert report["pooled"]["difference"] == pytest.approx(0.75)
    assert report["pooled"]["contingency"]["candidate_only_success"] == 9
    assert report["pooled"]["contingency"]["source_only_success"] == 0
    assert report["per_seed"]["43"]["difference"] == pytest.approx(0.75)
    assert report["seed42_query_catalog_reproduction"][
        "seed42_rows_reproduce_prior_catalog"
    ]
    assert report["secondary_prospective_seeds"]["eval_seeds"] == [43, 44]
    assert report["secondary_prospective_seeds"]["is_primary_decision"] is False

    text = paths["output"].read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text == json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert "NaN" not in text and "Infinity" not in text


def test_fail_decision(tmp_path: Path) -> None:
    source = {42: [T, T, T, T], 43: [T, T, T, T], 44: [T, T, T, T]}
    candidate = {42: [F, F, F, F], 43: [F, F, F, F], 44: [F, F, F, F]}
    paths = _write_case(tmp_path, source, candidate)
    report = _run(paths)

    assert report["decision"]["outcome"] == "fail"
    assert report["decision"]["upper_one_sided_95"] < -0.05
    assert report["pooled"]["difference"] == pytest.approx(-1.0)
    assert report["pooled"]["exact_two_sided_mcnemar_p"] == pytest.approx(2.0**-11)


def test_inconclusive_decision_with_zero_difference(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [F, T, T, F], 43: [T, T, F, F], 44: [F, T, F, T]}
    paths = _write_case(tmp_path, source, candidate)
    report = _run(paths)

    assert report["decision"]["outcome"] == "inconclusive"
    assert report["pooled"]["difference"] == pytest.approx(0.0)
    assert report["pooled"]["exact_two_sided_mcnemar_p"] == pytest.approx(1.0)
    assert report["decision"]["lower_one_sided_95"] <= -0.05
    assert report["decision"]["upper_one_sided_95"] >= -0.05


def test_bootstrap_is_deterministic(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [T, T, T, F], 43: [T, T, F, T], 44: [F, T, F, T]}
    first = _run(_write_case(tmp_path / "a", source, candidate))
    second = _run(_write_case(tmp_path / "b", source, candidate))
    assert first["primary_bootstrap"] == second["primary_bootstrap"]


def test_candidate_checkpoint_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [T, T, T, F], 43: [T, T, F, T], 44: [F, T, F, T]}

    def mutate(config: dict[str, Any]) -> None:
        config["candidate"]["checkpoint_sha256"] = "c" * 64

    paths = _write_case(tmp_path, source, candidate, config_mutator=mutate)
    with pytest.raises(analyzer.ValidationError, match="candidate checkpoint sha256"):
        _run(paths)
    assert not paths["output"].exists()


def test_prior_aggregate_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [T, T, T, F], 43: [T, T, F, T], 44: [F, T, F, T]}

    def mutate(config: dict[str, Any]) -> None:
        config["prior_screen"]["aggregate_sha256"] = "d" * 64

    paths = _write_case(tmp_path, source, candidate, config_mutator=mutate)
    with pytest.raises(analyzer.ValidationError, match="prior aggregate sha256"):
        _run(paths)


def test_wrong_eval_seeds_fail_closed(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F]}
    candidate = {42: [T, T, T, F], 43: [T, T, F, T]}
    paths = _write_case(tmp_path, source, candidate)
    with pytest.raises(analyzer.ValidationError, match="eval seeds"):
        _run(paths)


def test_short_paired_outcomes_fail_closed(tmp_path: Path) -> None:
    source = {42: [T, F, T], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [T, T, T], 43: [T, T, F, T], 44: [F, T, F, T]}
    paths = _write_case(tmp_path, source, candidate)
    with pytest.raises(analyzer.ValidationError, match="outcomes, expected 4"):
        _run(paths)


def test_model_order_mismatch_fails_closed(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [T, T, T, F], 43: [T, T, F, T], 44: [F, T, F, T]}
    paths = _write_case(tmp_path, source, candidate)
    aggregate = json.loads(paths["aggregate"].read_text(encoding="utf-8"))
    aggregate["models"].reverse()
    _dump(paths["aggregate"], aggregate)
    with pytest.raises(analyzer.ValidationError, match="model order/names"):
        _run(paths)


def test_seed42_query_catalog_mismatch_fails_closed(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [T, T, T, F], 43: [T, T, F, T], 44: [F, T, F, T]}
    paths = _write_case(tmp_path, source, candidate, fresh_catalog_shift=1)
    with pytest.raises(analyzer.ValidationError, match="query-catalog rows"):
        _run(paths)
    assert not paths["output"].exists()


def test_seed42_outcome_change_is_reported_but_not_a_gate(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [T, T, T, F], 43: [T, T, F, T], 44: [F, T, F, T]}
    paths = _write_case(
        tmp_path,
        source,
        candidate,
        prior_source=[F, F, T, F],
        prior_candidate=[T, T, T, F],
    )
    report = _run(paths)

    reproduction = report["seed42_outcome_reproduction"]
    assert reproduction["is_validity_gate"] is False
    assert reproduction["source"]["outcomes_reproduce_prior_aggregate"] is False
    assert reproduction["source"]["prior_successes"] == 1
    assert reproduction["source"]["fresh_successes"] == 2
    assert reproduction["candidate"]["outcomes_reproduce_prior_aggregate"] is True
    assert report["decision"]["outcome"] in {"pass", "fail", "inconclusive"}


def test_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = {42: [T, F, T, F], 43: [T, T, F, F], 44: [T, F, F, T]}
    candidate = {42: [T, T, T, F], 43: [T, T, F, T], 44: [F, T, F, T]}
    paths = _write_case(tmp_path, source, candidate)
    paths["output"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(paths)
