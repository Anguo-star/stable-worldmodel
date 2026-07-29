#!/usr/bin/env python3
"""Score paired-replay LeWM checkpoints on frozen Door rule futures.

The pre-existing ContextWorld evaluator hard-codes a synthetic-only training
provenance contract.  Replay50 checkpoints intentionally contain 50% original
data, so this entry keeps the established frozen assets, model adapter, scorer,
summary, and rule-switch gates while validating the actual replay50 training
report instead of weakening that old contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


EXPECTED_STEPS = 1024
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_PER_DEVICE = 128
ALLOWED_OBJECTIVES = (
    "native_lewm",
    "lewm_conditional_sigreg_0p09",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_training_provenance(
    *,
    checkpoint: Path,
    training_report: Path,
    expected_objective: str,
    stablewm_ref: str,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    training_report = training_report.resolve()
    _require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
    _require(
        training_report.is_file(),
        f"Missing training report: {training_report}",
    )
    report = json.loads(training_report.read_text(encoding="utf-8"))
    checkpoint_sha256 = _sha256(checkpoint)
    artifacts = report["artifacts"]
    training = report["training"]
    model = report["model"]
    objective = model["training_objective"]
    paired = report["data"]["paired_batch_execution"]
    visible = paired["visible_batch_audit"]

    _require(report.get("passed") is True, "Training report did not pass")
    _require(
        report.get("profile") == "passage_formal",
        "Checkpoint is not passage_formal",
    )
    _require(
        training.get("training_complete") is True
        and int(training.get("global_step", -1)) == EXPECTED_STEPS,
        "Training did not complete the frozen 1,024-step budget",
    )
    _require(
        int(training.get("world_size", -1)) == EXPECTED_WORLD_SIZE
        and int(training.get("devices", -1)) == EXPECTED_WORLD_SIZE
        and int(training.get("batch_size_per_device", -1))
        == EXPECTED_BATCH_PER_DEVICE,
        "Training topology differs from 8 GPUs x batch 128",
    )
    _require(
        report.get("save_load_exact") is True,
        "Final checkpoint failed the save/load equality audit",
    )
    _require(
        Path(artifacts["pretrained"]).resolve() == checkpoint
        and artifacts["pretrained_sha256"] == checkpoint_sha256,
        "Training report checkpoint identity mismatch",
    )
    _require(
        report["stable_worldmodel"]["commit"] == stablewm_ref,
        "Training and scoring StableWorldModel commits differ",
    )
    _require(
        expected_objective in ALLOWED_OBJECTIVES,
        f"Unsupported expected objective: {expected_objective}",
    )
    _require(
        objective["name"] == expected_objective,
        "Checkpoint objective does not match the registered method",
    )
    _require(
        float(objective["regularizer_weight"]) == 0.09,
        "Registered comparison requires regularizer weight 0.09",
    )
    _require(
        visible.get("passed") is True
        and visible.get("same_initial_pixels_for_every_pair") is True
        and visible.get("same_full_action_sequence_for_every_pair") is True,
        "Visible-condition pair audit did not pass",
    )
    _require(
        paired.get("uses_rule_value_in_loss") is False
        and paired.get("uses_pair_id_in_loss") is False,
        "Privileged pair metadata reached the training loss",
    )
    _require(
        model["input_boundary"]["privileged_fields_at_model_boundary"] == [],
        "Privileged fields reached the model boundary",
    )
    _require(
        report["logger"]["backend"] == "swanlab"
        and report["logger"]["initialized"] is True,
        "Formal checkpoint was not tracked by SwanLab",
    )

    config_path = Path(artifacts["pretrained_config"]).resolve()
    _require(config_path.is_file(), f"Missing checkpoint config: {config_path}")
    config_sha256 = _sha256(config_path)
    _require(
        config_sha256 == artifacts["pretrained_config_sha256"],
        "Checkpoint config hash mismatch",
    )
    checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
    embedded = checkpoint_config["contextworld_benchmark"]
    _require(
        embedded["training_objective"] == objective,
        "Checkpoint and report embed different objectives",
    )
    _require(
        embedded["data"]["paired_batch_execution"] == paired,
        "Checkpoint and report embed different paired-data receipts",
    )

    return {
        "passed": True,
        "training_report": str(training_report),
        "training_report_sha256": _sha256(training_report),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_config": str(config_path),
        "checkpoint_config_sha256": config_sha256,
        "run_name": report["run_name"],
        "training_seed": int(training["seed_before_model_initialization"]),
        "stable_worldmodel_commit": stablewm_ref,
        "objective": objective,
        "initialization_checkpoint": report["initialization_checkpoint"],
        "paired_batch_execution": paired,
        "loss_trace": training["loss_trace"],
        "logger": report["logger"],
    }


def _registered_screen(summary: dict[str, Any]) -> dict[str, Any]:
    target_accuracy = float(
        summary["overall"]["same_history_two_target_accuracy"]
    )
    history_win_rate = float(
        summary["overall"]["matching_vs_opposite_history_win_rate"]
    )
    strata = {
        f"{rule}/{cell}": float(
            values["same_history_two_target_accuracy"]
        )
        for rule, rule_summary in summary["by_true_rule"].items()
        for cell, values in rule_summary[
            "by_eval_seed_and_direction"
        ].items()
    }
    worst_stratum = min(strata.values())
    thresholds = {
        "correct_target_selection_rate_min": 0.95,
        "correct_history_win_rate_min": 0.95,
        "worst_stratum_correct_rate_min": 0.80,
    }
    checks = {
        "frozen_rule_switch_v2_passed": (
            summary["decision"]["passed"] is True
        ),
        "correct_target_selection_rate": (
            target_accuracy
            >= thresholds["correct_target_selection_rate_min"]
        ),
        "correct_history_win_rate": (
            history_win_rate
            >= thresholds["correct_history_win_rate_min"]
        ),
        "worst_stratum_correct_rate": (
            worst_stratum
            >= thresholds["worst_stratum_correct_rate_min"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "metrics": {
            "correct_target_selection_rate": target_accuracy,
            "correct_history_win_rate": history_win_rate,
            "worst_stratum_correct_rate": worst_stratum,
            "strata": dict(sorted(strata.items())),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextworld-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument(
        "--expected-objective",
        choices=ALLOWED_OBJECTIVES,
        required=True,
    )
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--stablewm-repo", required=True)
    parser.add_argument("--stablewm-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contextworld_repo = args.contextworld_repo.resolve()
    if str(contextworld_repo) not in sys.path:
        sys.path.insert(0, str(contextworld_repo))

    from contextworld.benchmarks.adapters import StableWorldModelLeWMAdapter
    from contextworld.evaluation.hidden_passage_validation import (
        load_validation_assets,
        score_validation_assets,
        summarize_validation_records,
    )
    from contextworld.paths import resolve_contextworld_path
    from contextworld.synthesis.manifest import write_json

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    catalog = resolve_contextworld_path(
        args.catalog or config["artifacts"]["catalog"],
        repo_root=contextworld_repo,
    )
    normalizer = resolve_contextworld_path(
        args.normalizer or config["adapter"]["normalizer"],
        repo_root=contextworld_repo,
    )
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    _require(not output.exists(), f"Refusing to overwrite {output}")
    _require(
        _sha256(normalizer) == config["adapter"]["normalizer_sha256"],
        "Frozen normalizer hash mismatch",
    )
    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    _require(
        catalog_payload["benchmark"] == config["benchmark"],
        "Catalog benchmark differs from the frozen scoring protocol",
    )
    evaluation = config["evaluation"]
    _require(
        list(map(int, evaluation["eval_seeds"]))
        == [42, 43, 44, 45, 46, 47]
        and int(evaluation["unique_queries_per_seed"]) == 50,
        "Door screen requires the frozen 50 x 6 query matrix",
    )
    training_provenance = validate_training_provenance(
        checkpoint=checkpoint,
        training_report=args.training_report,
        expected_objective=args.expected_objective,
        stablewm_ref=args.stablewm_ref,
    )
    assets, data_audit = load_validation_assets(
        catalog,
        repo_root=contextworld_repo,
    )
    adapter = StableWorldModelLeWMAdapter.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=contextworld_repo,
        stablewm_repo=args.stablewm_repo,
        stablewm_ref=args.stablewm_ref,
        device=args.device,
    )
    scored = score_validation_assets(
        adapter,
        assets,
        batch_size=int(args.batch_size or evaluation["batch_size"]),
    )
    summary = summarize_validation_records(
        scored["records"],
        eval_seeds=evaluation["eval_seeds"],
        unique_queries_per_seed=int(
            evaluation["unique_queries_per_seed"]
        ),
        gates=config["gates"],
    )
    screen = _registered_screen(summary)
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "method_id": args.method_id,
        "registered_screen": screen,
        "identity": {
            "scoring_config": str(config_path),
            "scoring_config_sha256": _sha256(config_path),
            "catalog": str(catalog),
            "catalog_sha256": _sha256(catalog),
            "normalizer": str(normalizer),
            "normalizer_sha256": _sha256(normalizer),
            "evaluation_script": str(Path(__file__).resolve()),
            "evaluation_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "training_provenance": training_provenance,
        "model": adapter.metadata,
        "data_audit": data_audit,
        "score_audit": scored["score_audit"],
        "summary": summary,
        "records": scored["records"],
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "method_id": args.method_id,
                "registered_screen": screen,
                "frozen_rule_switch_v2": summary["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
