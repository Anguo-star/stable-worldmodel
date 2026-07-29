#!/usr/bin/env python3
"""Audit the three-training-seed conditional-SIGReg stability extension.

Each checkpoint is evaluated against the same frozen 300-query Door catalog,
the same frozen 300-query original-domain CEM catalog, and the same rollout
catalog.  CEM noninferiority is tested separately for every checkpoint against
the frozen native-SIGReg-0.09 control.  Pooled counts across training seeds are
descriptive only; they are not treated as independent training repetitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from analyze_conditional_sigreg_screen import (
    BOOTSTRAP_RESAMPLES,
    EVAL_SEEDS,
    NONINFERIORITY_MARGIN,
    REPO_ROOT,
    ROLLOUT_HORIZONS,
    _door_result,
    _load_json,
    _planning_result,
    _require,
    _rollout_result,
    _sha256,
    _training_result,
    paired_stratified_bootstrap,
)


TRAINING_SEEDS = (3072, 4096, 5120)
METHODS = ("paired_native_0p09", "conditional_sigreg_0p09")
STABLE_WORLDMODEL_COMMIT = (
    "ad2bc44579f2b5b65c004fd2c9d8edc8ebaa43ce"
)


def _variant_specifications(
    artifact_root: Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    reports = artifact_root / "training/reports"
    screen = (
        artifact_root
        / "evaluation/history3/conditional_sigreg_screen_v1"
    )
    multiseed = (
        artifact_root
        / "evaluation/history3/conditional_sigreg_multiseed_v1"
    )
    specifications: dict[int, dict[str, dict[str, Any]]] = {
        3072: {
            "paired_native_0p09": {
                "ability": screen / "ability/paired_native",
                "door": screen / "door/paired_native.json",
                "method_id": "paired_native",
                "training": reports
                / "h3_passage_replay50_paired_native_sigreg0p09_"
                "passage_formal_s3072.json",
            },
            "conditional_sigreg_0p09": {
                "ability": screen / "ability/conditional_sigreg_0p09",
                "door": screen / "door/conditional_sigreg_0p09.json",
                "method_id": "conditional_sigreg_0p09",
                "training": reports
                / "h3_passage_replay50_conditional_sigreg0p09_"
                "passage_formal_s3072.json",
            },
        }
    }
    for seed in (4096, 5120):
        specifications[seed] = {
            "paired_native_0p09": {
                "ability": multiseed / f"ability/paired_native_s{seed}",
                "door": multiseed / f"door/paired_native_s{seed}.json",
                "method_id": f"paired_native_s{seed}",
                "training": reports
                / "h3_passage_replay50_paired_native_sigreg0p09_"
                f"passage_formal_s{seed}.json",
            },
            "conditional_sigreg_0p09": {
                "ability": multiseed
                / f"ability/conditional_sigreg_0p09_s{seed}",
                "door": multiseed
                / f"door/conditional_sigreg_0p09_s{seed}.json",
                "method_id": f"conditional_sigreg_0p09_s{seed}",
                "training": reports
                / "h3_passage_replay50_conditional_sigreg0p09_"
                f"passage_formal_s{seed}.json",
            },
        }
    return specifications


def _evaluate_variant(
    specification: dict[str, Any],
    *,
    training_seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], frozenset[str], dict[str, Any]]:
    ability = Path(specification["ability"])
    cem_records, cem, cem_provenance = _planning_result(
        ability / "planning_original_heldout"
    )
    rollout, rollout_ids, rollout_provenance = _rollout_result(
        ability / "rollout_error.json"
    )
    training, training_provenance = _training_result(
        Path(specification["training"])
    )
    door, door_provenance = _door_result(
        Path(specification["door"]),
        method_id=str(specification["method_id"]),
    )
    _require(
        training["training_seed"] == training_seed,
        f"Training seed mismatch for {specification['method_id']}",
    )
    _require(
        training["swanlab_initialized"] is True,
        f"SwanLab was not initialized for {specification['method_id']}",
    )
    checkpoint_sha256 = training["checkpoint_sha256"]
    _require(
        checkpoint_sha256
        == door_provenance["checkpoint_sha256"]
        == cem_provenance["inputs"]["checkpoint"]["sha256"]
        == rollout_provenance["inputs"]["checkpoint"]["sha256"],
        f"Checkpoint identity mismatch for {specification['method_id']}",
    )
    _require(
        training_provenance["sha256"]
        == door_provenance["training_report_sha256"],
        f"Training-report identity mismatch for {specification['method_id']}",
    )
    stable_commits = {
        door_provenance["stable_worldmodel_commit"],
        cem_provenance["inputs"]["stable_worldmodel"]["commit"],
        rollout_provenance["inputs"]["stable_worldmodel"]["commit"],
    }
    _require(
        stable_commits == {STABLE_WORLDMODEL_COMMIT},
        f"Stable-WorldModel commit mismatch for {specification['method_id']}",
    )
    metrics = {
        "training": training,
        "door_rule_use": door,
        "original_domain_real_environment_cem": cem,
        "original_domain_rollout": rollout,
    }
    provenance = {
        "training": training_provenance,
        "door": door_provenance,
        "cem": cem_provenance,
        "rollout": rollout_provenance,
    }
    return metrics, cem_records, rollout_ids, provenance


def summarize_variants(
    variants: dict[str, dict[str, dict[str, Any]]],
    comparisons: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Create descriptive cross-seed summaries and registered decisions."""
    summary: dict[str, Any] = {}
    for method in METHODS:
        rows = [variants[str(seed)][method] for seed in TRAINING_SEEDS]
        cem_successes = sum(
            int(row["original_domain_real_environment_cem"]["successes"])
            for row in rows
        )
        door_rows = [row["door_rule_use"] for row in rows]
        summary[method] = {
            "training_checkpoints": len(rows),
            "door_checkpoints_passed": sum(
                int(row["passed"]) for row in door_rows
            ),
            "door_correct_target_selection_rate": {
                "mean": fmean(
                    row["correct_target_selection_rate"]
                    for row in door_rows
                ),
                "minimum": min(
                    row["correct_target_selection_rate"]
                    for row in door_rows
                ),
            },
            "door_correct_history_win_rate": {
                "mean": fmean(
                    row["correct_history_win_rate"] for row in door_rows
                ),
                "minimum": min(
                    row["correct_history_win_rate"] for row in door_rows
                ),
            },
            "door_worst_stratum_correct_rate": min(
                row["worst_seed_direction_rule_accuracy"]
                for row in door_rows
            ),
            "cem_descriptive_across_checkpoints": {
                "queries": 900,
                "successes": cem_successes,
                "success_rate": cem_successes / 900,
                "inference_allowed": False,
                "reason": (
                    "The pooled query count does not create 900 independent "
                    "training repetitions."
                ),
            },
            "rollout_mean_native_latent_mse_across_checkpoints": {
                str(horizon): fmean(
                    row["original_domain_rollout"][str(horizon)][
                        "mean_native_latent_mse"
                    ]
                    for row in rows
                )
                for horizon in ROLLOUT_HORIZONS
            },
        }
    conditional_noninferior = all(
        comparisons[str(seed)]["conditional_sigreg_0p09_vs_native_0p09"][
            "noninferior"
        ]
        for seed in TRAINING_SEEDS
    )
    conditional_door_passes = (
        summary["conditional_sigreg_0p09"]["door_checkpoints_passed"] == 3
    )
    paired_door_fails = (
        summary["paired_native_0p09"]["door_checkpoints_passed"] == 0
    )
    decision = {
        "conditional_door_passes_all_training_seeds": conditional_door_passes,
        "paired_native_door_fails_all_training_seeds": paired_door_fails,
        "conditional_cem_noninferior_at_every_training_seed": (
            conditional_noninferior
        ),
        "pairing_only_explanation_rejected": bool(
            conditional_door_passes and paired_door_fails
        ),
        "optimization_seed_stability_screen_passes": bool(
            conditional_door_passes
            and paired_door_fails
            and conditional_noninferior
        ),
        "cross_task_method_claim_allowed": False,
    }
    return {"methods": summary, "decision": decision}


def analyze(*, artifact_root: Path) -> dict[str, Any]:
    specifications = _variant_specifications(artifact_root)
    screen_root = (
        artifact_root
        / "evaluation/history3/conditional_sigreg_screen_v1/ability"
    )
    native_records, native_cem, native_cem_provenance = _planning_result(
        screen_root / "native_sigreg_0p09_reference/planning_original_heldout"
    )
    native_rollout, native_rollout_ids, native_rollout_provenance = (
        _rollout_result(
            screen_root / "native_sigreg_0p09_reference/rollout_error.json"
        )
    )
    reference_path = (
        REPO_ROOT
        / "research/conditional_dynamics_representation/results/"
        "sigreg_replay50_falsification_summary.json"
    )
    reference = _load_json(reference_path)
    native_reference = reference["models"][
        "H3_PassageReplay50_SIGReg0p09"
    ]
    native_checkpoint = native_cem_provenance["inputs"]["checkpoint"]["sha256"]
    _require(
        native_checkpoint
        == native_rollout_provenance["inputs"]["checkpoint"]["sha256"]
        == native_reference["checkpoint_sha256"],
        "Frozen native-SIGReg-0.09 control identity mismatch",
    )

    variants: dict[str, dict[str, dict[str, Any]]] = {}
    records: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    provenance: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    reference_cem_inputs = native_cem_provenance["inputs"]
    reference_rollout_inputs = native_rollout_provenance["inputs"]
    for training_seed in TRAINING_SEEDS:
        seed_key = str(training_seed)
        variants[seed_key] = {}
        records[seed_key] = {}
        provenance[seed_key] = {}
        for method in METHODS:
            metrics, cem_records, rollout_ids, method_provenance = (
                _evaluate_variant(
                    specifications[training_seed][method],
                    training_seed=training_seed,
                )
            )
            _require(
                set(cem_records) == set(native_records),
                f"CEM query identities differ for {method}, seed {seed_key}",
            )
            _require(
                rollout_ids == native_rollout_ids,
                f"Rollout query identities differ for {method}, seed {seed_key}",
            )
            cem_inputs = method_provenance["cem"]["inputs"]
            rollout_inputs = method_provenance["rollout"]["inputs"]
            _require(
                cem_inputs["catalog"] == reference_cem_inputs["catalog"]
                and cem_inputs["normalizer"]
                == reference_cem_inputs["normalizer"]
                and cem_inputs["stable_worldmodel"]
                == reference_cem_inputs["stable_worldmodel"],
                f"CEM inputs changed for {method}, seed {seed_key}",
            )
            _require(
                rollout_inputs["catalog"]
                == reference_rollout_inputs["catalog"]
                and rollout_inputs["normalizer"]
                == reference_rollout_inputs["normalizer"]
                and rollout_inputs["stable_worldmodel"]
                == reference_rollout_inputs["stable_worldmodel"],
                f"Rollout inputs changed for {method}, seed {seed_key}",
            )
            variants[seed_key][method] = metrics
            records[seed_key][method] = cem_records
            provenance[seed_key][method] = method_provenance
        comparisons[seed_key] = {
            "paired_native_0p09_vs_native_0p09": (
                paired_stratified_bootstrap(
                    native_records,
                    records[seed_key]["paired_native_0p09"],
                )
            ),
            "conditional_sigreg_0p09_vs_native_0p09": (
                paired_stratified_bootstrap(
                    native_records,
                    records[seed_key]["conditional_sigreg_0p09"],
                )
            ),
            "conditional_sigreg_0p09_vs_paired_native_0p09": (
                paired_stratified_bootstrap(
                    records[seed_key]["paired_native_0p09"],
                    records[seed_key]["conditional_sigreg_0p09"],
                )
            ),
        }

    summaries = summarize_variants(variants, comparisons)
    return {
        "schema_version": 1,
        "benchmark": "tworoom_conditional_sigreg_multiseed_v1",
        "status": "completed_three_training_seed_stability_extension",
        "date": "2026-07-29",
        "protocol": {
            "training_seeds": list(TRAINING_SEEDS),
            "initial_weights_fixed": True,
            "data_split_seed": 3072,
            "formal_training": (
                "8 GPUs, global batch 1024, 1024 optimizer steps per checkpoint"
            ),
            "door": "50 queries x 6 eval seeds; frozen rule-switch-v2",
            "utility": "50 queries x 6 eval seeds; real-environment CEM",
            "rollout_horizons_action_blocks": list(ROLLOUT_HORIZONS),
            "eval_seeds": list(EVAL_SEEDS),
            "noninferiority_margin_absolute": NONINFERIORITY_MARGIN,
            "bootstrap_resamples_per_checkpoint": BOOTSTRAP_RESAMPLES,
            "excluded_metrics": [
                "spearman_candidate_ranking",
                "generic_candidate_order_correlation",
            ],
        },
        "frozen_native_sigreg_0p09_control": {
            "training_seed": 3072,
            "checkpoint_sha256": native_checkpoint,
            "door_rule_use": dict(native_reference["door"]),
            "original_domain_real_environment_cem": native_cem,
            "original_domain_rollout": native_rollout,
        },
        "variants_by_training_seed": variants,
        "paired_cem_comparisons_by_training_seed": comparisons,
        "descriptive_summary": summaries["methods"],
        "decision": {
            **summaries["decision"],
            "claim_boundary": (
                "Three optimization seeds on one hidden-dynamics task establish "
                "a stable candidate mechanism, not cross-task superiority over "
                "sufficiently tuned native SIGReg."
            ),
            "closed_loop_boundary": (
                "The blocked mode of the one-passage Door environment makes an "
                "opposite-room goal unreachable, so 'did not cross while "
                "blocked' is not a valid rule-inference result. A second task "
                "must give both hidden modes reachable but different optimal "
                "actions."
            ),
        },
        "provenance": {
            "reference_summary": {
                "path": str(reference_path.resolve()),
                "sha256": _sha256(reference_path),
            },
            "native_control": {
                "cem": native_cem_provenance,
                "rollout": native_rollout_provenance,
            },
            "variants_by_training_seed": provenance,
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
            "conditional_sigreg_multiseed_v1.json"
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
                "stability_screen_passed": result["decision"][
                    "optimization_seed_stability_screen_passes"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
