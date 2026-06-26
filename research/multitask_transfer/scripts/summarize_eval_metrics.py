#!/usr/bin/env python3
"""Summarize multitask-transfer eval metrics.

The script reads raw `*_metrics.txt` files produced by `scripts/plan/eval_wm.py`
and writes reproducible tables for the research ledger.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHECKPOINT_ROOTS = (
    Path("/opt/huawei/explorer-env/dataset/ag_data/outputs/stablewm/checkpoints"),
    Path("/home/ag/dataset/ag_data/outputs/stablewm/checkpoints"),
    Path("/opt/workspace/explorer-env/dataset/ag_data/outputs/stablewm/checkpoints"),
)
CHECKPOINT_ROOT = Path(
    os.environ.get("STABLEWM_CHECKPOINT_ROOT", "")
) if os.environ.get("STABLEWM_CHECKPOINT_ROOT") else next(
    (path for path in DEFAULT_CHECKPOINT_ROOTS if path.exists()),
    DEFAULT_CHECKPOINT_ROOTS[0],
)
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = RESEARCH_ROOT / "tables"
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    task: str
    training_regime: str
    checkpoint_dir: str
    epoch: int
    fairness_view: str


EXPERIMENTS = [
    ExperimentSpec(
        "mt3_epoch10_pusht",
        "pusht",
        "three_task",
        "lewm_mt3_lance",
        10,
        "multitask_anchor",
    ),
    ExperimentSpec(
        "mt3_epoch10_reacher",
        "reacher",
        "three_task",
        "lewm_mt3_lance",
        10,
        "multitask_anchor",
    ),
    ExperimentSpec(
        "mt3_epoch10_tworoom",
        "tworoom",
        "three_task",
        "lewm_mt3_lance",
        10,
        "multitask_anchor",
    ),
    ExperimentSpec(
        "mt3_epoch30_pusht",
        "pusht",
        "three_task",
        "lewm_mt3_lance",
        30,
        "long_multitask",
    ),
    ExperimentSpec(
        "mt3_epoch30_reacher",
        "reacher",
        "three_task",
        "lewm_mt3_lance",
        30,
        "long_multitask",
    ),
    ExperimentSpec(
        "mt3_epoch30_tworoom",
        "tworoom",
        "three_task",
        "lewm_mt3_lance",
        30,
        "long_multitask",
    ),
    ExperimentSpec(
        "mt4_epoch10_pusht",
        "pusht",
        "four_task",
        "lewm_mt4_lance",
        10,
        "four_task_mid_training",
    ),
    ExperimentSpec(
        "mt4_epoch10_reacher",
        "reacher",
        "four_task",
        "lewm_mt4_lance",
        10,
        "four_task_mid_training",
    ),
    ExperimentSpec(
        "mt4_epoch10_tworoom",
        "tworoom",
        "four_task",
        "lewm_mt4_lance",
        10,
        "four_task_mid_training",
    ),
    ExperimentSpec(
        "mt4_epoch10_cube",
        "cube",
        "four_task",
        "lewm_mt4_lance",
        10,
        "four_task_mid_training",
    ),
    ExperimentSpec(
        "mt4_epoch30_pusht",
        "pusht",
        "four_task",
        "lewm_mt4_lance",
        30,
        "four_task_final",
    ),
    ExperimentSpec(
        "mt4_epoch30_reacher",
        "reacher",
        "four_task",
        "lewm_mt4_lance",
        30,
        "four_task_final",
    ),
    ExperimentSpec(
        "mt4_epoch30_tworoom",
        "tworoom",
        "four_task",
        "lewm_mt4_lance",
        30,
        "four_task_final",
    ),
    ExperimentSpec(
        "mt4_epoch30_cube",
        "cube",
        "four_task",
        "lewm_mt4_lance",
        30,
        "four_task_final",
    ),
    ExperimentSpec(
        "mt4_epoch4_pusht",
        "pusht",
        "four_task",
        "lewm_mt4_lance",
        4,
        "four_task_early_diagnostic",
    ),
    ExperimentSpec(
        "pusht_single_epoch10",
        "pusht",
        "single_task",
        "lewm_pusht_lance",
        10,
        "per_task_exposure_matched",
    ),
    ExperimentSpec(
        "pusht_single_epoch30",
        "pusht",
        "single_task",
        "lewm_pusht_lance",
        30,
        "compute_matched",
    ),
    ExperimentSpec(
        "pusht_single_epoch33",
        "pusht",
        "single_task",
        "lewm_pusht_lance",
        33,
        "long_single_task",
    ),
    ExperimentSpec(
        "reacher_single_epoch10",
        "reacher",
        "single_task",
        "lewm_reacher_lance",
        10,
        "per_task_exposure_matched",
    ),
    ExperimentSpec(
        "reacher_single_epoch33",
        "reacher",
        "single_task",
        "lewm_reacher_lance",
        33,
        "compute_matched",
    ),
    ExperimentSpec(
        "tworoom_single_epoch30",
        "tworoom",
        "single_task",
        "lewm_tworoom_lance",
        30,
        "per_task_exposure_matched",
    ),
    ExperimentSpec(
        "tworoom_single_epoch80",
        "tworoom",
        "single_task",
        "lewm_tworoom_lance",
        80,
        "long_single_task",
    ),
]


# Train lengths measured from the active Lance datasets with train_split=0.9,
# frameskip=5, num_steps=4, seed=3072.
TRAIN_LENGTHS = {
    "pusht": 1_783_549,
    "reacher": 1_638_001,
    "tworoom": 657_729,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_metrics(path: Path) -> dict:
    text = path.read_text()
    match = re.search(r"==== JSON ====\s*(\{.*\})\s*$", text, re.S)
    if match:
        payload = json.loads(match.group(1))
        metrics = payload["metrics"]
        episode_successes = [bool(x) for x in metrics["episode_successes"]]
        return {
            "success_rate": float(metrics["success_rate"]),
            "successes": int(sum(episode_successes)),
            "num_eval": len(episode_successes),
            "weights_path": payload.get("weights_path", ""),
            "eval_dataset_path": payload.get("eval_dataset_path")
            or payload.get("lance_path", ""),
        }

    success_match = re.search(r"'success_rate':\s*([0-9.]+)", text)
    episode_match = re.search(
        r"'episode_successes':\s*array\(\[(.*?)\]\)", text, re.S
    )
    if not success_match or not episode_match:
        raise ValueError(f"Cannot parse metrics from {path}")
    tokens = re.findall(r"\bTrue\b|\bFalse\b", episode_match.group(1))
    successes = sum(token == "True" for token in tokens)
    return {
        "success_rate": float(success_match.group(1)),
        "successes": successes,
        "num_eval": len(tokens),
        "weights_path": "",
        "eval_dataset_path": "",
    }


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for spec in EXPERIMENTS:
        for seed in SEEDS:
            metrics_path = (
                CHECKPOINT_ROOT
                / spec.checkpoint_dir
                / "eval_results"
                / f"{spec.task}_epoch{spec.epoch}_num50_seed{seed}_metrics.txt"
            )
            if not metrics_path.exists():
                rows.append(
                    {
                        "experiment_id": spec.experiment_id,
                        "task": spec.task,
                        "training_regime": spec.training_regime,
                        "checkpoint_dir": spec.checkpoint_dir,
                        "epoch": spec.epoch,
                        "seed": seed,
                        "status": "missing",
                        "fairness_view": spec.fairness_view,
                        "success_rate": "",
                        "successes": "",
                        "num_eval": "",
                        "metrics_path": str(metrics_path),
                        "metrics_sha256": "",
                        "weights_path": "",
                        "eval_dataset_path": "",
                    }
                )
                continue

            parsed = parse_metrics(metrics_path)
            rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "task": spec.task,
                    "training_regime": spec.training_regime,
                    "checkpoint_dir": spec.checkpoint_dir,
                    "epoch": spec.epoch,
                    "seed": seed,
                    "status": "complete",
                    "fairness_view": spec.fairness_view,
                    "success_rate": parsed["success_rate"],
                    "successes": parsed["successes"],
                    "num_eval": parsed["num_eval"],
                    "metrics_path": str(metrics_path),
                    "metrics_sha256": sha256(metrics_path),
                    "weights_path": parsed["weights_path"],
                    "eval_dataset_path": parsed["eval_dataset_path"],
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for spec in EXPERIMENTS:
        group = [
            row
            for row in rows
            if row["experiment_id"] == spec.experiment_id
            and row["status"] == "complete"
        ]
        if not group:
            out.append(
                {
                    "experiment_id": spec.experiment_id,
                    "task": spec.task,
                    "training_regime": spec.training_regime,
                    "epoch": spec.epoch,
                    "status": "missing",
                }
            )
            continue

        rates = [float(row["success_rate"]) for row in group]
        successes = sum(int(row["successes"]) for row in group)
        num_eval = sum(int(row["num_eval"]) for row in group)
        out.append(
            {
                "experiment_id": spec.experiment_id,
                "task": spec.task,
                "training_regime": spec.training_regime,
                "epoch": spec.epoch,
                "fairness_view": spec.fairness_view,
                "seeds": " ".join(str(row["seed"]) for row in group),
                "mean_success_rate": round(sum(rates) / len(rates), 4),
                "sample_sd_pp": round(
                    statistics.stdev(rates) if len(rates) > 1 else 0.0, 4
                ),
                "pooled_success_rate": round(100.0 * successes / num_eval, 4),
                "successes": successes,
                "num_eval": num_eval,
                "status": "complete",
            }
        )
    return out


def exposure_rows() -> list[dict]:
    max_len = max(TRAIN_LENGTHS.values())
    rows = []
    for mt3_epoch in (10, 30):
        for task, train_len in TRAIN_LENGTHS.items():
            rows.append(
                {
                    "task": task,
                    "single_task_train_len": train_len,
                    "mt3_per_epoch_task_samples": max_len,
                    "mt3_epoch": mt3_epoch,
                    "equivalent_single_task_epochs": round(
                        mt3_epoch * max_len / train_len, 4
                    ),
                    "notes": "BalancedConcatDataset repeats shorter datasets modulo length.",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = collect_rows()
    summary = summarize(rows)
    exposure = exposure_rows()

    write_csv(TABLE_DIR / "eval_runs.csv", rows)
    write_csv(TABLE_DIR / "eval_summary.csv", summary)
    write_csv(TABLE_DIR / "exposure_equivalence.csv", exposure)

    summary_md = markdown_table(
        summary,
        [
            "experiment_id",
            "task",
            "training_regime",
            "epoch",
            "mean_success_rate",
            "sample_sd_pp",
            "successes",
            "num_eval",
            "fairness_view",
        ],
    )
    (TABLE_DIR / "eval_summary.md").write_text(summary_md)

    exposure_md = markdown_table(
        exposure,
        [
            "task",
            "single_task_train_len",
            "mt3_per_epoch_task_samples",
            "mt3_epoch",
            "equivalent_single_task_epochs",
            "notes",
        ],
    )
    (TABLE_DIR / "exposure_equivalence.md").write_text(exposure_md)

    missing = [row for row in rows if row["status"] != "complete"]
    if missing:
        print(f"completed with {len(missing)} missing metric files")
    else:
        print("completed with all configured metric files present")
    print(TABLE_DIR / "eval_summary.md")


if __name__ == "__main__":
    main()
