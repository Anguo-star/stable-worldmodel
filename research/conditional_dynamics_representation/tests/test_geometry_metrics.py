from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_checkpoint_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("geometry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


def test_effective_rank_distinguishes_line_and_plane() -> None:
    line = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    square = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    assert np.isclose(geometry.effective_rank(line), 1.0)
    assert np.isclose(geometry.effective_rank(square), 2.0)


def test_paired_ratio_detects_local_contraction() -> None:
    passable = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    blocked = passable + np.asarray([0.01, 0.0])
    summary = geometry.paired_and_unrelated_mse(passable, blocked)
    assert summary["paired_mean_mse"] < summary["unrelated_mean_mse"]
    assert summary["paired_to_unrelated_ratio"] < 1.0e-4


def test_prediction_summary_requires_directional_switch() -> None:
    target_passable = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    target_blocked = np.asarray([[-1.0, 0.0], [-1.0, 0.0]])
    summary = geometry.prediction_summary(
        predicted_passable_history=target_passable,
        predicted_blocked_history=target_blocked,
        predicted_no_attempt_history=np.zeros((2, 2)),
        target_passable=target_passable,
        target_blocked=target_blocked,
    )
    assert summary["passable_history_target_accuracy"] == 1.0
    assert summary["blocked_history_target_accuracy"] == 1.0
    assert summary["correct_rule_switch_rate"] == 1.0
    assert summary["difference_alignment_cosine_mean"] == 1.0


def test_pldm_scope_registers_native_pldm_adapters(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "training/runs/checkpoints"
    runs = [
        "h3_origheldout_s3072",
        *[
            run
            for seed in (3072, 4096, 5120)
            for run in (
                "h3_passage_mixed_rules_pldm_objective_"
                f"passage_formal_s{seed}",
                "h3_passage_mixed_rules_pldm_objective_"
                "fixed_representation_passage_formal_"
                f"s{seed}",
            )
        ],
    ]
    for run in runs:
        path = checkpoint_root / run
        path.mkdir(parents=True)
        filename = (
            "weights_final_step_6420.pt"
            if run == "h3_origheldout_s3072"
            else "weights_epoch_4.pt"
        )
        (path / filename).write_bytes(b"test")

    specs = geometry.checkpoint_specs(
        tmp_path,
        scope="pldm",
        tiny_epochs=(1,),
        formal_epochs=(4,),
        include_single_rule=False,
    )

    assert len(specs) == 7
    assert specs[0].training_method == "lewm"
    assert {spec.training_method for spec in specs[1:]} == {"pldm"}
    assert {spec.family for spec in specs[1:]} == {
        "pldm_joint",
        "pldm_fixed",
    }


def test_checkpoint_specs_support_candidate_run_pattern(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "training/runs/checkpoints"
    original = checkpoint_root / "h3_origheldout_s3072"
    original.mkdir(parents=True)
    (original / "weights_final_step_6420.pt").write_bytes(b"original")
    for seed in (3072, 4096):
        run = checkpoint_root / f"candidate_formal_s{seed}"
        run.mkdir(parents=True)
        for epoch in (1, 4):
            (run / f"weights_epoch_{epoch}.pt").write_bytes(b"candidate")

    specs = geometry.checkpoint_specs(
        tmp_path,
        scope="candidate",
        tiny_epochs=(1,),
        formal_epochs=(1, 4),
        include_single_rule=False,
        candidate_run_pattern="candidate_formal_s{seed}",
        candidate_seeds=(3072, 4096),
        candidate_training_method="lewm",
        candidate_label="SIGReg 2.05",
    )

    assert len(specs) == 5
    assert specs[0].family == "original"
    assert {spec.seed for spec in specs[1:]} == {3072, 4096}
    assert {spec.epoch for spec in specs[1:]} == {1, 4}
    assert all(spec.family == "candidate_joint" for spec in specs[1:])
    assert all("SIGReg 2.05" in spec.label for spec in specs[1:])
