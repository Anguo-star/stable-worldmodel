from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_sigreg_replay50_falsification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sigreg_replay_falsification_analysis", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _planning_rows(successes: dict[int, list[bool]]) -> dict[str, dict]:
    return {
        f"s{seed}-q{index:02d}": {
            "evaluation_id": f"s{seed}-q{index:02d}",
            "eval_seed": seed,
            "success": success,
            "final_distance": float(not success),
        }
        for seed, values in successes.items()
        for index, success in enumerate(values)
    }


def test_paired_bootstrap_recognizes_identical_candidate_as_noninferior() -> None:
    control = _planning_rows(
        {
            42: [True, False, True, False],
            43: [False, True, True, False],
        }
    )

    result = MODULE._paired_stratified_bootstrap(
        control,
        control,
        expected_seeds=(42, 43),
        resamples=1000,
        confidence=0.95,
        seed=7,
        margin=0.05,
    )

    assert result["point_difference"] == 0.0
    assert result["one_sided_lower_confidence_bound"] == 0.0
    assert result["noninferior"] is True
    assert result["discordant_pairs"] == {
        "candidate_only_success": 0,
        "control_only_success": 0,
    }


def test_paired_bootstrap_rejects_large_success_drop() -> None:
    control = _planning_rows(
        {
            42: [True] * 20,
            43: [True] * 20,
        }
    )
    candidate = _planning_rows(
        {
            42: [False] * 10 + [True] * 10,
            43: [False] * 10 + [True] * 10,
        }
    )

    result = MODULE._paired_stratified_bootstrap(
        control,
        candidate,
        expected_seeds=(42, 43),
        resamples=1000,
        confidence=0.95,
        seed=7,
        margin=0.05,
    )

    assert result["point_difference"] == -0.5
    assert result["one_sided_lower_confidence_bound"] < -0.05
    assert result["noninferior"] is False
    assert result["discordant_pairs"]["control_only_success"] == 20


def test_planning_summary_uses_real_environment_success() -> None:
    rows = _planning_rows(
        {
            42: [True, False],
            43: [True, True],
        }
    )

    summary = MODULE._planning_summary(rows, expected_seeds=(42, 43))

    assert summary["successes"] == 3
    assert summary["success_rate"] == 0.75
    assert summary["by_eval_seed"]["42"]["success_rate"] == 0.5
    assert summary["by_eval_seed"]["43"]["success_rate"] == 1.0
