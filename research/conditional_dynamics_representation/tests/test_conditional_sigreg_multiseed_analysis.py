from __future__ import annotations

import importlib
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    MODULE = importlib.import_module("analyze_conditional_sigreg_multiseed")
finally:
    sys.path.pop(0)


def _variant(*, door_passed: bool, successes: int):
    return {
        "door_rule_use": {
            "passed": door_passed,
            "correct_target_selection_rate": 1.0 if door_passed else 0.5,
            "correct_history_win_rate": 1.0 if door_passed else 0.6,
            "worst_seed_direction_rule_accuracy": (
                1.0 if door_passed else 0.0
            ),
        },
        "original_domain_real_environment_cem": {
            "successes": successes,
        },
        "original_domain_rollout": {
            str(horizon): {"mean_native_latent_mse": float(horizon)}
            for horizon in MODULE.ROLLOUT_HORIZONS
        },
    }


def test_summary_requires_every_training_seed_to_pass() -> None:
    variants = {
        str(seed): {
            "paired_native_0p09": _variant(
                door_passed=False,
                successes=280,
            ),
            "conditional_sigreg_0p09": _variant(
                door_passed=True,
                successes=282,
            ),
        }
        for seed in MODULE.TRAINING_SEEDS
    }
    comparisons = {
        str(seed): {
            "conditional_sigreg_0p09_vs_native_0p09": {
                "noninferior": True,
            }
        }
        for seed in MODULE.TRAINING_SEEDS
    }

    result = MODULE.summarize_variants(variants, comparisons)

    assert result["decision"]["optimization_seed_stability_screen_passes"]
    assert result["methods"]["conditional_sigreg_0p09"][
        "cem_descriptive_across_checkpoints"
    ]["successes"] == 846
    assert result["methods"]["conditional_sigreg_0p09"][
        "cem_descriptive_across_checkpoints"
    ]["inference_allowed"] is False


def test_summary_fails_if_one_conditional_checkpoint_is_inferior() -> None:
    variants = {
        str(seed): {
            "paired_native_0p09": _variant(
                door_passed=False,
                successes=280,
            ),
            "conditional_sigreg_0p09": _variant(
                door_passed=True,
                successes=282,
            ),
        }
        for seed in MODULE.TRAINING_SEEDS
    }
    comparisons = {
        str(seed): {
            "conditional_sigreg_0p09_vs_native_0p09": {
                "noninferior": seed != 5120,
            }
        }
        for seed in MODULE.TRAINING_SEEDS
    }

    result = MODULE.summarize_variants(variants, comparisons)

    assert not result["decision"][
        "optimization_seed_stability_screen_passes"
    ]
