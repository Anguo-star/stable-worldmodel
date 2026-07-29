from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_conditional_sigreg_screen.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_conditional_sigreg_screen",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _records(successes: dict[int, set[int]]):
    return {
        f"original-s{seed}-e{index:03d}": {
            "eval_seed": seed,
            "success": index in successes.get(seed, set()),
        }
        for seed in MODULE.EVAL_SEEDS
        for index in range(50)
    }


def test_paired_bootstrap_uses_query_level_differences() -> None:
    control = _records({seed: set(range(45)) for seed in MODULE.EVAL_SEEDS})
    candidate = _records({seed: set(range(46)) for seed in MODULE.EVAL_SEEDS})

    result = MODULE.paired_stratified_bootstrap(
        control,
        candidate,
        resamples=2_000,
        seed=7,
    )

    assert result["point_difference"] == 0.02
    assert result["discordant_pairs"] == {
        "candidate_only_success": 6,
        "control_only_success": 0,
    }
    assert result["noninferior"] is True


def test_paired_bootstrap_rejects_unpaired_queries() -> None:
    control = _records({})
    candidate = _records({})
    candidate.pop(next(iter(candidate)))

    try:
        MODULE.paired_stratified_bootstrap(control, candidate, resamples=10)
    except RuntimeError as error:
        assert "not paired" in str(error)
    else:
        raise AssertionError("Expected an unpaired-query failure")
