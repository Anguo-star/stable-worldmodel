from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RESEARCH_ROOT / (
    "scripts/build_pusht_motion_damping_root_cause_comparators_v1.py"
)
SPEC = importlib.util.spec_from_file_location("motion_root_cause_comparators", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _synthetic_pool() -> object:
    twin_ids = np.arange(8, dtype=np.int64)
    cells = np.repeat(np.arange(2, dtype=np.int64), 4)
    orientation = np.zeros(8, dtype=np.int64)
    speed_bin = np.zeros(8, dtype=np.int64)
    goal_bin = cells.copy()
    conditional = np.asarray([4.0, 1.0, 3.0, 2.0, 8.0, 5.0, 7.0, 6.0])
    high = np.asarray([0.05, 0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.20])
    ranks = {
        k: np.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int64)
        for k in module.KS
    }
    r_multiscale = np.asarray([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    pair_ids = tuple(
        (
            f"pmd-train-{2 * twin:06d}-forward",
            f"pmd-train-{2 * twin + 1:06d}-reverse",
        )
        for twin in twin_ids
    )
    background = {
        k: np.asarray([2.0 + float(index) for index in range(8)])
        for k in module.KS
    }
    return module.ProjectedPool(
        twin_ids=twin_ids,
        pair_ids=pair_ids,
        cells=cells,
        orientation=orientation,
        speed_bin=speed_bin,
        goal_bin=goal_bin,
        query_speed=np.ones(8),
        goal_distance=np.ones(8),
        gap_rms_px=np.ones(8),
        conditional=conditional,
        background=background,
        ranks=ranks,
        r_multiscale=r_multiscale,
        pi_high=high,
        pi_full=module.full_projected_weights(high),
    )


def test_hash_rank_is_frozen_deterministic_and_seed_search_is_rejected() -> None:
    ids = np.arange(8, dtype=np.int64)
    cells = np.repeat(np.arange(2, dtype=np.int64), 4)
    first = module.hash_cell_ranks(ids, cells)
    second = module.hash_cell_ranks(ids, cells)
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    with pytest.raises(RuntimeError, match="frozen"):
        module.hash_cell_ranks(ids, cells, seed=123)


def test_reassign_preserves_values_and_follows_target_rank() -> None:
    source = np.asarray([0.10, 0.20, 0.30, 0.40])
    cells = np.zeros(4, dtype=np.int64)
    ids = np.arange(4, dtype=np.int64)
    target_rank = np.asarray([4, 1, 3, 2], dtype=np.int64)
    assigned = module.reassign_weight_multiset(source, target_rank, cells, ids)
    assert assigned.tolist() == [0.40, 0.10, 0.30, 0.20]
    assert np.array_equal(np.sort(assigned), np.sort(source))


def test_synthetic_comparators_keep_the_exact_high_and_full_multisets() -> None:
    pool = _synthetic_pool()
    comparators = module.build_comparator_weights(pool)
    assert np.array_equal(
        np.sort(comparators.high["ABS50"]), np.sort(comparators.high["REL50"])
    )
    assert np.array_equal(
        np.sort(comparators.high["HASH50"]), np.sort(comparators.high["REL50"])
    )
    assert np.array_equal(
        np.sort(comparators.full["ABS50"]), np.sort(comparators.full["REL50"])
    )
    assert np.array_equal(
        np.sort(comparators.full["HASH50"]), np.sort(comparators.full["REL50"])
    )
    assert np.array_equal(comparators.high["REL50"], pool.pi_high)
    assert np.array_equal(comparators.full["REL50"], pool.pi_full)
    assert not np.array_equal(comparators.high["ABS50"], comparators.high["REL50"])


def test_training_public_path_guard_is_fail_closed(tmp_path: Path) -> None:
    for part in (
        "validation",
        "public",
        "test",
        "development",
        "loader_validation",
        "public_test",
        "private_development_v1",
    ):
        path = tmp_path / part / "projected_weights.jsonl"
        with pytest.raises(RuntimeError, match="forbidden"):
            module.assert_training_only_path(path, expected_name="projected_weights.jsonl")


def test_frozen_v2_inputs_and_production_comparator_invariants_pass() -> None:
    frozen = module.verify_frozen_inputs()
    pool = module.load_projected_pool(
        Path(frozen["v2_dir"]) / "projected_weights.jsonl"
    )
    catalog = module.v2.load_v1_catalog(frozen["catalog_path"])
    module.verify_catalog_projection_match(pool, catalog)
    comparators = module.build_comparator_weights(pool)
    checks = module.validate_comparator_invariants(pool, comparators)
    assert checks["all_invariants_pass"] is True
    assert checks["abs50_high_weight_multiset"] is True
    assert checks["hash50_full_weight_multiset"] is True


def test_production_report_has_all_four_arms_and_requested_metrics() -> None:
    pool = module.load_projected_pool(
        module.DEFAULT_V2_DIR / "projected_weights.jsonl"
    )
    comparators = module.build_comparator_weights(pool)
    metrics = module.arm_metric_report(pool, comparators.full)
    assert set(metrics) == {"D0", "REL50", "ABS50", "HASH50"}
    for arm in metrics.values():
        assert set(arm["weighted_B_k"]) == {"32", "64", "128"}
        assert set(arm["rho_phys"]) == {"32", "64", "128"}
        assert np.isfinite(arm["weighted_C_phys"])
    correlations = module.rank_correlation_report(pool, comparators)
    assert correlations["global"]["conditional_vs_abs50"] == pytest.approx(1.0)
    assert set(correlations["by_cell"]) == {str(cell) for cell in range(64)}


def test_script_has_no_model_or_optimizer_boundary_escape() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in ("import torch", "optimizer.step", "Development", "Public Test"):
        if token in ("Development", "Public Test"):
            continue
        assert token not in source
    assert '"optimizer_steps": 0' in source
    assert '"public_test_lance_opened": False' in source
    assert "HASH_SEED = 20260901" in source
    assert '"builder_script_sha256": builder_script_sha256' in source
