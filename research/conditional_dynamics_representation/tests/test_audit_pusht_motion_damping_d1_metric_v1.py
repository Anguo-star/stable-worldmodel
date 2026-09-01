from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / (
    "scripts/audit_pusht_motion_damping_d1_metric_v1.py"
)
SPEC = importlib.util.spec_from_file_location("d1_metric", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d1
SPEC.loader.exec_module(d1)


def test_ratio_of_means_is_not_mean_local_score() -> None:
    conditional = np.asarray([1.0, 9.0])
    background = np.asarray([1.0, 1.0])
    weights = np.asarray([0.9, 0.1])
    result = d1.aggregate_relative_energy(conditional, background, weights)
    expected_rho = 1.8 / (1.8 + 1.0)
    expected_mean_score = 0.9 * 0.5 + 0.1 * 0.9
    assert np.isclose(result["rho_phys_ratio_of_means"], expected_rho)
    assert np.isclose(
        result["weighted_mean_local_score_diagnostic_only"], expected_mean_score
    )
    assert not np.isclose(expected_rho, expected_mean_score)


def test_exact_neighbors_exclude_both_twin_directions_and_break_ties() -> None:
    descriptors = np.asarray([[0.0], [0.0], [1.0], [1.0], [-1.0], [-1.0]])
    twins = np.asarray([0, 0, 1, 1, 2, 2])
    pair_indices = np.arange(6)
    neighbors, distances = d1.exact_leave_twin_out_neighbors(
        descriptors,
        twins,
        max_k=2,
        pair_indices=pair_indices,
        chunk_size=2,
    )
    assert neighbors[0].tolist() == [2, 3]
    assert neighbors[1].tolist() == [2, 3]
    assert np.allclose(distances[0], [1.0, 1.0])
    for row in range(6):
        assert not np.any(twins[neighbors[row]] == twins[row])


def test_background_variance_matches_population_definition() -> None:
    means = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    neighbors = np.asarray([[2, 3], [2, 3], [0, 1], [0, 1]])
    result = d1.background_variance_by_k(means, neighbors, [2])[2]
    assert np.allclose(result[:2], [1.0, 1.0])
    assert np.allclose(result[2:], [0.0, 0.0])


def test_robust_descriptor_scaling_zeros_constant_dimensions() -> None:
    values = np.asarray([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaled, median, iqr, active = d1.robust_scale_descriptors(values)
    assert active.tolist() == [True, False]
    assert iqr[1] == 0.0
    assert np.all(scaled[:, 1] == 0.0)
    assert median[1] == 5.0


def test_coverage_cells_have_exact_equal_frequency_quota() -> None:
    orientation = np.repeat(np.arange(2), 16)
    speed = np.concatenate([np.arange(16), np.arange(16)]).astype(float)
    distance = np.tile(np.asarray([3.0, 2.0, 1.0, 0.0]), 8)
    ids = np.arange(32)
    cells, speed_bin, goal_bin = d1.assign_coverage_cells(
        orientation, speed, distance, bins=2, stable_ids=ids
    )
    unique, counts = np.unique(cells, return_counts=True)
    assert unique.tolist() == list(range(8))
    assert counts.tolist() == [4] * 8
    assert set(speed_bin.tolist()) == {0, 1}
    assert set(goal_bin.tolist()) == {0, 1}


def test_selection_uses_stable_id_for_equal_scores() -> None:
    scores = np.ones(8)
    cells = np.repeat([0, 1], 4)
    stable_ids = np.asarray([3, 1, 2, 0, 7, 5, 6, 4])
    selected = d1.select_top_per_cell(
        scores, cells, per_cell=2, stable_ids=stable_ids
    )
    assert selected.tolist() == [3, 1, 7, 5]


def test_projected_weights_are_exact_half_natural_half_high() -> None:
    weights = d1.projected_weights(8, np.asarray([0, 2]))
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights[[1, 3, 4, 5, 6, 7]], 0.5 / 8)
    assert np.allclose(weights[[0, 2]], 0.5 / 8 + 0.5 / 2)


def test_gates_fail_closed_on_unstable_or_non_improving_candidate() -> None:
    passed = d1.evaluate_gates(
        spearman_32=0.95,
        spearman_128=0.96,
        jaccard_32=0.85,
        jaccard_128=0.86,
        rho_d0=0.1,
        rho_r50=0.2,
        coverage_ok=True,
        invariants_ok=True,
    )
    assert passed["passed"] is True
    failed = d1.evaluate_gates(
        spearman_32=0.89,
        spearman_128=0.96,
        jaccard_32=0.85,
        jaccard_128=0.79,
        rho_d0=0.2,
        rho_r50=0.2,
        coverage_ok=True,
        invariants_ok=True,
    )
    assert failed["passed"] is False
    assert failed["checks"]["spearman_k32_vs_k64_at_least_0p90"] is False
    assert failed["checks"]["r50_rho_phys_strictly_above_d0"] is False
