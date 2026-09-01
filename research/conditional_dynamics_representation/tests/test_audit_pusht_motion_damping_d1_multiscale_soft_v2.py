from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RESEARCH_ROOT / (
    "scripts/audit_pusht_motion_damping_d1_multiscale_soft_v2.py"
)
SPEC = importlib.util.spec_from_file_location("d1_multiscale_soft_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d1v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d1v2
SPEC.loader.exec_module(d1v2)


KS = (32, 64, 128)


def _catalog_row(
    *,
    twin_id: int,
    cell: int,
    orientation: int,
    speed_bin: int,
    goal_bin: int,
    conditional: float,
    background: dict[int, float],
) -> dict[str, Any]:
    score = {
        k: conditional / (conditional + background[k] + d1v2.TAU) for k in background
    }
    return {
        "twin_id": twin_id,
        "pair_ids": [
            f"pmd-train-{2 * twin_id:06d}-forward",
            f"pmd-train-{2 * twin_id + 1:06d}-reverse",
        ],
        "coverage_cell": cell,
        "orientation_bin": orientation,
        "speed_bin": speed_bin,
        "goal_distance_bin": goal_bin,
        "query_speed": 10.0 + twin_id,
        "goal_distance": 100.0 + twin_id,
        "gap_rms_px": 2.5 + 0.1 * twin_id,
        "future_gap_min_px": 2.5 + 0.1 * twin_id,
        "conditional_energy_physical": conditional,
        "background_future_variation": {str(k): background[k] for k in background},
        "relative_conditional_score": {str(k): score[k] for k in background},
    }


def _write_synthetic_catalog(path: Path, *, corrupt: str | None = None) -> Path:
    rows = []
    for twin_id in range(8):
        cell = twin_id // 4
        conditional = 1.0 + twin_id
        background = {32: 4.0 + twin_id, 64: 8.0 + twin_id, 128: 16.0 + twin_id}
        row = _catalog_row(
            twin_id=twin_id,
            cell=cell,
            orientation=0,
            speed_bin=0,
            goal_bin=cell,
            conditional=conditional,
            background=background,
        )
        if corrupt == "twin_id" and twin_id == 3:
            row["twin_id"] = 99
        if corrupt == "cell" and twin_id == 3:
            row["coverage_cell"] = 1
            row["goal_distance_bin"] = 1
        if corrupt == "score" and twin_id == 3:
            row["relative_conditional_score"]["64"] = 0.5
        if corrupt == "conditional" and twin_id == 3:
            row["conditional_energy_physical"] = 0.0
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _load_synthetic(path: Path) -> Any:
    return d1v2.load_v1_catalog(
        path,
        expected_twin_count=8,
        expected_cells=2,
        expected_twins_per_cell=4,
    )


def _gate_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "rho_d0_by_k": {32: 0.20, 64: 0.10, 128: 0.05},
        "rho_candidate_by_k": {32: 0.21, 64: 0.11, 128: 0.06},
        "weighted_conditional_d0": 2.0,
        "weighted_conditional_candidate": 2.1,
        "total_variation_by_dropped_k": {32: 0.01, 64: 0.02, 128: 0.03},
        "high_arm_cell_mass_exact": True,
        "full_cell_mass_exact": True,
        "all_twins_positive_weight": True,
        "catalog_scores_reproduced": True,
        "coverage_ok": True,
        "frozen_inputs_ok": True,
        "invariants_ok": True,
    }
    kwargs.update(overrides)
    return kwargs


def test_frozen_input_sha256_constants_match_the_v1_training_artifacts() -> None:
    catalog_sha = d1v2.file_sha256(d1v2.DEFAULT_V1_CATALOG)
    summary_sha = d1v2.file_sha256(d1v2.DEFAULT_V1_SUMMARY)
    assert catalog_sha == d1v2.EXPECTED_V1_CATALOG_SHA256
    assert summary_sha == d1v2.EXPECTED_V1_SUMMARY_SHA256
    frozen = json.loads(d1v2.DEFAULT_V1_SUMMARY.read_text(encoding="utf-8"))
    identity = frozen["frozen_identity"]
    assert (
        identity["manifest_train_table_sha256"]
        == d1v2.EXPECTED_TRAIN_LANCE_DIRECTORY_SHA256
    )
    assert (
        identity["verified_train_lance_directory_sha256"]
        == d1v2.EXPECTED_TRAIN_LANCE_DIRECTORY_SHA256
    )
    assert identity["twin_count"] == 4096
    assert identity["train_pair_count"] == 8192


def test_only_the_frozen_training_table_is_accepted(tmp_path: Path) -> None:
    train = tmp_path / "train.lance"
    train.mkdir()
    assert d1v2.assert_training_only_paths(train) == train.resolve()
    for forbidden in ("validation.lance", "loader_validation.lance", "test.lance"):
        path = tmp_path / forbidden
        path.mkdir()
        with pytest.raises(RuntimeError):
            d1v2.assert_training_only_paths(path)


def test_catalog_loader_accepts_the_frozen_shape_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    catalog = _load_synthetic(_write_synthetic_catalog(tmp_path / "good.jsonl"))
    assert catalog.twin_ids.tolist() == list(range(8))
    assert catalog.coverage_cells.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert set(catalog.background) == set(KS)
    assert np.all(catalog.conditional > 0.0)

    for corrupt in ("twin_id", "cell", "conditional"):
        path = _write_synthetic_catalog(tmp_path / f"{corrupt}.jsonl", corrupt=corrupt)
        with pytest.raises(RuntimeError):
            _load_synthetic(path)


def test_recomputed_scores_catch_a_transcribed_catalog_score(tmp_path: Path) -> None:
    catalog = _load_synthetic(_write_synthetic_catalog(tmp_path / "good.jsonl"))
    recomputed = d1v2.recompute_relative_scores(
        catalog.conditional, catalog.background
    )
    clean = d1v2.verify_catalog_scores(recomputed, catalog.catalog_score)
    assert clean["consistent"] is True
    assert clean["max_abs_deviation"] == 0.0

    bad = _load_synthetic(
        _write_synthetic_catalog(tmp_path / "score.jsonl", corrupt="score")
    )
    dirty = d1v2.verify_catalog_scores(
        d1v2.recompute_relative_scores(bad.conditional, bad.background),
        bad.catalog_score,
    )
    assert dirty["consistent"] is False
    assert dirty["max_abs_deviation"] > 0.0


def test_stable_ranks_run_low_to_high_and_break_score_ties_by_twin_id() -> None:
    cells = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    twin_ids = np.arange(8)
    scores = np.asarray([0.4, 0.1, 0.3, 0.2, 9.0, 7.0, 8.0, 6.0])
    ranks = d1v2.stable_cell_ranks(scores, cells, stable_ids=twin_ids)
    assert ranks.tolist() == [4, 1, 3, 2, 4, 2, 3, 1]

    tied = np.asarray([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0])
    tied_ranks = d1v2.stable_cell_ranks(tied, cells, stable_ids=twin_ids)
    assert tied_ranks.tolist() == [1, 2, 3, 4, 1, 2, 3, 4]

    # A lower twin_id must keep the lower rank even when the array order is shuffled.
    shuffled_ids = np.asarray([3, 2, 1, 0, 7, 6, 5, 4])
    shuffled = d1v2.stable_cell_ranks(tied, cells, stable_ids=shuffled_ids)
    assert shuffled.tolist() == [4, 3, 2, 1, 2, 1, 4, 3]


def test_multiscale_rank_is_the_arithmetic_mean_of_the_three_scales() -> None:
    ranks_by_k = {
        32: np.asarray([1, 2, 3, 4]),
        64: np.asarray([2, 1, 4, 3]),
        128: np.asarray([3, 3, 2, 2]),
    }
    r_ms = d1v2.multiscale_rank(ranks_by_k, KS)
    assert np.allclose(r_ms, [2.0, 2.0, 3.0, 3.0])
    kept = d1v2.multiscale_rank(ranks_by_k, [32, 128])
    assert np.allclose(kept, [2.0, 2.5, 2.5, 3.0])


def test_high_arm_carries_exactly_one_over_n_cells_per_cell() -> None:
    cells = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    r_ms = np.asarray([1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0])
    high = d1v2.high_arm_distribution(r_ms, cells)
    assert np.isclose(high.sum(), 1.0, atol=1e-15)
    for cell in (0, 1):
        assert abs(float(high[cells == cell].sum()) - 0.5) <= d1v2.MASS_TOLERANCE
    # In-cell mass is proportional to r_ms; the bottom-ranked twin keeps mass.
    assert np.allclose(high[:4], 0.5 * np.asarray([1, 2, 3, 4]) / 10.0)
    assert np.allclose(high[4:], 0.125)
    assert np.all(high > 0.0)

    report = d1v2.cell_mass_report(high, cells)
    assert report["cell_count"] == 2
    assert report["exactly_uniform"] is True
    assert report["target_mass_per_cell"] == 0.5

    with pytest.raises(RuntimeError):
        d1v2.high_arm_distribution(np.asarray([0.0, 1.0, 1.0, 1.0]), np.zeros(4, int))


def test_full_weights_are_half_uniform_plus_half_high_arm() -> None:
    cells = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    r_ms = np.asarray([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    high = d1v2.high_arm_distribution(r_ms, cells)
    full = d1v2.full_projected_weights(high)
    uniform = d1v2.uniform_weights(8)
    assert np.allclose(full, 0.5 * uniform + 0.5 * high)
    assert np.isclose(full.sum(), 1.0, atol=1e-15)
    assert np.all(full > 0.0)
    # Equal-sized cells keep the full projected mass at 1/n_cells as well.
    full_report = d1v2.cell_mass_report(full, cells)
    assert full_report["exactly_uniform"] is True
    for cell in (0, 1):
        assert abs(float(full[cells == cell].sum()) - 0.5) <= d1v2.MASS_TOLERANCE
    support = d1v2.positive_weight_support(np.arange(8.0), full)
    assert support["covers_full_pool"] is True
    assert support["positive_weight_twins"] == 8


def test_leave_one_scale_out_total_variation_is_zero_when_scales_agree() -> None:
    cells = np.asarray([0, 0, 0, 0])
    ranks_by_k = {k: np.asarray([1, 2, 3, 4]) for k in KS}
    main = d1v2.high_arm_distribution(d1v2.multiscale_rank(ranks_by_k, KS), cells)
    dropped = d1v2.leave_one_scale_out_high_arms(ranks_by_k, cells)
    assert set(dropped) == set(KS)
    for k in KS:
        assert d1v2.total_variation(main, dropped[k]) == pytest.approx(0.0, abs=1e-15)


def test_leave_one_scale_out_total_variation_grows_with_a_disagreeing_scale() -> None:
    cells = np.zeros(4, dtype=np.int64)
    ranks_by_k = {
        32: np.asarray([1, 2, 3, 4]),
        64: np.asarray([1, 2, 3, 4]),
        128: np.asarray([4, 3, 2, 1]),
    }
    main = d1v2.high_arm_distribution(d1v2.multiscale_rank(ranks_by_k, KS), cells)
    dropped = d1v2.leave_one_scale_out_high_arms(ranks_by_k, cells)
    # Dropping the agreeing scales leaves the blend nearly unchanged; dropping the
    # disagreeing one moves the most mass.
    tv = {k: d1v2.total_variation(main, dropped[k]) for k in KS}
    assert tv[128] > tv[32] > 0.0
    assert tv[32] == pytest.approx(tv[64])


def test_total_variation_rejects_unnormalized_input() -> None:
    with pytest.raises(RuntimeError):
        d1v2.total_variation(np.asarray([0.5, 0.4]), np.asarray([0.5, 0.5]))


def test_gates_pass_only_when_every_section_3_5_check_holds() -> None:
    result = d1v2.evaluate_gates(**_gate_kwargs())
    assert result["passed"] is True
    assert set(result["checks"]) == {
        "rho_phys_k32_strictly_above_d0",
        "rho_phys_k64_strictly_above_d0",
        "rho_phys_k128_strictly_above_d0",
        "weighted_conditional_energy_strictly_above_d0",
        "leave_out_k32_high_arm_total_variation_at_most_0p10",
        "leave_out_k64_high_arm_total_variation_at_most_0p10",
        "leave_out_k128_high_arm_total_variation_at_most_0p10",
        "high_arm_cell_mass_exactly_uniform",
        "full_cell_mass_exactly_uniform",
        "all_twins_have_positive_natural_weight",
        "v1_catalog_scores_reproduced",
        "coverage_cells_exact",
        "frozen_input_sha256_match",
        "training_only_invariants",
    }


@pytest.mark.parametrize("scale", [32, 64, 128])
def test_no_go_when_any_single_scale_rho_fails(scale: int) -> None:
    rho = {32: 0.21, 64: 0.11, 128: 0.06}
    rho[scale] = {32: 0.20, 64: 0.10, 128: 0.05}[scale]
    result = d1v2.evaluate_gates(**_gate_kwargs(rho_candidate_by_k=rho))
    assert result["passed"] is False
    assert result["checks"][f"rho_phys_k{scale}_strictly_above_d0"] is False
    for other in KS:
        if other != scale:
            assert result["checks"][f"rho_phys_k{other}_strictly_above_d0"] is True


def test_no_go_when_weighted_conditional_energy_does_not_strictly_rise() -> None:
    result = d1v2.evaluate_gates(
        **_gate_kwargs(weighted_conditional_candidate=2.0)
    )
    assert result["passed"] is False
    assert result["checks"]["weighted_conditional_energy_strictly_above_d0"] is False
    # Every relative share still rose: a ratio-only gain cannot rescue the run.
    for k in KS:
        assert result["checks"][f"rho_phys_k{k}_strictly_above_d0"] is True


def test_no_go_when_a_leave_one_scale_out_total_variation_exceeds_the_gate() -> None:
    result = d1v2.evaluate_gates(
        **_gate_kwargs(total_variation_by_dropped_k={32: 0.01, 64: 0.1001, 128: 0.03})
    )
    assert result["passed"] is False
    assert (
        result["checks"]["leave_out_k64_high_arm_total_variation_at_most_0p10"]
        is False
    )
    boundary = d1v2.evaluate_gates(
        **_gate_kwargs(total_variation_by_dropped_k={32: 0.10, 64: 0.10, 128: 0.10})
    )
    assert boundary["passed"] is True


def test_total_variation_gate_reads_the_high_arm_not_the_shrunk_full_mixture() -> None:
    high_main = np.asarray([0.35, 0.35, 0.15, 0.15])
    high_dropped = np.asarray([0.25, 0.25, 0.25, 0.25])
    high_tv = d1v2.total_variation(high_main, high_dropped)
    full_tv = d1v2.total_variation(
        d1v2.full_projected_weights(high_main),
        d1v2.full_projected_weights(high_dropped),
    )
    assert high_tv == pytest.approx(0.20)
    assert full_tv == pytest.approx(0.10)
    # The natural half halves the apparent movement, so gating the full mixture
    # would pass a high arm that is twice as unstable as the gate allows.
    assert (
        d1v2.evaluate_gates(
            **_gate_kwargs(
                total_variation_by_dropped_k={32: high_tv, 64: 0.01, 128: 0.01}
            )
        )["passed"]
        is False
    )
    assert (
        d1v2.evaluate_gates(
            **_gate_kwargs(
                total_variation_by_dropped_k={32: full_tv, 64: 0.01, 128: 0.01}
            )
        )["passed"]
        is True
    )


@pytest.mark.parametrize(
    "flag",
    [
        "high_arm_cell_mass_exact",
        "full_cell_mass_exact",
        "all_twins_positive_weight",
        "catalog_scores_reproduced",
        "coverage_ok",
        "frozen_inputs_ok",
        "invariants_ok",
    ],
)
def test_no_go_when_any_structural_invariant_is_false(flag: str) -> None:
    result = d1v2.evaluate_gates(**_gate_kwargs(**{flag: False}))
    assert result["passed"] is False


def test_end_to_end_soft_exposure_raises_all_three_shares_on_a_synthetic_pool(
    tmp_path: Path,
) -> None:
    catalog = _load_synthetic(_write_synthetic_catalog(tmp_path / "good.jsonl"))
    score = d1v2.recompute_relative_scores(catalog.conditional, catalog.background)
    ranks_by_k = {
        k: d1v2.stable_cell_ranks(
            score[k], catalog.coverage_cells, stable_ids=catalog.twin_ids
        )
        for k in KS
    }
    high = d1v2.high_arm_distribution(
        d1v2.multiscale_rank(ranks_by_k, KS), catalog.coverage_cells
    )
    full = d1v2.full_projected_weights(high)
    d0 = d1v2.uniform_weights(len(catalog.twin_ids))
    rho_d0 = {}
    rho_full = {}
    for k in KS:
        rho_d0[k] = d1v2.aggregate_relative_energy(
            catalog.conditional, catalog.background[k], d0
        )["rho_phys_ratio_of_means"]
        rho_full[k] = d1v2.aggregate_relative_energy(
            catalog.conditional, catalog.background[k], full
        )["rho_phys_ratio_of_means"]
    assert all(rho_full[k] > rho_d0[k] for k in KS)
    conditional_d0 = d1v2.aggregate_relative_energy(
        catalog.conditional, catalog.background[64], d0
    )["weighted_conditional_energy"]
    conditional_full = d1v2.aggregate_relative_energy(
        catalog.conditional, catalog.background[64], full
    )["weighted_conditional_energy"]
    assert conditional_full > conditional_d0
    assert d1v2.cell_mass_report(high, catalog.coverage_cells)["exactly_uniform"]
    assert np.all(full > 0.0)


def test_the_v2_audit_declares_a_zero_optimizer_training_only_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        '"optimizer_steps": 0',
        '"schedule_generated": False',
        '"development_lance_opened": False',
        '"public_test_lance_opened": False',
        '"model_loaded": False',
    ):
        assert token in source
    assert "import torch" not in source
    assert "optimizer.step" not in source
    # Development/Test tables appear only inside the fail-closed path guard.
    for table in ("validation.lance", "loader_validation.lance", "test.lance"):
        assert source.count(f'"{table}"') == 1
    assert "forbidden" in source
