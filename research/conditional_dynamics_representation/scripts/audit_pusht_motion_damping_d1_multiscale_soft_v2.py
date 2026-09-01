#!/usr/bin/env python3
"""Training-only D1-0 v2 audit: multiscale soft exposure (D1-MS50) over the frozen v1 catalog.

Implements D1_CONSTRUCTION_PLAN_ZH.md section 3.5.  This audit loads no model, runs no
optimizer step, never opens Development/Public Test tables and never emits an integer
schedule.  It only reuses the frozen v1 per-twin catalog, rebuilds the multiscale rank
exposure distribution and reports the section 3.5 gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


import numpy as np


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = RESEARCH_ROOT / "scripts/audit_pusht_motion_damping_d1_metric_v1.py"
V1_ARTIFACT_DIR = RESEARCH_ROOT / (
    "artifacts/pusht_motion_damping_d1_metric_v1/d1_0_training_only_v1_final"
)
DEFAULT_V1_CATALOG = V1_ARTIFACT_DIR / "per_twin_catalog.jsonl"
DEFAULT_V1_SUMMARY = V1_ARTIFACT_DIR / "summary.json"
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / (
    "artifacts/pusht_motion_damping_d1_multiscale_soft_v2/d1_0_training_only_v2"
)


def _load_v1_module() -> Any:
    """Import the frozen v1 audit script for its pure helper functions."""

    name = "pusht_motion_damping_d1_metric_v1"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen v1 audit script: {V1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1_module()

# Pure helpers reused verbatim from the frozen v1 audit; v2 must not redefine them.
file_sha256 = v1.file_sha256
directory_sha256 = v1.directory_sha256
aggregate_relative_energy = v1.aggregate_relative_energy
weighted_shift_metrics = v1.weighted_shift_metrics
weighted_quantiles = v1.weighted_quantiles
distribution_summary = v1._distribution_summary
_require = v1._require
_finite = v1._finite
_json_dump = v1._json_dump

TAU = v1.TAU
KS = v1.KS
MODES = v1.MODES
SEPARATION_PX = v1.SEPARATION_PX
EXPECTED_TWIN_COUNT = v1.EXPECTED_TWIN_COUNT
EXPECTED_PAIR_COUNT = v1.EXPECTED_PAIR_COUNT
EXPECTED_MANIFEST_SHA256 = v1.EXPECTED_MANIFEST_SHA256
EXPECTED_RELEASE_CONFIG_SHA256 = v1.EXPECTED_RELEASE_CONFIG_SHA256

EXPECTED_V1_CATALOG_SHA256 = (
    "c84df85632c4f4d81728393e22ca553773e1a5992cccc79b5b798f288c5dbb99"
)
EXPECTED_V1_SUMMARY_SHA256 = (
    "ab88f532758a8f5cf21307bd381a8b30f0883ac94d93f207a6b279c9d945e63a"
)
EXPECTED_TRAIN_LANCE_DIRECTORY_SHA256 = (
    "085a4d7bb60f5ec31215c3bad452c130ab90bd04b7cd80573211848bb2a13b05"
)

CANDIDATE_ID = "D1-MS50"
COVERAGE_BINS = 4
COVERAGE_CELLS = COVERAGE_BINS**3
TWINS_PER_CELL = 64
NATURAL_FRACTION = 0.5
TOTAL_VARIATION_GATE = 0.10
MASS_TOLERANCE = 1.0e-12
SCORE_RELATIVE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class CatalogArrays:
    """Per-twin quantities read back from the frozen v1 catalog."""

    twin_ids: np.ndarray
    pair_ids: tuple[tuple[str, ...], ...]
    coverage_cells: np.ndarray
    orientation_bin: np.ndarray
    speed_bin: np.ndarray
    goal_distance_bin: np.ndarray
    query_speed: np.ndarray
    goal_distance: np.ndarray
    gap_rms_px: np.ndarray
    future_gap_min_px: np.ndarray
    conditional: np.ndarray
    background: dict[int, np.ndarray]
    catalog_score: dict[int, np.ndarray]


def assert_training_only_paths(train_lance: Path) -> Path:
    """Fail closed if anything but the frozen Training table is addressed."""

    train_lance = train_lance.expanduser().resolve()
    _require(train_lance.name == "train.lance", "D1-0 v2 accepts only train.lance")
    forbidden = {
        (train_lance.parent / name).resolve()
        for name in ("loader_validation.lance", "validation.lance", "test.lance")
    }
    _require(train_lance not in forbidden, "Development/Test table is forbidden")
    _require(train_lance.is_dir(), f"missing frozen Training table: {train_lance}")
    return train_lance


def load_v1_catalog(
    path: Path,
    *,
    ks: Iterable[int] = KS,
    expected_twin_count: int = EXPECTED_TWIN_COUNT,
    expected_cells: int = COVERAGE_CELLS,
    expected_twins_per_cell: int = TWINS_PER_CELL,
    bins: int = COVERAGE_BINS,
) -> CatalogArrays:
    """Read the frozen v1 per-twin catalog and re-verify its structural invariants."""

    path = Path(path).expanduser().resolve()
    _require(path.is_file(), f"missing frozen v1 catalog: {path}")
    ks = tuple(int(k) for k in ks)

    twin_ids: list[int] = []
    pair_ids: list[tuple[str, ...]] = []
    cells: list[int] = []
    orientation: list[int] = []
    speed_bin: list[int] = []
    goal_bin: list[int] = []
    speed: list[float] = []
    goal_distance: list[float] = []
    gap_rms: list[float] = []
    gap_min: list[float] = []
    conditional: list[float] = []
    background: dict[int, list[float]] = {k: [] for k in ks}
    catalog_score: dict[int, list[float]] = {k: [] for k in ks}

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if not line.strip():
                raise RuntimeError(f"blank catalog line at {line_number}")
            row = json.loads(line)
            twin_ids.append(int(row["twin_id"]))
            pair_ids.append(tuple(str(value) for value in row["pair_ids"]))
            cells.append(int(row["coverage_cell"]))
            orientation.append(int(row["orientation_bin"]))
            speed_bin.append(int(row["speed_bin"]))
            goal_bin.append(int(row["goal_distance_bin"]))
            speed.append(float(row["query_speed"]))
            goal_distance.append(float(row["goal_distance"]))
            gap_rms.append(float(row["gap_rms_px"]))
            gap_min.append(float(row["future_gap_min_px"]))
            conditional.append(float(row["conditional_energy_physical"]))
            for k in ks:
                background[k].append(float(row["background_future_variation"][str(k)]))
                catalog_score[k].append(
                    float(row["relative_conditional_score"][str(k)])
                )

    count = len(twin_ids)
    _require(
        count == expected_twin_count,
        f"catalog has {count} twins, expected {expected_twin_count}",
    )
    twin_id_array = np.asarray(twin_ids, dtype=np.int64)
    _require(
        np.array_equal(twin_id_array, np.arange(count, dtype=np.int64)),
        "catalog twin_id is not contiguous and ascending from zero",
    )
    _require(
        all(len(value) == 2 for value in pair_ids),
        "every twin must carry exactly two directed pair ids",
    )

    cell_array = np.asarray(cells, dtype=np.int64)
    orientation_array = np.asarray(orientation, dtype=np.int64)
    speed_bin_array = np.asarray(speed_bin, dtype=np.int64)
    goal_bin_array = np.asarray(goal_bin, dtype=np.int64)
    _require(
        np.array_equal(
            cell_array,
            orientation_array * bins * bins + speed_bin_array * bins + goal_bin_array,
        ),
        "coverage_cell does not match orientation x speed x goal bins",
    )
    unique_cells, cell_counts = np.unique(cell_array, return_counts=True)
    _require(
        len(unique_cells) == expected_cells
        and np.array_equal(unique_cells, np.arange(expected_cells, dtype=np.int64)),
        f"catalog does not cover exactly {expected_cells} contiguous cells",
    )
    _require(
        np.all(cell_counts == expected_twins_per_cell),
        f"every coverage cell must hold exactly {expected_twins_per_cell} twins",
    )

    speed_array = np.asarray(speed, dtype=np.float64)
    goal_array = np.asarray(goal_distance, dtype=np.float64)
    gap_array = np.asarray(gap_rms, dtype=np.float64)
    gap_min_array = np.asarray(gap_min, dtype=np.float64)
    conditional_array = np.asarray(conditional, dtype=np.float64)
    background_arrays = {
        k: np.asarray(background[k], dtype=np.float64) for k in ks
    }
    score_arrays = {k: np.asarray(catalog_score[k], dtype=np.float64) for k in ks}

    for name, value in (
        ("catalog query_speed", speed_array),
        ("catalog goal_distance", goal_array),
        ("catalog gap_rms_px", gap_array),
        ("catalog future_gap_min_px", gap_min_array),
        ("catalog C_phys", conditional_array),
    ):
        _finite(name, value)
    for k in ks:
        _finite(f"catalog B_{k}", background_arrays[k])
        _finite(f"catalog s_{k}", score_arrays[k])
        _require(np.all(background_arrays[k] >= 0.0), f"catalog B_{k} is negative")
        _require(
            np.all((score_arrays[k] > 0.0) & (score_arrays[k] <= 1.0)),
            f"catalog s_{k} is outside (0, 1]",
        )
    _require(np.all(conditional_array > 0.0), "catalog C_phys is not strictly positive")
    _require(np.all(speed_array > 0.0), "catalog query_speed is not positive")
    _require(np.all(goal_array > 0.0), "catalog goal_distance is not positive")
    _require(
        np.all(gap_min_array >= SEPARATION_PX),
        "catalog violates the absolute future separation invariant",
    )

    return CatalogArrays(
        twin_ids=twin_id_array,
        pair_ids=tuple(pair_ids),
        coverage_cells=cell_array,
        orientation_bin=orientation_array,
        speed_bin=speed_bin_array,
        goal_distance_bin=goal_bin_array,
        query_speed=speed_array,
        goal_distance=goal_array,
        gap_rms_px=gap_array,
        future_gap_min_px=gap_min_array,
        conditional=conditional_array,
        background=background_arrays,
        catalog_score=score_arrays,
    )


def recompute_relative_scores(
    conditional: np.ndarray,
    background: dict[int, np.ndarray],
    *,
    tau: float = TAU,
) -> dict[int, np.ndarray]:
    """s_k = C_phys / (C_phys + B_k + tau), recomputed rather than transcribed."""

    conditional = np.asarray(conditional, dtype=np.float64)
    _finite("C_phys", conditional)
    _require(np.all(conditional > 0.0), "C_phys is not strictly positive")
    scores: dict[int, np.ndarray] = {}
    for k, value in background.items():
        value = np.asarray(value, dtype=np.float64)
        _finite(f"B_{k}", value)
        _require(value.shape == conditional.shape, f"B_{k} shape mismatch")
        _require(np.all(value >= 0.0), f"B_{k} is negative")
        score = conditional / (conditional + value + tau)
        _finite(f"s_{k}", score)
        scores[int(k)] = score
    return scores


def verify_catalog_scores(
    recomputed: dict[int, np.ndarray],
    catalog: dict[int, np.ndarray],
    *,
    relative_tolerance: float = SCORE_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Cross-check the recomputed s_k against the frozen catalog transcription."""

    _require(set(recomputed) == set(catalog), "score scale sets differ")
    deviations: dict[str, float] = {}
    consistent = True
    for k in sorted(recomputed):
        left = np.asarray(recomputed[k], dtype=np.float64)
        right = np.asarray(catalog[k], dtype=np.float64)
        _require(left.shape == right.shape, f"s_{k} shape mismatch")
        deviation = float(np.max(np.abs(left - right))) if left.size else 0.0
        deviations[str(k)] = deviation
        consistent = consistent and bool(
            np.allclose(left, right, rtol=relative_tolerance, atol=0.0)
        )
    return {
        "max_abs_deviation_by_k": deviations,
        "max_abs_deviation": max(deviations.values()) if deviations else 0.0,
        "relative_tolerance": float(relative_tolerance),
        "consistent": bool(consistent),
    }


def stable_cell_ranks(
    scores: np.ndarray,
    cells: np.ndarray,
    *,
    stable_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Rank each coverage cell from lowest (1) to highest score, ties by ascending id."""

    scores = np.asarray(scores, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    count = len(scores)
    _require(cells.shape == scores.shape, "rank shape mismatch")
    _finite("rank scores", scores)
    if stable_ids is None:
        stable_ids = np.arange(count, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    _require(stable_ids.shape == scores.shape, "stable_ids shape mismatch")
    _require(
        len(np.unique(stable_ids)) == count, "stable_ids must be unique for tie breaks"
    )

    ranks = np.zeros(count, dtype=np.int64)
    for cell in sorted(np.unique(cells).tolist()):
        indices = np.flatnonzero(cells == cell)
        order = indices[np.lexsort((stable_ids[indices], scores[indices]))]
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    _require(np.all(ranks >= 1), "rank assignment is incomplete")
    return ranks


def multiscale_rank(
    ranks_by_k: dict[int, np.ndarray], ks: Iterable[int] | None = None
) -> np.ndarray:
    """r_ms(u) = arithmetic mean of the per-scale stable ranks."""

    keys = sorted(ranks_by_k) if ks is None else [int(k) for k in ks]
    _require(len(keys) > 0, "multiscale rank needs at least one scale")
    stacked = np.stack(
        [np.asarray(ranks_by_k[k], dtype=np.float64) for k in keys], axis=0
    )
    _finite("stable ranks", stacked)
    _require(np.all(stacked >= 1.0), "stable ranks must start at one")
    return stacked.mean(axis=0)


def high_arm_distribution(rank_scores: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Soft high-arm distribution: exactly 1/n_cells per cell, in-cell mass ~ r_ms."""

    rank_scores = np.asarray(rank_scores, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    _require(rank_scores.shape == cells.shape, "high-arm shape mismatch")
    _finite("high-arm rank scores", rank_scores)
    _require(np.all(rank_scores > 0.0), "high-arm rank scores must be positive")

    unique_cells = np.unique(cells)
    cell_mass = 1.0 / len(unique_cells)
    weights = np.zeros(len(rank_scores), dtype=np.float64)
    for cell in unique_cells.tolist():
        indices = np.flatnonzero(cells == cell)
        total = float(rank_scores[indices].sum())
        _require(total > 0.0, f"cell {cell} has zero rank mass")
        weights[indices] = cell_mass * rank_scores[indices] / total
    _finite("high-arm weights", weights)
    _require(np.all(weights > 0.0), "high-arm weights must be strictly positive")
    _require(
        math.isclose(float(weights.sum()), 1.0, abs_tol=MASS_TOLERANCE),
        "high-arm weights do not sum to one",
    )
    return weights


def uniform_weights(count: int) -> np.ndarray:
    _require(count > 0, "weight count must be positive")
    return np.full(count, 1.0 / count, dtype=np.float64)


def full_projected_weights(
    high_weights: np.ndarray, *, natural_fraction: float = NATURAL_FRACTION
) -> np.ndarray:
    """pi_MS50 = natural_fraction * Uniform_all + (1 - natural_fraction) * pi_high."""

    high_weights = np.asarray(high_weights, dtype=np.float64)
    _require(
        0.0 < natural_fraction < 1.0, "natural fraction must lie strictly in (0, 1)"
    )
    _require(
        math.isclose(float(high_weights.sum()), 1.0, abs_tol=MASS_TOLERANCE),
        "high-arm weights do not sum to one",
    )
    count = len(high_weights)
    weights = (
        natural_fraction * uniform_weights(count)
        + (1.0 - natural_fraction) * high_weights
    )
    _finite("projected weights", weights)
    _require(np.all(weights > 0.0), "every twin must keep a positive natural weight")
    _require(
        math.isclose(float(weights.sum()), 1.0, abs_tol=MASS_TOLERANCE),
        "projected weights do not sum to one",
    )
    return weights


def cell_mass_report(weights: np.ndarray, cells: np.ndarray) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    unique_cells = np.unique(cells)
    masses = np.asarray(
        [float(weights[cells == cell].sum()) for cell in unique_cells.tolist()],
        dtype=np.float64,
    )
    target = 1.0 / len(unique_cells)
    deviation = float(np.max(np.abs(masses - target)))
    return {
        "cell_count": int(len(unique_cells)),
        "target_mass_per_cell": float(target),
        "cell_mass_min": float(masses.min()),
        "cell_mass_max": float(masses.max()),
        "max_abs_deviation_from_target": deviation,
        "exactly_uniform": bool(deviation <= MASS_TOLERANCE),
    }


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Total variation distance between two distributions over the same support."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    _require(left.shape == right.shape, "total variation shape mismatch")
    _finite("total variation left", left)
    _finite("total variation right", right)
    _require(np.all(left >= 0.0) and np.all(right >= 0.0), "distributions are negative")
    _require(
        math.isclose(float(left.sum()), 1.0, abs_tol=1.0e-9)
        and math.isclose(float(right.sum()), 1.0, abs_tol=1.0e-9),
        "total variation inputs are not normalized",
    )
    return 0.5 * float(np.sum(np.abs(left - right)))


def leave_one_scale_out_high_arms(
    ranks_by_k: dict[int, np.ndarray], cells: np.ndarray
) -> dict[int, np.ndarray]:
    """Rebuild pi_high from the remaining two ranks after dropping one scale."""

    keys = sorted(ranks_by_k)
    _require(len(keys) >= 3, "leave-one-scale-out needs at least three scales")
    result: dict[int, np.ndarray] = {}
    for dropped in keys:
        kept = [k for k in keys if k != dropped]
        result[int(dropped)] = high_arm_distribution(
            multiscale_rank(ranks_by_k, kept), cells
        )
    return result


def positive_weight_support(
    values: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    _require(values.shape == weights.shape, "support shape mismatch")
    mask = weights > 0.0
    _require(bool(mask.any()), "no twin carries positive weight")
    return {
        "positive_weight_twins": int(mask.sum()),
        "covers_full_pool": bool(mask.all()),
        "support_min": float(np.min(values[mask])),
        "support_max": float(np.max(values[mask])),
    }


def evaluate_gates(
    *,
    rho_d0_by_k: dict[int, float],
    rho_candidate_by_k: dict[int, float],
    weighted_conditional_d0: float,
    weighted_conditional_candidate: float,
    total_variation_by_dropped_k: dict[int, float],
    high_arm_cell_mass_exact: bool,
    full_cell_mass_exact: bool,
    all_twins_positive_weight: bool,
    catalog_scores_reproduced: bool,
    coverage_ok: bool,
    frozen_inputs_ok: bool,
    invariants_ok: bool,
) -> dict[str, Any]:
    """Section 3.5 gates.  Any failure is a no-go; no result-dependent thresholds."""

    _require(
        set(rho_d0_by_k) == set(rho_candidate_by_k), "rho scale sets differ from D0"
    )
    checks: dict[str, bool] = {}
    for k in sorted(rho_candidate_by_k):
        checks[f"rho_phys_k{k}_strictly_above_d0"] = bool(
            rho_candidate_by_k[k] > rho_d0_by_k[k]
        )
    checks["weighted_conditional_energy_strictly_above_d0"] = bool(
        weighted_conditional_candidate > weighted_conditional_d0
    )
    for k in sorted(total_variation_by_dropped_k):
        checks[f"leave_out_k{k}_high_arm_total_variation_at_most_0p10"] = bool(
            total_variation_by_dropped_k[k] <= TOTAL_VARIATION_GATE
        )
    checks["high_arm_cell_mass_exactly_uniform"] = bool(high_arm_cell_mass_exact)
    checks["full_cell_mass_exactly_uniform"] = bool(full_cell_mass_exact)
    checks["all_twins_have_positive_natural_weight"] = bool(all_twins_positive_weight)
    checks["v1_catalog_scores_reproduced"] = bool(catalog_scores_reproduced)
    checks["coverage_cells_exact"] = bool(coverage_ok)
    checks["frozen_input_sha256_match"] = bool(frozen_inputs_ok)
    checks["training_only_invariants"] = bool(invariants_ok)
    return {"checks": checks, "passed": all(checks.values())}


def candidate_summary(
    *,
    name: str,
    weights: np.ndarray,
    d0_weights: np.ndarray,
    catalog: CatalogArrays,
    conditional: np.ndarray,
    background: dict[int, np.ndarray],
    score: dict[int, np.ndarray],
    high_weights: np.ndarray | None,
) -> dict[str, Any]:
    aggregate_by_k: dict[str, Any] = {}
    conditional_values: list[float] = []
    for k in sorted(background):
        aggregate = aggregate_relative_energy(conditional, background[k], weights)
        aggregate_by_k[str(k)] = aggregate
        conditional_values.append(aggregate["weighted_conditional_energy"])
    _require(
        all(
            math.isclose(value, conditional_values[0], rel_tol=1.0e-12, abs_tol=0.0)
            for value in conditional_values
        ),
        "weighted C_phys must not depend on the background scale",
    )

    covariates: dict[str, Any] = {}
    for label, values in (
        ("query_speed", catalog.query_speed),
        ("goal_distance", catalog.goal_distance),
        ("response_gap_px", catalog.gap_rms_px),
    ):
        covariates[label] = {
            **distribution_summary(values, weights),
            "positive_weight_support": positive_weight_support(values, weights),
            "shift_from_d0": weighted_shift_metrics(
                values, d0_weights, values, weights
            ),
        }

    summary: dict[str, Any] = {
        "name": name,
        "weighted_conditional_energy": conditional_values[0],
        "weighted_background_variation_by_k": {
            key: value["weighted_background_variation"]
            for key, value in aggregate_by_k.items()
        },
        "rho_phys_ratio_of_means_by_k": {
            key: value["rho_phys_ratio_of_means"]
            for key, value in aggregate_by_k.items()
        },
        "aggregate_by_k": aggregate_by_k,
        "local_score_by_k": {
            str(k): distribution_summary(score[k], weights) for k in sorted(score)
        },
        "response_gap_rms_px": float(
            math.sqrt(
                float(
                    np.dot(
                        weights / weights.sum(), np.square(catalog.gap_rms_px)
                    )
                )
            )
        ),
        "covariates": covariates,
        "coverage": {
            "full_arm": cell_mass_report(weights, catalog.coverage_cells),
            "all_twins_have_positive_weight": bool(np.all(weights > 0.0)),
            "twins_with_zero_weight": int(np.sum(weights <= 0.0)),
        },
        "projected_training": {
            "optimizer_steps": 0,
            "schedule_generated": False,
            "mode_balance": {MODES[0]: 0.5, MODES[1]: 0.5},
            "direction_balance": {"forward": 0.5, "reverse": 0.5},
        },
    }
    if high_weights is not None:
        summary["coverage"]["high_arm"] = cell_mass_report(
            high_weights, catalog.coverage_cells
        )
        summary["coverage"]["high_arm_min_weight"] = float(np.min(high_weights))
        summary["coverage"]["high_arm_max_weight"] = float(np.max(high_weights))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-catalog", type=Path, default=DEFAULT_V1_CATALOG)
    parser.add_argument("--v1-summary", type=Path, default=DEFAULT_V1_SUMMARY)
    parser.add_argument("--train-lance", type=Path, default=v1.DEFAULT_TRAIN_LANCE)
    parser.add_argument("--manifest", type=Path, default=v1.DEFAULT_MANIFEST)
    parser.add_argument(
        "--release-config", type=Path, default=v1.DEFAULT_RELEASE_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    v1_catalog = args.v1_catalog.expanduser().resolve()
    v1_summary = args.v1_summary.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    release_config = args.release_config.expanduser().resolve()
    train_lance = assert_training_only_paths(args.train_lance)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    v1_catalog_sha256 = file_sha256(v1_catalog)
    v1_summary_sha256 = file_sha256(v1_summary)
    manifest_sha256 = file_sha256(manifest_path)
    release_config_sha256 = file_sha256(release_config)
    # Recomputed, not transcribed from the manifest declaration.
    train_lance_sha256 = directory_sha256(train_lance)

    observed_sha = {
        "v1_per_twin_catalog": v1_catalog_sha256,
        "v1_summary": v1_summary_sha256,
        "manifest": manifest_sha256,
        "release_config": release_config_sha256,
        "train_lance_directory": train_lance_sha256,
    }
    expected_sha = {
        "v1_per_twin_catalog": EXPECTED_V1_CATALOG_SHA256,
        "v1_summary": EXPECTED_V1_SUMMARY_SHA256,
        "manifest": EXPECTED_MANIFEST_SHA256,
        "release_config": EXPECTED_RELEASE_CONFIG_SHA256,
        "train_lance_directory": EXPECTED_TRAIN_LANCE_DIRECTORY_SHA256,
    }
    for key, expected in expected_sha.items():
        _require(observed_sha[key] == expected, f"frozen {key} SHA256 changed")
    frozen_inputs_ok = observed_sha == expected_sha

    frozen_summary = json.loads(v1_summary.read_text(encoding="utf-8"))
    frozen_identity = frozen_summary["frozen_identity"]
    _require(
        str(frozen_identity["manifest_train_table_sha256"]) == train_lance_sha256,
        "recomputed train.lance directory SHA256 disagrees with the frozen manifest",
    )
    _require(
        int(frozen_identity["twin_count"]) == EXPECTED_TWIN_COUNT
        and int(frozen_identity["train_pair_count"]) == EXPECTED_PAIR_COUNT,
        "frozen v1 summary reports an unexpected Training size",
    )

    catalog = load_v1_catalog(v1_catalog)
    twin_count = len(catalog.twin_ids)
    recomputed_score = recompute_relative_scores(catalog.conditional, catalog.background)
    consistency = verify_catalog_scores(recomputed_score, catalog.catalog_score)
    _require(
        consistency["consistent"],
        "recomputed s_k disagrees with the frozen v1 catalog",
    )

    ranks_by_k = {
        k: stable_cell_ranks(
            recomputed_score[k], catalog.coverage_cells, stable_ids=catalog.twin_ids
        )
        for k in KS
    }
    for k in KS:
        counts = np.bincount(ranks_by_k[k], minlength=TWINS_PER_CELL + 1)[1:]
        _require(
            np.all(counts == COVERAGE_CELLS),
            f"stable ranks for k={k} are not a per-cell permutation of 1..{TWINS_PER_CELL}",
        )
    r_multiscale = multiscale_rank(ranks_by_k, KS)
    high_weights = high_arm_distribution(r_multiscale, catalog.coverage_cells)
    full_weights = full_projected_weights(high_weights)
    d0_weights = uniform_weights(twin_count)

    leave_out_high = leave_one_scale_out_high_arms(ranks_by_k, catalog.coverage_cells)
    total_variation_by_dropped_k = {
        k: total_variation(high_weights, value) for k, value in leave_out_high.items()
    }

    summaries = {
        "D0": candidate_summary(
            name="D0",
            weights=d0_weights,
            d0_weights=d0_weights,
            catalog=catalog,
            conditional=catalog.conditional,
            background=catalog.background,
            score=recomputed_score,
            high_weights=None,
        ),
        CANDIDATE_ID: candidate_summary(
            name=CANDIDATE_ID,
            weights=full_weights,
            d0_weights=d0_weights,
            catalog=catalog,
            conditional=catalog.conditional,
            background=catalog.background,
            score=recomputed_score,
            high_weights=high_weights,
        ),
    }
    rho_d0_by_k = {
        k: summaries["D0"]["rho_phys_ratio_of_means_by_k"][str(k)] for k in KS
    }
    rho_candidate_by_k = {
        k: summaries[CANDIDATE_ID]["rho_phys_ratio_of_means_by_k"][str(k)] for k in KS
    }
    high_cell_report = cell_mass_report(high_weights, catalog.coverage_cells)
    full_cell_report = cell_mass_report(full_weights, catalog.coverage_cells)
    coverage_ok = bool(
        high_cell_report["cell_count"] == COVERAGE_CELLS
        and full_cell_report["cell_count"] == COVERAGE_CELLS
        and np.all(np.bincount(catalog.coverage_cells) == TWINS_PER_CELL)
    )
    invariants_ok = bool(
        np.all(catalog.conditional > 0.0)
        and np.all(catalog.future_gap_min_px >= SEPARATION_PX)
        and np.all(full_weights > 0.0)
        and np.all(high_weights > 0.0)
        and twin_count == EXPECTED_TWIN_COUNT
    )
    gates = evaluate_gates(
        rho_d0_by_k=rho_d0_by_k,
        rho_candidate_by_k=rho_candidate_by_k,
        weighted_conditional_d0=summaries["D0"]["weighted_conditional_energy"],
        weighted_conditional_candidate=summaries[CANDIDATE_ID][
            "weighted_conditional_energy"
        ],
        total_variation_by_dropped_k=total_variation_by_dropped_k,
        high_arm_cell_mass_exact=high_cell_report["exactly_uniform"],
        full_cell_mass_exact=full_cell_report["exactly_uniform"],
        all_twins_positive_weight=bool(np.all(full_weights > 0.0)),
        catalog_scores_reproduced=consistency["consistent"],
        coverage_ok=coverage_ok,
        frozen_inputs_ok=frozen_inputs_ok,
        invariants_ok=invariants_ok,
    )

    summary = {
        "schema_version": 1,
        "audit_id": "pusht_motion_damping_d1_multiscale_soft_v2",
        "stage": "D1-0_training_only_multiscale_soft_v2",
        "candidate_id": CANDIDATE_ID,
        "status": "passed_go" if gates["passed"] else "failed_no_go",
        "frozen_identity": {
            "release_id": frozen_identity["release_id"],
            "manifest_sha256": manifest_sha256,
            "release_config_sha256": release_config_sha256,
            "manifest_train_table_sha256": str(
                frozen_identity["manifest_train_table_sha256"]
            ),
            "verified_train_lance_directory_sha256": train_lance_sha256,
            "v1_per_twin_catalog_sha256": v1_catalog_sha256,
            "v1_summary_sha256": v1_summary_sha256,
            "train_pair_count": EXPECTED_PAIR_COUNT,
            "twin_count": twin_count,
        },
        "metric": {
            "ks": list(KS),
            "tau_px2": TAU,
            "selection_score": "s_k = C_phys/(C_phys+B_k+tau)",
            "rank_rule": (
                "per coverage cell, ascending s_k gets rank 1..64; score ties "
                "break by ascending twin_id"
            ),
            "multiscale_rank": "r_ms = (r_32 + r_64 + r_128) / 3",
            "high_arm": "pi_high(u|cell) = r_ms(u) / sum_{v in cell} r_ms(v)",
            "projected": (
                "pi_MS50 = 0.5 * Uniform_all_twins + 0.5 * pi_high; "
                "B_k is treated as a bandwidth nuisance, never merged across k"
            ),
            "aggregate": "rho_phys,k = E_pi[C_phys] / (E_pi[C_phys] + E_pi[B_k])",
        },
        "catalog_consistency": consistency,
        "multiscale_rank": {
            "r_ms_min": float(r_multiscale.min()),
            "r_ms_max": float(r_multiscale.max()),
            "r_ms_mean": float(r_multiscale.mean()),
            "mean_abs_rank_deviation_from_r_ms": {
                str(k): float(np.mean(np.abs(ranks_by_k[k] - r_multiscale)))
                for k in KS
            },
            "distinct_r_ms_values": int(len(np.unique(r_multiscale))),
        },
        "candidates": summaries,
        "comparison_vs_d0": {
            "rho_phys_by_k": {
                str(k): {
                    "d0": rho_d0_by_k[k],
                    CANDIDATE_ID: rho_candidate_by_k[k],
                    "absolute_delta": rho_candidate_by_k[k] - rho_d0_by_k[k],
                    "relative_delta": (
                        (rho_candidate_by_k[k] - rho_d0_by_k[k]) / rho_d0_by_k[k]
                    ),
                }
                for k in KS
            },
            "weighted_conditional_energy": {
                "d0": summaries["D0"]["weighted_conditional_energy"],
                CANDIDATE_ID: summaries[CANDIDATE_ID]["weighted_conditional_energy"],
            },
            "weighted_background_variation_by_k": {
                str(k): {
                    "d0": summaries["D0"]["weighted_background_variation_by_k"][str(k)],
                    CANDIDATE_ID: summaries[CANDIDATE_ID][
                        "weighted_background_variation_by_k"
                    ][str(k)],
                }
                for k in KS
            },
        },
        "leave_one_scale_out": {
            f"dropped_k{k}": {
                "kept_scales": [scale for scale in KS if scale != k],
                "high_arm_total_variation_vs_main": total_variation_by_dropped_k[k],
                "high_arm_max_abs_weight_difference": float(
                    np.max(np.abs(high_weights - leave_out_high[k]))
                ),
                "gate_at_most": TOTAL_VARIATION_GATE,
                "gate_applies_to": "high_arm_distribution_not_full_pi_ms50",
            }
            for k in KS
        },
        "gates": gates,
        "evidence_boundary": {
            "development_lance_opened": False,
            "public_test_lance_opened": False,
            "optimizer_steps": 0,
            "model_loaded": False,
            "schedule_generated": False,
            "pixels_decoded": False,
            "claim": (
                "D1-0 v2 only tests whether a multiscale soft Training exposure "
                "distribution is stable across the pre-registered neighbourhood "
                "scales while raising the absolute conditional numerator and all "
                "three relative shares; it is recipe engineering informed by v1 "
                "Training results, not an independent confirmation of stability, "
                "and it says nothing about rho_lat, V_grad or native Development."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": 1,
        "audit_id": summary["audit_id"],
        "candidate_id": CANDIDATE_ID,
        "inputs": {
            "v1_catalog": str(v1_catalog),
            "v1_summary": str(v1_summary),
            "train_lance": str(train_lance),
            "manifest": str(manifest_path),
            "release_config": str(release_config),
        },
        "expected_input_sha256": expected_sha,
        "output_dir": str(output_dir),
        "ks": list(KS),
        "tau": TAU,
        "selection": {
            "coverage_cells": COVERAGE_CELLS,
            "twins_per_cell": TWINS_PER_CELL,
            "rank_low_to_high": [1, TWINS_PER_CELL],
            "natural_weight": NATURAL_FRACTION,
            "high_arm_weight": 1.0 - NATURAL_FRACTION,
            "stable_tie_break": "ascending_twin_id",
            "hard_pool": False,
            "temperature": None,
            "top_fraction": None,
        },
        "gates": {
            "rho_phys_strictly_above_d0_for_every_k": list(KS),
            "weighted_conditional_energy_strictly_above_d0": True,
            "leave_one_scale_out_high_arm_total_variation_at_most": (
                TOTAL_VARIATION_GATE
            ),
            "cell_mass_tolerance": MASS_TOLERANCE,
        },
    }
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    weights_path = output_dir / "projected_weights.jsonl"
    receipt_path = output_dir / "receipt.json"
    _json_dump(config_path, config)
    _json_dump(summary_path, summary)

    natural_weight = NATURAL_FRACTION / twin_count
    with weights_path.open("x", encoding="utf-8") as stream:
        for twin in range(twin_count):
            row = {
                "twin_id": int(catalog.twin_ids[twin]),
                "pair_ids": list(catalog.pair_ids[twin]),
                "coverage_cell": int(catalog.coverage_cells[twin]),
                "orientation_bin": int(catalog.orientation_bin[twin]),
                "speed_bin": int(catalog.speed_bin[twin]),
                "goal_distance_bin": int(catalog.goal_distance_bin[twin]),
                "query_speed": float(catalog.query_speed[twin]),
                "goal_distance": float(catalog.goal_distance[twin]),
                "gap_rms_px": float(catalog.gap_rms_px[twin]),
                "conditional_energy_physical": float(catalog.conditional[twin]),
                "background_future_variation": {
                    str(k): float(catalog.background[k][twin]) for k in KS
                },
                "relative_conditional_score": {
                    str(k): float(recomputed_score[k][twin]) for k in KS
                },
                "stable_rank_by_k": {
                    str(k): int(ranks_by_k[k][twin]) for k in KS
                },
                "r_multiscale": float(r_multiscale[twin]),
                "pi_natural_uniform": float(natural_weight),
                "pi_high": float(high_weights[twin]),
                "pi_ms50": float(full_weights[twin]),
                "pi_high_leave_one_scale_out": {
                    str(k): float(leave_out_high[k][twin]) for k in KS
                },
            }
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    receipt = {
        "schema_version": 1,
        "audit_id": summary["audit_id"],
        "candidate_id": CANDIDATE_ID,
        "status": summary["status"],
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "input_sha256": {
            **observed_sha,
            "v1_audit_script": file_sha256(V1_SCRIPT),
            "audit_script": file_sha256(Path(__file__).resolve()),
        },
        "expected_input_sha256": expected_sha,
        "frozen_input_sha256_match": frozen_inputs_ok,
        "output_sha256": {
            "config.json": file_sha256(config_path),
            "summary.json": file_sha256(summary_path),
            "projected_weights.jsonl": file_sha256(weights_path),
        },
        "gates": gates,
        "development_lance_opened": False,
        "public_test_lance_opened": False,
        "optimizer_steps": 0,
        "model_loaded": False,
        "schedule_generated": False,
        "pixels_decoded": False,
    }
    _json_dump(receipt_path, receipt)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "rho_phys_d0": {str(k): rho_d0_by_k[k] for k in KS},
                f"rho_phys_{CANDIDATE_ID}": {
                    str(k): rho_candidate_by_k[k] for k in KS
                },
                "weighted_conditional_energy": {
                    "d0": summaries["D0"]["weighted_conditional_energy"],
                    CANDIDATE_ID: summaries[CANDIDATE_ID][
                        "weighted_conditional_energy"
                    ],
                },
                "leave_one_scale_out_total_variation": {
                    str(k): total_variation_by_dropped_k[k] for k in KS
                },
            },
            indent=2,
        )
    )
    if not gates["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
