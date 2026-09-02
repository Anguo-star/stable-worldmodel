#!/usr/bin/env python3
"""Build deterministic, Training-only exposure comparators for Motion D1-MS50.

The comparator is deliberately a permutation experiment.  Within each frozen
coverage cell it keeps the exact D1-MS50 ``pi_high`` multiset and only changes
which twin receives each value:

* ABS50 orders twins by ``conditional_energy_physical``;
* HASH50 orders twins by ``sha256(f"{HASH_SEED}|{twin_id}")``.

The script never opens a model, optimizer, pixel table, Development split or
Public Test split.  It only reads the frozen v1 catalog and the frozen v2
projected-weights artifact (plus their immutable provenance metadata).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
V2_SCRIPT = RESEARCH_ROOT / "scripts/audit_pusht_motion_damping_d1_multiscale_soft_v2.py"
DEFAULT_V2_DIR = RESEARCH_ROOT / (
    "artifacts/pusht_motion_damping_d1_multiscale_soft_v2/d1_0_training_only_v2"
)
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / (
    "artifacts/pusht_motion_damping_root_cause_comparators_v1/comparators_v1_final"
)

EXPECTED_V2_SHA256 = {
    "config.json": "e1fcfdf41c1756116eb5471b77e7ba45c097e77b1987e678257ca1dfbe41d786",
    "summary.json": "05d745ebddaab4a9a8ec7a1dfa7bd504b27efb217c5205561361dc0d683d614e",
    "projected_weights.jsonl": (
        "6a45c6f18e1eeb61f184c9977b81b08f4de384ee4c9c36cfa41e186dca755afa"
    ),
    "receipt.json": "7d7f37ec89d41f9b5e7c5a5c1f4c134459e00fcd71deb1fabac713dc26b85ac5",
}
EXPECTED_INPUT_SHA256 = {
    "manifest": "48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b",
    "release_config": "1795717d8bfa1d1cfcc01a69931b6241b0e7759dcf43553a6c4cca225ec9326b",
    "train_lance_directory": (
        "085a4d7bb60f5ec31215c3bad452c130ab90bd04b7cd80573211848bb2a13b05"
    ),
    "v1_per_twin_catalog": (
        "c84df85632c4f4d81728393e22ca553773e1a5992cccc79b5b798f288c5dbb99"
    ),
    "v1_summary": (
        "ab88f532758a8f5cf21307bd381a8b30f0883ac94d93f207a6b279c9d945e63a"
    ),
}

EXPECTED_TWINS = 4096
EXPECTED_CELLS = 64
TWINS_PER_CELL = 64
KS = (32, 64, 128)
NATURAL_FRACTION = 0.5
HASH_SEED = 20260901
MASS_TOLERANCE = 1.0e-12
STRICT_WEIGHT_TOLERANCE = 0.0


def _load_v2_module() -> Any:
    """Import only the frozen v2 module for its catalog metric helpers."""

    module_name = "pusht_motion_damping_d1_multiscale_soft_v2_for_comparator"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, V2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen v2 audit script: {V2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load_v2_module()
aggregate_relative_energy = v2.aggregate_relative_energy
file_sha256 = v2.file_sha256
full_projected_weights = v2.full_projected_weights
stable_cell_ranks = v2.stable_cell_ranks
uniform_weights = v2.uniform_weights


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite(name: str, values: np.ndarray | float) -> None:
    require(bool(np.isfinite(values).all()), f"{name} contains non-finite values")


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def assert_training_only_path(path: Path, *, expected_name: str | None = None) -> Path:
    """Reject validation/public inputs before looking at file contents."""

    resolved = Path(path).expanduser().resolve()
    if expected_name is not None:
        require(resolved.name == expected_name, f"expected {expected_name}: {resolved}")
    forbidden = {"validation", "test", "public", "development"}
    path_tokens = {
        token
        for part in resolved.parts
        for token in re.split(r"[^a-z0-9]+", part.lower())
        if token
    }
    require(
        path_tokens.isdisjoint(forbidden),
        f"Development/Public input is forbidden: {resolved}",
    )
    return resolved


@dataclass(frozen=True)
class ProjectedPool:
    """The frozen v2 per-twin values needed by this permutation audit."""

    twin_ids: np.ndarray
    pair_ids: tuple[tuple[str, str], ...]
    cells: np.ndarray
    orientation: np.ndarray
    speed_bin: np.ndarray
    goal_bin: np.ndarray
    query_speed: np.ndarray
    goal_distance: np.ndarray
    gap_rms_px: np.ndarray
    conditional: np.ndarray
    background: dict[int, np.ndarray]
    ranks: dict[int, np.ndarray]
    r_multiscale: np.ndarray
    pi_high: np.ndarray
    pi_full: np.ndarray


@dataclass(frozen=True)
class ComparatorWeights:
    """Full and high-arm weights plus the deterministic target ranks."""

    high: dict[str, np.ndarray]
    full: dict[str, np.ndarray]
    ranks: dict[str, np.ndarray]
    hash_hex: tuple[str, ...]


def verify_frozen_inputs(v2_dir: Path = DEFAULT_V2_DIR) -> dict[str, Any]:
    """Verify the immutable v2 artifact and its Training-only provenance.

    The large Training Lance directory is verified by its frozen SHA recorded in
    the already audited v2 receipt; this comparator does not rescan or decode it.
    """

    v2_dir = Path(v2_dir).expanduser().resolve()
    require(v2_dir.is_dir(), f"missing frozen v2 directory: {v2_dir}")
    observed = {
        name: file_sha256(v2_dir / name) for name in EXPECTED_V2_SHA256
    }
    require(observed == EXPECTED_V2_SHA256, "frozen v2 artifact SHA256 changed")

    config = json.loads((v2_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((v2_dir / "summary.json").read_text(encoding="utf-8"))
    receipt = json.loads((v2_dir / "receipt.json").read_text(encoding="utf-8"))
    require(
        config.get("candidate_id") == summary.get("candidate_id") == receipt.get("candidate_id") == "D1-MS50",
        "frozen candidate identity changed",
    )
    require(summary.get("status") == receipt.get("status") == "passed_go", "v2 is not passed_go")
    require(summary.get("gates", {}).get("passed") is True, "v2 gates are not passed")
    require(receipt.get("gates", {}).get("passed") is True, "v2 receipt gates are not passed")
    require(
        receipt.get("output_sha256", {}) == {
            "config.json": observed["config.json"],
            "summary.json": observed["summary.json"],
            "projected_weights.jsonl": observed["projected_weights.jsonl"],
        },
        "v2 receipt output hashes changed",
    )

    expected_inputs = config.get("expected_input_sha256")
    require(expected_inputs == EXPECTED_INPUT_SHA256, "v2 expected input SHA256 changed")
    require(receipt.get("expected_input_sha256") == EXPECTED_INPUT_SHA256, "v2 receipt input pin changed")
    recorded = receipt.get("input_sha256", {})
    for key, expected in EXPECTED_INPUT_SHA256.items():
        require(recorded.get(key) == expected, f"v2 receipt input hash changed: {key}")

    inputs = config.get("inputs", {})
    train_lance = assert_training_only_path(
        Path(inputs["train_lance"]), expected_name="train.lance"
    )
    manifest = assert_training_only_path(Path(inputs["manifest"]), expected_name="manifest.json")
    release_config = assert_training_only_path(
        Path(inputs["release_config"]), expected_name="pusht_motion_damping_icl_release_v1.yaml"
    )
    catalog_path = assert_training_only_path(
        Path(inputs["v1_catalog"]), expected_name="per_twin_catalog.jsonl"
    )
    summary_path = assert_training_only_path(Path(inputs["v1_summary"]), expected_name="summary.json")
    require(catalog_path.is_file(), f"missing frozen catalog: {catalog_path}")
    require(summary_path.is_file(), f"missing frozen v1 summary: {summary_path}")
    require(file_sha256(catalog_path) == EXPECTED_INPUT_SHA256["v1_per_twin_catalog"], "catalog SHA256 changed")
    require(file_sha256(summary_path) == EXPECTED_INPUT_SHA256["v1_summary"], "v1 summary SHA256 changed")
    require(train_lance.is_dir(), f"missing frozen Training table: {train_lance}")
    require(manifest.is_file(), f"missing frozen Training manifest: {manifest}")
    require(release_config.is_file(), f"missing frozen release config: {release_config}")

    summary_boundary = summary.get("evidence_boundary", {})
    require(
        summary_boundary.get("development_lance_opened") is False
        and summary_boundary.get("public_test_lance_opened") is False
        and summary_boundary.get("optimizer_steps") == 0
        and summary_boundary.get("model_loaded") is False
        and summary_boundary.get("schedule_generated") is False
        and summary_boundary.get("pixels_decoded") is False,
        "v2 evidence boundary changed",
    )
    require(
        receipt.get("development_lance_opened") is False
        and receipt.get("public_test_lance_opened") is False
        and receipt.get("optimizer_steps") == 0
        and receipt.get("model_loaded") is False
        and receipt.get("schedule_generated") is False
        and receipt.get("pixels_decoded") is False,
        "v2 receipt evidence boundary changed",
    )
    return {
        "v2_dir": v2_dir,
        "observed_v2_sha256": observed,
        "config": config,
        "summary": summary,
        "receipt": receipt,
        "train_lance": train_lance,
        "manifest": manifest,
        "release_config": release_config,
        "catalog_path": catalog_path,
        "summary_path": summary_path,
    }


def load_projected_pool(path: Path) -> ProjectedPool:
    """Load and structurally re-audit the frozen v2 projected rows."""

    path = assert_training_only_path(Path(path), expected_name="projected_weights.jsonl")
    require(path.is_file(), f"missing projected weights: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            require(bool(line.strip()), f"blank projected row at line {line_number}")
            row = json.loads(line)
            require(isinstance(row, dict), f"projected row {line_number} is not an object")
            rows.append(row)
    require(len(rows) == EXPECTED_TWINS, f"projected rows {len(rows)} != {EXPECTED_TWINS}")

    twin_ids = np.asarray([int(row["twin_id"]) for row in rows], dtype=np.int64)
    require(np.array_equal(twin_ids, np.arange(EXPECTED_TWINS)), "twin ids are not contiguous")
    pair_ids = tuple(tuple(str(value) for value in row["pair_ids"]) for row in rows)
    require(all(len(value) == 2 for value in pair_ids), "every twin must have two pair ids")
    for twin_id, observed in enumerate(pair_ids):
        expected = (
            f"pmd-train-{2 * twin_id:06d}-forward",
            f"pmd-train-{2 * twin_id + 1:06d}-reverse",
        )
        require(observed == expected, f"pair-to-twin mapping changed at twin {twin_id}")

    cells = np.asarray([int(row["coverage_cell"]) for row in rows], dtype=np.int64)
    orientation = np.asarray([int(row["orientation_bin"]) for row in rows], dtype=np.int64)
    speed_bin = np.asarray([int(row["speed_bin"]) for row in rows], dtype=np.int64)
    goal_bin = np.asarray([int(row["goal_distance_bin"]) for row in rows], dtype=np.int64)
    query_speed = np.asarray([float(row["query_speed"]) for row in rows], dtype=np.float64)
    goal_distance = np.asarray([float(row["goal_distance"]) for row in rows], dtype=np.float64)
    gap_rms = np.asarray([float(row["gap_rms_px"]) for row in rows], dtype=np.float64)
    conditional = np.asarray(
        [float(row["conditional_energy_physical"]) for row in rows], dtype=np.float64
    )
    background = {
        k: np.asarray(
            [float(row["background_future_variation"][str(k)]) for row in rows],
            dtype=np.float64,
        )
        for k in KS
    }
    ranks = {
        k: np.asarray([int(row["stable_rank_by_k"][str(k)]) for row in rows], dtype=np.int64)
        for k in KS
    }
    r_multiscale = np.asarray([float(row["r_multiscale"]) for row in rows], dtype=np.float64)
    pi_high = np.asarray([float(row["pi_high"]) for row in rows], dtype=np.float64)
    pi_full = np.asarray([float(row["pi_ms50"]) for row in rows], dtype=np.float64)

    require(
        np.array_equal(cells, orientation * 16 + speed_bin * 4 + goal_bin),
        "coverage-cell identity changed",
    )
    unique_cells, cell_counts = np.unique(cells, return_counts=True)
    require(
        np.array_equal(unique_cells, np.arange(EXPECTED_CELLS))
        and np.all(cell_counts == TWINS_PER_CELL),
        "coverage is not 64 cells x 64 twins",
    )
    for name, values in (
        ("query_speed", query_speed),
        ("goal_distance", goal_distance),
        ("gap_rms_px", gap_rms),
        ("conditional_energy_physical", conditional),
        ("r_multiscale", r_multiscale),
        ("pi_high", pi_high),
        ("pi_ms50", pi_full),
    ):
        _finite(name, values)
    for k in KS:
        _finite(f"B_{k}", background[k])
        require(np.all(background[k] >= 0.0), f"B_{k} is negative")
        require(np.all((ranks[k] >= 1) & (ranks[k] <= TWINS_PER_CELL)), f"bad stable rank k={k}")
        for cell in range(EXPECTED_CELLS):
            require(
                np.array_equal(np.sort(ranks[k][cells == cell]), np.arange(1, TWINS_PER_CELL + 1)),
                f"rank permutation changed in cell {cell}, k={k}",
            )
    require(np.all(conditional > 0.0), "C_phys must be strictly positive")
    require(np.all(query_speed > 0.0) and np.all(goal_distance > 0.0), "non-positive covariate")
    require(np.all(pi_high > 0.0) and np.all(pi_full > 0.0), "non-positive source weight")
    require(math.isclose(float(pi_high.sum()), 1.0, abs_tol=MASS_TOLERANCE), "pi_high is not normalized")
    require(math.isclose(float(pi_full.sum()), 1.0, abs_tol=MASS_TOLERANCE), "pi_ms50 is not normalized")
    require(
        np.allclose(
            pi_full,
            NATURAL_FRACTION / EXPECTED_TWINS + (1.0 - NATURAL_FRACTION) * pi_high,
            rtol=0.0,
            atol=1.0e-15,
        ),
        "pi_ms50 is not the frozen half-uniform mixture",
    )
    require(
        np.allclose(
            r_multiscale,
            np.stack([ranks[k] for k in KS], axis=0).mean(axis=0),
            rtol=0.0,
            atol=1.0e-15,
        ),
        "r_multiscale changed",
    )
    for cell in range(EXPECTED_CELLS):
        mask = cells == cell
        require(
            math.isclose(float(pi_high[mask].sum()), 1.0 / EXPECTED_CELLS, abs_tol=1.0e-15),
            f"pi_high cell mass changed in cell {cell}",
        )
        expected_high = (1.0 / EXPECTED_CELLS) * r_multiscale[mask] / float(r_multiscale[mask].sum())
        require(
            np.allclose(pi_high[mask], expected_high, rtol=0.0, atol=1.0e-15),
            f"pi_high values changed in cell {cell}",
        )
    return ProjectedPool(
        twin_ids=twin_ids,
        pair_ids=pair_ids,
        cells=cells,
        orientation=orientation,
        speed_bin=speed_bin,
        goal_bin=goal_bin,
        query_speed=query_speed,
        goal_distance=goal_distance,
        gap_rms_px=gap_rms,
        conditional=conditional,
        background=background,
        ranks=ranks,
        r_multiscale=r_multiscale,
        pi_high=pi_high,
        pi_full=pi_full,
    )


def verify_catalog_projection_match(pool: ProjectedPool, catalog: Any) -> None:
    """Require projected metadata and Training catalog physics to agree exactly."""

    require(np.array_equal(pool.twin_ids, catalog.twin_ids), "catalog/projected twin ids differ")
    require(pool.pair_ids == catalog.pair_ids, "catalog/projected pair ids differ")
    for name, left, right in (
        ("coverage cells", pool.cells, catalog.coverage_cells),
        ("orientation bins", pool.orientation, catalog.orientation_bin),
        ("speed bins", pool.speed_bin, catalog.speed_bin),
        ("goal bins", pool.goal_bin, catalog.goal_distance_bin),
    ):
        require(np.array_equal(left, right), f"catalog/projected {name} differ")
    for name, left, right in (
        ("query speed", pool.query_speed, catalog.query_speed),
        ("goal distance", pool.goal_distance, catalog.goal_distance),
        ("gap rms", pool.gap_rms_px, catalog.gap_rms_px),
        ("C_phys", pool.conditional, catalog.conditional),
    ):
        require(np.array_equal(left, right), f"catalog/projected {name} differ")
    for k in KS:
        require(
            np.array_equal(pool.background[k], catalog.background[k]),
            f"catalog/projected B_{k} differ",
        )


def _hash_digest(seed: int, twin_id: int) -> bytes:
    require(int(seed) == HASH_SEED, f"hash seed is frozen at {HASH_SEED}")
    return hashlib.sha256(f"{HASH_SEED}|{int(twin_id)}".encode("ascii")).digest()


def hash_cell_ranks(
    twin_ids: np.ndarray,
    cells: np.ndarray,
    *,
    seed: int = HASH_SEED,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Rank sha256(seed|twin_id) within each cell, low digest first."""

    twin_ids = np.asarray(twin_ids, dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    require(twin_ids.shape == cells.shape, "hash rank shape mismatch")
    require(len(np.unique(twin_ids)) == len(twin_ids), "hash rank ids are not unique")
    digests = tuple(_hash_digest(seed, int(twin_id)) for twin_id in twin_ids)
    ranks = np.zeros(len(twin_ids), dtype=np.int64)
    for cell in sorted(np.unique(cells).tolist()):
        indices = np.flatnonzero(cells == cell).tolist()
        order = sorted(indices, key=lambda index: (digests[index], int(twin_ids[index])))
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    require(np.all(ranks >= 1), "hash rank assignment incomplete")
    return ranks, tuple(value.hex() for value in digests)


def reassign_weight_multiset(
    source_high: np.ndarray,
    target_ranks: np.ndarray,
    cells: np.ndarray,
    twin_ids: np.ndarray,
) -> np.ndarray:
    """Assign each cell's sorted source weights to target ranks, without changing values."""

    source_high = np.asarray(source_high, dtype=np.float64)
    target_ranks = np.asarray(target_ranks, dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    twin_ids = np.asarray(twin_ids, dtype=np.int64)
    require(
        source_high.shape == target_ranks.shape == cells.shape == twin_ids.shape,
        "weight reassignment shape mismatch",
    )
    _finite("source high weights", source_high)
    require(np.all(source_high > 0.0), "source high weights must be positive")
    require(np.all(target_ranks >= 1), "target ranks must be positive")
    result = np.empty_like(source_high)
    for cell in sorted(np.unique(cells).tolist()):
        indices = np.flatnonzero(cells == cell)
        source_order = indices[np.lexsort((twin_ids[indices], source_high[indices]))]
        target_order = indices[np.lexsort((twin_ids[indices], target_ranks[indices]))]
        require(len(source_order) == len(target_order), f"cell {cell} size mismatch")
        result[target_order] = source_high[source_order]
    _finite("reassigned high weights", result)
    require(np.all(result > 0.0), "reassigned weights must be positive")
    return result


def build_comparator_weights(
    pool: ProjectedPool,
    *,
    hash_seed: int = HASH_SEED,
) -> ComparatorWeights:
    """Construct REL50, ABS50 and HASH50 from the frozen source multiset."""

    require(int(hash_seed) == HASH_SEED, f"HASH50 seed is frozen at {HASH_SEED}")
    abs_rank = stable_cell_ranks(
        pool.conditional, pool.cells, stable_ids=pool.twin_ids
    )
    hash_rank, hash_hex = hash_cell_ranks(pool.twin_ids, pool.cells, seed=hash_seed)
    rel_rank = stable_cell_ranks(
        pool.pi_high, pool.cells, stable_ids=pool.twin_ids
    )
    abs_high = reassign_weight_multiset(
        pool.pi_high, abs_rank, pool.cells, pool.twin_ids
    )
    hash_high = reassign_weight_multiset(
        pool.pi_high, hash_rank, pool.cells, pool.twin_ids
    )
    return ComparatorWeights(
        high={
            "REL50": pool.pi_high.copy(),
            "ABS50": abs_high,
            "HASH50": hash_high,
        },
        full={
            "D0": uniform_weights(len(pool.twin_ids)),
            "REL50": pool.pi_full.copy(),
            "ABS50": full_projected_weights(abs_high),
            "HASH50": full_projected_weights(hash_high),
        },
        ranks={"REL50": rel_rank, "ABS50": abs_rank, "HASH50": hash_rank},
        hash_hex=hash_hex,
    )


def _strict_multiset_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return bool(left.shape == right.shape and np.array_equal(np.sort(left), np.sort(right)))


def _cell_mass(weights: np.ndarray, cells: np.ndarray, cell: int) -> float:
    return float(np.asarray(weights, dtype=np.float64)[cells == cell].sum())


def _distribution_stats(weights: np.ndarray) -> dict[str, float]:
    values = np.sort(np.asarray(weights, dtype=np.float64))
    _finite("distribution weights", values)
    require(np.all(values > 0.0), "distribution weights must be positive")
    ess = 1.0 / float(np.sum(np.square(values), dtype=np.float64))
    entropy = -float(np.sum(values * np.log(values), dtype=np.float64))
    return {
        "ess": ess,
        "entropy_nats": entropy,
        "min": float(values[0]),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(values[-1]),
    }


def validate_comparator_invariants(
    pool: ProjectedPool,
    comparators: ComparatorWeights,
) -> dict[str, Any]:
    """Fail closed unless permutation, coverage and distribution invariants hold."""

    require(set(comparators.full) == {"D0", "REL50", "ABS50", "HASH50"}, "full arm set changed")
    require(set(comparators.high) == {"REL50", "ABS50", "HASH50"}, "high arm set changed")
    require(len(pool.twin_ids) == EXPECTED_TWINS, "unexpected comparator twin count")
    require(len(np.unique(pool.cells)) == EXPECTED_CELLS, "unexpected comparator cell count")
    checks: dict[str, bool] = {
        "training_pool_shape_4096": len(pool.twin_ids) == EXPECTED_TWINS,
        "coverage_shape_64x64": bool(
            np.all(np.bincount(pool.cells, minlength=EXPECTED_CELLS) == TWINS_PER_CELL)
        ),
    }
    source_high = comparators.high["REL50"]
    source_full = comparators.full["REL50"]
    for arm, weights in comparators.full.items():
        checks[f"{arm.lower()}_full_positive"] = bool(np.all(weights > 0.0))
        checks[f"{arm.lower()}_full_normalized"] = math.isclose(
            float(weights.sum()), 1.0, abs_tol=MASS_TOLERANCE
        )
        for cell in range(EXPECTED_CELLS):
            checks[f"{arm.lower()}_full_cell_{cell}_mass"] = math.isclose(
                _cell_mass(weights, pool.cells, cell),
                1.0 / EXPECTED_CELLS,
                abs_tol=1.0e-15,
            )
    for arm, weights in comparators.high.items():
        checks[f"{arm.lower()}_high_positive"] = bool(np.all(weights > 0.0))
        checks[f"{arm.lower()}_high_normalized"] = math.isclose(
            float(weights.sum()), 1.0, abs_tol=MASS_TOLERANCE
        )
        for cell in range(EXPECTED_CELLS):
            mask = pool.cells == cell
            checks[f"{arm.lower()}_high_cell_{cell}_mass"] = math.isclose(
                float(weights[mask].sum()), 1.0 / EXPECTED_CELLS, abs_tol=1.0e-15
            )
            checks[f"{arm.lower()}_high_cell_{cell}_multiset"] = _strict_multiset_equal(
                weights[mask], source_high[mask]
            )
            checks[f"{arm.lower()}_full_cell_{cell}_multiset"] = _strict_multiset_equal(
                comparators.full[arm][mask], source_full[mask]
            )
    for arm in ("ABS50", "HASH50"):
        checks[f"{arm.lower()}_full_weight_multiset"] = _strict_multiset_equal(
            comparators.full[arm], source_full
        )
        checks[f"{arm.lower()}_high_weight_multiset"] = _strict_multiset_equal(
            comparators.high[arm], source_high
        )
        source_stats = _distribution_stats(source_full)
        arm_stats = _distribution_stats(comparators.full[arm])
        checks[f"{arm.lower()}_ess_exact"] = source_stats["ess"] == arm_stats["ess"]
        checks[f"{arm.lower()}_entropy_exact"] = (
            source_stats["entropy_nats"] == arm_stats["entropy_nats"]
        )
    checks["ABS50_rank_is_stable_conditional_rank"] = bool(
        np.array_equal(
            comparators.ranks["ABS50"],
            stable_cell_ranks(pool.conditional, pool.cells, stable_ids=pool.twin_ids),
        )
    )
    expected_hash_rank, expected_hash_hex = hash_cell_ranks(
        pool.twin_ids, pool.cells, seed=HASH_SEED
    )
    checks["HASH50_rank_is_frozen_sha256_rank"] = bool(
        np.array_equal(comparators.ranks["HASH50"], expected_hash_rank)
        and comparators.hash_hex == expected_hash_hex
    )
    checks["REL50_source_unchanged"] = bool(
        np.array_equal(comparators.high["REL50"], pool.pi_high)
        and np.array_equal(comparators.full["REL50"], pool.pi_full)
    )
    checks["all_invariants_pass"] = all(checks.values())
    require(checks["all_invariants_pass"], "one or more comparator invariants failed")
    return checks


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    _finite("rank-correlation values", values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Spearman rho without importing a second statistics dependency."""

    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - float(left_rank.mean())
    right_centered = right_rank - float(right_rank.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    require(denominator > 0.0, "rank correlation is undefined for a constant vector")
    return float(np.dot(left_centered, right_centered) / denominator)


def rank_correlation_report(pool: ProjectedPool, comparators: ComparatorWeights) -> dict[str, Any]:
    """Report global and per-cell ordering agreement, never as a success claim."""

    labels = ("REL50", "ABS50", "HASH50")
    pairs = (("REL50", "ABS50"), ("REL50", "HASH50"), ("ABS50", "HASH50"))
    global_values: dict[str, float] = {}
    for left, right in pairs:
        global_values[f"{left.lower()}_vs_{right.lower()}"] = spearman_rank_correlation(
            comparators.ranks[left], comparators.ranks[right]
        )
    # Comparator ranks are within-cell ranks.  Re-rank C_phys within the same
    # cells before pooling; correlating raw C_phys with local ranks would mostly
    # measure the arbitrary cell numbering instead of the assignment rule.
    conditional_rank = stable_cell_ranks(
        pool.conditional, pool.cells, stable_ids=pool.twin_ids
    )
    global_values["conditional_vs_rel50"] = spearman_rank_correlation(
        conditional_rank, comparators.ranks["REL50"]
    )
    global_values["conditional_vs_abs50"] = spearman_rank_correlation(
        conditional_rank, comparators.ranks["ABS50"]
    )
    global_values["conditional_vs_hash50"] = spearman_rank_correlation(
        conditional_rank, comparators.ranks["HASH50"]
    )

    by_cell: dict[str, dict[str, float]] = {}
    for cell in range(EXPECTED_CELLS):
        mask = pool.cells == cell
        values: dict[str, float] = {}
        for left, right in pairs:
            values[f"{left.lower()}_vs_{right.lower()}"] = spearman_rank_correlation(
                comparators.ranks[left][mask], comparators.ranks[right][mask]
            )
        values["conditional_vs_rel50"] = spearman_rank_correlation(
            pool.conditional[mask], comparators.ranks["REL50"][mask]
        )
        values["conditional_vs_abs50"] = spearman_rank_correlation(
            pool.conditional[mask], comparators.ranks["ABS50"][mask]
        )
        values["conditional_vs_hash50"] = spearman_rank_correlation(
            pool.conditional[mask], comparators.ranks["HASH50"][mask]
        )
        by_cell[str(cell)] = values

    keys = tuple(global_values)
    by_cell_summary = {
        key: {
            "min": float(min(by_cell[str(cell)][key] for cell in range(EXPECTED_CELLS))),
            "mean": float(np.mean([by_cell[str(cell)][key] for cell in range(EXPECTED_CELLS)])),
            "median": float(np.median([by_cell[str(cell)][key] for cell in range(EXPECTED_CELLS)])),
            "max": float(max(by_cell[str(cell)][key] for cell in range(EXPECTED_CELLS))),
        }
        for key in keys
    }
    return {
        "definition": "Spearman correlation of low-to-high within-cell ranks; descriptive only",
        "global": global_values,
        "by_cell": by_cell,
        "by_cell_summary": by_cell_summary,
    }


def arm_metric_report(pool: ProjectedPool, weights: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Report weighted C_phys, B_k and rho_phys for one exposure arm."""

    result: dict[str, Any] = {}
    for arm, arm_weights in weights.items():
        arm_weights = np.asarray(arm_weights, dtype=np.float64)
        require(arm_weights.shape == pool.conditional.shape, f"{arm} weight shape mismatch")
        aggregates = {
            str(k): aggregate_relative_energy(pool.conditional, pool.background[k], arm_weights)
            for k in KS
        }
        result[arm] = {
            "weighted_C_phys": aggregates["64"]["weighted_conditional_energy"],
            "weighted_B_k": {
                str(k): aggregates[str(k)]["weighted_background_variation"] for k in KS
            },
            "rho_phys": {
                str(k): aggregates[str(k)]["rho_phys_ratio_of_means"] for k in KS
            },
            "aggregate_by_k": aggregates,
            "weight_distribution": _distribution_stats(arm_weights),
        }
    return result


def build_summary(
    pool: ProjectedPool,
    comparators: ComparatorWeights,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    metrics = arm_metric_report(pool, comparators.full)
    return {
        "schema_version": 1,
        "builder_id": "pusht_motion_damping_root_cause_comparators_v1",
        "status": "completed_training_only_comparator",
        "candidate_id": "D1-MS50",
        "hash_seed": HASH_SEED,
        "dimensions": {
            "twins": int(len(pool.twin_ids)),
            "coverage_cells": int(len(np.unique(pool.cells))),
            "twins_per_cell": int(TWINS_PER_CELL),
            "ks": list(KS),
        },
        "arms": metrics,
        "rank_correlations": rank_correlation_report(pool, comparators),
        "invariants": dict(checks),
        "construction": {
            "REL50": "frozen D1-MS50 projected pi_ms50",
            "ABS50": "within-cell stable ascending conditional_energy_physical rank receives sorted REL50 pi_high multiset",
            "HASH50": "within-cell stable ascending sha256(UTF8(f'{HASH_SEED}|twin_id')) rank receives sorted REL50 pi_high multiset",
            "full_projection": "0.5 * Uniform_all_twins + 0.5 * high_arm",
            "comparator_role": "exposure comparator only; no causal or ICL success claim",
        },
        "evidence_boundary": {
            "training_only": True,
            "development_lance_opened": False,
            "public_test_lance_opened": False,
            "pixels_decoded": False,
            "model_loaded": False,
            "optimizer_steps": 0,
            "schedule_generated": False,
            "claim": "This artifact only constructs and audits matched exposure comparators; it does not establish a model, latent, gradient, calibration, or native ICL improvement.",
        },
    }


def build_projected_rows(pool: ProjectedPool, comparators: ComparatorWeights) -> list[dict[str, Any]]:
    """Serialize one row per twin while retaining all source quantities."""

    rows: list[dict[str, Any]] = []
    for index, twin_id in enumerate(pool.twin_ids.tolist()):
        rows.append(
            {
                "twin_id": int(twin_id),
                "pair_ids": list(pool.pair_ids[index]),
                "coverage_cell": int(pool.cells[index]),
                "orientation_bin": int(pool.orientation[index]),
                "speed_bin": int(pool.speed_bin[index]),
                "goal_distance_bin": int(pool.goal_bin[index]),
                "query_speed": float(pool.query_speed[index]),
                "goal_distance": float(pool.goal_distance[index]),
                "gap_rms_px": float(pool.gap_rms_px[index]),
                "conditional_energy_physical": float(pool.conditional[index]),
                "background_future_variation": {
                    str(k): float(pool.background[k][index]) for k in KS
                },
                "stable_rank_by_k": {str(k): int(pool.ranks[k][index]) for k in KS},
                "r_multiscale": float(pool.r_multiscale[index]),
                "pi_high_rel50": float(comparators.high["REL50"][index]),
                "pi_ms50_rel50": float(comparators.full["REL50"][index]),
                "pi_high_abs50": float(comparators.high["ABS50"][index]),
                "pi_abs50": float(comparators.full["ABS50"][index]),
                "pi_high_hash50": float(comparators.high["HASH50"][index]),
                "pi_hash50": float(comparators.full["HASH50"][index]),
                "rank_rel50": int(comparators.ranks["REL50"][index]),
                "rank_abs50": int(comparators.ranks["ABS50"][index]),
                "rank_hash50": int(comparators.ranks["HASH50"][index]),
                "hash_sha256": comparators.hash_hex[index],
            }
        )
    return rows


def write_artifacts(
    output_dir: Path,
    *,
    frozen: Mapping[str, Any],
    pool: ProjectedPool,
    comparators: ComparatorWeights,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an exclusive comparator artifact directory and receipt."""

    output_dir = Path(output_dir).expanduser().resolve()
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config.json"
    weights_path = output_dir / "projected_weights.jsonl"
    summary_path = output_dir / "summary.json"
    receipt_path = output_dir / "receipt.json"
    builder_script_sha256 = file_sha256(Path(__file__).resolve())
    config = {
        "schema_version": 1,
        "builder_id": "pusht_motion_damping_root_cause_comparators_v1",
        "builder_script_sha256": builder_script_sha256,
        "candidate_id": "D1-MS50",
        "source_v2_dir": str(frozen["v2_dir"]),
        "source_input_sha256": {
            **frozen["observed_v2_sha256"],
            "v1_per_twin_catalog": EXPECTED_INPUT_SHA256["v1_per_twin_catalog"],
            "v1_summary": EXPECTED_INPUT_SHA256["v1_summary"],
            "train_lance_directory": EXPECTED_INPUT_SHA256["train_lance_directory"],
            "manifest": EXPECTED_INPUT_SHA256["manifest"],
            "release_config": EXPECTED_INPUT_SHA256["release_config"],
        },
        "hash_seed": HASH_SEED,
        "arms": ["D0", "REL50", "ABS50", "HASH50"],
        "rank_weight_multiset_source": "D1-MS50 pi_high",
        "abs_rank": "ascending conditional_energy_physical; ties ascending twin_id",
        "hash_rank": f"ascending sha256(UTF8('{HASH_SEED}|twin_id')); ties ascending twin_id",
        "full_projection": "0.5 * Uniform_all_twins + 0.5 * high_arm",
        "evidence_boundary": summary["evidence_boundary"],
    }
    _json_dump(config_path, config)
    with weights_path.open("x", encoding="utf-8") as stream:
        for row in build_projected_rows(pool, comparators):
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _json_dump(summary_path, summary)
    receipt = {
        "schema_version": 1,
        "builder_id": summary["builder_id"],
        "candidate_id": summary["candidate_id"],
        "status": summary["status"],
        "hash_seed": HASH_SEED,
        "input_sha256": config["source_input_sha256"],
        "builder_script_sha256": builder_script_sha256,
        "output_sha256": {
            "config.json": file_sha256(config_path),
            "projected_weights.jsonl": file_sha256(weights_path),
            "summary.json": file_sha256(summary_path),
        },
        "development_lance_opened": False,
        "public_test_lance_opened": False,
        "pixels_decoded": False,
        "model_loaded": False,
        "optimizer_steps": 0,
        "schedule_generated": False,
        "invariants": summary["invariants"],
    }
    _json_dump(receipt_path, receipt)
    return {
        "output_dir": output_dir,
        "config": config_path,
        "projected_weights": weights_path,
        "summary": summary_path,
        "receipt": receipt_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify inputs and construct the report in memory without writing artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frozen = verify_frozen_inputs(args.v2_dir)
    catalog = v2.load_v1_catalog(frozen["catalog_path"])
    projected_path = Path(args.v2_dir).expanduser().resolve() / "projected_weights.jsonl"
    pool = load_projected_pool(projected_path)
    verify_catalog_projection_match(pool, catalog)
    comparators = build_comparator_weights(pool)
    checks = validate_comparator_invariants(pool, comparators)
    summary = build_summary(pool, comparators, checks)
    if args.check_only:
        print(json.dumps({"status": summary["status"], "summary": summary}, indent=2, sort_keys=True))
        return 0
    paths = write_artifacts(
        args.output_dir,
        frozen=frozen,
        pool=pool,
        comparators=comparators,
        summary=summary,
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(paths["output_dir"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
