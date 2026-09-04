#!/usr/bin/env python3
"""D0/D1 raw-pixel baseline adapter for pusht_motion_damping D2 P1b (v1).

Narrow scope (frozen by
configs/pusht_motion_damping_d2_v1_pre_p1b_execution_addendum_v1.yaml,
``pixel_baseline_resolution.decision = targeted_adapter_not_a_new_audit_branch``):

  * Reuse the frozen raw-pixel auditor's lossless Lance PNG decoder and its
    pixel C / B / rho math verbatim.  No new renderer, resize, crop or
    normalization is defined here.
  * Select a deterministic coverage-stratified Training-only panel of exactly
    ``TWINS_PER_CELL`` D0 twins from each of the 64 frozen coverage cells using
    ``catalog_identity.baseline_pixel_panel_seed``.  Ordering is a stable
    SHA256 of (seed, cell, twin_id) only -- never an outcome field.
  * Report C_pixel, B_pixel and ratio-of-means rho_pixel under two weightings
    of the SAME queries and the SAME fixed neighbor graph: D0-uniform, and the
    frozen realized D1-MS50 multiplicity.  Both are standardized to equal mass
    per coverage cell.
  * Report simultaneous (max-statistic) intervals for the three D1-D0 rho
    differences from a stratified twin cluster bootstrap with a frozen seed.

Evidence boundary (immutable):
  claim_scope        = frozen_training_only_raw_pixel_d0_d1_reweighting_baseline
  d2_pass_fail_claim = not_made  (this is a baseline adapter, not a D2 gate)
  latent_status      = not_measured
  binding_cause      = not_claimed

No model is loaded, no optimizer step is taken, no Development or Public split
is opened, and ContextWorld is read only through the frozen Training Lance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Evidence boundary -- immutable module-level constants (read by tests)
# ---------------------------------------------------------------------------
MODEL_LOADED: bool = False
OPTIMIZER_STEPS: int = 0
DEVELOPMENT_SPLIT_OPENED: bool = False
PUBLIC_SPLIT_OPENED: bool = False
CONTEXT_WORLD_MODIFIED: bool = False
CLAIM_SCOPE: str = "frozen_training_only_raw_pixel_d0_d1_reweighting_baseline"
D2_PASS_FAIL_CLAIM: str = "not_made"
LATENT_STATUS: str = "not_measured"
BINDING_CAUSE_STATUS: str = "not_claimed"
PIXEL_UNITS: str = "normalized_rgb_mse_per_pixel_channel"

# Frozen identities taken from the addendum (verified at run time).
PANEL_SEED: int = 2026090404          # catalog_identity.baseline_pixel_panel_seed
BOOTSTRAP_SEED: int = 2026090405      # catalog_identity.bootstrap_seed
N_COVERAGE_CELLS: int = 64
TWINS_PER_CELL: int = 4
N_D0_TWINS: int = 4096
KS: tuple[int, ...] = (32, 64, 128)
N_BOOTSTRAP: int = 2048
CI_LEVEL: float = 0.95
DIRECTIONS: tuple[str, ...] = ("forward", "reverse")
MODES: tuple[str, ...] = ("faster_decay", "no_extra_decay")
CONDITION_WEIGHTS: tuple[float, float] = (0.5, 0.5)
QUERY_STEP: int = 10
FUTURE_STEP: int = 15

# Repo-relative paths of inputs whose SHA256 the addendum pins.
RAW_PIXEL_AUDITOR_REL = (
    "research/conditional_dynamics_representation/scripts/"
    "audit_icl_training_raw_pixel_visibility_v1.py"
)
PRIOR_PIXEL_RESULT_REL = (
    "research/conditional_dynamics_representation/artifacts/"
    "icl_training_raw_pixel_visibility_v1/training_only_v1/per_task.jsonl"
)
D1_MULTIPLICITY_REL = (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_schedule_v1/d1_ms50_schedule_v1_final/multiplicity.jsonl"
)
D1_PROJECTED_WEIGHTS_REL = (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_multiscale_soft_v2/d1_0_training_only_v2/"
    "projected_weights.jsonl"
)

_PAIR_ID_RE = re.compile(r"pmd-train-(\d{6})-(forward|reverse)")


# ---------------------------------------------------------------------------
# Frozen auditor reuse
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_raw_pixel_auditor(path: Path | None = None) -> Any:
    """Import the frozen raw-pixel auditor module and reuse its primitives."""
    target = Path(path) if path is not None else repo_root() / RAW_PIXEL_AUDITOR_REL
    spec = importlib.util.spec_from_file_location(
        "audit_icl_training_raw_pixel_visibility_v1", target
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import frozen raw-pixel auditor: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDITOR = load_raw_pixel_auditor()

# Reused verbatim from the frozen auditor -- no re-derivation here.
file_sha256 = AUDITOR.file_sha256
training_only_guard = AUDITOR.training_only_guard
exclusive_mkdir = AUDITOR.exclusive_mkdir
normalize_pixels = AUDITOR.normalize_pixels
decode_pixel_blob = AUDITOR.decode_pixel_blob
pixel_conditional_variance = AUDITOR.pixel_conditional_variance
pixel_mean_displacement = AUDITOR.pixel_mean_displacement
ratio_of_means = AUDITOR.ratio_of_means
robust_scale = AUDITOR.robust_scale
leave_cluster_out_knn = AUDITOR.leave_cluster_out_knn
background_variance_by_k = AUDITOR.background_variance_by_k
load_config = AUDITOR.load_config
_read_selected_push_rows = AUDITOR._read_selected_push_rows
_resolve = AUDITOR._resolve


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Input hash verification
# ---------------------------------------------------------------------------

def addendum_expected_hashes(addendum: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Map input name -> (repo-relative path, expected sha256) from the addendum."""
    expected: dict[str, tuple[str, str]] = {}
    for name, item in dict(addendum.get("parent_identity", {})).items():
        expected[name] = (str(item["path"]), str(item["sha256"]))
    identity = dict(
        addendum.get("pixel_baseline_resolution", {}).get("required_identity", {})
    )
    _require(bool(identity), "addendum lacks pixel_baseline_resolution.required_identity")
    pinned = {
        "projected_weights": (D1_PROJECTED_WEIGHTS_REL, "projected_weights"),
        "pixel_config": (
            "research/conditional_dynamics_representation/configs/"
            "icl_training_raw_pixel_visibility_v1.yaml",
            "pixel_config",
        ),
        "motion_training_manifest": (
            "../ContextWorld/artifacts/synthesis/"
            "pusht_motion_damping_h3_release_v4/manifest.json",
            "motion_training_manifest",
        ),
        "raw_pixel_auditor": (RAW_PIXEL_AUDITOR_REL, "raw_pixel_auditor"),
        "prior_pixel_result": (PRIOR_PIXEL_RESULT_REL, "prior_pixel_result"),
        "d1_multiplicity": (D1_MULTIPLICITY_REL, "d1_multiplicity"),
    }
    for name, (fallback_path, field_prefix) in pinned.items():
        expected[name] = (
            str(identity.get(f"{field_prefix}_path", fallback_path)),
            str(identity[f"{field_prefix}_sha256"]),
        )
    return expected


def verify_input_hashes(
    root: Path, expected: dict[str, tuple[str, str]]
) -> dict[str, str]:
    """Verify every pinned input hash; raise on any mismatch or missing file."""
    observed: dict[str, str] = {}
    for name in sorted(expected):
        rel, want = expected[name]
        path = (Path(root) / rel).resolve()
        training_only_guard(path)
        _require(path.exists(), f"pinned input missing: {name} -> {path}")
        got = file_sha256(path)
        _require(
            got == want,
            f"sha256 mismatch for {name}: expected {want}, observed {got}",
        )
        observed[name] = got
    return observed


def assert_training_only_inputs(paths: Iterable[Path]) -> list[str]:
    """Apply the frozen forbidden-split guard to every input path."""
    checked: list[str] = []
    for path in paths:
        resolved = Path(path).resolve()
        training_only_guard(resolved)
        checked.append(str(resolved))
    return checked


# ---------------------------------------------------------------------------
# Deterministic coverage-stratified panel selection
# ---------------------------------------------------------------------------

def panel_rank_key(seed: int, coverage_cell: int, twin_id: int) -> str:
    """Stable outcome-independent ordering key for one twin."""
    payload = f"{int(seed)}:{int(coverage_cell)}:{int(twin_id)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_pair_ids(pair_ids: Sequence[str], twin_id: int) -> dict[str, int]:
    """Return {direction: pair_index}; require exactly one forward and one reverse."""
    by_direction: dict[str, int] = {}
    for pair_id in pair_ids:
        match = _PAIR_ID_RE.fullmatch(str(pair_id))
        _require(match is not None, f"unexpected pair_id: {pair_id}")
        pair_index = int(match.group(1))
        direction = match.group(2)
        _require(direction not in by_direction, f"duplicate direction for twin {twin_id}")
        _require(
            pair_index // 2 == int(twin_id),
            f"pair_id {pair_id} does not belong to twin {twin_id}",
        )
        by_direction[direction] = pair_index
    _require(
        set(by_direction) == set(DIRECTIONS),
        f"twin {twin_id} lacks both forward/reverse directions: {sorted(by_direction)}",
    )
    return by_direction


def select_panel(
    multiplicity_rows: Sequence[dict[str, Any]],
    *,
    seed: int = PANEL_SEED,
    twins_per_cell: int = TWINS_PER_CELL,
    expected_cells: int = N_COVERAGE_CELLS,
    expected_twins: int | None = N_D0_TWINS,
) -> list[dict[str, Any]]:
    """Exactly `twins_per_cell` twins per coverage cell, SHA256-ordered."""
    by_cell: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[int] = set()
    for row in multiplicity_rows:
        twin_id = int(row["twin_id"])
        _require(twin_id not in seen, f"duplicate twin_id {twin_id}")
        seen.add(twin_id)
        by_cell[int(row["coverage_cell"])].append(row)
    if expected_twins is not None:
        _require(
            len(seen) == int(expected_twins),
            f"expected {expected_twins} D0 twins, found {len(seen)}",
        )
    _require(
        len(by_cell) == int(expected_cells),
        f"expected {expected_cells} coverage cells, found {len(by_cell)}",
    )

    panel: list[dict[str, Any]] = []
    for cell in sorted(by_cell):
        members = by_cell[cell]
        _require(
            len(members) >= twins_per_cell,
            f"coverage cell {cell} has {len(members)} twins < {twins_per_cell}",
        )
        ordered = sorted(
            members,
            key=lambda row: (
                panel_rank_key(seed, cell, int(row["twin_id"])),
                int(row["twin_id"]),
            ),
        )
        for row in ordered[:twins_per_cell]:
            twin_id = int(row["twin_id"])
            mass = float(row["realized_pi_ms50"])
            _require(
                math.isfinite(mass) and mass > 0.0,
                f"twin {twin_id} has non-positive realized_pi_ms50 {mass}",
            )
            panel.append(
                {
                    "coverage_cell": cell,
                    "twin_id": twin_id,
                    "pair_ids": [str(v) for v in row["pair_ids"]],
                    "pair_index_by_direction": parse_pair_ids(row["pair_ids"], twin_id),
                    "realized_pi_ms50": mass,
                }
            )
    _require(
        len(panel) == int(expected_cells) * twins_per_cell,
        "panel size mismatch after stratified selection",
    )
    return panel


def panel_hash(panel: Sequence[dict[str, Any]]) -> str:
    lines = [
        f"{entry['coverage_cell']}:{entry['twin_id']}:{pair_id}"
        for entry in panel
        for pair_id in sorted(entry["pair_ids"])
    ]
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Equal-cell weighting (D0 uniform and frozen realized D1-MS50)
# ---------------------------------------------------------------------------

def equal_cell_weights(
    occurrence_cells: Sequence[int],
    occurrence_mass: Sequence[float],
    occurrence_queries: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Equal mass per coverage cell; within-cell mass proportional to twin mass.

    Within one twin the mass is split equally across its query directions.
    Returns (query_indices, weights) with weights summing to 1.
    """
    cells = [int(c) for c in occurrence_cells]
    mass = [float(m) for m in occurrence_mass]
    _require(len(cells) == len(mass) == len(occurrence_queries), "occurrence length mismatch")
    _require(bool(cells), "no occurrences")
    totals: dict[int, float] = defaultdict(float)
    for cell, value in zip(cells, mass):
        _require(math.isfinite(value) and value > 0.0, f"non-positive mass {value}")
        totals[cell] += value
    n_cells = len(totals)
    indices: list[int] = []
    weights: list[float] = []
    for cell, value, queries in zip(cells, mass, occurrence_queries):
        _require(bool(queries), f"occurrence in cell {cell} has no queries")
        share = (1.0 / n_cells) * (value / totals[cell]) / float(len(queries))
        for query in queries:
            indices.append(int(query))
            weights.append(share)
    weight_array = np.asarray(weights, dtype=np.float64)
    _require(
        math.isclose(float(weight_array.sum()), 1.0, abs_tol=1e-12),
        "equal-cell weights do not sum to 1",
    )
    return np.asarray(indices, dtype=np.int64), weight_array


def panel_weightings(
    panel: Sequence[dict[str, Any]], query_index_by_twin: dict[int, list[int]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cells = [int(entry["coverage_cell"]) for entry in panel]
    queries = [query_index_by_twin[int(entry["twin_id"])] for entry in panel]
    d0 = equal_cell_weights(cells, [1.0] * len(panel), queries)
    d1 = equal_cell_weights(
        cells, [float(entry["realized_pi_ms50"]) for entry in panel], queries
    )
    return {"d0_uniform": d0, "d1_ms50_realized": d1}


# ---------------------------------------------------------------------------
# Estimates on the shared fixed neighbor graph
# ---------------------------------------------------------------------------

def rho_components(
    C: np.ndarray, B: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Ratio-of-means rho with numerator and denominator reported separately."""
    c_sel = np.asarray(C, dtype=np.float64)[indices]
    b_sel = np.asarray(B, dtype=np.float64)[indices]
    numerator = float(np.dot(weights, c_sel))
    background = float(np.dot(weights, b_sel))
    denominator = numerator + background
    _require(denominator > 0.0, "rho denominator is zero")
    rho = ratio_of_means(c_sel, b_sel, weights)
    _require(
        math.isclose(rho, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12),
        "rho disagrees with reported numerator/denominator",
    )
    return {
        "rho_pixel": float(rho),
        "numerator_expected_C_pixel": numerator,
        "expected_B_pixel": background,
        "denominator_expected_C_plus_B": denominator,
    }


def build_neighbor_graph(
    descriptors: np.ndarray, twin_ids: np.ndarray, *, max_k: int = max(KS)
) -> np.ndarray:
    """One fixed leave-whole-twin-out neighbor graph shared by D0 and D1."""
    scaled, _, _ = robust_scale(np.asarray(descriptors, dtype=np.float64))
    neighbors, _ = leave_cluster_out_knn(
        scaled, np.asarray(twin_ids, dtype=np.int64), max_k=int(max_k)
    )
    return neighbors


# ---------------------------------------------------------------------------
# Stratified twin cluster bootstrap with simultaneous max-statistic intervals
# ---------------------------------------------------------------------------

def _draw_matrices(
    panel: Sequence[dict[str, Any]], query_index_by_twin: dict[int, list[int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cell_ids, mass_matrix (n_cells,T), query_matrix (n_cells,T,Q))."""
    by_cell: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in panel:
        by_cell[int(entry["coverage_cell"])].append(entry)
    cells = sorted(by_cell)
    sizes = {len(by_cell[cell]) for cell in cells}
    _require(len(sizes) == 1, "bootstrap requires equal twins per coverage cell")
    per_cell = sizes.pop()
    query_counts = {
        len(query_index_by_twin[int(entry["twin_id"])]) for entry in panel
    }
    _require(len(query_counts) == 1, "bootstrap requires equal queries per twin")
    per_twin = query_counts.pop()
    mass = np.empty((len(cells), per_cell), dtype=np.float64)
    queries = np.empty((len(cells), per_cell, per_twin), dtype=np.int64)
    for row, cell in enumerate(cells):
        entries = sorted(by_cell[cell], key=lambda item: int(item["twin_id"]))
        for column, entry in enumerate(entries):
            mass[row, column] = float(entry["realized_pi_ms50"])
            queries[row, column] = np.asarray(
                query_index_by_twin[int(entry["twin_id"])], dtype=np.int64
            )
    return np.asarray(cells, dtype=np.int64), mass, queries


def _rho_from_draws(
    draws: np.ndarray,
    mass: np.ndarray,
    queries: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Vectorised equal-cell weighted rho for a stack of twin draws."""
    n_cells = mass.shape[0]
    cell_axis = np.arange(n_cells, dtype=np.int64)[None, :, None]
    selected_mass = mass[cell_axis, draws]                       # (Bn, cells, T)
    cell_total = selected_mass.sum(axis=2, keepdims=True)
    _require(bool(np.all(cell_total > 0.0)), "bootstrap cell mass is zero")
    share = selected_mass / cell_total / float(n_cells)
    selected_queries = queries[cell_axis, draws]                 # (Bn, cells, T, Q)
    weights = share[..., None] / float(queries.shape[2])
    numerator = np.sum(weights * C[selected_queries], axis=(1, 2, 3))
    background = np.sum(weights * B[selected_queries], axis=(1, 2, 3))
    denominator = numerator + background
    _require(bool(np.all(denominator > 0.0)), "bootstrap rho denominator is zero")
    return numerator / denominator


def stratified_cluster_bootstrap_differences(
    panel: Sequence[dict[str, Any]],
    query_index_by_twin: dict[int, list[int]],
    C: np.ndarray,
    backgrounds: dict[int, np.ndarray],
    *,
    ks: Sequence[int] = KS,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = CI_LEVEL,
) -> dict[str, Any]:
    """D1-D0 rho differences with marginal and simultaneous max-statistic bands."""
    _, mass, queries = _draw_matrices(panel, query_index_by_twin)
    n_cells, per_cell = mass.shape
    uniform = np.ones_like(mass)
    C = np.asarray(C, dtype=np.float64)

    identity = np.tile(
        np.arange(per_cell, dtype=np.int64)[None, None, :], (1, n_cells, 1)
    )
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, per_cell, size=(int(n_resamples), n_cells, per_cell))

    observed: list[float] = []
    replicates: list[np.ndarray] = []
    for k in ks:
        B = np.asarray(backgrounds[int(k)], dtype=np.float64)
        d0_obs = _rho_from_draws(identity, uniform, queries, C, B)[0]
        d1_obs = _rho_from_draws(identity, mass, queries, C, B)[0]
        observed.append(float(d1_obs - d0_obs))
        d0 = _rho_from_draws(draws, uniform, queries, C, B)
        d1 = _rho_from_draws(draws, mass, queries, C, B)
        replicates.append(d1 - d0)

    observed_array = np.asarray(observed, dtype=np.float64)
    replicate_array = np.stack(replicates, axis=1)               # (Bn, K)
    standard_error = replicate_array.std(axis=0, ddof=1)
    _require(
        bool(np.all(standard_error > 0.0)),
        "degenerate bootstrap: zero standard error for a rho difference",
    )
    max_statistic = np.max(
        np.abs(replicate_array - observed_array[None, :]) / standard_error[None, :],
        axis=1,
    )
    ordered = np.sort(max_statistic)
    index = min(len(ordered) - 1, int(math.ceil(ci_level * len(ordered))) - 1)
    critical = float(ordered[max(0, index)])

    alpha = 1.0 - ci_level
    lo_index = max(0, int(math.floor(alpha / 2 * n_resamples)))
    hi_index = min(n_resamples - 1, int(math.ceil((1 - alpha / 2) * n_resamples)))

    per_k: dict[str, Any] = {}
    for position, k in enumerate(ks):
        column = np.sort(replicate_array[:, position])
        per_k[str(int(k))] = {
            "rho_difference_d1_minus_d0": float(observed_array[position]),
            "bootstrap_standard_error": float(standard_error[position]),
            "marginal_percentile_ci_lo": float(column[lo_index]),
            "marginal_percentile_ci_hi": float(column[hi_index]),
            "simultaneous_ci_lo": float(
                observed_array[position] - critical * standard_error[position]
            ),
            "simultaneous_ci_hi": float(
                observed_array[position] + critical * standard_error[position]
            ),
        }
    return {
        "method": "stratified_within_coverage_cell_twin_cluster_bootstrap",
        "simultaneity": "studentized_max_statistic_over_the_three_k_differences",
        "ci_level": float(ci_level),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "max_statistic_critical_value": critical,
        "per_k": per_k,
    }


# ---------------------------------------------------------------------------
# Panel measurement from frozen Training Lance
# ---------------------------------------------------------------------------

def measure_panel(
    panel: Sequence[dict[str, Any]],
    *,
    train_lance: Path,
    pixel_column: str,
    image_shape: Sequence[int],
    query_step: int = QUERY_STEP,
    future_step: int = FUTURE_STEP,
) -> dict[str, Any]:
    """Decode only the required query/future rows and build C, means, descriptors."""
    pair_ids: list[str] = []
    for entry in panel:
        pair_ids.extend(
            entry["pair_ids"] if isinstance(entry["pair_ids"], list) else []
        )
    steps = sorted({int(query_step), int(future_step)})
    records = _read_selected_push_rows(
        train_lance=Path(train_lance),
        pair_ids=sorted(set(pair_ids)),
        steps=steps,
        pixel_column=str(pixel_column),
    )
    weights = np.asarray(CONDITION_WEIGHTS, dtype=np.float64)
    shape = tuple(int(v) for v in image_shape)

    conditional: list[float] = []
    means: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    twin_ids: list[int] = []
    query_ids: list[str] = []
    query_index_by_twin: dict[int, list[int]] = defaultdict(list)
    rows_returned = 0
    pixel_frames_decoded = 0
    query_action_abs_max = 0.0
    query_pixel_mismatch_max = 0.0

    for entry in panel:
        twin_id = int(entry["twin_id"])
        for direction in DIRECTIONS:
            pair_id = f"pmd-train-{entry['pair_index_by_direction'][direction]:06d}-{direction}"
            by_mode = records.get(pair_id, {})
            _require(set(by_mode) == set(MODES), f"incomplete modes for {pair_id}")
            for mode in MODES:
                _require(
                    set(by_mode[mode]) == set(steps),
                    f"incomplete steps for {pair_id}/{mode}",
                )
                rows_returned += len(by_mode[mode])
            query_pixel_mismatch_max = max(
                query_pixel_mismatch_max,
                float(
                    by_mode[MODES[0]][query_step]["pixels"]
                    != by_mode[MODES[1]][query_step]["pixels"]
                ),
            )
            _require(
                query_pixel_mismatch_max == 0.0,
                f"query pixels differ across modes for {pair_id}",
            )
            x_pixel = decode_pixel_blob(by_mode[MODES[0]][query_step]["pixels"], shape)
            futures = np.stack(
                [
                    decode_pixel_blob(by_mode[mode][future_step]["pixels"], shape)
                    for mode in MODES
                ]
            )
            pixel_frames_decoded += 1 + len(MODES)
            conditional.append(
                float(pixel_conditional_variance(futures[None], x_pixel[None], weights)[0])
            )
            means.append(
                pixel_mean_displacement(futures[None], x_pixel[None], weights)[0].astype(
                    np.float32
                )
            )
            query = by_mode[MODES[0]][query_step]["physics"]
            goal = by_mode[MODES[0]][query_step]["goal"]
            action = np.asarray(
                by_mode[MODES[0]][query_step]["action"], dtype=np.float64
            )
            reverse_action = np.asarray(
                by_mode[MODES[1]][query_step]["action"], dtype=np.float64
            )
            _require(
                float(np.max(np.abs(action - reverse_action))) <= 1.0e-9,
                f"query actions differ across modes for {pair_id}",
            )
            query_action_abs_max = max(
                query_action_abs_max, float(np.max(np.abs(action)))
            )
            x_phys = np.asarray(query[6:8], dtype=np.float64)
            theta = float(query[10])
            goal_relative = np.asarray(goal[2:4], dtype=np.float64) - x_phys
            descriptors.append(
                np.concatenate(
                    [
                        x_phys,
                        [float(query[8]), float(query[9]), math.sin(theta), math.cos(theta)],
                        goal_relative,
                        action,
                    ]
                )
            )
            query_index_by_twin[twin_id].append(len(conditional) - 1)
            twin_ids.append(twin_id)
            query_ids.append(pair_id)

    C = np.asarray(conditional, dtype=np.float64)
    _require(bool(np.all(np.isfinite(C))), "C_pixel contains non-finite values")
    return {
        "C": C,
        "means": np.stack(means),
        "descriptors": np.stack(descriptors),
        "twin_ids": np.asarray(twin_ids, dtype=np.int64),
        "query_ids": query_ids,
        "query_index_by_twin": {k: list(v) for k, v in query_index_by_twin.items()},
        "read_counts": {
            "pair_ids_requested": len(set(pair_ids)),
            "lance_rows_requested": len(set(pair_ids)) * len(MODES) * len(steps),
            "lance_rows_returned": rows_returned,
            "steps_read": steps,
            "pixel_frames_decoded": pixel_frames_decoded,
        },
        "query_action_abs_max": query_action_abs_max,
        "query_pixel_mismatch_max": query_pixel_mismatch_max,
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "q00": float(np.min(v)),
        "q50": float(np.median(v)),
        "q100": float(np.max(v)),
        "n": int(len(v)),
    }


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def assert_no_nan(obj: Any, path: str = "$") -> None:
    """Refuse to emit any non-finite number anywhere in the artifact."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert_no_nan(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            assert_no_nan(value, f"{path}[{index}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, float):
        _require(math.isfinite(obj), f"non-finite value at {path}")
    elif isinstance(obj, (np.floating, np.integer)):
        _require(math.isfinite(float(obj)), f"non-finite value at {path}")


def write_exclusive_json(path: Path, payload: Any) -> None:
    """Write once; refuse overwrite and refuse NaN/Inf."""
    assert_no_nan(payload)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def evidence_receipt() -> dict[str, Any]:
    return {
        "development_split_opened": DEVELOPMENT_SPLIT_OPENED,
        "public_or_test_split_opened": PUBLIC_SPLIT_OPENED,
        "context_world_modified": CONTEXT_WORLD_MODIFIED,
        "model_loaded": MODEL_LOADED,
        "optimizer_steps": OPTIMIZER_STEPS,
        "gradients_computed": False,
        "reads": "frozen_training_lance_query_and_future_rows_only",
        "pixel_source": "lossless_training_png_decoded_to_224x224_rgb_float64_0_1",
        "decoder_and_math": "reused_verbatim_from_frozen_raw_pixel_auditor",
        "claim_scope": CLAIM_SCOPE,
        "d2_pass_fail_claim": D2_PASS_FAIL_CLAIM,
        "latent_status": LATENT_STATUS,
        "binding_cause_status": BINDING_CAUSE_STATUS,
        "pixel_units": PIXEL_UNITS,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D0/D1 raw-pixel baseline adapter for pusht_motion_damping P1b"
    )
    parser.add_argument("--addendum", required=True, type=Path)
    parser.add_argument("--pixel-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true",
                        help="Verify hashes and build the panel; decode no pixels")
    args = parser.parse_args(argv)

    root = repo_root()
    addendum_path = Path(args.addendum).resolve()
    config_path = Path(args.pixel_config).resolve()
    multiplicity_path = (root / D1_MULTIPLICITY_REL).resolve()
    projected_path = (root / D1_PROJECTED_WEIGHTS_REL).resolve()
    checked_inputs = assert_training_only_inputs(
        [addendum_path, config_path, multiplicity_path, projected_path]
    )

    addendum = load_config(addendum_path)
    expected = addendum_expected_hashes(addendum)
    observed_hashes = verify_input_hashes(root, expected)
    observed_hashes["addendum"] = file_sha256(addendum_path)
    _require(
        file_sha256(config_path) == observed_hashes["pixel_config"],
        "--pixel-config does not match the frozen addendum identity",
    )
    observed_hashes["builder_script"] = file_sha256(Path(__file__).resolve())

    catalog = dict(addendum.get("catalog_identity", {}))
    panel_seed = int(catalog.get("baseline_pixel_panel_seed", PANEL_SEED))
    bootstrap_seed = int(catalog.get("bootstrap_seed", BOOTSTRAP_SEED))
    _require(panel_seed == PANEL_SEED, "addendum panel seed drifted from frozen constant")
    _require(bootstrap_seed == BOOTSTRAP_SEED, "addendum bootstrap seed drifted")

    multiplicity_rows = read_jsonl(multiplicity_path)
    projected_rows = read_jsonl(projected_path)
    _require(
        len(projected_rows) == len(multiplicity_rows),
        "projected_weights and multiplicity row counts differ",
    )
    for left, right in zip(projected_rows, multiplicity_rows):
        _require(
            int(left["twin_id"]) == int(right["twin_id"])
            and int(left["coverage_cell"]) == int(right["coverage_cell"])
            and list(left["pair_ids"]) == list(right["pair_ids"]),
            f"D1 twin identity mismatch at twin {right['twin_id']}",
        )

    panel = select_panel(multiplicity_rows, seed=panel_seed)
    identity = {
        "panel_seed": panel_seed,
        "twins_per_coverage_cell": TWINS_PER_CELL,
        "coverage_cells": N_COVERAGE_CELLS,
        "n_twins": len(panel),
        "n_queries": len(panel) * len(DIRECTIONS),
        "directions": list(DIRECTIONS),
        "panel_sha256": panel_hash(panel),
        "selection_rule": "sha256(seed:coverage_cell:twin_id) ascending, twin_id tie-break",
        "selection_uses_outcomes": False,
    }

    output_dir = Path(args.output_dir).resolve()
    exclusive_mkdir(output_dir)

    common = {
        "schema_version": 1,
        "adapter_id": "pusht_motion_damping_d2_pixel_baseline_v1",
        "addendum_id": str(addendum.get("addendum_id", "")),
        "panel_identity": identity,
        "input_sha256": observed_hashes,
        "expected_input_sha256": {k: v[1] for k, v in expected.items()},
        "training_only_checked_paths": checked_inputs,
        "evidence_receipt": evidence_receipt(),
    }

    with (output_dir / "panel.jsonl").open("x", encoding="utf-8") as handle:
        for entry in panel:
            handle.write(
                json.dumps(
                    {
                        "coverage_cell": entry["coverage_cell"],
                        "twin_id": entry["twin_id"],
                        "pair_ids": entry["pair_ids"],
                        "realized_pi_ms50": entry["realized_pi_ms50"],
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )

    if args.check_only:
        payload = dict(common, status="check_ok", pixels_decoded=False)
        write_exclusive_json(output_dir / "pixel_baseline_v1.json", payload)
        print(json.dumps({"status": "check_ok", "output_dir": str(output_dir)}))
        return 0

    task_cfg = dict(load_config(config_path)["tasks"]["motion"])
    train_lance = _resolve(root, task_cfg["train_lance"])
    training_only_guard(train_lance)
    _require(train_lance.exists(), f"frozen Training Lance missing: {train_lance}")

    measured = measure_panel(
        panel,
        train_lance=train_lance,
        pixel_column=str(task_cfg.get("pixel_column", "pixels")),
        image_shape=task_cfg.get("image_shape", [224, 224, 3]),
        query_step=int(task_cfg.get("query_step", QUERY_STEP)),
        future_step=int(task_cfg.get("future_step", FUTURE_STEP)),
    )

    neighbors = build_neighbor_graph(measured["descriptors"], measured["twin_ids"])
    backgrounds = background_variance_by_k(measured["means"], neighbors, KS)
    weightings = panel_weightings(panel, measured["query_index_by_twin"])

    estimates: dict[str, Any] = {}
    for name, (indices, weights) in weightings.items():
        estimates[name] = {
            "C_pixel_weighted_mean": float(np.dot(weights, measured["C"][indices])),
            "C_pixel_unweighted": _summary(measured["C"]),
            "by_k": {
                str(k): dict(
                    rho_components(measured["C"], backgrounds[k], indices, weights),
                    B_pixel_unweighted=_summary(backgrounds[k]),
                )
                for k in KS
            },
        }

    differences = stratified_cluster_bootstrap_differences(
        panel,
        measured["query_index_by_twin"],
        measured["C"],
        backgrounds,
        seed=bootstrap_seed,
    )

    payload = dict(
        common,
        status="ok",
        pixels_decoded=True,
        ks=list(KS),
        neighbor_graph={
            "shared_by_d0_and_d1": True,
            "exclusion": "leave_whole_twin_out",
            "max_k": int(max(KS)),
            "descriptor": (
                "frozen PushT physical query descriptor "
                "(block position, block velocity, sin/cos theta, goal-relative) "
                "with the exact query action appended"
            ),
        },
        weighting={
            "d0_uniform": "equal mass per coverage cell, uniform over twins and directions",
            "d1_ms50_realized": (
                "equal mass per coverage cell, within-cell mass proportional to frozen "
                "realized_pi_ms50, split equally across the twin's two directions"
            ),
        },
        estimates=estimates,
        rho_differences_d1_minus_d0=differences,
        read_counts=measured["read_counts"],
        query_action=
        {
            "included_in_descriptor": True,
            "abs_max_observed": measured["query_action_abs_max"],
            "is_exactly_zero": measured["query_action_abs_max"] == 0.0,
        },
        query_pixel_mismatch_max=measured["query_pixel_mismatch_max"],
    )
    write_exclusive_json(output_dir / "pixel_baseline_v1.json", payload)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "panel_sha256": identity["panel_sha256"],
                "n_queries": identity["n_queries"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
