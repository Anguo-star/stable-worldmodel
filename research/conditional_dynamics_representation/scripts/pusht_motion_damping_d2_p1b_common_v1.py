#!/usr/bin/env python3
"""Pure pre-outcome primitives shared by the Motion D2 P1b pilot scripts.

Everything in this module is deterministic, Training-only and rollout-free.  It
freezes the identities, the D0 coverage cut points, the action construction and
the candidate/quota plan declared by
``pusht_motion_damping_d2_v1_pre_p1b_execution_addendum_v1.yaml``.

Structural admission (history/query contact, canonical query match, inherited
separation, playfield bounds) needs a simulator rollout and is deliberately NOT
implemented here: the design artifact commits the ordered candidate stream and
the quota algorithm, never a claim that a candidate already passed contact.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
RESEARCH_ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"

ADDENDUM_PATH = RESEARCH_ROOT / (
    "configs/pusht_motion_damping_d2_v1_pre_p1b_execution_addendum_v1.yaml"
)

SCHEMA_VERSION = 1
DESIGN_ID = "pusht_motion_damping_d2_p1b_design_v1"

# --- addendum: parent_identity -------------------------------------------------
PARENT_IDENTITY: dict[str, tuple[str, str]] = {
    "construction_plan": (
        "research/conditional_dynamics_representation/D2_CONSTRUCTION_PLAN_ZH.md",
        "d3cd799bf6a964e90958adf733dd3f9e63ec7735e49179ac74fd4ff82303a450",
    ),
    "preregistration": (
        "research/conditional_dynamics_representation/configs/"
        "pusht_motion_damping_d2_preregistration_v1.yaml",
        "bfb01960427125d974614dd7eb24379030a662f52353bda923cedc00f33c7937",
    ),
    "p1a_script": (
        "research/conditional_dynamics_representation/scripts/"
        "probe_pusht_motion_damping_d2_p1a_v1.py",
        "ca9a9fc4c94d3d3055dce53d7d00fe7aa9365856c19a3af7ed9bdc3ff8b1f97e",
    ),
    "p1a_result": (
        "research/conditional_dynamics_representation/artifacts/"
        "pusht_motion_damping_d2_p1a_v1/training_only_cpu_probe_v2.json",
        "e2453321c4b76f43ffc65a287d7df08094396be114f927bbb5793d0f812dbd42",
    ),
}

# --- addendum: pixel_baseline_resolution.required_identity ---------------------
# Only the entries the addendum binds to an explicit path can be verified here.
PIXEL_IDENTITY_WITH_PATH: dict[str, tuple[str, str]] = {
    "projected_weights": (
        "research/conditional_dynamics_representation/artifacts/"
        "pusht_motion_damping_d1_multiscale_soft_v2/d1_0_training_only_v2/"
        "projected_weights.jsonl",
        "6a45c6f18e1eeb61f184c9977b81b08f4de384ee4c9c36cfa41e186dca755afa",
    ),
    "pixel_config": (
        "research/conditional_dynamics_representation/configs/"
        "icl_training_raw_pixel_visibility_v1.yaml",
        "2e06b3f06a24414091bd1de44433a92e083ffdabbbde54ea7d8db78e9d38059d",
    ),
    "motion_training_manifest": (
        "../ContextWorld/artifacts/synthesis/"
        "pusht_motion_damping_h3_release_v4/manifest.json",
        "48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b",
    ),
    "raw_pixel_auditor": (
        "research/conditional_dynamics_representation/scripts/"
        "audit_icl_training_raw_pixel_visibility_v1.py",
        "7bdfb5552a3b4a0fea650bf642d064256ec61ef95acd8684651365a2133db50a",
    ),
    "prior_pixel_result": (
        "research/conditional_dynamics_representation/artifacts/"
        "icl_training_raw_pixel_visibility_v1/training_only_v1/per_task.jsonl",
        "6f54214e1966e2888fd20d0728587bdcbdf5840d74fdb03a862b5c91cd1db7b1",
    ),
    "d1_multiplicity": (
        "research/conditional_dynamics_representation/artifacts/"
        "pusht_motion_damping_d1_schedule_v1/d1_ms50_schedule_v1_final/"
        "multiplicity.jsonl",
        "4be57c44b5e9485902edabdfbfb1c629b4bf433ed375ab00c783ae0ed187abb8",
    ),
}
PIXEL_IDENTITY_UNBOUND: dict[str, str] = {}

PROJECTED_WEIGHTS_PATH = REPO_ROOT / PIXEL_IDENTITY_WITH_PATH["projected_weights"][0]

# --- addendum: catalog_identity ------------------------------------------------
PILOT_CATALOG_SEED = 2026090401
CATALOG_SPLIT_TAG = "train"
CANDIDATE_WINDOWS: dict[str, tuple[int, int]] = {
    "calibration": (0, 8191),
    "sealed_holdout": (8192, 16383),
}
SPLITS = tuple(CANDIDATE_WINDOWS)
IDENTITY_PREFIX = "pmd-d2-v1"
SPLIT_IDENTITY_TOKEN = {"calibration": "cal", "sealed_holdout": "seal"}

# --- addendum: p1b_design ------------------------------------------------------
GROUPS_PER_SPLIT = 192
COVERAGE_CELLS = 64
COVERAGE_BINS = 4
GROUPS_PER_CELL_PER_SPLIT = 3
DIRECTIONS = ("forward", "reverse")
STRATUM_IDS = ("low_approach", "mid_approach", "mid_tangent_assisted")
PLANNER_SCALE_BY_STRATUM = {
    "low_approach": 0.25,
    "mid_approach": 0.625,
    "mid_tangent_assisted": 0.625,
}
TANGENT_MAGNITUDE_DEGREES = 15.0
AUDIT_ONLY_ACTION_ROTATION_DEGREES = (-2.0, 2.0)
AUDIT_ONLY_AGENT_TANGENT_SHIFT_PX = (-1.0, 1.0)
ADMISSION_MAY_USE_ONLY = (
    "coverage_quota",
    "action_norm_and_box",
    "history_contact_free",
    "canonical_query_match",
    "query_start_clear",
    "both_conditions_contact",
    "equal_first_contact_raw_step",
    "inherited_history_and_future_separation",
    "playfield_bounds",
)
ADMISSION_MUST_NOT_USE = (
    "gamma_magnitude",
    "rho",
    "probe_score",
    "model_or_gradient",
    "development",
    "public_test",
)
# Pre-outcome gates this module can decide without any rollout.
PRE_OUTCOME_ADMISSION_KEYS = ("action_norm_and_box",)
ROLLOUT_REQUIRED_ADMISSION_KEYS = tuple(
    key
    for key in ADMISSION_MAY_USE_ONLY
    if key not in PRE_OUTCOME_ADMISSION_KEYS and key != "coverage_quota"
)

QUERY_RAW_STEPS = 5
PLANNER_DENOMINATOR = 100.0
NORM_ATOL = 1.0e-12

# --- forbidden split guard -----------------------------------------------------
FORBIDDEN_PATH_SUBSTRINGS = (
    "loader_validation",
    "public_test",
    "public-test",
    "development",
)
FORBIDDEN_PATH_PARTS = (
    "validation",
    "validation.lance",
    "loader_validation",
    "loader_validation.lance",
    "development",
    "development.lance",
    "public_test",
    "public_test.lance",
    "test.lance",
    "dev.lance",
)


class ForbiddenSplitError(RuntimeError):
    """Raised when a path would touch Development or Public Test evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def reject_forbidden_path(path: str | Path, *, name: str = "path") -> Path:
    """Reject any path token that could open Development/Public Test data."""

    resolved = Path(path).expanduser()
    text = resolved.as_posix().casefold()
    for token in FORBIDDEN_PATH_SUBSTRINGS:
        if token in text:
            raise ForbiddenSplitError(
                f"{name} contains forbidden split token {token!r}: {resolved}"
            )
    for part in resolved.parts:
        if part.casefold() in FORBIDDEN_PATH_PARTS:
            raise ForbiddenSplitError(
                f"{name} contains forbidden split component {part!r}: {resolved}"
            )
    return resolved


# --- hashing / json ------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic JSON text.  NaN/Infinity are rejected, never emitted."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_json_exclusive(path: Path, payload: Any) -> str:
    """Write pretty deterministic JSON, refusing to overwrite or append."""

    path = reject_forbidden_path(path, name="output").resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- declared identities -------------------------------------------------------
def verify_declared_identities(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify every SHA256 identity the addendum binds to a concrete path."""

    verified: dict[str, dict[str, str]] = {}
    for group, table in (
        ("parent_identity", PARENT_IDENTITY),
        ("pixel_baseline_resolution", PIXEL_IDENTITY_WITH_PATH),
    ):
        for key, (relative, expected) in table.items():
            path = reject_forbidden_path(repo_root / relative, name=key)
            require(path.is_file(), f"{group}.{key} is missing: {relative}")
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(
                    f"{group}.{key} SHA256 changed: {relative} "
                    f"expected={expected} observed={observed}"
                )
            verified[f"{group}.{key}"] = {"path": relative, "sha256": observed}
    addendum = reject_forbidden_path(ADDENDUM_PATH, name="addendum")
    require(addendum.is_file(), "addendum is missing")
    return {
        "addendum_id": "pusht_motion_damping_d2_v1_pre_p1b_execution_addendum_v1",
        "addendum_path": str(addendum.relative_to(repo_root)),
        "addendum_sha256": sha256_file(addendum),
        "verified": verified,
        "declared_without_path_not_verifiable_here": dict(PIXEL_IDENTITY_UNBOUND),
        "common_module_sha256": sha256_file(THIS_SOURCE),
    }


# --- D0 coverage cut points ----------------------------------------------------
D0_ROW_FIELDS = (
    "twin_id",
    "coverage_cell",
    "orientation_bin",
    "speed_bin",
    "goal_distance_bin",
    "query_speed",
    "goal_distance",
)


def load_d0_coverage_rows(path: Path = PROJECTED_WEIGHTS_PATH) -> list[dict[str, Any]]:
    """Read the frozen full D0 projected-weights JSONL coverage metadata."""

    path = reject_forbidden_path(path, name="projected_weights").resolve()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            row = {field: raw[field] for field in D0_ROW_FIELDS}
            require(
                math.isfinite(float(row["query_speed"]))
                and math.isfinite(float(row["goal_distance"])),
                "D0 coverage row has a non-finite descriptor",
            )
            rows.append(row)
    require(bool(rows), "D0 projected weights file is empty")
    twin_ids = [int(row["twin_id"]) for row in rows]
    require(
        twin_ids == list(range(len(rows))),
        "D0 projected weights must be the full contiguous twin stream",
    )
    for row in rows:
        cell = (
            int(row["orientation_bin"]) * COVERAGE_BINS * COVERAGE_BINS
            + int(row["speed_bin"]) * COVERAGE_BINS
            + int(row["goal_distance_bin"])
        )
        require(cell == int(row["coverage_cell"]), "D0 coverage cell is inconsistent")
    return rows


def _cut_points(lower: Sequence[float], upper: Sequence[float], label: str) -> float:
    top = max(lower)
    bottom = min(upper)
    require(top < bottom, f"D0 {label} bins are not separable")
    return 0.5 * (top + bottom)


def infer_d0_cut_points(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Freeze numeric cut points reproducing the D0 64-cell assignment.

    The D0 audit used equal-frequency bins.  P1b must never recompute
    equal-frequency bins on its own outcomes, so the boundaries are turned into
    fixed numeric thresholds here and reused verbatim for new candidates.
    """

    rows = list(rows)
    orientations = sorted({int(row["orientation_bin"]) for row in rows})
    require(
        orientations == list(range(COVERAGE_BINS)),
        "D0 orientation bins must be contiguous 0..3",
    )
    speed_cuts: dict[str, list[float]] = {}
    goal_cuts: dict[str, list[float]] = {}
    for orient in orientations:
        by_speed_bin = {
            index: [
                float(row["query_speed"])
                for row in rows
                if int(row["orientation_bin"]) == orient
                and int(row["speed_bin"]) == index
            ]
            for index in range(COVERAGE_BINS)
        }
        for index, values in by_speed_bin.items():
            require(bool(values), f"D0 speed bin {orient}/{index} is empty")
        speed_cuts[str(orient)] = [
            _cut_points(by_speed_bin[index], by_speed_bin[index + 1], "speed")
            for index in range(COVERAGE_BINS - 1)
        ]
        for speed_bin in range(COVERAGE_BINS):
            by_goal_bin = {
                index: [
                    float(row["goal_distance"])
                    for row in rows
                    if int(row["orientation_bin"]) == orient
                    and int(row["speed_bin"]) == speed_bin
                    and int(row["goal_distance_bin"]) == index
                ]
                for index in range(COVERAGE_BINS)
            }
            for index, values in by_goal_bin.items():
                require(
                    bool(values),
                    f"D0 goal bin {orient}/{speed_bin}/{index} is empty",
                )
            goal_cuts[f"{orient}-{speed_bin}"] = [
                _cut_points(by_goal_bin[index], by_goal_bin[index + 1], "goal")
                for index in range(COVERAGE_BINS - 1)
            ]
    cuts = {
        "reference": "frozen_d0_64_cells",
        "bins": COVERAGE_BINS,
        "cells": COVERAGE_CELLS,
        "cell_formula": "orientation_bin*16 + speed_bin*4 + goal_distance_bin",
        "rule": "fixed_numeric_thresholds_inferred_once_never_refit_on_p1b",
        "speed_cut_points_by_orientation": speed_cuts,
        "goal_distance_cut_points_by_orientation_and_speed_bin": goal_cuts,
        "source_twin_count": len(rows),
    }
    cuts["sha256"] = canonical_sha256(cuts)
    return cuts


def _bin_index(value: float, cut_points: Sequence[float]) -> int:
    index = 0
    for cut in cut_points:
        if float(value) > float(cut):
            index += 1
        else:
            break
    return index


def assign_d0_cell(
    *,
    orientation_bin: int,
    query_speed: float,
    goal_distance: float,
    cut_points: dict[str, Any],
) -> dict[str, int]:
    """Assign one candidate to a frozen D0 cell using the frozen cut points."""

    orientation_bin = int(orientation_bin)
    require(
        0 <= orientation_bin < COVERAGE_BINS,
        f"orientation bin out of range: {orientation_bin}",
    )
    require(
        math.isfinite(float(query_speed)) and math.isfinite(float(goal_distance)),
        "coverage descriptors must be finite",
    )
    speed_bin = _bin_index(
        query_speed, cut_points["speed_cut_points_by_orientation"][str(orientation_bin)]
    )
    goal_bin = _bin_index(
        goal_distance,
        cut_points["goal_distance_cut_points_by_orientation_and_speed_bin"][
            f"{orientation_bin}-{speed_bin}"
        ],
    )
    return {
        "orientation_bin": orientation_bin,
        "speed_bin": speed_bin,
        "goal_distance_bin": goal_bin,
        "coverage_cell": orientation_bin * COVERAGE_BINS * COVERAGE_BINS
        + speed_bin * COVERAGE_BINS
        + goal_bin,
    }


def cut_points_reproduce_d0(
    rows: Iterable[dict[str, Any]], cut_points: dict[str, Any]
) -> bool:
    """Return True only when the cut points reproduce every D0 assignment."""

    for row in rows:
        assigned = assign_d0_cell(
            orientation_bin=int(row["orientation_bin"]),
            query_speed=float(row["query_speed"]),
            goal_distance=float(row["goal_distance"]),
            cut_points=cut_points,
        )
        if assigned["coverage_cell"] != int(row["coverage_cell"]):
            return False
    return True


def orientation_bin_from_block_angle(block_angle: float) -> int:
    """Recover the D0 orientation bin from geometry alone."""

    quarter = 0.5 * math.pi
    return int(round(float(block_angle) / quarter)) % COVERAGE_BINS


# --- frozen action construction ------------------------------------------------
def planner_base_vector(query_snapshot: Sequence[float]) -> np.ndarray:
    """clip((block_position + block_velocity - agent_position)/100, -1, 1)."""

    query = np.asarray(query_snapshot, dtype=np.float64)
    require(query.shape == (12,), "query snapshot must be 12-D")
    require(bool(np.all(np.isfinite(query))), "query snapshot must be finite")
    return np.clip(
        (query[6:8] + query[8:10] - query[0:2]) / PLANNER_DENOMINATOR, -1.0, 1.0
    )


def rotate_preserving_norm(vector: Sequence[float], degrees: float) -> np.ndarray:
    """Rotate a planar vector by ``degrees`` without changing its norm."""

    vector = np.asarray(vector, dtype=np.float64)
    require(vector.shape == (2,), "rotation expects a planar vector")
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated = np.asarray(
        [
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ],
        dtype=np.float64,
    )
    before = float(np.linalg.norm(vector))
    after = float(np.linalg.norm(rotated))
    require(
        abs(before - after) <= max(NORM_ATOL, NORM_ATOL * before),
        "tangent rotation changed the planner-vector norm",
    )
    return rotated


def tangent_angle_degrees(*, stratum_id: str, coverage_cell: int, split: str) -> float:
    """Frozen tangent angle: cell parity, with the opposite sign on holdout."""

    require(stratum_id in STRATUM_IDS, f"unknown action stratum {stratum_id!r}")
    require(split in CANDIDATE_WINDOWS, f"unknown split {split!r}")
    if stratum_id != "mid_tangent_assisted":
        return 0.0
    signed = (
        TANGENT_MAGNITUDE_DEGREES
        if int(coverage_cell) % 2 == 0
        else -TANGENT_MAGNITUDE_DEGREES
    )
    return signed if split == "calibration" else -signed


def build_query_actions(
    query_snapshot: Sequence[float],
    *,
    stratum_id: str,
    coverage_cell: int,
    split: str,
    extra_rotation_degrees: float = 0.0,
) -> np.ndarray:
    """Five identical raw actions from the frozen planner/rotation/scale rule."""

    base = planner_base_vector(query_snapshot)
    degrees = (
        tangent_angle_degrees(
            stratum_id=stratum_id, coverage_cell=coverage_cell, split=split
        )
        + float(extra_rotation_degrees)
    )
    rotated = rotate_preserving_norm(base, degrees)
    scaled = float(PLANNER_SCALE_BY_STRATUM[stratum_id]) * rotated
    return np.repeat(scaled[None, :], QUERY_RAW_STEPS, axis=0)


def action_norm_and_box(actions: np.ndarray) -> dict[str, Any]:
    """The only structural admission gate decidable before any rollout."""

    actions = np.asarray(actions, dtype=np.float64)
    require(actions.shape == (QUERY_RAW_STEPS, 2), "query actions must be 5x2")
    norms = np.linalg.norm(actions, axis=1)
    identical = bool(np.all(actions == actions[0]))
    inside_box = bool(np.all(np.abs(actions) <= 1.0))
    inside_ball = bool(np.all(norms <= 1.0 + NORM_ATOL))
    return {
        "per_step_l2_norm": [float(value) for value in norms],
        "maximum_l2_norm": float(norms.max(initial=0.0)),
        "five_identical_raw_actions": identical,
        "inside_component_box": inside_box,
        "inside_l2_unit_ball": inside_ball,
        "no_clipping_required": inside_box,
        "passed": bool(identical and inside_box and inside_ball),
    }


def local_tangent_unit(query_snapshot: Sequence[float]) -> list[float]:
    """Unit tangent used only by the addendum's audit-only geometry shift."""

    base = planner_base_vector(query_snapshot)
    norm = float(np.linalg.norm(base))
    require(norm > 0.0, "planner vector is degenerate; no local tangent exists")
    unit = base / norm
    return [float(-unit[1]), float(unit[0])]


# --- seed-bearing identities ---------------------------------------------------
def seed_bearing_template_id(*, split: str, group_index: int, direction: str) -> str:
    require(split in SPLIT_IDENTITY_TOKEN, f"unknown split {split!r}")
    require(direction in DIRECTIONS, f"unknown direction {direction!r}")
    require(int(group_index) >= 0, "group index must be non-negative")
    return (
        f"{IDENTITY_PREFIX}-{PILOT_CATALOG_SEED}-"
        f"{SPLIT_IDENTITY_TOKEN[split]}-{int(group_index):06d}-{direction}"
    )


def rename_to_seed_bearing_identity(template: Any, template_id: str) -> Any:
    """Replace a ContextWorld catalog id with the frozen pmd-d2-v1 identity."""

    require(
        str(template_id).startswith(f"{IDENTITY_PREFIX}-{PILOT_CATALOG_SEED}-"),
        f"refusing a non seed-bearing identity: {template_id!r}",
    )
    return replace(template, template_id=str(template_id))


def candidate_window(split: str) -> tuple[int, int]:
    require(split in CANDIDATE_WINDOWS, f"unknown split {split!r}")
    return CANDIDATE_WINDOWS[split]


def windows_are_disjoint() -> bool:
    (low_a, high_a) = CANDIDATE_WINDOWS["calibration"]
    (low_b, high_b) = CANDIDATE_WINDOWS["sealed_holdout"]
    return high_a < low_b or high_b < low_a


# --- quota plan ----------------------------------------------------------------
def stratum_for_rank(rank: int) -> str:
    """Frozen action stratum for one of the three per-cell quota slots."""

    require(
        0 <= int(rank) < GROUPS_PER_CELL_PER_SPLIT,
        f"rank out of the frozen per-cell quota: {rank}",
    )
    return STRATUM_IDS[int(rank)]


def stratum_for_encounter(encounter_rank: int) -> str:
    """Assign a candidate's action before rollout, cycling within its cell.

    Contact admission depends on the action, so a stratum may never be assigned
    after observing admission.  The candidate encounter rank is determined only
    by the frozen increasing-index stream and its geometry-only coverage cell.
    """

    require(int(encounter_rank) >= 0, "encounter rank must be non-negative")
    return STRATUM_IDS[int(encounter_rank) % len(STRATUM_IDS)]


def quota_plan() -> dict[str, Any]:
    """The committed quota/fill algorithm, independent of any outcome."""

    return {
        "groups_per_split": GROUPS_PER_SPLIT,
        "coverage_cells": COVERAGE_CELLS,
        "groups_per_cell_per_split": GROUPS_PER_CELL_PER_SPLIT,
        "action_strata_per_cell": list(STRATUM_IDS),
        "stratum_by_within_cell_rank": {
            str(rank): stratum_for_rank(rank)
            for rank in range(GROUPS_PER_CELL_PER_SPLIT)
        },
        "fill_rule": (
            "assign each candidate a stratum from its geometry-only within-cell "
            "encounter rank modulo three, then scan the frozen increasing-index "
            "stream and admit the first passing candidate for every cell/stratum"
        ),
        "quota_exhaustion": "valid_p1b_structural_failure_no_new_window_or_seed",
        "admission_may_use_only": list(ADMISSION_MAY_USE_ONLY),
        "admission_must_not_use": list(ADMISSION_MUST_NOT_USE),
        "pre_outcome_admission_keys": list(PRE_OUTCOME_ADMISSION_KEYS),
        "rollout_required_admission_keys": list(ROLLOUT_REQUIRED_ADMISSION_KEYS),
        "rejected_candidate_accounting_required": True,
        "robustness_is_audit_only": {
            "action_rotation_degrees": list(AUDIT_ONLY_ACTION_ROTATION_DEGREES),
            "agent_geometry_shift_px_along_local_tangent": list(
                AUDIT_ONLY_AGENT_TANGENT_SHIFT_PX
            ),
            "usage": "recorded_never_used_for_admission_or_ranking",
        },
    }


def assign_quota(
    candidates: Sequence[dict[str, Any]],
    *,
    split: str,
    structurally_admitted: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Apply the frozen quota algorithm to an ordered candidate stream.

    ``candidates`` must already be ordered by increasing group index and must
    carry ``group_index``, ``coverage_cell``, the pre-rollout
    ``action_stratum``, and ``pre_outcome_passed``.  The
    optional ``structurally_admitted`` mask is what a later rollout evaluator
    supplies; when it is omitted no candidate is claimed to have passed contact
    and the result is a plan, not an accepted manifest.
    """

    require(split in CANDIDATE_WINDOWS, f"unknown split {split!r}")
    if structurally_admitted is None:
        mask: list[bool | None] = [None] * len(candidates)
    else:
        require(
            len(structurally_admitted) == len(candidates),
            "structural mask length mismatch",
        )
        mask = [bool(value) for value in structurally_admitted]

    previous = -1
    filled: dict[int, dict[str, dict[str, Any] | None]] = {
        cell: {stratum: None for stratum in STRATUM_IDS}
        for cell in range(COVERAGE_CELLS)
    }
    rejected: list[dict[str, Any]] = []
    for candidate, admitted in zip(candidates, mask, strict=True):
        group_index = int(candidate["group_index"])
        require(group_index > previous, "candidate stream is not strictly increasing")
        previous = group_index
        cell = int(candidate["coverage_cell"])
        require(0 <= cell < COVERAGE_CELLS, f"coverage cell out of range: {cell}")
        stratum = str(candidate["action_stratum"])
        require(stratum in STRATUM_IDS, f"unknown candidate action stratum: {stratum}")
        if filled[cell][stratum] is not None:
            rejected.append(
                {
                    "group_index": group_index,
                    "coverage_cell": cell,
                    "action_stratum": stratum,
                    "rejection_reason": "coverage_action_quota_already_full",
                }
            )
            continue
        if not bool(candidate["pre_outcome_passed"]):
            rejected.append(
                {
                    "group_index": group_index,
                    "coverage_cell": cell,
                    "action_stratum": stratum,
                    "rejection_reason": str(
                        candidate.get("rejection_reason", "action_norm_and_box")
                    ),
                }
            )
            continue
        if admitted is False:
            rejected.append(
                {
                    "group_index": group_index,
                    "coverage_cell": cell,
                    "action_stratum": stratum,
                    "rejection_reason": str(
                        candidate.get("rejection_reason", "structural_admission")
                    ),
                }
            )
            continue
        filled[cell][stratum] = {
            **candidate,
            "within_cell_rank": STRATUM_IDS.index(stratum),
            "structurally_admitted": admitted,
        }
    accepted = [
        filled[cell][stratum]
        for cell in range(COVERAGE_CELLS)
        for stratum in STRATUM_IDS
        if filled[cell][stratum] is not None
    ]
    complete = all(
        filled[cell][stratum] is not None
        for cell in range(COVERAGE_CELLS)
        for stratum in STRATUM_IDS
    )
    return {
        "split": split,
        "quota_complete": bool(complete),
        "accepted_count": len(accepted),
        "accepted": accepted,
        "per_cell_counts": {
            str(cell): sum(value is not None for value in filled[cell].values())
            for cell in range(COVERAGE_CELLS)
        },
        "per_cell_action_counts": {
            str(cell): {
                stratum: int(filled[cell][stratum] is not None)
                for stratum in STRATUM_IDS
            }
            for cell in range(COVERAGE_CELLS)
        },
        "underfilled_cells": [
            cell
            for cell in range(COVERAGE_CELLS)
            if any(filled[cell][stratum] is None for stratum in STRATUM_IDS)
        ],
        "rejected_count": len(rejected),
        "rejected": rejected,
        "structural_admission_supplied": structurally_admitted is not None,
    }


def ensure_contextworld_on_path() -> None:
    if str(CONTEXTWORLD_ROOT) not in sys.path:
        sys.path.insert(0, str(CONTEXTWORLD_ROOT))


__all__ = [
    "ADMISSION_MAY_USE_ONLY",
    "ADMISSION_MUST_NOT_USE",
    "CANDIDATE_WINDOWS",
    "COVERAGE_CELLS",
    "DESIGN_ID",
    "DIRECTIONS",
    "ForbiddenSplitError",
    "GROUPS_PER_CELL_PER_SPLIT",
    "GROUPS_PER_SPLIT",
    "PILOT_CATALOG_SEED",
    "PROJECTED_WEIGHTS_PATH",
    "SCHEMA_VERSION",
    "STRATUM_IDS",
    "action_norm_and_box",
    "assign_d0_cell",
    "assign_quota",
    "build_query_actions",
    "canonical_json",
    "canonical_sha256",
    "candidate_window",
    "cut_points_reproduce_d0",
    "ensure_contextworld_on_path",
    "infer_d0_cut_points",
    "load_d0_coverage_rows",
    "local_tangent_unit",
    "orientation_bin_from_block_angle",
    "planner_base_vector",
    "quota_plan",
    "reject_forbidden_path",
    "rename_to_seed_bearing_identity",
    "require",
    "rotate_preserving_norm",
    "seed_bearing_template_id",
    "sha256_file",
    "stratum_for_rank",
    "stratum_for_encounter",
    "tangent_angle_degrees",
    "verify_declared_identities",
    "windows_are_disjoint",
    "write_json_exclusive",
]
