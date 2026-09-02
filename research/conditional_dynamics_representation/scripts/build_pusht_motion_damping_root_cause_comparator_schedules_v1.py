#!/usr/bin/env python3
"""Build deterministic Training-only schedules for four Motion comparators.

The schedule is an index-only construction.  It consumes the frozen
``D0/REL50/ABS50/HASH50`` projected weights, keeps the D1 budget and coverage
layout, and never opens a model or a data table.  Each batch contains complete
twin groups: both source pair rows for every twin are emitted together.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARATOR_DIR = RESEARCH_ROOT / (
    "artifacts/pusht_motion_damping_root_cause_comparators_v1/comparators_v1_final"
)
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / (
    "artifacts/pusht_motion_damping_root_cause_comparator_schedules_v1/"
    "comparator_schedules_v1_final"
)

# These are the four files in the immutable comparator artifact.  Keeping the
# values here makes a modified or substituted comparator fail before any
# schedule construction occurs.
EXPECTED_COMPARATOR_SHA256 = {
    "config.json": "5fd0569117777e0a14fd404520c24ba1983e57addd3e760321085feb51a830a8",
    "summary.json": "0407789c060b493a47940e116b669ac3f8a37485f58221da87d3f6174b737ec4",
    "projected_weights.jsonl": "55e74fc4e3e27dba2f8ee999ea5f361f5163ff46af79c9a2f0306336fd8964d3",
    # Updated when the comparator artifact is resealed with its builder pin.
    "receipt.json": "0209220090f6d4c731e76862a3df4420d79611ae170a0bf41b018efa7f88474e",
}
COMPARATOR_FILES = tuple(EXPECTED_COMPARATOR_SHA256)

ARMS = ("D0", "REL50", "ABS50", "HASH50")
COMPARATOR_HIGH_FIELDS = {
    "D0": None,
    "REL50": "pi_high_rel50",
    "ABS50": "pi_high_abs50",
    "HASH50": "pi_high_hash50",
}
COMPARATOR_FULL_FIELDS = {
    "D0": None,
    "REL50": "pi_ms50_rel50",
    "ABS50": "pi_abs50",
    "HASH50": "pi_hash50",
}
COMPARATOR_RANK_FIELDS = {
    "D0": "rank_rel50",
    "REL50": "rank_rel50",
    "ABS50": "rank_abs50",
    "HASH50": "rank_hash50",
}

EXPECTED_TWINS = 4096
EXPECTED_CELLS = 64
TWINS_PER_CELL = 64
KS = (32, 64, 128)
CONSTRUCTION_SEED = 20260831
TRAINING_SEED = 14321
CYCLES = 32
BATCHES_PER_CYCLE = 256
TWINS_PER_BATCH = 16
ARM_SLOTS_PER_BATCH = 8
ARM_SLOTS_PER_CYCLE = BATCHES_PER_CYCLE * ARM_SLOTS_PER_BATCH
TOTAL_ARM_SLOTS = CYCLES * ARM_SLOTS_PER_CYCLE
ARM_SLOTS_PER_CELL = TOTAL_ARM_SLOTS // EXPECTED_CELLS
ARM_SLOTS_PER_CELL_CYCLE = ARM_SLOTS_PER_CYCLE // EXPECTED_CELLS
NATURAL_COUNT_PER_TWIN = TOTAL_ARM_SLOTS // EXPECTED_TWINS
TOTAL_TWIN_SLOTS = 2 * TOTAL_ARM_SLOTS
HIGH_TV_BOUND = EXPECTED_TWINS / (2.0 * TOTAL_ARM_SLOTS)
FULL_TV_BOUND = HIGH_TV_BOUND / 2.0
MASS_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class ComparatorPool:
    """The frozen comparator rows needed to construct and audit schedules."""

    twin_ids: np.ndarray
    pair_ids: tuple[tuple[str, str], ...]
    cells: np.ndarray
    orientation: np.ndarray
    speed_bin: np.ndarray
    goal_bin: np.ndarray
    ranks: dict[str, np.ndarray]
    high_weights: dict[str, np.ndarray]
    full_weights: dict[str, np.ndarray]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(seed: int, *parts: Any) -> bytes:
    payload = "|".join([str(int(seed)), *(str(value) for value in parts)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _path_tokens(path: Path) -> set[str]:
    return {
        token
        for part in Path(path).parts
        for token in re.split(r"[^a-z0-9]+", part.lower())
        if token
    }


def assert_training_only_path(path: Path, *, expected_name: str | None = None) -> Path:
    """Reject held-out or development paths before opening their contents."""

    resolved = Path(path).expanduser().resolve()
    if expected_name is not None:
        require(resolved.name == expected_name, f"expected {expected_name}: {resolved}")
    forbidden = {"development", "public", "test", "validation"}
    require(
        _path_tokens(resolved).isdisjoint(forbidden),
        f"Development/Public/Test/Validation input is forbidden: {resolved}",
    )
    return resolved


def _finite(name: str, values: np.ndarray) -> None:
    require(bool(np.isfinite(values).all()), f"{name} contains non-finite values")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def verify_frozen_inputs(
    comparator_dir: Path = DEFAULT_COMPARATOR_DIR,
) -> dict[str, Any]:
    """Pin the comparator artifact and its training-only evidence boundary."""

    comparator_dir = assert_training_only_path(Path(comparator_dir))
    require(comparator_dir.is_dir(), f"missing comparator directory: {comparator_dir}")
    observed: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name in COMPARATOR_FILES:
        path = assert_training_only_path(comparator_dir / name, expected_name=name)
        require(path.is_file(), f"missing comparator artifact file: {path}")
        paths[name] = path
        observed[name] = file_sha256(path)
    require(observed == EXPECTED_COMPARATOR_SHA256, "frozen comparator artifact SHA256 changed")

    config = _load_json(paths["config.json"])
    summary = _load_json(paths["summary.json"])
    receipt = _load_json(paths["receipt.json"])
    require(
        config.get("builder_id")
        == summary.get("builder_id")
        == receipt.get("builder_id")
        == "pusht_motion_damping_root_cause_comparators_v1",
        "comparator builder identity changed",
    )
    require(
        config.get("candidate_id")
        == summary.get("candidate_id")
        == receipt.get("candidate_id")
        == "D1-MS50",
        "comparator candidate identity changed",
    )
    require(
        summary.get("status")
        == receipt.get("status")
        == "completed_training_only_comparator",
        "comparator status changed",
    )
    require(config.get("arms") == list(ARMS), "comparator arm set changed")
    require(
        receipt.get("output_sha256")
        == {
            "config.json": observed["config.json"],
            "summary.json": observed["summary.json"],
            "projected_weights.jsonl": observed["projected_weights.jsonl"],
        },
        "comparator receipt output pin changed",
    )
    require(
        config.get("source_input_sha256") == receipt.get("input_sha256"),
        "comparator source input pin changed",
    )
    for invariant_name, invariant_value in summary.get("invariants", {}).items():
        require(invariant_value is True, f"comparator invariant failed: {invariant_name}")
    for invariant_name, invariant_value in receipt.get("invariants", {}).items():
        require(invariant_value is True, f"comparator receipt invariant failed: {invariant_name}")

    boundaries = (config.get("evidence_boundary"), summary.get("evidence_boundary"), receipt)
    for boundary in boundaries:
        require(isinstance(boundary, dict), "comparator evidence boundary is missing")
        require(
            boundary.get("training_only") in (True, None),
            "comparator is not Training-only",
        )
        require(boundary.get("development_lance_opened") is False, "Development input was opened")
        require(boundary.get("public_test_lance_opened") is False, "Public Test input was opened")
        require(boundary.get("model_loaded") is False, "model loading escaped comparator boundary")
        require(boundary.get("pixels_decoded") is False, "pixel decoding escaped comparator boundary")
        require(
            boundary.get("optimizer_steps", boundary.get("optimizer_steps_run")) == 0,
            "optimizer steps escaped comparator boundary",
        )
        require(boundary.get("schedule_generated") is False, "comparator already contains a schedule")

    source_v2_dir = config.get("source_v2_dir")
    require(isinstance(source_v2_dir, str), "comparator source v2 path is missing")
    assert_training_only_path(Path(source_v2_dir))
    comparator_builder_sha256 = receipt.get("builder_script_sha256")
    require(
        isinstance(comparator_builder_sha256, str) and len(comparator_builder_sha256) == 64,
        "comparator receipt is missing its builder script SHA256",
    )
    require(
        comparator_builder_sha256 == config.get("builder_script_sha256"),
        "comparator builder script pin differs between config and receipt",
    )
    return {
        "comparator_dir": comparator_dir,
        "paths": paths,
        "observed_sha256": observed,
        "config": config,
        "summary": summary,
        "receipt": receipt,
        "comparator_builder_sha256": comparator_builder_sha256,
    }


def _weight_array(rows: Sequence[Mapping[str, Any]], field: str | None, *, default: float) -> np.ndarray:
    if field is None:
        values = [default] * len(rows)
    else:
        values = [float(row[field]) for row in rows]
    result = np.asarray(values, dtype=np.float64)
    _finite(field or "uniform", result)
    return result


def load_comparator_pool(path: Path) -> ComparatorPool:
    """Load projected comparator rows and recheck their structural identities."""

    path = assert_training_only_path(Path(path), expected_name="projected_weights.jsonl")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            require(bool(line.strip()), f"blank comparator row at line {line_number}")
            row = json.loads(line)
            require(isinstance(row, dict), f"comparator row {line_number} is not an object")
            rows.append(row)
    require(len(rows) == EXPECTED_TWINS, f"comparator rows {len(rows)} != {EXPECTED_TWINS}")

    twin_ids = np.asarray([int(row["twin_id"]) for row in rows], dtype=np.int64)
    require(np.array_equal(twin_ids, np.arange(EXPECTED_TWINS)), "twin ids are not contiguous")
    pair_ids = tuple(tuple(str(value) for value in row["pair_ids"]) for row in rows)
    require(all(len(pair) == 2 for pair in pair_ids), "each twin must have two pair ids")
    for twin_id, pair in enumerate(pair_ids):
        expected = (
            f"pmd-train-{2 * twin_id:06d}-forward",
            f"pmd-train-{2 * twin_id + 1:06d}-reverse",
        )
        require(pair == expected, f"pair-to-twin identity changed at twin {twin_id}")

    cells = np.asarray([int(row["coverage_cell"]) for row in rows], dtype=np.int64)
    orientation = np.asarray([int(row["orientation_bin"]) for row in rows], dtype=np.int64)
    speed_bin = np.asarray([int(row["speed_bin"]) for row in rows], dtype=np.int64)
    goal_bin = np.asarray([int(row["goal_distance_bin"]) for row in rows], dtype=np.int64)
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

    ranks = {
        arm: np.asarray([int(row[field]) for row in rows], dtype=np.int64)
        for arm, field in COMPARATOR_RANK_FIELDS.items()
    }
    for arm, rank_values in ranks.items():
        require(
            np.all((rank_values >= 1) & (rank_values <= TWINS_PER_CELL)),
            f"{arm} comparator ranks are out of range",
        )
        for cell in range(EXPECTED_CELLS):
            require(
                np.array_equal(
                    np.sort(rank_values[cells == cell]), np.arange(1, TWINS_PER_CELL + 1)
                ),
                f"{arm} comparator ranks are not a cell permutation",
            )

    high_weights = {
        arm: _weight_array(
            rows,
            COMPARATOR_HIGH_FIELDS[arm],
            default=1.0 / EXPECTED_TWINS,
        )
        for arm in ARMS
    }
    full_weights = {
        arm: _weight_array(
            rows,
            COMPARATOR_FULL_FIELDS[arm],
            default=1.0 / EXPECTED_TWINS,
        )
        for arm in ARMS
    }
    for arm in ARMS:
        high = high_weights[arm]
        full = full_weights[arm]
        require(np.all(high > 0.0), f"{arm} high weights are not positive")
        require(np.all(full > 0.0), f"{arm} full weights are not positive")
        require(math.isclose(float(high.sum()), 1.0, abs_tol=MASS_TOLERANCE), f"{arm} high mass changed")
        require(math.isclose(float(full.sum()), 1.0, abs_tol=MASS_TOLERANCE), f"{arm} full mass changed")
        require(
            np.allclose(full, 0.5 / EXPECTED_TWINS + 0.5 * high, rtol=0.0, atol=1.0e-15),
            f"{arm} full projection changed",
        )
        for cell in range(EXPECTED_CELLS):
            mask = cells == cell
            require(
                math.isclose(float(high[mask].sum()), 1.0 / EXPECTED_CELLS, abs_tol=1.0e-15),
                f"{arm} high cell mass changed in cell {cell}",
            )
            require(
                math.isclose(float(full[mask].sum()), 1.0 / EXPECTED_CELLS, abs_tol=1.0e-15),
                f"{arm} full cell mass changed in cell {cell}",
            )

    # The D0 arm is a true uniform control, while the three comparator arms
    # retain the source high-weight multiset and only reassign its exposure.
    require(np.all(high_weights["D0"] == 1.0 / EXPECTED_TWINS), "D0 high arm is not uniform")
    require(np.all(full_weights["D0"] == 1.0 / EXPECTED_TWINS), "D0 full arm is not uniform")
    for arm in ("ABS50", "HASH50"):
        require(
            np.array_equal(np.sort(high_weights[arm]), np.sort(high_weights["REL50"])),
            f"{arm} high-weight multiset changed",
        )
        require(
            np.array_equal(np.sort(full_weights[arm]), np.sort(full_weights["REL50"])),
            f"{arm} full-weight multiset changed",
        )
    return ComparatorPool(
        twin_ids=twin_ids,
        pair_ids=pair_ids,
        cells=cells,
        orientation=orientation,
        speed_bin=speed_bin,
        goal_bin=goal_bin,
        ranks=ranks,
        high_weights=high_weights,
        full_weights=full_weights,
    )


def load_projected_pool(path: Path) -> ComparatorPool:
    """Compatibility name for callers of the preceding D1 schedule builder."""

    return load_comparator_pool(path)


def largest_remainder_counts(
    probabilities: np.ndarray,
    cells: np.ndarray,
    stable_ids: np.ndarray,
    *,
    slots_per_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hamilton integerization independently within every coverage cell."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    require(probabilities.shape == cells.shape == stable_ids.shape, "largest-remainder shape mismatch")
    require(slots_per_cell > 0, "slots_per_cell must be positive")
    counts = np.zeros(len(probabilities), dtype=np.int64)
    desired = np.zeros(len(probabilities), dtype=np.float64)
    for cell in sorted(np.unique(cells).tolist()):
        indices = np.flatnonzero(cells == cell)
        local = probabilities[indices]
        require(np.all(local >= 0.0) and float(local.sum()) > 0.0, "invalid cell mass")
        target = slots_per_cell * local / float(local.sum())
        base = np.floor(target).astype(np.int64)
        remaining = int(slots_per_cell - int(base.sum()))
        require(0 <= remaining <= len(indices), "invalid Hamilton remainder count")
        fraction = target - base
        order = np.lexsort((stable_ids[indices], -fraction))
        base[order[:remaining]] += 1
        require(int(base.sum()) == slots_per_cell, "Hamilton cell total mismatch")
        counts[indices] = base
        desired[indices] = target
    return counts, desired


def natural_cycle_incidence(
    cells: np.ndarray,
    stable_ids: np.ndarray,
    *,
    cycles: int = CYCLES,
    seed: int = CONSTRUCTION_SEED,
) -> np.ndarray:
    """Build complementary natural-anchor halves for every cell."""

    require(cycles > 0 and cycles % 2 == 0, "natural cycles must be positive and even")
    cells = np.asarray(cells, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    require(cells.shape == stable_ids.shape, "natural incidence shape mismatch")
    require(len(np.unique(stable_ids)) == len(stable_ids), "natural ids are not unique")
    incidence = np.zeros((len(stable_ids), cycles), dtype=bool)
    for cell in sorted(np.unique(cells).tolist()):
        indices = np.flatnonzero(cells == cell)
        require(len(indices) % 2 == 0, "natural cell size must be even")
        half = len(indices) // 2
        for cycle_pair in range(cycles // 2):
            ordered = sorted(
                indices.tolist(),
                key=lambda index: (
                    stable_hash(seed, "natural", cell, cycle_pair, int(stable_ids[index])),
                    int(stable_ids[index]),
                ),
            )
            incidence[ordered[:half], 2 * cycle_pair] = True
            incidence[ordered[half:], 2 * cycle_pair + 1] = True
    return incidence


def high_cycle_incidence(
    counts: np.ndarray,
    cells: np.ndarray,
    stable_ids: np.ndarray,
    *,
    cycles: int = CYCLES,
    seed: int = CONSTRUCTION_SEED,
) -> np.ndarray:
    """Allocate Hamilton counts to least-loaded cycles with stable ties."""

    counts = np.asarray(counts, dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    require(counts.shape == cells.shape == stable_ids.shape, "high incidence shape mismatch")
    require(np.all((counts >= 0) & (counts <= cycles)), "high count exceeds cycle support")
    require(len(np.unique(stable_ids)) == len(stable_ids), "high ids are not unique")
    incidence = np.zeros((len(counts), cycles), dtype=bool)
    for cell in sorted(np.unique(cells).tolist()):
        indices = np.flatnonzero(cells == cell)
        total = int(counts[indices].sum())
        require(total % cycles == 0, "high cell total is not cycle-balanced")
        target_load = total // cycles
        loads = np.zeros(cycles, dtype=np.int64)
        ordered = sorted(
            indices.tolist(),
            key=lambda index: (
                -int(counts[index]),
                stable_hash(seed, "high-twin-order", cell, int(stable_ids[index])),
                int(stable_ids[index]),
            ),
        )
        for index in ordered:
            count = int(counts[index])
            cycle_order = sorted(
                range(cycles),
                key=lambda cycle: (
                    int(loads[cycle]),
                    stable_hash(seed, "high-cycle-choice", cell, int(stable_ids[index]), cycle),
                    cycle,
                ),
            )
            selected = cycle_order[:count]
            incidence[index, selected] = True
            loads[selected] += 1
        require(np.all(loads == target_load), f"high cycle allocation is imbalanced in cell {cell}")
    return incidence


def deterministic_blocks(
    twin_ids: Iterable[int],
    *,
    block_size: int,
    seed: int,
    label: str,
    cycle: int,
) -> list[tuple[int, ...]]:
    ordered = sorted(
        (int(value) for value in twin_ids),
        key=lambda twin: (stable_hash(seed, label, cycle, twin), twin),
    )
    require(len(ordered) % block_size == 0, "arm cannot be divided into blocks")
    return [tuple(ordered[start : start + block_size]) for start in range(0, len(ordered), block_size)]


def match_disjoint_blocks(
    high_blocks: Sequence[Sequence[int]],
    natural_blocks: Sequence[Sequence[int]],
    *,
    seed: int,
    cycle: int,
) -> list[int]:
    """Find a deterministic perfect matching without cross-arm duplicates."""

    require(len(high_blocks) == len(natural_blocks), "arm block count mismatch")
    count = len(high_blocks)
    high_sets = [set(block) for block in high_blocks]
    natural_sets = [set(block) for block in natural_blocks]
    adjacency: list[list[int]] = []
    for high_index in range(count):
        allowed = [
            natural_index
            for natural_index in range(count)
            if high_sets[high_index].isdisjoint(natural_sets[natural_index])
        ]
        allowed.sort(
            key=lambda natural_index: (
                stable_hash(seed, "block-match", cycle, high_index, natural_index),
                natural_index,
            )
        )
        require(allowed, f"high block {high_index} has no conflict-free match")
        adjacency.append(allowed)

    natural_owner = [-1] * count

    def augment(high_index: int, seen: set[int]) -> bool:
        for natural_index in adjacency[high_index]:
            if natural_index in seen:
                continue
            seen.add(natural_index)
            owner = natural_owner[natural_index]
            if owner < 0 or augment(owner, seen):
                natural_owner[natural_index] = high_index
                return True
        return False

    for high_index in range(count):
        require(augment(high_index, set()), "no conflict-free perfect block matching")
    high_to_natural = [-1] * count
    for natural_index, high_index in enumerate(natural_owner):
        high_to_natural[high_index] = natural_index
    require(all(value >= 0 for value in high_to_natural), "incomplete block matching")
    return high_to_natural


def expand_hidden_rows(twin_ids: Sequence[int]) -> list[int]:
    return [4 * int(twin) + offset for twin in twin_ids for offset in range(4)]


def match_disjoint_blocks_multi(
    high_blocks_by_arm: Mapping[str, Sequence[Sequence[int]]],
    natural_blocks: Sequence[Sequence[int]],
    *,
    seed: int,
    cycle: int,
) -> list[int]:
    """Match natural blocks that are disjoint from every arm's high block.

    Matching is performed on the shared abstract block layout.  Consequently
    comparator arms differ only in the twin instantiated at a weighted slot,
    while the batch/cell signature and the natural half stay identical.
    """

    require(high_blocks_by_arm, "at least one arm is required for block matching")
    counts = {len(blocks) for blocks in high_blocks_by_arm.values()}
    require(len(counts) == 1 and len(natural_blocks) in counts, "multi-arm block count mismatch")
    count = len(natural_blocks)
    natural_sets = [set(block) for block in natural_blocks]
    adjacency: list[list[int]] = []
    for high_index in range(count):
        allowed = [
            natural_index
            for natural_index in range(count)
            if all(
                set(high_blocks[high_index]).isdisjoint(natural_sets[natural_index])
                for high_blocks in high_blocks_by_arm.values()
            )
        ]
        allowed.sort(
            key=lambda natural_index: (
                stable_hash(seed, "multi-block-match", cycle, high_index, natural_index),
                natural_index,
            )
        )
        require(allowed, f"shared high block {high_index} has no conflict-free match")
        adjacency.append(allowed)

    natural_owner = [-1] * count

    def augment(high_index: int, seen: set[int]) -> bool:
        for natural_index in adjacency[high_index]:
            if natural_index in seen:
                continue
            seen.add(natural_index)
            owner = natural_owner[natural_index]
            if owner < 0 or augment(owner, seen):
                natural_owner[natural_index] = high_index
                return True
        return False

    for high_index in range(count):
        require(augment(high_index, set()), "no shared conflict-free perfect block matching")
    high_to_natural = [-1] * count
    for natural_index, high_index in enumerate(natural_owner):
        high_to_natural[high_index] = natural_index
    require(all(value >= 0 for value in high_to_natural), "incomplete shared block matching")
    return high_to_natural


@dataclass(frozen=True)
class SharedBatch:
    cycle: int
    batch_in_cycle: int
    abstract_high_slots_by_arm: Mapping[str, tuple[int, ...]]
    natural_twin_ids: tuple[int, ...]


@dataclass(frozen=True)
class SharedSchedulePlan:
    """One abstract weighted-slot layout shared by all comparator arms."""

    abstract_cells: np.ndarray
    abstract_ranks: np.ndarray
    abstract_high_incidence_by_arm: Mapping[str, np.ndarray]
    natural_incidence: np.ndarray
    batches: tuple[SharedBatch, ...]


def rank_slot_mapping(pool: ComparatorPool, arm: str) -> np.ndarray:
    """Map each (cell, source-weight-rank) slot to the arm's twin id."""

    require(arm in ARMS, f"unknown rank-mapping arm: {arm}")
    rank_values = pool.ranks[arm]
    mapping = np.full(EXPECTED_TWINS, -1, dtype=np.int64)
    for index, twin_id in enumerate(pool.twin_ids.tolist()):
        slot = int(pool.cells[index]) * TWINS_PER_CELL + int(rank_values[index]) - 1
        require(0 <= slot < EXPECTED_TWINS, f"{arm} rank slot out of range")
        require(mapping[slot] < 0, f"{arm} rank slot is not unique")
        mapping[slot] = int(twin_id)
    require(np.all(mapping >= 0), f"{arm} rank slot mapping is incomplete")
    return mapping


def instantiate_high_incidence(
    abstract_incidence: np.ndarray,
    rank_to_twin: np.ndarray,
) -> np.ndarray:
    """Instantiate abstract (cell, source-rank) slots as concrete twin rows."""

    abstract_incidence = np.asarray(abstract_incidence, dtype=bool)
    rank_to_twin = np.asarray(rank_to_twin, dtype=np.int64)
    require(abstract_incidence.shape[0] == len(rank_to_twin), "abstract mapping shape mismatch")
    require(np.array_equal(np.sort(rank_to_twin), np.arange(len(rank_to_twin))), "rank mapping ids changed")
    incidence = np.zeros_like(abstract_incidence)
    incidence[rank_to_twin] = abstract_incidence
    return incidence


def weight_slot_mapping(
    pool: ComparatorPool,
    arm: str,
    source_order_by_cell: Mapping[int, Sequence[int]],
    target_counts: np.ndarray,
    source_counts: np.ndarray,
) -> np.ndarray:
    """Map source rank slots to an arm while preserving Hamilton tie counts.

    Comparator weights are a multiset permutation.  Equal floating weights can
    have different twin-id remainder winners after permutation, so equal-weight
    slots are matched by realized target count before twin id.  This preserves
    both the exact exposure target and the common weight-rank signature.
    """

    require(arm in ARMS, f"unknown weight-slot arm: {arm}")
    mapping = np.full(EXPECTED_TWINS, -1, dtype=np.int64)
    arm_weights = pool.high_weights[arm]
    source_weights = pool.high_weights["REL50"]
    for cell in range(EXPECTED_CELLS):
        source_indices = [int(index) for index in source_order_by_cell[cell]]
        target_indices = np.flatnonzero(pool.cells == cell).tolist()
        target_indices.sort(key=lambda index: (float(arm_weights[index]), int(pool.twin_ids[index])))
        require(len(source_indices) == len(target_indices) == TWINS_PER_CELL, "weight rank cell size changed")
        start = 0
        while start < TWINS_PER_CELL:
            source_weight = float(source_weights[source_indices[start]])
            source_end = start + 1
            while source_end < TWINS_PER_CELL and float(source_weights[source_indices[source_end]]) == source_weight:
                source_end += 1
            target_end = start + 1
            while target_end < TWINS_PER_CELL and float(arm_weights[target_indices[target_end]]) == source_weight:
                target_end += 1
            require(source_end == target_end, f"{arm} equal-weight multiset changed in cell {cell}")
            source_group = source_indices[start:source_end]
            target_group = target_indices[start:target_end]
            source_group.sort(key=lambda index: (int(source_counts[index]), int(pool.twin_ids[index])))
            target_group.sort(key=lambda index: (int(target_counts[index]), int(pool.twin_ids[index])))
            source_position = {
                int(index): start + offset
                for offset, index in enumerate(source_indices[start:source_end])
            }
            for source_index, target_index in zip(source_group, target_group, strict=True):
                slot = cell * TWINS_PER_CELL + source_position[int(source_index)]
                mapping[slot] = int(pool.twin_ids[target_index])
            start = source_end
    require(np.all(mapping >= 0), f"{arm} weight-slot mapping is incomplete")
    require(np.array_equal(np.sort(mapping), np.arange(EXPECTED_TWINS)), f"{arm} weight-slot mapping ids changed")
    return mapping


def build_abstract_high_incidences(
    pool: ComparatorPool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build shared comparator rank slots plus a separate uniform D0 layout."""

    abstract_cells = np.repeat(np.arange(EXPECTED_CELLS, dtype=np.int64), TWINS_PER_CELL)
    abstract_ids = np.arange(EXPECTED_TWINS, dtype=np.int64)
    target_counts: dict[str, np.ndarray] = {}
    for arm in ARMS:
        target_counts[arm], _ = largest_remainder_counts(
            pool.high_weights[arm], pool.cells, pool.twin_ids, slots_per_cell=ARM_SLOTS_PER_CELL
        )
    rel_counts = target_counts["REL50"]
    source_order_by_cell: dict[int, list[int]] = {}
    for cell in range(EXPECTED_CELLS):
        source_order_by_cell[cell] = sorted(
            np.flatnonzero(pool.cells == cell).tolist(),
            key=lambda index: (float(pool.high_weights["REL50"][index]), int(pool.twin_ids[index])),
        )
    weighted_abstract_counts = np.zeros(EXPECTED_TWINS, dtype=np.int64)
    for cell in range(EXPECTED_CELLS):
        for local_rank, index in enumerate(source_order_by_cell[cell]):
            weighted_abstract_counts[cell * TWINS_PER_CELL + local_rank] = int(rel_counts[index])
    rank_to_twin = {
        "D0": rank_slot_mapping(pool, "REL50"),
        "REL50": weight_slot_mapping(
            pool, "REL50", source_order_by_cell, target_counts["REL50"], rel_counts
        ),
        "ABS50": weight_slot_mapping(
            pool, "ABS50", source_order_by_cell, target_counts["ABS50"], rel_counts
        ),
        "HASH50": weight_slot_mapping(
            pool, "HASH50", source_order_by_cell, target_counts["HASH50"], rel_counts
        ),
    }

    # All three comparator arms share this source weight-rank slot schedule.
    # Their target twin identities differ only through the frozen permutation.
    weighted_abstract_incidence = high_cycle_incidence(
        weighted_abstract_counts,
        abstract_cells,
        abstract_ids,
        cycles=CYCLES,
        seed=CONSTRUCTION_SEED,
    )
    d0_counts, _ = largest_remainder_counts(
        np.full(EXPECTED_TWINS, 1.0 / EXPECTED_TWINS),
        abstract_cells,
        abstract_ids,
        slots_per_cell=ARM_SLOTS_PER_CELL,
    )
    d0_abstract_incidence = high_cycle_incidence(
        d0_counts,
        abstract_cells,
        abstract_ids,
        cycles=CYCLES,
        seed=CONSTRUCTION_SEED,
    )
    abstract_incidence = {
        "D0": d0_abstract_incidence,
        "REL50": weighted_abstract_incidence,
        "ABS50": weighted_abstract_incidence.copy(),
        "HASH50": weighted_abstract_incidence.copy(),
    }
    actual_incidence = {
        arm: instantiate_high_incidence(abstract_incidence[arm], rank_to_twin[arm])
        for arm in ARMS
    }
    return abstract_incidence, actual_incidence, rank_to_twin


def deterministic_cell_blocks(
    twin_ids: Iterable[int],
    cells: np.ndarray,
    *,
    block_size: int,
    seed: int,
    label: str,
    cycle: int,
) -> list[tuple[int, ...]]:
    """Partition each cell into stable blocks, preserving batch cell signatures."""

    twin_ids = [int(value) for value in twin_ids]
    cells = np.asarray(cells, dtype=np.int64)
    require(len(twin_ids) == len(cells), "cell block shape mismatch")
    blocks: list[tuple[int, ...]] = []
    for cell in sorted(np.unique(cells).tolist()):
        local = [twin for twin, local_cell in zip(twin_ids, cells.tolist(), strict=True) if local_cell == cell]
        ordered = sorted(
            local,
            key=lambda twin: (stable_hash(seed, label, cycle, cell, twin), twin),
        )
        require(len(ordered) % block_size == 0, "cell cannot be divided into blocks")
        blocks.extend(
            tuple(ordered[start : start + block_size])
            for start in range(0, len(ordered), block_size)
        )
    return blocks


def build_shared_plan(
    pool: ComparatorPool,
    abstract_high_incidence_by_arm: Mapping[str, np.ndarray],
    natural_incidence: np.ndarray,
    rank_to_twin: Mapping[str, np.ndarray],
    *,
    seed: int = CONSTRUCTION_SEED,
) -> SharedSchedulePlan:
    """Build common cell/rank blocks and jointly conflict-free natural matches."""

    n = len(pool.twin_ids)
    expected_shape = (n, CYCLES)
    require(set(abstract_high_incidence_by_arm) == set(ARMS), "abstract incidence arm set changed")
    for arm in ARMS:
        require(
            abstract_high_incidence_by_arm[arm].shape == expected_shape,
            f"{arm} abstract high incidence shape mismatch",
        )
    require(natural_incidence.shape == expected_shape, "natural incidence shape mismatch")
    require(set(rank_to_twin) == set(ARMS), "rank mapping arm set changed")
    abstract_cells = np.repeat(np.arange(EXPECTED_CELLS, dtype=np.int64), TWINS_PER_CELL)
    abstract_ranks = np.tile(np.arange(1, TWINS_PER_CELL + 1, dtype=np.int64), EXPECTED_CELLS)
    batches: list[SharedBatch] = []
    for cycle in range(CYCLES):
        abstract_ids_by_arm = {
            arm: np.flatnonzero(abstract_high_incidence_by_arm[arm][:, cycle]).tolist()
            for arm in ARMS
        }
        natural_ids = pool.twin_ids[natural_incidence[:, cycle]].tolist()
        require(
            all(len(abstract_ids_by_arm[arm]) == ARM_SLOTS_PER_CYCLE for arm in ARMS),
            "bad shared weighted cycle size",
        )
        require(len(natural_ids) == ARM_SLOTS_PER_CYCLE, "bad shared natural cycle size")
        high_blocks_by_arm = {
            arm: deterministic_cell_blocks(
                abstract_ids_by_arm[arm],
                np.asarray([abstract_cells[slot] for slot in abstract_ids_by_arm[arm]], dtype=np.int64),
                block_size=ARM_SLOTS_PER_BATCH,
                seed=seed,
                label="abstract-high-block-order",
                cycle=cycle,
            )
            for arm in ARMS
        }
        natural_blocks = deterministic_cell_blocks(
            natural_ids,
            pool.cells[natural_incidence[:, cycle]],
            block_size=ARM_SLOTS_PER_BATCH,
            seed=seed,
            label="natural-block-order",
            cycle=cycle,
        )
        mapped_blocks = {
            arm: [
                tuple(int(rank_to_twin[arm][slot]) for slot in block)
                for block in high_blocks_by_arm[arm]
            ]
            for arm in ARMS
        }
        matching = match_disjoint_blocks_multi(
            mapped_blocks, natural_blocks, seed=seed, cycle=cycle
        )
        for batch_in_cycle in range(BATCHES_PER_CYCLE):
            batches.append(
                SharedBatch(
                    cycle=cycle,
                    batch_in_cycle=batch_in_cycle,
                    abstract_high_slots_by_arm={
                        arm: tuple(int(slot) for slot in high_blocks_by_arm[arm][batch_in_cycle])
                        for arm in ARMS
                    },
                    natural_twin_ids=tuple(int(twin) for twin in natural_blocks[matching[batch_in_cycle]]),
                )
            )
    require(len(batches) == CYCLES * BATCHES_PER_CYCLE, "shared plan batch count changed")
    return SharedSchedulePlan(
        abstract_cells=abstract_cells,
        abstract_ranks=abstract_ranks,
        abstract_high_incidence_by_arm=abstract_high_incidence_by_arm,
        natural_incidence=natural_incidence,
        batches=tuple(batches),
    )


def build_schedule(
    pool: ComparatorPool,
    high_incidence: np.ndarray,
    natural_incidence: np.ndarray,
    *,
    arm: str = "D0",
    exposure_weights: np.ndarray | None = None,
    high_weights: np.ndarray | None = None,
    rank_to_twin: np.ndarray | None = None,
    shared_plan: SharedSchedulePlan | None = None,
    seed: int = CONSTRUCTION_SEED,
) -> list[dict[str, Any]]:
    """Build one arm schedule, optionally instantiating a shared slot plan."""

    require(arm in ARMS, f"unknown schedule arm: {arm}")
    expected_shape = (len(pool.twin_ids), CYCLES)
    require(high_incidence.shape == natural_incidence.shape == expected_shape, "schedule incidence shape mismatch")
    if exposure_weights is None:
        exposure_weights = pool.full_weights[arm]
    exposure_weights = np.asarray(exposure_weights, dtype=np.float64)
    require(exposure_weights.shape == pool.twin_ids.shape, "schedule exposure-weight shape mismatch")
    _finite("schedule exposure weights", exposure_weights)
    if high_weights is None:
        high_weights = pool.high_weights[arm]
    high_weights = np.asarray(high_weights, dtype=np.float64)
    require(high_weights.shape == pool.twin_ids.shape, "schedule high-weight shape mismatch")
    _finite("schedule high weights", high_weights)
    if shared_plan is not None:
        require(rank_to_twin is not None, "shared plan requires rank-to-twin mapping")
        require(
            shared_plan.abstract_high_incidence_by_arm[arm].shape == high_incidence.shape,
            "shared plan incidence mismatch",
        )
    schedule: list[dict[str, Any]] = []
    for cycle in range(CYCLES):
        if shared_plan is not None:
            cycle_batches = [
                (index, batch)
                for index, batch in enumerate(shared_plan.batches)
                if batch.cycle == cycle
            ]
            for _, batch in cycle_batches:
                abstract_high_slots = batch.abstract_high_slots_by_arm[arm]
                high_block = tuple(int(rank_to_twin[slot]) for slot in abstract_high_slots)
                natural_block = batch.natural_twin_ids
                twins = [*high_block, *natural_block]
                require(len(set(twins)) == TWINS_PER_BATCH, "cross-arm batch conflict")
                schedule_index = cycle * BATCHES_PER_CYCLE + batch.batch_in_cycle
                schedule.append(
                    {
                        "arm": arm,
                        "schedule_index": schedule_index,
                        "optimizer_step": schedule_index + 1,
                        "cycle": cycle,
                        "batch_in_cycle": batch.batch_in_cycle,
                        "high_twin_ids": list(high_block),
                        "natural_twin_ids": list(natural_block),
                        "twin_ids": twins,
                        "pair_ids": [list(pool.pair_ids[int(twin)]) for twin in twins],
                        "exposure_weights": [float(exposure_weights[int(twin)]) for twin in twins],
                        "high_exposure_weights": [float(high_weights[int(twin)]) for twin in high_block]
                        + [0.0] * ARM_SLOTS_PER_BATCH,
                        "high_slot_ids": list(abstract_high_slots),
                        "high_slot_signature": [
                            [
                                int(shared_plan.abstract_cells[slot]),
                                int(shared_plan.abstract_ranks[slot]),
                            ]
                            for slot in abstract_high_slots
                        ],
                        "hidden_row_indices": expand_hidden_rows(twins),
                    }
                )
            continue
        high_ids = pool.twin_ids[high_incidence[:, cycle]].tolist()
        natural_ids = pool.twin_ids[natural_incidence[:, cycle]].tolist()
        require(len(high_ids) == ARM_SLOTS_PER_CYCLE, "bad weighted-arm cycle size")
        require(len(natural_ids) == ARM_SLOTS_PER_CYCLE, "bad natural-arm cycle size")
        high_blocks = deterministic_blocks(
            high_ids, block_size=ARM_SLOTS_PER_BATCH, seed=seed, label=f"{arm}-high-block-order", cycle=cycle
        )
        natural_blocks = deterministic_blocks(
            natural_ids, block_size=ARM_SLOTS_PER_BATCH, seed=seed, label="natural-block-order", cycle=cycle
        )
        matching = match_disjoint_blocks(high_blocks, natural_blocks, seed=seed, cycle=cycle)
        for batch_in_cycle, high_block in enumerate(high_blocks):
            natural_block = natural_blocks[matching[batch_in_cycle]]
            twins = [*high_block, *natural_block]
            require(len(set(twins)) == TWINS_PER_BATCH, "cross-arm batch conflict")
            schedule_index = cycle * BATCHES_PER_CYCLE + batch_in_cycle
            schedule.append(
                {
                    "arm": arm,
                    "schedule_index": schedule_index,
                    "optimizer_step": schedule_index + 1,
                    "cycle": cycle,
                    "batch_in_cycle": batch_in_cycle,
                    "high_twin_ids": list(high_block),
                    "natural_twin_ids": list(natural_block),
                    "twin_ids": twins,
                    "pair_ids": [list(pool.pair_ids[int(twin)]) for twin in twins],
                    "exposure_weights": [float(exposure_weights[int(twin)]) for twin in twins],
                    "high_exposure_weights": [float(high_weights[int(twin)]) for twin in high_block]
                    + [0.0] * ARM_SLOTS_PER_BATCH,
                    "hidden_row_indices": expand_hidden_rows(twins),
                }
            )
    return schedule


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape, "TV shape mismatch")
    require(
        math.isclose(float(left.sum()), 1.0, abs_tol=MASS_TOLERANCE)
        and math.isclose(float(right.sum()), 1.0, abs_tol=MASS_TOLERANCE),
        "TV inputs are not normalized",
    )
    return 0.5 * float(np.abs(left - right).sum())


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        label: float(value)
        for label, value in zip(
            ("min", "q25", "median", "q75", "max"),
            np.quantile(np.asarray(values, dtype=np.float64), [0.0, 0.25, 0.5, 0.75, 1.0]),
            strict=True,
        )
    }


def _error_report(target: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    error = np.asarray(realized, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    _finite("realized exposure error", error)
    return {
        "max_abs": float(np.max(np.abs(error))),
        "mean_abs": float(np.mean(np.abs(error))),
        "sum_abs": float(np.sum(np.abs(error), dtype=np.float64)),
        "total_variation": total_variation(target, realized),
    }


def audit_arm_schedule(
    pool: ComparatorPool,
    arm: str,
    schedule: Sequence[Mapping[str, Any]],
    high_incidence: np.ndarray,
    natural_incidence: np.ndarray,
    high_counts: np.ndarray,
    desired_high_counts: np.ndarray,
    full_counts: np.ndarray,
    desired_full_counts: np.ndarray,
    *,
    shared_plan: SharedSchedulePlan | None = None,
    rank_to_twin: np.ndarray | None = None,
) -> dict[str, Any]:
    """Audit one schedule against its incidences, target counts and pairs."""

    require(len(schedule) == CYCLES * BATCHES_PER_CYCLE, f"{arm} schedule batch count changed")
    n = len(pool.twin_ids)
    realized_high_counts = np.zeros(n, dtype=np.int64)
    realized_natural_counts = np.zeros(n, dtype=np.int64)
    pair_integrity = True
    batch_integrity = True
    for index, row in enumerate(schedule):
        cycle = index // BATCHES_PER_CYCLE
        batch_in_cycle = index % BATCHES_PER_CYCLE
        expected_twins = [*row["high_twin_ids"], *row["natural_twin_ids"]]
        expected_pairs = [list(pool.pair_ids[int(twin)]) for twin in expected_twins]
        row_twins = [int(twin) for twin in row["twin_ids"]]
        row_pairs = [[str(value) for value in pair] for pair in row["pair_ids"]]
        expected_weights = [float(pool.full_weights[arm][int(twin)]) for twin in expected_twins]
        expected_slots: tuple[int, ...] = ()
        expected_signature: list[list[int]] = []
        if shared_plan is not None:
            require(rank_to_twin is not None, "shared plan audit requires rank-to-twin mapping")
            expected_slots = shared_plan.batches[index].abstract_high_slots_by_arm[arm]
            expected_signature = [
                [
                    int(shared_plan.abstract_cells[slot]),
                    int(shared_plan.abstract_ranks[slot]),
                ]
                for slot in expected_slots
            ]
        batch_integrity = batch_integrity and (
            row.get("arm") == arm
            and row.get("schedule_index") == index
            and row.get("optimizer_step") == index + 1
            and row.get("cycle") == cycle
            and row.get("batch_in_cycle") == batch_in_cycle
            and len(row["high_twin_ids"]) == ARM_SLOTS_PER_BATCH
            and len(row["natural_twin_ids"]) == ARM_SLOTS_PER_BATCH
            and len(row_twins) == TWINS_PER_BATCH
            and len(set(row_twins)) == TWINS_PER_BATCH
            and set(row["high_twin_ids"]).isdisjoint(row["natural_twin_ids"])
            and row_twins == expected_twins
            and row.get("hidden_row_indices") == expand_hidden_rows(expected_twins)
            and row_pairs == expected_pairs
            and np.allclose(row["exposure_weights"], expected_weights, rtol=0.0, atol=0.0)
            and (
                shared_plan is None
                or (
                    tuple(int(slot) for slot in row["high_slot_ids"]) == expected_slots
                    and row["high_slot_signature"] == expected_signature
                    and [int(rank_to_twin[slot]) for slot in expected_slots]
                    == [int(twin) for twin in row["high_twin_ids"]]
                )
            )
        )
        pair_integrity = pair_integrity and all(
            pair == list(pool.pair_ids[int(twin)]) for twin, pair in zip(row_twins, row_pairs, strict=True)
        )
        realized_high_counts[np.asarray(row["high_twin_ids"], dtype=np.int64)] += 1
        realized_natural_counts[np.asarray(row["natural_twin_ids"], dtype=np.int64)] += 1

    expected_high_ids = [pool.twin_ids[high_incidence[:, index]].tolist() for index in range(CYCLES)]
    expected_natural_ids = [pool.twin_ids[natural_incidence[:, index]].tolist() for index in range(CYCLES)]
    row_high_ids = [
        [int(twin) for twin in row["high_twin_ids"]]
        for row in schedule
    ]
    row_natural_ids = [
        [int(twin) for twin in row["natural_twin_ids"]]
        for row in schedule
    ]
    for cycle in range(CYCLES):
        observed_high = row_high_ids[cycle * BATCHES_PER_CYCLE : (cycle + 1) * BATCHES_PER_CYCLE]
        observed_natural = row_natural_ids[cycle * BATCHES_PER_CYCLE : (cycle + 1) * BATCHES_PER_CYCLE]
        require(
            sorted(twin for block in observed_high for twin in block) == sorted(expected_high_ids[cycle]),
            f"{arm} weighted incidence changed in cycle {cycle}",
        )
        require(
            sorted(twin for block in observed_natural for twin in block) == sorted(expected_natural_ids[cycle]),
            f"{arm} natural incidence changed in cycle {cycle}",
        )

    realized_total_counts = realized_high_counts + realized_natural_counts
    expected_high_counts, expected_desired_high = largest_remainder_counts(
        pool.high_weights[arm], pool.cells, pool.twin_ids, slots_per_cell=ARM_SLOTS_PER_CELL
    )
    expected_full_counts, expected_desired_full = largest_remainder_counts(
        pool.full_weights[arm], pool.cells, pool.twin_ids, slots_per_cell=2 * ARM_SLOTS_PER_CELL
    )
    require(np.array_equal(realized_high_counts, high_incidence.sum(axis=1)), f"{arm} high incidence counts changed")
    require(np.array_equal(realized_natural_counts, natural_incidence.sum(axis=1)), f"{arm} natural incidence counts changed")
    require(np.array_equal(realized_high_counts, high_counts), f"{arm} realized high counts differ from Hamilton target")
    require(np.array_equal(realized_total_counts, full_counts), f"{arm} realized full counts differ from target")
    realized_high = realized_high_counts.astype(np.float64) / TOTAL_ARM_SLOTS
    realized_full = realized_total_counts.astype(np.float64) / TOTAL_TWIN_SLOTS
    high_error = _error_report(pool.high_weights[arm], realized_high)
    full_error = _error_report(pool.full_weights[arm], realized_full)
    per_cell_high = all(
        int(realized_high_counts[pool.cells == cell].sum()) == ARM_SLOTS_PER_CELL for cell in range(EXPECTED_CELLS)
    )
    per_cell_full = all(
        int(realized_total_counts[pool.cells == cell].sum()) == 2 * ARM_SLOTS_PER_CELL
        for cell in range(EXPECTED_CELLS)
    )
    per_cycle_cell = all(
        int(high_incidence[pool.cells == cell, cycle].sum()) == ARM_SLOTS_PER_CELL_CYCLE
        and int(natural_incidence[pool.cells == cell, cycle].sum()) == ARM_SLOTS_PER_CELL_CYCLE
        for cell in range(EXPECTED_CELLS)
        for cycle in range(CYCLES)
    )
    orientation_cycle = all(
        int(high_incidence[pool.orientation == orientation, cycle].sum())
        + int(natural_incidence[pool.orientation == orientation, cycle].sum())
        == 2 * ARM_SLOTS_PER_CELL_CYCLE * 16
        for orientation in range(4)
        for cycle in range(CYCLES)
    )
    checks = {
        "schedule_has_8192_batches": len(schedule) == CYCLES * BATCHES_PER_CYCLE,
        "batch_arm_quota_and_unique_twins_exact": batch_integrity,
        "pair_integrity_exact": pair_integrity,
        "natural_multiplicity_exactly_16": bool(np.all(realized_natural_counts == NATURAL_COUNT_PER_TWIN)),
        "high_multiplicity_matches_largest_remainder": bool(np.array_equal(realized_high_counts, high_counts)),
        "full_multiplicity_matches_largest_remainder": bool(np.array_equal(realized_total_counts, full_counts)),
        "largest_remainder_high_target_recorded": bool(
            np.array_equal(high_counts, expected_high_counts)
            and np.allclose(desired_high_counts, expected_desired_high, rtol=0.0, atol=0.0)
        ),
        "largest_remainder_full_target_recorded": bool(
            np.array_equal(full_counts, expected_full_counts)
            and np.allclose(desired_full_counts, expected_desired_full, rtol=0.0, atol=0.0)
        ),
        "per_cell_high_slots_exact": per_cell_high,
        "per_cell_full_slots_exact": per_cell_full,
        "per_arm_cycle_cell_slots_exact": per_cycle_cell,
        "orientation_cycle_slots_exact": orientation_cycle,
        "positive_full_support": bool(np.all(realized_full > 0.0)),
        "high_realized_tv_below_one_over_32": high_error["total_variation"] < HIGH_TV_BOUND,
        "full_realized_tv_below_one_over_64": full_error["total_variation"] < FULL_TV_BOUND,
    }
    require(all(checks.values()), f"one or more {arm} schedule invariants failed")
    return {
        "checks": checks,
        "realized_high_counts": realized_high_counts,
        "realized_natural_counts": realized_natural_counts,
        "realized_total_counts": realized_total_counts,
        "desired_high_counts": desired_high_counts,
        "desired_full_counts": desired_full_counts,
        "high_error": high_error,
        "full_error": full_error,
    }


def _multiplicity_rows(
    pool: ComparatorPool,
    arm: str,
    audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    realized_high_counts = np.asarray(audit["realized_high_counts"], dtype=np.int64)
    realized_natural_counts = np.asarray(audit["realized_natural_counts"], dtype=np.int64)
    realized_total_counts = np.asarray(audit["realized_total_counts"], dtype=np.int64)
    desired_high_counts = np.asarray(audit["desired_high_counts"], dtype=np.float64)
    desired_full_counts = np.asarray(audit["desired_full_counts"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, twin_id in enumerate(pool.twin_ids.tolist()):
        rows.append(
            {
                "arm": arm,
                "twin_id": int(twin_id),
                "pair_ids": list(pool.pair_ids[index]),
                "coverage_cell": int(pool.cells[index]),
                "projected_high_weight": float(pool.high_weights[arm][index]),
                "projected_full_weight": float(pool.full_weights[arm][index]),
                "desired_high_count": float(desired_high_counts[index]),
                "realized_high_count": int(realized_high_counts[index]),
                "realized_natural_count": int(realized_natural_counts[index]),
                "desired_full_count": float(desired_full_counts[index]),
                "realized_full_count": int(realized_total_counts[index]),
                "realized_high_weight": float(realized_high_counts[index] / TOTAL_ARM_SLOTS),
                "realized_full_weight": float(realized_total_counts[index] / TOTAL_TWIN_SLOTS),
            }
        )
    return rows


def audit_shared_layout(
    pool: ComparatorPool,
    schedules: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Check the common per-batch cell/rank signatures across all arms."""

    require(set(schedules) == set(ARMS), "shared-layout arm set changed")
    length = CYCLES * BATCHES_PER_CYCLE
    require(all(len(schedules[arm]) == length for arm in ARMS), "shared-layout batch count changed")

    def coverage_signature(row: Mapping[str, Any]) -> tuple[int, ...]:
        return tuple(sorted(int(pool.cells[int(twin)]) for twin in row["twin_ids"]))

    def natural_signature(row: Mapping[str, Any]) -> tuple[int, ...]:
        return tuple(int(twin) for twin in row["natural_twin_ids"])

    def rank_signature(row: Mapping[str, Any]) -> tuple[int, ...]:
        return tuple(int(slot) for slot in row["high_slot_ids"])

    reference = schedules["REL50"]
    coverage_same = all(
        all(coverage_signature(schedules[arm][index]) == coverage_signature(reference[index]) for index in range(length))
        for arm in ARMS
    )
    natural_same = all(
        all(natural_signature(schedules[arm][index]) == natural_signature(reference[index]) for index in range(length))
        for arm in ARMS
    )
    comparator_rank_same = all(
        rank_signature(schedules["REL50"][index])
        == rank_signature(schedules["ABS50"][index])
        == rank_signature(schedules["HASH50"][index])
        for index in range(length)
    )
    quota_same = all(
        all(
            len(schedules[arm][index]["high_twin_ids"]) == ARM_SLOTS_PER_BATCH
            and len(schedules[arm][index]["natural_twin_ids"]) == ARM_SLOTS_PER_BATCH
            and len(schedules[arm][index]["twin_ids"]) == TWINS_PER_BATCH
            for index in range(length)
        )
        for arm in ARMS
    )
    index_same = all(
        all(
            schedules[arm][index]["schedule_index"] == reference[index]["schedule_index"]
            and schedules[arm][index]["optimizer_step"] == reference[index]["optimizer_step"]
            and schedules[arm][index]["cycle"] == reference[index]["cycle"]
            and schedules[arm][index]["batch_in_cycle"] == reference[index]["batch_in_cycle"]
            for index in range(length)
        )
        for arm in ARMS
    )
    checks = {
        "shared_batch_coverage_cell_signature_exact": coverage_same,
        "shared_natural_batch_signature_exact": natural_same,
        "shared_comparator_rank_weight_signature_exact": comparator_rank_same,
        "shared_arm_quota_exact": quota_same,
        "shared_batch_index_cycle_signature_exact": index_same,
        "same_budget_and_coverage_structure": coverage_same and quota_same and index_same,
    }
    require(all(checks.values()), "shared schedule layout invariant failed")
    return {
        "checks": checks,
        "comparison_arms": ["REL50", "ABS50", "HASH50"],
        "d0_uniform_rank_signature_exempt": True,
        "cross_arm_conflict_rule": "natural block is matched only when disjoint from every arm's weighted block",
        "cross_arm_conflict_checks": "all arms are checked jointly before each shared natural block is assigned",
    }


def build_summary(
    pool: ComparatorPool,
    audits: Mapping[str, Mapping[str, Any]],
    frozen: Mapping[str, Any],
    shared_audit: Mapping[str, Any],
) -> dict[str, Any]:
    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        audit = audits[arm]
        realized_high_counts = np.asarray(audit["realized_high_counts"], dtype=np.int64)
        realized_natural_counts = np.asarray(audit["realized_natural_counts"], dtype=np.int64)
        realized_total_counts = np.asarray(audit["realized_total_counts"], dtype=np.int64)
        arm_summary[arm] = {
            "projected_weight_source": "comparator projected_weights.jsonl",
            "multiplicity": {
                "natural_count_per_twin": NATURAL_COUNT_PER_TWIN,
                "high_count_quantiles": _quantiles(realized_high_counts),
                "full_count_quantiles": _quantiles(realized_total_counts),
                "high_count_sum": int(realized_high_counts.sum()),
                "natural_count_sum": int(realized_natural_counts.sum()),
                "full_count_sum": int(realized_total_counts.sum()),
            },
            "exposure_realized_error": {
                "high": audit["high_error"],
                "full": audit["full_error"],
            },
            "checks": audit["checks"],
        }
    all_checks = {
        f"{arm.lower()}_{name}": value
        for arm in ARMS
        for name, value in audits[arm]["checks"].items()
    }
    all_checks["four_arm_set_exact"] = True
    all_checks.update({str(name): bool(value) for name, value in shared_audit["checks"].items()})
    all_checks["frozen_comparator_inputs_verified"] = True
    all_checks["training_only_boundary_exact"] = True
    return {
        "schema_version": 1,
        "builder_id": "pusht_motion_damping_root_cause_comparator_schedules_v1",
        "candidate_id": "D1-MS50",
        "status": "passed_go",
        "construction_seed": CONSTRUCTION_SEED,
        "training_seed": TRAINING_SEED,
        "source_comparator_sha256": frozen["observed_sha256"],
        "dimensions": {
            "arms": list(ARMS),
            "twins": EXPECTED_TWINS,
            "coverage_cells": EXPECTED_CELLS,
            "twins_per_cell": TWINS_PER_CELL,
            "cycles": CYCLES,
            "batches_per_cycle": BATCHES_PER_CYCLE,
            "optimizer_steps": CYCLES * BATCHES_PER_CYCLE,
            "twins_per_batch": TWINS_PER_BATCH,
            "hidden_rows_per_batch": 4 * TWINS_PER_BATCH,
            "total_twin_exposures_per_arm": TOTAL_TWIN_SLOTS,
            "total_hidden_rows_per_arm": 4 * TOTAL_TWIN_SLOTS,
        },
        "arms": arm_summary,
        "shared_layout_audit": dict(shared_audit),
        "gates": {"checks": all_checks, "passed": all(all_checks.values())},
        "construction": {
            "D0": "uniform high component plus shared natural anchor",
            "REL50": "comparator pi_ms50_rel50 with shared natural anchor",
            "ABS50": "comparator pi_abs50 with shared natural anchor",
            "HASH50": "comparator pi_hash50 with shared natural anchor",
            "integerization": "per-cell Hamilton largest remainder; descending fractional remainder; twin_id tie break",
            "cycle_allocation": "descending multiplicity; least-loaded cycle; SHA256 tie break",
            "batch_construction": "deterministic block order plus conflict-free perfect matching",
            "pair_integrity": "both forward/reverse source pair rows stay in one complete twin batch",
            "exposure_only_difference": "budget, seed, cell quotas and natural incidence are shared; only weighted exposure differs by arm",
        },
        "evidence_boundary": {
            "training_only": True,
            "development_lance_opened": False,
            "public_test_lance_opened": False,
            "validation_lance_opened": False,
            "pixels_decoded": False,
            "model_loaded": False,
            "optimizer_steps_run": 0,
            "schedule_generated": True,
            "claim": "This artifact only constructs and audits matched Training exposure schedules; it does not establish a model, latent, gradient, calibration, or native ICL improvement.",
        },
    }


def write_artifacts(
    output_dir: Path,
    *,
    frozen: Mapping[str, Any],
    pool: ComparatorPool,
    schedules: Mapping[str, Sequence[Mapping[str, Any]]],
    audits: Mapping[str, Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Path]:
    """Write an exclusive schedule artifact and a complete receipt."""

    output_dir = Path(output_dir).expanduser().resolve()
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    receipt_path = output_dir / "receipt.json"
    schedule_paths = {arm: output_dir / f"schedule_{arm}.jsonl" for arm in ARMS}
    multiplicity_paths = {arm: output_dir / f"multiplicity_{arm}.jsonl" for arm in ARMS}
    output_paths: dict[str, Path] = {
        "config.json": config_path,
        "summary.json": summary_path,
        "receipt.json": receipt_path,
    }
    output_paths.update({f"schedule_{arm}.jsonl": path for arm, path in schedule_paths.items()})
    output_paths.update({f"multiplicity_{arm}.jsonl": path for arm, path in multiplicity_paths.items()})

    config = {
        "schema_version": 1,
        "builder_id": "pusht_motion_damping_root_cause_comparator_schedules_v1",
        "candidate_id": "D1-MS50",
        "source_comparator_dir": str(frozen["comparator_dir"]),
        "source_comparator_sha256": frozen["observed_sha256"],
        "comparator_builder_script_sha256": frozen["comparator_builder_sha256"],
        "schedule_builder_script_sha256": file_sha256(Path(__file__).resolve()),
        "arms": list(ARMS),
        "construction_seed": CONSTRUCTION_SEED,
        "training_seed": TRAINING_SEED,
        "cycles": CYCLES,
        "batches_per_cycle": BATCHES_PER_CYCLE,
        "optimizer_steps": CYCLES * BATCHES_PER_CYCLE,
        "twins_per_batch": TWINS_PER_BATCH,
        "weighted_slots_per_batch": ARM_SLOTS_PER_BATCH,
        "natural_slots_per_batch": ARM_SLOTS_PER_BATCH,
        "coverage_cells": EXPECTED_CELLS,
        "twins_per_cell": TWINS_PER_CELL,
        "integerization": "per-cell Hamilton; descending remainder; twin_id tie break",
        "cycle_allocation": "descending multiplicity; least-loaded cycle; SHA256 tie break",
        "schedule_files": {arm: path.name for arm, path in schedule_paths.items()},
        "multiplicity_files": {arm: path.name for arm, path in multiplicity_paths.items()},
        "evidence_boundary": summary["evidence_boundary"],
    }
    json_dump(config_path, config)
    for arm in ARMS:
        with multiplicity_paths[arm].open("x", encoding="utf-8") as stream:
            for row in _multiplicity_rows(pool, arm, audits[arm]):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with schedule_paths[arm].open("x", encoding="utf-8") as stream:
            for row in schedules[arm]:
                stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    json_dump(summary_path, summary)

    output_sha256 = {
        name: file_sha256(path)
        for name, path in output_paths.items()
        if name != "receipt.json"
    }
    receipt = {
        "schema_version": 1,
        "builder_id": summary["builder_id"],
        "candidate_id": summary["candidate_id"],
        "status": summary["status"],
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "input_sha256": {
            **{f"comparator_{name}": value for name, value in frozen["observed_sha256"].items()},
            "comparator_builder_script": frozen["comparator_builder_sha256"],
            "schedule_builder_script": file_sha256(Path(__file__).resolve()),
        },
        "output_sha256": output_sha256,
        "comparator_builder_script_sha256": frozen["comparator_builder_sha256"],
        "schedule_builder_script_sha256": file_sha256(Path(__file__).resolve()),
        "gates": summary["gates"],
        "development_lance_opened": False,
        "public_test_lance_opened": False,
        "validation_lance_opened": False,
        "pixels_decoded": False,
        "model_loaded": False,
        "optimizer_steps_run": 0,
        "schedule_generated": True,
        "evidence_boundary": summary["evidence_boundary"],
    }
    json_dump(receipt_path, receipt)
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator-dir", type=Path, default=DEFAULT_COMPARATOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify inputs and construct all reports in memory without writing artifacts",
    )
    return parser.parse_args(argv)


def construct(
    comparator_dir: Path = DEFAULT_COMPARATOR_DIR,
) -> tuple[dict[str, Any], ComparatorPool, dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Construct all four schedules and return their audited in-memory state."""

    frozen = verify_frozen_inputs(comparator_dir)
    pool = load_comparator_pool(frozen["paths"]["projected_weights.jsonl"])
    natural_incidence = natural_cycle_incidence(pool.cells, pool.twin_ids)
    abstract_incidence, actual_incidence, rank_to_twin = build_abstract_high_incidences(pool)
    shared_plan = build_shared_plan(
        pool,
        abstract_incidence,
        natural_incidence,
        rank_to_twin,
    )
    schedules: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        high_counts, desired_high_counts = largest_remainder_counts(
            pool.high_weights[arm], pool.cells, pool.twin_ids, slots_per_cell=ARM_SLOTS_PER_CELL
        )
        full_counts, desired_full_counts = largest_remainder_counts(
            pool.full_weights[arm], pool.cells, pool.twin_ids, slots_per_cell=2 * ARM_SLOTS_PER_CELL
        )
        require(np.all(high_counts <= CYCLES), f"{arm} Hamilton count exceeds cycle support")
        require(np.all(full_counts == high_counts + NATURAL_COUNT_PER_TWIN), f"{arm} full/high target mismatch")
        high_incidence = actual_incidence[arm]
        require(
            np.array_equal(high_incidence.sum(axis=1), high_counts),
            f"{arm} shared slot incidence does not realize Hamilton counts",
        )
        schedules[arm] = build_schedule(
            pool,
            high_incidence,
            natural_incidence,
            arm=arm,
            exposure_weights=pool.full_weights[arm],
            high_weights=pool.high_weights[arm],
            rank_to_twin=rank_to_twin[arm],
            shared_plan=shared_plan,
        )
        audits[arm] = audit_arm_schedule(
            pool,
            arm,
            schedules[arm],
            high_incidence,
            natural_incidence,
            high_counts,
            desired_high_counts,
            full_counts,
            desired_full_counts,
            shared_plan=shared_plan,
            rank_to_twin=rank_to_twin[arm],
        )
    shared_audit = audit_shared_layout(pool, schedules)
    summary = build_summary(pool, audits, frozen, shared_audit)
    require(summary["gates"]["passed"] is True, "one or more four-arm schedule invariants failed")
    return frozen, pool, schedules, audits, summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frozen, pool, schedules, audits, summary = construct(args.comparator_dir)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "arms": list(ARMS),
                    "gates": summary["gates"],
                    "exposure_realized_error": {
                        arm: summary["arms"][arm]["exposure_realized_error"] for arm in ARMS
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    paths = write_artifacts(
        args.output_dir,
        frozen=frozen,
        pool=pool,
        schedules=schedules,
        audits=audits,
        summary=summary,
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
