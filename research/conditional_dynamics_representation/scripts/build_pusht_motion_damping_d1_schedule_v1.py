#!/usr/bin/env python3
"""Build the frozen Training-only D1-MS50 integer exposure schedule."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = RESEARCH_ROOT / "scripts/audit_pusht_motion_damping_d1_metric_v1.py"
V2_SCRIPT = RESEARCH_ROOT / (
    "scripts/audit_pusht_motion_damping_d1_multiscale_soft_v2.py"
)
DEFAULT_V2_DIR = RESEARCH_ROOT / (
    "artifacts/pusht_motion_damping_d1_multiscale_soft_v2/"
    "d1_0_training_only_v2"
)
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / (
    "artifacts/pusht_motion_damping_d1_schedule_v1/d1_ms50_schedule_v1_final"
)

EXPECTED_V2_SHA256 = {
    "config.json": "e1fcfdf41c1756116eb5471b77e7ba45c097e77b1987e678257ca1dfbe41d786",
    "summary.json": "05d745ebddaab4a9a8ec7a1dfa7bd504b27efb217c5205561361dc0d683d614e",
    "projected_weights.jsonl": (
        "6a45c6f18e1eeb61f184c9977b81b08f4de384ee4c9c36cfa41e186dca755afa"
    ),
    "receipt.json": "7d7f37ec89d41f9b5e7c5a5c1f4c134459e00fcd71deb1fabac713dc26b85ac5",
}
EXPECTED_TRAIN_LANCE_SHA256 = (
    "085a4d7bb60f5ec31215c3bad452c130ab90bd04b7cd80573211848bb2a13b05"
)
EXPECTED_TWINS = 4096
EXPECTED_CELLS = 64
TWINS_PER_CELL = 64
KS = (32, 64, 128)
CONSTRUCTION_SEED = 20260831
CYCLES = 32
BATCHES_PER_CYCLE = 256
TWINS_PER_BATCH = 16
ARM_SLOTS_PER_BATCH = 8
ARM_SLOTS_PER_CYCLE = BATCHES_PER_CYCLE * ARM_SLOTS_PER_BATCH
TOTAL_ARM_SLOTS = CYCLES * ARM_SLOTS_PER_CYCLE
ARM_SLOTS_PER_CELL = TOTAL_ARM_SLOTS // EXPECTED_CELLS
ARM_SLOTS_PER_CELL_CYCLE = ARM_SLOTS_PER_CYCLE // EXPECTED_CELLS
NATURAL_COUNT_PER_TWIN = TOTAL_ARM_SLOTS // EXPECTED_TWINS
HIGH_TV_BOUND = EXPECTED_TWINS / (2.0 * TOTAL_ARM_SLOTS)
FULL_TV_BOUND = HIGH_TV_BOUND / 2.0
MASS_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class ProjectedPool:
    twin_ids: np.ndarray
    pair_ids: tuple[tuple[str, str], ...]
    cells: np.ndarray
    orientation: np.ndarray
    speed_bin: np.ndarray
    goal_bin: np.ndarray
    conditional: np.ndarray
    background: dict[int, np.ndarray]
    ranks: dict[int, np.ndarray]
    r_multiscale: np.ndarray
    pi_high: np.ndarray
    pi_full: np.ndarray


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_hash(seed: int, *parts: Any) -> bytes:
    payload = "|".join([str(int(seed)), *(str(value) for value in parts)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def assert_training_only_path(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    require(path.name == "train.lance", "builder accepts only train.lance")
    forbidden = {
        (path.parent / name).resolve()
        for name in ("loader_validation.lance", "validation.lance", "test.lance")
    }
    require(path not in forbidden, "Development/Public Test table is forbidden")
    require(path.is_dir(), f"missing frozen Training table: {path}")
    return path


def _finite(name: str, values: np.ndarray) -> None:
    require(bool(np.isfinite(values).all()), f"{name} contains non-finite values")


def load_projected_pool(path: Path) -> ProjectedPool:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    require(len(rows) == EXPECTED_TWINS, "unexpected projected twin count")
    twin_ids = np.asarray([row["twin_id"] for row in rows], dtype=np.int64)
    require(
        np.array_equal(twin_ids, np.arange(EXPECTED_TWINS)),
        "projected twin ids are not contiguous",
    )
    pair_ids = tuple(tuple(str(value) for value in row["pair_ids"]) for row in rows)
    require(all(len(value) == 2 for value in pair_ids), "invalid source pair ids")
    for twin_id, observed_pair_ids in enumerate(pair_ids):
        expected_pair_ids = (
            f"pmd-train-{2 * twin_id:06d}-forward",
            f"pmd-train-{2 * twin_id + 1:06d}-reverse",
        )
        require(
            observed_pair_ids == expected_pair_ids,
            f"source pair-to-twin row mapping changed at twin {twin_id}",
        )
    cells = np.asarray([row["coverage_cell"] for row in rows], dtype=np.int64)
    orientation = np.asarray([row["orientation_bin"] for row in rows], dtype=np.int64)
    speed_bin = np.asarray([row["speed_bin"] for row in rows], dtype=np.int64)
    goal_bin = np.asarray([row["goal_distance_bin"] for row in rows], dtype=np.int64)
    conditional = np.asarray(
        [row["conditional_energy_physical"] for row in rows], dtype=np.float64
    )
    background = {
        k: np.asarray(
            [row["background_future_variation"][str(k)] for row in rows],
            dtype=np.float64,
        )
        for k in KS
    }
    ranks = {
        k: np.asarray(
            [row["stable_rank_by_k"][str(k)] for row in rows], dtype=np.int64
        )
        for k in KS
    }
    r_multiscale = np.asarray(
        [row["r_multiscale"] for row in rows], dtype=np.float64
    )
    pi_high = np.asarray([row["pi_high"] for row in rows], dtype=np.float64)
    pi_full = np.asarray([row["pi_ms50"] for row in rows], dtype=np.float64)

    for name, values in (
        ("conditional", conditional),
        ("r_multiscale", r_multiscale),
        ("pi_high", pi_high),
        ("pi_full", pi_full),
    ):
        _finite(name, values)
    for k in KS:
        _finite(f"background_{k}", background[k])
        require(np.all(background[k] >= 0.0), f"negative B_{k}")
    require(np.all(conditional > 0.0), "conditional energy must be positive")
    require(np.all(pi_high > 0.0) and np.all(pi_full > 0.0), "non-positive weights")
    require(math.isclose(float(pi_high.sum()), 1.0, abs_tol=MASS_TOLERANCE), "bad high mass")
    require(math.isclose(float(pi_full.sum()), 1.0, abs_tol=MASS_TOLERANCE), "bad full mass")
    require(
        np.allclose(
            pi_full,
            0.5 / EXPECTED_TWINS + 0.5 * pi_high,
            rtol=0.0,
            atol=1.0e-15,
        ),
        "pi_MS50 is not the frozen half-uniform mixture",
    )
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
    require(
        np.allclose(
            r_multiscale,
            np.stack([ranks[k] for k in KS], axis=0).mean(axis=0),
            rtol=0.0,
            atol=1.0e-15,
        ),
        "multiscale rank changed",
    )
    for cell in range(EXPECTED_CELLS):
        mask = cells == cell
        require(
            math.isclose(float(pi_high[mask].sum()), 1.0 / EXPECTED_CELLS, abs_tol=1e-15),
            f"high mass changed in cell {cell}",
        )
        for k in KS:
            require(
                np.array_equal(np.sort(ranks[k][mask]), np.arange(1, 65)),
                f"rank permutation changed for cell {cell}, k={k}",
            )
        expected = (1.0 / EXPECTED_CELLS) * r_multiscale[mask] / float(
            r_multiscale[mask].sum()
        )
        require(
            np.allclose(pi_high[mask], expected, rtol=0.0, atol=1e-15),
            f"soft high weights changed in cell {cell}",
        )
    return ProjectedPool(
        twin_ids=twin_ids,
        pair_ids=pair_ids,
        cells=cells,
        orientation=orientation,
        speed_bin=speed_bin,
        goal_bin=goal_bin,
        conditional=conditional,
        background=background,
        ranks=ranks,
        r_multiscale=r_multiscale,
        pi_high=pi_high,
        pi_full=pi_full,
    )


def largest_remainder_counts(
    probabilities: np.ndarray,
    cells: np.ndarray,
    stable_ids: np.ndarray,
    *,
    slots_per_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    require(
        probabilities.shape == cells.shape == stable_ids.shape,
        "largest-remainder shape mismatch",
    )
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
    require(cycles > 0 and cycles % 2 == 0, "natural cycles must be positive and even")
    cells = np.asarray(cells, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
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
    counts = np.asarray(counts, dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    require(np.all((counts >= 0) & (counts <= cycles)), "high count exceeds cycle support")
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
                    stable_hash(
                        seed,
                        "high-cycle-choice",
                        cell,
                        int(stable_ids[index]),
                        cycle,
                    ),
                    cycle,
                ),
            )
            selected = cycle_order[:count]
            incidence[index, selected] = True
            loads[selected] += 1
        require(
            np.all(loads == target_load),
            f"high cycle allocation is imbalanced in cell {cell}",
        )
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


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape, "TV shape mismatch")
    require(
        math.isclose(float(left.sum()), 1.0, abs_tol=1.0e-12)
        and math.isclose(float(right.sum()), 1.0, abs_tol=1.0e-12),
        "TV inputs are not normalized",
    )
    return 0.5 * float(np.abs(left - right).sum())


def build_schedule(
    pool: ProjectedPool,
    high_incidence: np.ndarray,
    natural_incidence: np.ndarray,
    *,
    seed: int = CONSTRUCTION_SEED,
) -> list[dict[str, Any]]:
    require(
        high_incidence.shape == natural_incidence.shape == (EXPECTED_TWINS, CYCLES),
        "schedule incidence shape mismatch",
    )
    schedule: list[dict[str, Any]] = []
    for cycle in range(CYCLES):
        high_ids = pool.twin_ids[high_incidence[:, cycle]].tolist()
        natural_ids = pool.twin_ids[natural_incidence[:, cycle]].tolist()
        require(len(high_ids) == ARM_SLOTS_PER_CYCLE, "bad high-arm cycle size")
        require(len(natural_ids) == ARM_SLOTS_PER_CYCLE, "bad natural-arm cycle size")
        high_blocks = deterministic_blocks(
            high_ids,
            block_size=ARM_SLOTS_PER_BATCH,
            seed=seed,
            label="high-block-order",
            cycle=cycle,
        )
        natural_blocks = deterministic_blocks(
            natural_ids,
            block_size=ARM_SLOTS_PER_BATCH,
            seed=seed,
            label="natural-block-order",
            cycle=cycle,
        )
        matching = match_disjoint_blocks(
            high_blocks, natural_blocks, seed=seed, cycle=cycle
        )
        for batch_in_cycle, high_block in enumerate(high_blocks):
            natural_block = natural_blocks[matching[batch_in_cycle]]
            twins = [*high_block, *natural_block]
            require(len(set(twins)) == TWINS_PER_BATCH, "cross-arm batch conflict")
            schedule_index = cycle * BATCHES_PER_CYCLE + batch_in_cycle
            schedule.append(
                {
                    "schedule_index": schedule_index,
                    "optimizer_step": schedule_index + 1,
                    "cycle": cycle,
                    "batch_in_cycle": batch_in_cycle,
                    "high_twin_ids": list(high_block),
                    "natural_twin_ids": list(natural_block),
                    "twin_ids": twins,
                    "hidden_row_indices": expand_hidden_rows(twins),
                }
            )
    return schedule


def verify_frozen_inputs(v2_dir: Path) -> dict[str, Any]:
    v2_dir = Path(v2_dir).expanduser().resolve()
    observed = {name: file_sha256(v2_dir / name) for name in EXPECTED_V2_SHA256}
    require(observed == EXPECTED_V2_SHA256, "frozen v2 artifact SHA256 changed")
    config = json.loads((v2_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((v2_dir / "summary.json").read_text(encoding="utf-8"))
    receipt = json.loads((v2_dir / "receipt.json").read_text(encoding="utf-8"))
    require(summary["status"] == receipt["status"] == "passed_go", "v2 is not go")
    require(
        summary["candidate_id"] == receipt["candidate_id"] == "D1-MS50",
        "v2 candidate identity changed",
    )
    require(summary["gates"]["passed"] and receipt["gates"]["passed"], "v2 gates failed")
    for name in ("config.json", "summary.json", "projected_weights.jsonl"):
        require(receipt["output_sha256"][name] == observed[name], f"v2 receipt mismatch: {name}")
    require(
        file_sha256(V1_SCRIPT) == receipt["input_sha256"]["v1_audit_script"],
        "v1 audit source changed after v2",
    )
    require(
        file_sha256(V2_SCRIPT) == receipt["input_sha256"]["audit_script"],
        "v2 audit source changed after formal output",
    )
    train_lance = assert_training_only_path(Path(config["inputs"]["train_lance"]))
    manifest = Path(config["inputs"]["manifest"]).expanduser().resolve()
    release_config = Path(config["inputs"]["release_config"]).expanduser().resolve()
    require(file_sha256(manifest) == receipt["input_sha256"]["manifest"], "manifest changed")
    require(
        file_sha256(release_config) == receipt["input_sha256"]["release_config"],
        "release config changed",
    )
    train_sha = directory_sha256(train_lance)
    require(train_sha == EXPECTED_TRAIN_LANCE_SHA256, "train.lance directory changed")
    require(
        train_sha == receipt["input_sha256"]["train_lance_directory"],
        "v2 receipt train identity changed",
    )
    for key in ("v1_catalog", "v1_summary"):
        input_path = Path(config["inputs"][key]).expanduser().resolve()
        receipt_key = "v1_per_twin_catalog" if key == "v1_catalog" else "v1_summary"
        require(file_sha256(input_path) == receipt["input_sha256"][receipt_key], f"{key} changed")
    boundary = summary["evidence_boundary"]
    require(
        boundary["development_lance_opened"] is False
        and boundary["public_test_lance_opened"] is False
        and boundary["optimizer_steps"] == 0
        and boundary["model_loaded"] is False
        and boundary["schedule_generated"] is False
        and boundary["pixels_decoded"] is False,
        "v2 evidence boundary changed",
    )
    return {
        "observed_v2_sha256": observed,
        "config": config,
        "summary": summary,
        "receipt": receipt,
        "train_lance": train_lance,
        "train_lance_sha256": train_sha,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        label: float(value)
        for label, value in zip(
            ("min", "q25", "median", "q75", "max"),
            np.quantile(np.asarray(values, dtype=np.float64), [0.0, 0.25, 0.5, 0.75, 1.0]),
            strict=True,
        )
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    v2_dir = args.v2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    frozen = verify_frozen_inputs(v2_dir)
    weights_path = v2_dir / "projected_weights.jsonl"
    pool = load_projected_pool(weights_path)
    high_counts, desired_high_counts = largest_remainder_counts(
        pool.pi_high,
        pool.cells,
        pool.twin_ids,
        slots_per_cell=ARM_SLOTS_PER_CELL,
    )
    require(np.all(high_counts <= CYCLES), "Hamilton count exceeds available cycles")
    natural_incidence = natural_cycle_incidence(pool.cells, pool.twin_ids)
    high_incidence = high_cycle_incidence(high_counts, pool.cells, pool.twin_ids)
    schedule = build_schedule(pool, high_incidence, natural_incidence)

    natural_counts = natural_incidence.sum(axis=1).astype(np.int64)
    realized_high_counts = high_incidence.sum(axis=1).astype(np.int64)
    total_counts = natural_counts + realized_high_counts
    realized_high = realized_high_counts.astype(np.float64) / TOTAL_ARM_SLOTS
    realized_full = total_counts.astype(np.float64) / (2 * TOTAL_ARM_SLOTS)
    high_tv = total_variation(pool.pi_high, realized_high)
    full_tv = total_variation(pool.pi_full, realized_full)

    v2_summary = frozen["summary"]
    d0 = v2_summary["candidates"]["D0"]
    weighted_conditional = float(np.dot(realized_full, pool.conditional))
    realized_by_k: dict[str, Any] = {}
    for k in KS:
        weighted_background = float(np.dot(realized_full, pool.background[k]))
        rho = weighted_conditional / (weighted_conditional + weighted_background)
        d0_rho = float(d0["rho_phys_ratio_of_means_by_k"][str(k)])
        realized_by_k[str(k)] = {
            "d0_rho_phys": d0_rho,
            "realized_rho_phys": rho,
            "absolute_delta": rho - d0_rho,
            "relative_delta": (rho - d0_rho) / d0_rho,
            "realized_weighted_background": weighted_background,
        }

    conflict_count = sum(
        len(set(row["high_twin_ids"]) & set(row["natural_twin_ids"]))
        for row in schedule
    )
    row_expansion_ok = all(
        row["hidden_row_indices"] == expand_hidden_rows(row["twin_ids"])
        for row in schedule
    )
    per_cell_high_ok = all(
        int(realized_high_counts[pool.cells == cell].sum()) == ARM_SLOTS_PER_CELL
        for cell in range(EXPECTED_CELLS)
    )
    per_cell_total_ok = all(
        int(total_counts[pool.cells == cell].sum()) == 2 * ARM_SLOTS_PER_CELL
        for cell in range(EXPECTED_CELLS)
    )
    per_cycle_cell_ok = all(
        int(high_incidence[pool.cells == cell, cycle].sum())
        == ARM_SLOTS_PER_CELL_CYCLE
        and int(natural_incidence[pool.cells == cell, cycle].sum())
        == ARM_SLOTS_PER_CELL_CYCLE
        for cell in range(EXPECTED_CELLS)
        for cycle in range(CYCLES)
    )
    orientation_cycle_ok = all(
        int(high_incidence[pool.orientation == orientation, cycle].sum())
        + int(natural_incidence[pool.orientation == orientation, cycle].sum())
        == 2 * ARM_SLOTS_PER_CELL_CYCLE * 16
        for orientation in range(4)
        for cycle in range(CYCLES)
    )
    batch_ok = all(
        len(row["high_twin_ids"]) == ARM_SLOTS_PER_BATCH
        and len(row["natural_twin_ids"]) == ARM_SLOTS_PER_BATCH
        and len(row["twin_ids"]) == TWINS_PER_BATCH
        and len(set(row["twin_ids"])) == TWINS_PER_BATCH
        and len(row["hidden_row_indices"]) == 4 * TWINS_PER_BATCH
        for row in schedule
    )
    checks = {
        "schedule_has_8192_batches": len(schedule) == CYCLES * BATCHES_PER_CYCLE,
        "natural_multiplicity_exactly_16": bool(
            np.all(natural_counts == NATURAL_COUNT_PER_TWIN)
        ),
        "high_multiplicity_matches_hamilton": bool(
            np.array_equal(realized_high_counts, high_counts)
        ),
        "high_multiplicity_within_cycle_support": bool(
            np.all((high_counts >= 0) & (high_counts <= CYCLES))
        ),
        "per_cell_high_slots_exact": per_cell_high_ok,
        "per_cell_total_slots_exact": per_cell_total_ok,
        "per_arm_cycle_cell_slots_exact": per_cycle_cell_ok,
        "orientation_cycle_slots_exact": orientation_cycle_ok,
        "batch_arm_quota_and_unique_twins_exact": batch_ok,
        "cross_arm_conflict_count_zero": conflict_count == 0,
        "four_row_expansion_exact": row_expansion_ok,
        "high_realized_tv_below_one_over_32": high_tv < HIGH_TV_BOUND,
        "full_realized_tv_below_one_over_64": full_tv < FULL_TV_BOUND,
        "realized_conditional_energy_above_d0": (
            weighted_conditional > float(d0["weighted_conditional_energy"])
        ),
        **{
            f"realized_rho_phys_k{k}_above_d0": (
                realized_by_k[str(k)]["realized_rho_phys"]
                > realized_by_k[str(k)]["d0_rho_phys"]
            )
            for k in KS
        },
        "frozen_v2_and_training_inputs_verified": True,
    }
    passed = all(checks.values())

    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "config.json"
    multiplicity_path = output_dir / "multiplicity.jsonl"
    schedule_path = output_dir / "schedule.jsonl"
    summary_path = output_dir / "summary.json"
    receipt_path = output_dir / "receipt.json"
    config = {
        "schema_version": 1,
        "builder_id": "pusht_motion_damping_d1_schedule_v1",
        "candidate_id": "D1-MS50",
        "v2_dir": str(v2_dir),
        "output_dir": str(output_dir),
        "construction_seed": CONSTRUCTION_SEED,
        "cycles": CYCLES,
        "batches_per_cycle": BATCHES_PER_CYCLE,
        "twins_per_batch": TWINS_PER_BATCH,
        "high_slots_per_batch": ARM_SLOTS_PER_BATCH,
        "natural_slots_per_batch": ARM_SLOTS_PER_BATCH,
        "integerization": "per-cell Hamilton; descending remainder; twin_id tie break",
        "cycle_allocation": "descending multiplicity; least-loaded cycle; SHA256 tie break",
        "input_sha256": frozen["observed_v2_sha256"],
    }
    json_dump(config_path, config)

    with multiplicity_path.open("x", encoding="utf-8") as stream:
        for index in range(EXPECTED_TWINS):
            row = {
                "twin_id": int(pool.twin_ids[index]),
                "pair_ids": list(pool.pair_ids[index]),
                "coverage_cell": int(pool.cells[index]),
                "orientation_bin": int(pool.orientation[index]),
                "speed_bin": int(pool.speed_bin[index]),
                "goal_distance_bin": int(pool.goal_bin[index]),
                "stable_rank_by_k": {str(k): int(pool.ranks[k][index]) for k in KS},
                "r_multiscale": float(pool.r_multiscale[index]),
                "projected_pi_high": float(pool.pi_high[index]),
                "projected_pi_ms50": float(pool.pi_full[index]),
                "desired_high_count": float(desired_high_counts[index]),
                "realized_high_count": int(realized_high_counts[index]),
                "realized_natural_count": int(natural_counts[index]),
                "realized_total_count": int(total_counts[index]),
                "realized_pi_high": float(realized_high[index]),
                "realized_pi_ms50": float(realized_full[index]),
            }
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with schedule_path.open("x", encoding="utf-8") as stream:
        for row in schedule:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    core_hashes = {
        "multiplicity.jsonl": file_sha256(multiplicity_path),
        "schedule.jsonl": file_sha256(schedule_path),
    }
    summary = {
        "schema_version": 1,
        "builder_id": "pusht_motion_damping_d1_schedule_v1",
        "candidate_id": "D1-MS50",
        "status": "passed_go" if passed else "failed_no_go",
        "construction_seed": CONSTRUCTION_SEED,
        "frozen_input_sha256": frozen["observed_v2_sha256"],
        "dimensions": {
            "twins": EXPECTED_TWINS,
            "coverage_cells": EXPECTED_CELLS,
            "cycles": CYCLES,
            "batches_per_cycle": BATCHES_PER_CYCLE,
            "optimizer_steps": CYCLES * BATCHES_PER_CYCLE,
            "twins_per_batch": TWINS_PER_BATCH,
            "hidden_rows_per_batch": 4 * TWINS_PER_BATCH,
            "total_twin_exposures": 2 * TOTAL_ARM_SLOTS,
            "total_hidden_rows": 4 * 2 * TOTAL_ARM_SLOTS,
        },
        "multiplicity": {
            "natural_count_per_twin": NATURAL_COUNT_PER_TWIN,
            "high_count_quantiles": _quantiles(realized_high_counts),
            "total_count_quantiles": _quantiles(total_counts),
            "high_count_sum": int(realized_high_counts.sum()),
            "natural_count_sum": int(natural_counts.sum()),
        },
        "realized_distribution": {
            "high_tv_vs_projection": high_tv,
            "high_tv_strict_bound": HIGH_TV_BOUND,
            "full_tv_vs_projection": full_tv,
            "full_tv_strict_bound": FULL_TV_BOUND,
            "weighted_conditional_energy": weighted_conditional,
            "d0_weighted_conditional_energy": float(d0["weighted_conditional_energy"]),
            "by_k": realized_by_k,
        },
        "batch_audit": {
            "cross_arm_conflict_count": conflict_count,
            "per_batch_high_slots": ARM_SLOTS_PER_BATCH,
            "per_batch_natural_slots": ARM_SLOTS_PER_BATCH,
            "four_rows_per_twin": True,
        },
        "gates": {"checks": checks, "passed": passed},
        "core_output_sha256": core_hashes,
        "evidence_boundary": {
            "training_only": True,
            "development_lance_opened": False,
            "public_test_lance_opened": False,
            "pixels_decoded": False,
            "model_loaded": False,
            "optimizer_steps_run": 0,
            "schedule_generated": passed,
            "schedule_authorized_for_next_gate": passed,
            "claim": (
                "This artifact only integerizes and audits the frozen D1-MS50 "
                "Training exposure. It does not establish latent, gradient, or "
                "native ICL improvement."
            ),
        },
    }
    json_dump(summary_path, summary)
    receipt = {
        "schema_version": 1,
        "builder_id": summary["builder_id"],
        "candidate_id": summary["candidate_id"],
        "status": summary["status"],
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "input_sha256": {
            **frozen["observed_v2_sha256"],
            "train_lance_directory": frozen["train_lance_sha256"],
            "v1_audit_script": file_sha256(V1_SCRIPT),
            "v2_audit_script": file_sha256(V2_SCRIPT),
            "builder_script": file_sha256(Path(__file__).resolve()),
        },
        "output_sha256": {
            "config.json": file_sha256(config_path),
            "summary.json": file_sha256(summary_path),
            **core_hashes,
        },
        "gates": summary["gates"],
        "development_lance_opened": False,
        "public_test_lance_opened": False,
        "pixels_decoded": False,
        "model_loaded": False,
        "optimizer_steps_run": 0,
        "schedule_generated": passed,
    }
    json_dump(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "high_tv": high_tv,
                "full_tv": full_tv,
                "weighted_conditional_energy": weighted_conditional,
                "rho_phys_by_k": {
                    k: realized_by_k[k]["realized_rho_phys"] for k in realized_by_k
                },
                "conflicts": conflict_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
