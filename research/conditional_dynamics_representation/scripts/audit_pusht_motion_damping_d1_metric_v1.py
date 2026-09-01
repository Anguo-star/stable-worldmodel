#!/usr/bin/env python3
"""Training-only D1-0 audit for Motion Damping relative conditional energy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import lance
import numpy as np
from scipy.stats import spearmanr, wasserstein_distance


ROOT = Path(__file__).resolve().parents[3]
CONTEXTWORLD_ROOT = ROOT.parent / "ContextWorld"
RELEASE_ROOT = CONTEXTWORLD_ROOT / (
    "artifacts/synthesis/pusht_motion_damping_h3_release_v4"
)
DEFAULT_MANIFEST = RELEASE_ROOT / "manifest.json"
DEFAULT_TRAIN_LANCE = RELEASE_ROOT / "train.lance"
DEFAULT_RELEASE_CONFIG = CONTEXTWORLD_ROOT / (
    "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml"
)
DEFAULT_OUTPUT_DIR = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_metric_v1/d1_0_training_only_v1_final"
)

EXPECTED_MANIFEST_SHA256 = (
    "48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b"
)
EXPECTED_RELEASE_CONFIG_SHA256 = (
    "1795717d8bfa1d1cfcc01a69931b6241b0e7759dcf43553a6c4cca225ec9326b"
)
EXPECTED_PAIR_COUNT = 8192
EXPECTED_TWIN_COUNT = 4096
EXPECTED_EPISODE_STEPS = 20
QUERY_STEP = 10
FUTURE_STEP = 15
MODES = ("faster_decay", "no_extra_decay")
KS = (32, 64, 128)
MAIN_K = 64
TAU = 1.0e-8
NEIGHBOR_CHUNK_SIZE = 128
SEPARATION_PX = 2.0
SPEARMAN_GATE = 0.90
JACCARD_GATE = 0.80
PAIR_ID_PATTERN = re.compile(r"^pmd-train-(\d{6})-(forward|reverse)$")


@dataclass(frozen=True)
class TrainingArrays:
    pair_ids: tuple[str, ...]
    pair_indices: np.ndarray
    twin_ids_directed: np.ndarray
    directions: tuple[str, ...]
    orientation_directed: np.ndarray
    descriptor_directed: np.ndarray
    mean_displacement_directed: np.ndarray
    conditional_component_directed: np.ndarray
    gap_directed: np.ndarray
    history_gap_directed: np.ndarray
    query_speed_directed: np.ndarray
    goal_distance_directed: np.ndarray
    query_action_max_abs: float
    manifest_train_table_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite(name: str, value: np.ndarray | float) -> None:
    if not bool(np.isfinite(value).all()):
        raise RuntimeError(f"{name} contains non-finite values")


def _fixed_list_numpy(table: Any, name: str, width: int) -> np.ndarray:
    column = table[name].combine_chunks()
    values = np.asarray(
        column.flatten().to_numpy(zero_copy_only=False), dtype=np.float64
    )
    result = values.reshape(len(column), width)
    _finite(name, result)
    return result


def _scalar_numpy(table: Any, name: str, dtype: Any) -> np.ndarray:
    return np.asarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _parse_pair_id(pair_id: str) -> tuple[int, str]:
    match = PAIR_ID_PATTERN.fullmatch(pair_id)
    if match is None:
        raise RuntimeError(f"unexpected pair_id: {pair_id}")
    return int(match.group(1)), match.group(2)


def load_training_arrays(
    *,
    train_lance: Path,
    manifest_path: Path,
    expected_pair_count: int = EXPECTED_PAIR_COUNT,
) -> TrainingArrays:
    """Read only the frozen Training table and derive directed pair records."""

    train_lance = train_lance.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    _require(train_lance.name == "train.lance", "D1-0 accepts only train.lance")
    forbidden = {
        (train_lance.parent / "loader_validation.lance").resolve(),
        (train_lance.parent / "validation.lance").resolve(),
    }
    _require(train_lance not in forbidden, "Development/Test table is forbidden")
    _require(train_lance.exists(), f"missing Training table: {train_lance}")
    _require(manifest_path.exists(), f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("passed") is True, "frozen manifest is not passed")
    train_manifest = manifest.get("splits", {}).get("train")
    _require(isinstance(train_manifest, dict), "manifest lacks splits.train")
    _require(train_manifest.get("passed") is True, "Training split audit is not passed")
    manifest_pairs = train_manifest.get("pairs")
    _require(
        isinstance(manifest_pairs, list)
        and len(manifest_pairs) == expected_pair_count,
        "unexpected Training manifest pair count",
    )
    manifest_by_index: dict[int, dict[str, Any]] = {}
    for row in manifest_pairs:
        index = int(row["pair_index"])
        _require(index not in manifest_by_index, "duplicate manifest pair_index")
        manifest_by_index[index] = row
    _require(
        sorted(manifest_by_index) == list(range(expected_pair_count)),
        "manifest pair indices are not contiguous",
    )

    columns = [
        "episode_idx",
        "step_idx",
        "action",
        "goal_state",
        "physics_state",
        "pair_id",
        "hidden_mode",
        "split",
        "catalog_index",
    ]
    table = lance.dataset(str(train_lance)).to_table(columns=columns)
    expected_episodes = 2 * expected_pair_count
    expected_rows = expected_episodes * EXPECTED_EPISODE_STEPS
    _require(table.num_rows == expected_rows, "unexpected Training row count")

    episode_idx = _scalar_numpy(table, "episode_idx", np.int64)
    step_idx = _scalar_numpy(table, "step_idx", np.int64)
    actions = _fixed_list_numpy(table, "action", 2)
    goals = _fixed_list_numpy(table, "goal_state", 7)
    physics = _fixed_list_numpy(table, "physics_state", 12)
    catalog_index = _fixed_list_numpy(table, "catalog_index", 1).reshape(-1)
    pair_ids_raw = np.asarray(table["pair_id"].to_pylist(), dtype=object)
    modes_raw = np.asarray(table["hidden_mode"].to_pylist(), dtype=object)
    splits_raw = np.asarray(table["split"].to_pylist(), dtype=object)

    order = np.lexsort((step_idx, episode_idx))
    episode_idx = episode_idx[order]
    step_idx = step_idx[order]
    actions = actions[order]
    goals = goals[order]
    physics = physics[order]
    catalog_index = catalog_index[order]
    pair_ids_raw = pair_ids_raw[order]
    modes_raw = modes_raw[order]
    splits_raw = splits_raw[order]

    _require(
        np.array_equal(
            episode_idx,
            np.repeat(np.arange(expected_episodes), EXPECTED_EPISODE_STEPS),
        ),
        "episode_idx is not a complete contiguous Training sequence",
    )
    _require(
        np.array_equal(
            step_idx,
            np.tile(np.arange(EXPECTED_EPISODE_STEPS), expected_episodes),
        ),
        "episodes do not contain exactly steps 0..19",
    )

    actions_e = actions.reshape(expected_episodes, EXPECTED_EPISODE_STEPS, 2)
    goals_e = goals.reshape(expected_episodes, EXPECTED_EPISODE_STEPS, 7)
    physics_e = physics.reshape(expected_episodes, EXPECTED_EPISODE_STEPS, 12)
    catalog_e = catalog_index.reshape(expected_episodes, EXPECTED_EPISODE_STEPS)
    pair_ids_e = pair_ids_raw.reshape(expected_episodes, EXPECTED_EPISODE_STEPS)
    modes_e = modes_raw.reshape(expected_episodes, EXPECTED_EPISODE_STEPS)
    splits_e = splits_raw.reshape(expected_episodes, EXPECTED_EPISODE_STEPS)

    episodes: dict[int, dict[str, int]] = {}
    episode_pair_id: dict[int, str] = {}
    for episode in range(expected_episodes):
        pair_values = {str(value) for value in pair_ids_e[episode]}
        mode_values = {str(value) for value in modes_e[episode]}
        split_values = {str(value) for value in splits_e[episode]}
        catalog_values = set(catalog_e[episode].tolist())
        _require(len(pair_values) == 1, f"episode {episode} changes pair_id")
        _require(len(mode_values) == 1, f"episode {episode} changes hidden_mode")
        _require(split_values == {"train"}, f"episode {episode} is not Training")
        _require(len(catalog_values) == 1, f"episode {episode} changes catalog_index")
        pair_id = pair_values.pop()
        pair_index, _ = _parse_pair_id(pair_id)
        _require(
            float(pair_index) == float(catalog_values.pop()),
            f"catalog_index mismatch for {pair_id}",
        )
        mode = mode_values.pop()
        _require(mode in MODES, f"unexpected hidden_mode: {mode}")
        by_mode = episodes.setdefault(pair_index, {})
        _require(mode not in by_mode, f"duplicate {pair_id}/{mode}")
        by_mode[mode] = episode
        episode_pair_id[episode] = pair_id

    _require(
        sorted(episodes) == list(range(expected_pair_count)),
        "Training pair indices are incomplete",
    )

    pair_ids: list[str] = []
    directions: list[str] = []
    orientation: list[int] = []
    descriptors: list[np.ndarray] = []
    mean_displacements: list[np.ndarray] = []
    conditional_components: list[float] = []
    gaps: list[float] = []
    history_gaps: list[float] = []
    query_speeds: list[float] = []
    goal_distances: list[float] = []
    query_action_max_abs = 0.0

    for pair_index in range(expected_pair_count):
        pair_episodes = episodes[pair_index]
        _require(set(pair_episodes) == set(MODES), "incomplete condition pair")
        manifest_row = manifest_by_index[pair_index]
        _require(
            manifest_row.get("audit", {}).get("passed") is True,
            f"manifest pair audit failed for pair {pair_index}",
        )
        manifest_pair_id = str(manifest_row["audit"]["template_id"])
        parsed_index, direction = _parse_pair_id(manifest_pair_id)
        _require(parsed_index == pair_index, "manifest template index mismatch")
        for mode in MODES:
            _require(
                episode_pair_id[pair_episodes[mode]] == manifest_pair_id,
                "Lance/manifest pair identity mismatch",
            )
        faster_episode = pair_episodes[MODES[0]]
        slower_episode = pair_episodes[MODES[1]]
        faster_physics = physics_e[faster_episode]
        slower_physics = physics_e[slower_episode]
        faster_actions = actions_e[faster_episode]
        slower_actions = actions_e[slower_episode]
        _require(
            np.array_equal(faster_actions, slower_actions),
            f"actions differ for {manifest_pair_id}",
        )
        _require(
            np.allclose(
                faster_physics[QUERY_STEP],
                slower_physics[QUERY_STEP],
                atol=1.0e-6,
                rtol=0.0,
            ),
            f"query physics differs for {manifest_pair_id}",
        )
        _require(
            np.allclose(
                goals_e[faster_episode],
                goals_e[slower_episode],
                atol=1.0e-6,
                rtol=0.0,
            ),
            f"goal state differs for {manifest_pair_id}",
        )
        query = faster_physics[QUERY_STEP]
        goal = goals_e[faster_episode, QUERY_STEP]
        future_0 = faster_physics[FUTURE_STEP]
        future_1 = slower_physics[FUTURE_STEP]
        x = query[6:8]
        y_0 = future_0[6:8]
        y_1 = future_1[6:8]
        gap_vector = y_1 - y_0
        gap = float(np.linalg.norm(gap_vector))
        manifest_gap = float(
            manifest_row["audit"]["future_gap"]["block_position_px"]
        )
        angle_gap = float(
            manifest_row["audit"]["future_gap"]["block_angle_rad"]
        )
        _require(gap >= SEPARATION_PX, "future separation gate failed")
        _require(
            math.isclose(gap, manifest_gap, abs_tol=1.0e-5, rel_tol=0.0),
            f"future gap cross-check failed for {manifest_pair_id}",
        )
        _require(abs(angle_gap) <= 1.0e-12, "Motion future angle gap is nonzero")
        _require(
            abs(float(future_1[10] - future_0[10])) <= 1.0e-6,
            "Lance future angle gap is nonzero",
        )
        mean_displacement = 0.5 * ((y_0 - x) + (y_1 - x))
        conditional_component = float(np.dot(gap_vector, gap_vector) / 4.0)
        theta = float(query[10])
        goal_relative = goal[2:4] - x
        descriptor = np.asarray(
            [
                query[6],
                query[7],
                query[8],
                query[9],
                math.sin(theta),
                math.cos(theta),
                goal_relative[0],
                goal_relative[1],
            ],
            dtype=np.float64,
        )
        query_action_max_abs = max(
            query_action_max_abs,
            float(np.max(np.abs(faster_actions[QUERY_STEP:FUTURE_STEP]))),
        )

        pair_ids.append(manifest_pair_id)
        directions.append(direction)
        orientation.append(int(manifest_row["orientation_bin"]))
        descriptors.append(descriptor)
        mean_displacements.append(mean_displacement)
        conditional_components.append(conditional_component)
        gaps.append(gap)
        history_gaps.append(
            float(
                manifest_row["audit"]["history_visible_response_gap"][
                    "block_position_px"
                ]
            )
        )
        query_speeds.append(float(np.linalg.norm(query[8:10])))
        goal_distances.append(float(np.linalg.norm(goal_relative)))

    pair_indices = np.arange(expected_pair_count, dtype=np.int64)
    twin_ids_directed = pair_indices // 2
    orientation_array = np.asarray(orientation, dtype=np.int64)
    descriptor_array = np.stack(descriptors)
    mean_array = np.stack(mean_displacements)
    conditional_array = np.asarray(conditional_components, dtype=np.float64)
    gap_array = np.asarray(gaps, dtype=np.float64)
    history_array = np.asarray(history_gaps, dtype=np.float64)
    speed_array = np.asarray(query_speeds, dtype=np.float64)
    distance_array = np.asarray(goal_distances, dtype=np.float64)
    for name, value in (
        ("descriptor", descriptor_array),
        ("mean_displacement", mean_array),
        ("conditional_component", conditional_array),
        ("gap", gap_array),
        ("history_gap", history_array),
        ("query_speed", speed_array),
        ("goal_distance", distance_array),
    ):
        _finite(name, value)
    _require(np.all(conditional_array > 0.0), "conditional component is not positive")
    _require(query_action_max_abs == 0.0, "Motion query action is not identically zero")

    for twin in range(expected_pair_count // 2):
        forward = 2 * twin
        reverse = forward + 1
        _require(
            directions[forward] == "forward" and directions[reverse] == "reverse",
            f"twin {twin} lacks forward/reverse order",
        )
        _require(
            orientation_array[forward] == orientation_array[reverse],
            f"twin {twin} changes orientation bin",
        )
        forward_template = manifest_by_index[forward]["template"]
        reverse_template = manifest_by_index[reverse]["template"]
        for field in (
            "goal_state",
            "simulator_seed",
            "visible_shape_id",
            "visible_shape_name",
        ):
            _require(
                forward_template[field] == reverse_template[field],
                f"twin {twin} changes shared manifest field {field}",
            )

    return TrainingArrays(
        pair_ids=tuple(pair_ids),
        pair_indices=pair_indices,
        twin_ids_directed=twin_ids_directed,
        directions=tuple(directions),
        orientation_directed=orientation_array,
        descriptor_directed=descriptor_array,
        mean_displacement_directed=mean_array,
        conditional_component_directed=conditional_array,
        gap_directed=gap_array,
        history_gap_directed=history_array,
        query_speed_directed=speed_array,
        goal_distance_directed=distance_array,
        query_action_max_abs=query_action_max_abs,
        manifest_train_table_sha256=str(train_manifest["table_sha256"]),
    )


def robust_scale_descriptors(
    descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    descriptors = np.asarray(descriptors, dtype=np.float64)
    _finite("descriptors", descriptors)
    quartiles = np.percentile(descriptors, [25.0, 50.0, 75.0], axis=0)
    median = quartiles[1]
    iqr = quartiles[2] - quartiles[0]
    active = iqr > 0.0
    scaled = np.zeros_like(descriptors, dtype=np.float64)
    scaled[:, active] = (descriptors[:, active] - median[active]) / iqr[active]
    _finite("scaled descriptors", scaled)
    return scaled, median, iqr, active


def exact_leave_twin_out_neighbors(
    descriptors: np.ndarray,
    twin_ids: np.ndarray,
    *,
    max_k: int,
    pair_indices: np.ndarray | None = None,
    chunk_size: int = NEIGHBOR_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic exact neighbors, breaking distance ties by pair index."""

    descriptors = np.asarray(descriptors, dtype=np.float64)
    twin_ids = np.asarray(twin_ids, dtype=np.int64)
    count = descriptors.shape[0]
    if pair_indices is None:
        pair_indices = np.arange(count, dtype=np.int64)
    pair_indices = np.asarray(pair_indices, dtype=np.int64)
    _require(descriptors.ndim == 2, "descriptors must be rank two")
    _require(twin_ids.shape == (count,), "twin_ids shape mismatch")
    _require(pair_indices.shape == (count,), "pair_indices shape mismatch")
    _require(max_k > 0 and max_k <= count - 2, "invalid max_k")
    _require(chunk_size > 0, "chunk_size must be positive")
    _finite("neighbor descriptors", descriptors)

    neighbors = np.empty((count, max_k), dtype=np.int64)
    distances = np.empty((count, max_k), dtype=np.float64)
    members = {
        int(twin): np.flatnonzero(twin_ids == twin)
        for twin in np.unique(twin_ids)
    }
    _require(all(len(value) == 2 for value in members.values()), "twins must have two directions")

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        query = descriptors[start:stop]
        differences = query[:, None, :] - descriptors[None, :, :]
        distance_sq = np.einsum(
            "qnd,qnd->qn",
            differences,
            differences,
            optimize=False,
        )
        for local, row in enumerate(range(start, stop)):
            distance_sq[local, members[int(twin_ids[row])]] = np.inf
            partial = np.argpartition(distance_sq[local], max_k - 1)[:max_k]
            cutoff = float(np.max(distance_sq[local, partial]))
            strict = np.flatnonzero(distance_sq[local] < cutoff)
            tied = np.flatnonzero(distance_sq[local] == cutoff)
            tied = tied[np.argsort(pair_indices[tied], kind="stable")]
            needed = max_k - len(strict)
            selected = np.concatenate([strict, tied[:needed]])
            order = np.lexsort(
                (pair_indices[selected], distance_sq[local, selected])
            )
            selected = selected[order]
            _require(len(selected) == max_k, "failed to select exact neighbors")
            _require(
                not np.any(twin_ids[selected] == twin_ids[row]),
                "leave-twin-out exclusion failed",
            )
            neighbors[row] = selected
            distances[row] = np.sqrt(distance_sq[local, selected])
    _finite("neighbor distances", distances)
    return neighbors, distances


def background_variance_by_k(
    mean_displacements: np.ndarray,
    neighbors: np.ndarray,
    ks: Iterable[int],
) -> dict[int, np.ndarray]:
    means = np.asarray(mean_displacements, dtype=np.float64)
    _finite("mean displacements", means)
    result: dict[int, np.ndarray] = {}
    for k in ks:
        _require(0 < k <= neighbors.shape[1], f"invalid k={k}")
        values = means[neighbors[:, :k]]
        center = values.mean(axis=1, keepdims=True)
        variance = np.mean(np.sum(np.square(values - center), axis=2), axis=1)
        _finite(f"B_{k}", variance)
        _require(np.all(variance >= 0.0), f"B_{k} is negative")
        result[int(k)] = variance
    return result


def aggregate_directed_to_twins(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    _require(values.shape[0] % 2 == 0, "directed array cannot form twins")
    return values.reshape(values.shape[0] // 2, 2, *values.shape[1:]).mean(axis=1)


def assign_coverage_cells(
    orientation: np.ndarray,
    speed: np.ndarray,
    goal_distance: np.ndarray,
    *,
    bins: int = 4,
    stable_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact equal-frequency orientation x speed x distance cells."""

    orientation = np.asarray(orientation, dtype=np.int64)
    speed = np.asarray(speed, dtype=np.float64)
    goal_distance = np.asarray(goal_distance, dtype=np.float64)
    count = len(orientation)
    if stable_ids is None:
        stable_ids = np.arange(count, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    _require(speed.shape == goal_distance.shape == orientation.shape, "coverage shape mismatch")
    _require(stable_ids.shape == orientation.shape, "stable_ids shape mismatch")
    _finite("coverage speed", speed)
    _finite("coverage goal distance", goal_distance)

    speed_bin = np.full(count, -1, dtype=np.int64)
    goal_bin = np.full(count, -1, dtype=np.int64)
    unique_orientation = np.unique(orientation)
    _require(
        np.array_equal(unique_orientation, np.arange(len(unique_orientation))),
        "orientation bins must be contiguous from zero",
    )
    for orient in unique_orientation:
        indices = np.flatnonzero(orientation == orient)
        _require(len(indices) % bins == 0, "orientation count is not divisible by bins")
        order = indices[np.lexsort((stable_ids[indices], speed[indices]))]
        width = len(indices) // bins
        speed_bin[order] = np.arange(len(indices)) // width
        for speed_value in range(bins):
            subgroup = np.flatnonzero(
                (orientation == orient) & (speed_bin == speed_value)
            )
            _require(len(subgroup) % bins == 0, "speed bin is not divisible by bins")
            subgroup_order = subgroup[
                np.lexsort((stable_ids[subgroup], goal_distance[subgroup]))
            ]
            goal_width = len(subgroup) // bins
            goal_bin[subgroup_order] = np.arange(len(subgroup)) // goal_width
    _require(np.all(speed_bin >= 0) and np.all(goal_bin >= 0), "coverage assignment incomplete")
    cells = orientation * bins * bins + speed_bin * bins + goal_bin
    return cells, speed_bin, goal_bin


def select_top_per_cell(
    scores: np.ndarray,
    cells: np.ndarray,
    *,
    per_cell: int,
    stable_ids: np.ndarray | None = None,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    count = len(scores)
    if stable_ids is None:
        stable_ids = np.arange(count, dtype=np.int64)
    stable_ids = np.asarray(stable_ids, dtype=np.int64)
    _finite("selection scores", scores)
    selected: list[int] = []
    for cell in sorted(np.unique(cells).tolist()):
        indices = np.flatnonzero(cells == cell)
        _require(len(indices) >= per_cell, f"cell {cell} is too small")
        order = indices[np.lexsort((stable_ids[indices], -scores[indices]))]
        selected.extend(order[:per_cell].tolist())
    return np.asarray(selected, dtype=np.int64)


def projected_weights(count: int, high_pool: np.ndarray | None) -> np.ndarray:
    _require(count > 0, "weight count must be positive")
    if high_pool is None:
        return np.full(count, 1.0 / count, dtype=np.float64)
    high_pool = np.asarray(high_pool, dtype=np.int64)
    _require(len(high_pool) > 0, "high pool is empty")
    _require(len(np.unique(high_pool)) == len(high_pool), "high pool has duplicates")
    _require(np.all((0 <= high_pool) & (high_pool < count)), "high pool is out of range")
    weights = np.full(count, 0.5 / count, dtype=np.float64)
    weights[high_pool] += 0.5 / len(high_pool)
    _require(math.isclose(float(weights.sum()), 1.0, abs_tol=1.0e-12), "weights do not sum to one")
    return weights


def aggregate_relative_energy(
    conditional: np.ndarray,
    background: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    conditional = np.asarray(conditional, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    _require(conditional.shape == background.shape == weights.shape, "aggregate shape mismatch")
    _finite("aggregate conditional", conditional)
    _finite("aggregate background", background)
    _finite("aggregate weights", weights)
    _require(np.all(conditional > 0.0), "conditional energy must be positive")
    _require(np.all(background >= 0.0), "background energy must be nonnegative")
    _require(np.all(weights >= 0.0), "weights must be nonnegative")
    _require(float(weights.sum()) > 0.0, "weights have zero mass")
    normalized = weights / weights.sum()
    mean_conditional = float(np.dot(normalized, conditional))
    mean_background = float(np.dot(normalized, background))
    denominator = mean_conditional + mean_background
    _require(denominator > 0.0, "aggregate denominator is not positive")
    local_score = conditional / (conditional + background + TAU)
    return {
        "weighted_conditional_energy": mean_conditional,
        "weighted_background_variation": mean_background,
        "rho_phys_ratio_of_means": mean_conditional / denominator,
        "weighted_mean_local_score_diagnostic_only": float(
            np.dot(normalized, local_score)
        ),
    }


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    _require(values.shape == weights.shape, "weighted quantile shape mismatch")
    order = np.lexsort((np.arange(len(values)), values))
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    result: dict[str, float] = {}
    for probability in probabilities:
        index = int(np.searchsorted(cumulative, probability, side="left"))
        index = min(index, len(sorted_values) - 1)
        result[f"q{int(round(100 * probability)):02d}"] = float(sorted_values[index])
    return result


def weighted_shift_metrics(
    reference_values: np.ndarray,
    reference_weights: np.ndarray,
    candidate_values: np.ndarray,
    candidate_weights: np.ndarray,
) -> dict[str, float]:
    reference_values = np.asarray(reference_values, dtype=np.float64)
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    reference_weights = np.asarray(reference_weights, dtype=np.float64)
    candidate_weights = np.asarray(candidate_weights, dtype=np.float64)
    reference_weights = reference_weights / reference_weights.sum()
    candidate_weights = candidate_weights / candidate_weights.sum()
    mean_reference = float(np.dot(reference_weights, reference_values))
    mean_candidate = float(np.dot(candidate_weights, candidate_values))
    variance_reference = float(
        np.dot(reference_weights, np.square(reference_values - mean_reference))
    )
    variance_candidate = float(
        np.dot(candidate_weights, np.square(candidate_values - mean_candidate))
    )
    pooled = math.sqrt(0.5 * (variance_reference + variance_candidate))
    if pooled == 0.0:
        _require(
            mean_reference == mean_candidate,
            "SMD is undefined for unequal constants",
        )
        smd = 0.0
    else:
        smd = (mean_candidate - mean_reference) / pooled

    support = np.unique(np.concatenate([reference_values, candidate_values]))
    reference_order = np.argsort(reference_values, kind="stable")
    candidate_order = np.argsort(candidate_values, kind="stable")
    reference_sorted = reference_values[reference_order]
    candidate_sorted = candidate_values[candidate_order]
    reference_cumulative = np.cumsum(reference_weights[reference_order])
    candidate_cumulative = np.cumsum(candidate_weights[candidate_order])
    reference_positions = np.searchsorted(reference_sorted, support, side="right") - 1
    candidate_positions = np.searchsorted(candidate_sorted, support, side="right") - 1
    reference_cdf = np.where(
        reference_positions >= 0,
        reference_cumulative[np.maximum(reference_positions, 0)],
        0.0,
    )
    candidate_cdf = np.where(
        candidate_positions >= 0,
        candidate_cumulative[np.maximum(candidate_positions, 0)],
        0.0,
    )
    return {
        "standardized_mean_difference": float(smd),
        "weighted_ks": float(np.max(np.abs(reference_cdf - candidate_cdf))),
        "weighted_wasserstein_1": float(
            wasserstein_distance(
                reference_values,
                candidate_values,
                u_weights=reference_weights,
                v_weights=candidate_weights,
            )
        ),
    }


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(np.asarray(left, dtype=np.int64).tolist())
    right_set = set(np.asarray(right, dtype=np.int64).tolist())
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def evaluate_gates(
    *,
    spearman_32: float,
    spearman_128: float,
    jaccard_32: float,
    jaccard_128: float,
    rho_d0: float,
    rho_r50: float,
    coverage_ok: bool,
    invariants_ok: bool,
) -> dict[str, Any]:
    checks = {
        "spearman_k32_vs_k64_at_least_0p90": spearman_32 >= SPEARMAN_GATE,
        "spearman_k128_vs_k64_at_least_0p90": spearman_128 >= SPEARMAN_GATE,
        "r50_pool_jaccard_k32_vs_k64_at_least_0p80": jaccard_32 >= JACCARD_GATE,
        "r50_pool_jaccard_k128_vs_k64_at_least_0p80": jaccard_128 >= JACCARD_GATE,
        "r50_rho_phys_strictly_above_d0": rho_r50 > rho_d0,
        "coverage_cells_exact": bool(coverage_ok),
        "training_only_invariants": bool(invariants_ok),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _distribution_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    normalized = weights / weights.sum()
    return {
        "weighted_mean": float(np.dot(normalized, values)),
        "weighted_std": float(
            math.sqrt(
                np.dot(
                    normalized,
                    np.square(values - np.dot(normalized, values)),
                )
            )
        ),
        "weighted_quantiles": weighted_quantiles(values, normalized),
        "support_min": float(np.min(values)),
        "support_max": float(np.max(values)),
    }


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _candidate_summary(
    *,
    name: str,
    weights: np.ndarray,
    conditional: np.ndarray,
    background: np.ndarray,
    local_score: np.ndarray,
    gap_rms_source: np.ndarray,
    speed: np.ndarray,
    goal_distance: np.ndarray,
    cells: np.ndarray,
    d0_weights: np.ndarray,
    selected: np.ndarray | None,
) -> dict[str, Any]:
    aggregate = aggregate_relative_energy(conditional, background, weights)
    unique_cells = sorted(np.unique(cells).tolist())
    cell_mass = [float(weights[cells == cell].sum()) for cell in unique_cells]
    cell_twin_counts = [int(np.sum(cells == cell)) for cell in unique_cells]
    return {
        "name": name,
        "selected_twin_count": 0 if selected is None else int(len(selected)),
        "selected_twin_ids": [] if selected is None else selected.tolist(),
        "aggregate": aggregate,
        "local_score": _distribution_summary(local_score, weights),
        "response_gap_px": _distribution_summary(gap_rms_source, weights),
        "response_gap_rms_px": float(
            math.sqrt(np.dot(weights / weights.sum(), np.square(gap_rms_source)))
        ),
        "query_speed": {
            **_distribution_summary(speed, weights),
            "shift_from_d0": weighted_shift_metrics(
                speed, d0_weights, speed, weights
            ),
        },
        "goal_distance": {
            **_distribution_summary(goal_distance, weights),
            "shift_from_d0": weighted_shift_metrics(
                goal_distance, d0_weights, goal_distance, weights
            ),
        },
        "coverage": {
            "cell_count": int(len(cell_mass)),
            "cell_mass_min": min(cell_mass),
            "cell_mass_max": max(cell_mass),
            "cell_twin_counts": cell_twin_counts,
            "cell_mass_exactly_uniform": bool(
                np.allclose(cell_mass, 1.0 / len(cell_mass), atol=1.0e-12, rtol=0.0)
            ),
            "all_twins_have_positive_weight": bool(np.all(weights > 0.0)),
        },
        "projected_training": {
            "optimizer_steps": 8192,
            "twins_per_batch": 16,
            "twin_slots": 131072,
            "condition_rows": 524288,
            "mode_balance": {MODES[0]: 0.5, MODES[1]: 0.5},
            "direction_balance": {"forward": 0.5, "reverse": 0.5},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-lance", type=Path, default=DEFAULT_TRAIN_LANCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--release-config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--neighbor-chunk-size", type=int, default=NEIGHBOR_CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    release_config = args.release_config.expanduser().resolve()
    train_lance = args.train_lance.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    _require(
        file_sha256(manifest_path) == EXPECTED_MANIFEST_SHA256,
        "frozen manifest SHA256 changed",
    )
    _require(
        file_sha256(release_config) == EXPECTED_RELEASE_CONFIG_SHA256,
        "frozen release config SHA256 changed",
    )

    arrays = load_training_arrays(
        train_lance=train_lance,
        manifest_path=manifest_path,
    )
    train_lance_sha256 = directory_sha256(train_lance)
    _require(
        train_lance_sha256 == arrays.manifest_train_table_sha256,
        "frozen train.lance directory SHA256 changed",
    )
    scaled, median, iqr, active = robust_scale_descriptors(
        arrays.descriptor_directed
    )
    neighbors, neighbor_distances = exact_leave_twin_out_neighbors(
        scaled,
        arrays.twin_ids_directed,
        max_k=max(KS),
        pair_indices=arrays.pair_indices,
        chunk_size=args.neighbor_chunk_size,
    )
    background_directed = background_variance_by_k(
        arrays.mean_displacement_directed, neighbors, KS
    )

    twin_count = len(arrays.pair_ids) // 2
    _require(twin_count == EXPECTED_TWIN_COUNT, "unexpected twin count")
    twin_ids = np.arange(twin_count, dtype=np.int64)
    conditional = aggregate_directed_to_twins(
        arrays.conditional_component_directed
    )
    gap = np.sqrt(4.0 * conditional)
    history_gap = aggregate_directed_to_twins(arrays.history_gap_directed)
    history_gap_min = arrays.history_gap_directed.reshape(twin_count, 2).min(axis=1)
    background = {
        k: aggregate_directed_to_twins(value)
        for k, value in background_directed.items()
    }
    score = {
        k: conditional / (conditional + background[k] + TAU)
        for k in KS
    }
    score_tau_zero = conditional / (conditional + background[MAIN_K])
    for k in KS:
        _finite(f"twin B_{k}", background[k])
        _finite(f"twin s_rel_{k}", score[k])
    _finite("tau-zero score", score_tau_zero)

    orientation = arrays.orientation_directed.reshape(twin_count, 2)[:, 0]
    speed = aggregate_directed_to_twins(arrays.query_speed_directed)
    goal_distance = aggregate_directed_to_twins(arrays.goal_distance_directed)
    cells, speed_bin, goal_bin = assign_coverage_cells(
        orientation, speed, goal_distance, stable_ids=twin_ids
    )
    unique_cells, cell_counts = np.unique(cells, return_counts=True)
    coverage_ok = bool(
        len(unique_cells) == 64
        and np.array_equal(unique_cells, np.arange(64))
        and np.all(cell_counts == 64)
    )
    _require(coverage_ok, "64-cell coverage construction failed")

    e50 = select_top_per_cell(
        4.0 * conditional, cells, per_cell=16, stable_ids=twin_ids
    )
    r50_by_k = {
        k: select_top_per_cell(
            score[k], cells, per_cell=16, stable_ids=twin_ids
        )
        for k in KS
    }
    r50 = r50_by_k[MAIN_K]
    _require(len(e50) == len(r50) == 1024, "candidate pool size mismatch")
    for pool in (e50, *r50_by_k.values()):
        counts = np.bincount(cells[pool], minlength=64)
        _require(np.all(counts == 16), "candidate cell quota mismatch")

    spearman_32 = float(spearmanr(score[MAIN_K], score[32]).statistic)
    spearman_128 = float(spearmanr(score[MAIN_K], score[128]).statistic)
    tau_zero_spearman = float(
        spearmanr(score[MAIN_K], score_tau_zero).statistic
    )
    jaccard_32 = jaccard(r50, r50_by_k[32])
    jaccard_128 = jaccard(r50, r50_by_k[128])
    tau_zero_pool = select_top_per_cell(
        score_tau_zero, cells, per_cell=16, stable_ids=twin_ids
    )
    tau_zero_jaccard = jaccard(r50, tau_zero_pool)
    tau_hit_rate = float(
        np.mean((conditional + background[MAIN_K]) <= TAU)
    )

    weights_d0 = projected_weights(twin_count, None)
    weights_e50 = projected_weights(twin_count, e50)
    weights_r50 = projected_weights(twin_count, r50)
    summaries = {
        "D0": _candidate_summary(
            name="D0",
            weights=weights_d0,
            conditional=conditional,
            background=background[MAIN_K],
            local_score=score[MAIN_K],
            gap_rms_source=gap,
            speed=speed,
            goal_distance=goal_distance,
            cells=cells,
            d0_weights=weights_d0,
            selected=None,
        ),
        "D1-E50": _candidate_summary(
            name="D1-E50",
            weights=weights_e50,
            conditional=conditional,
            background=background[MAIN_K],
            local_score=score[MAIN_K],
            gap_rms_source=gap,
            speed=speed,
            goal_distance=goal_distance,
            cells=cells,
            d0_weights=weights_d0,
            selected=e50,
        ),
        "D1-R50": _candidate_summary(
            name="D1-R50",
            weights=weights_r50,
            conditional=conditional,
            background=background[MAIN_K],
            local_score=score[MAIN_K],
            gap_rms_source=gap,
            speed=speed,
            goal_distance=goal_distance,
            cells=cells,
            d0_weights=weights_d0,
            selected=r50,
        ),
    }
    invariants_ok = bool(
        arrays.query_action_max_abs == 0.0
        and np.all(gap >= SEPARATION_PX)
        and np.all(conditional > 0.0)
        and all(
            candidate["coverage"]["cell_mass_exactly_uniform"]
            and candidate["coverage"]["all_twins_have_positive_weight"]
            for candidate in summaries.values()
        )
    )
    gates = evaluate_gates(
        spearman_32=spearman_32,
        spearman_128=spearman_128,
        jaccard_32=jaccard_32,
        jaccard_128=jaccard_128,
        rho_d0=summaries["D0"]["aggregate"]["rho_phys_ratio_of_means"],
        rho_r50=summaries["D1-R50"]["aggregate"]["rho_phys_ratio_of_means"],
        coverage_ok=coverage_ok,
        invariants_ok=invariants_ok,
    )

    summary = {
        "schema_version": 1,
        "audit_id": "pusht_motion_damping_d1_metric_v1",
        "stage": "D1-0_training_only_metric_feasibility",
        "status": "passed_go" if gates["passed"] else "failed_no_go",
        "frozen_identity": {
            "release_id": "contextworld_pusht_motion_damping_icl_history3_v1",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "release_config_sha256": EXPECTED_RELEASE_CONFIG_SHA256,
            "manifest_train_table_sha256": arrays.manifest_train_table_sha256,
            "verified_train_lance_directory_sha256": train_lance_sha256,
            "train_pair_count": len(arrays.pair_ids),
            "twin_count": twin_count,
        },
        "metric": {
            "main_k": MAIN_K,
            "sensitivity_k": [32, 128],
            "tau_px2": TAU,
            "conditional_component": "mean_direction(||y1-y0||^2/4)",
            "background": "mean_direction(population_variance_of_neighbor_condition_centers)",
            "selection_score": "C_phys/(C_phys+B_k+tau)",
            "aggregate": "E_pi[C_phys]/(E_pi[C_phys]+E_pi[B_k])",
            "descriptor_median": median.tolist(),
            "descriptor_iqr": iqr.tolist(),
            "descriptor_active": active.tolist(),
            "zero_iqr_dimensions": np.flatnonzero(~active).tolist(),
            "future_gap_min_direction_px": float(np.min(arrays.gap_directed)),
        },
        "stability": {
            "spearman_k32_vs_k64": spearman_32,
            "spearman_k128_vs_k64": spearman_128,
            "r50_pool_jaccard_k32_vs_k64": jaccard_32,
            "r50_pool_jaccard_k128_vs_k64": jaccard_128,
            "tau_zero_spearman_vs_tau_1e8": tau_zero_spearman,
            "tau_zero_pool_jaccard_vs_tau_1e8": tau_zero_jaccard,
            "tau_hit_rate": tau_hit_rate,
            "background_by_k": {
                str(k): _distribution_summary(background[k], weights_d0)
                for k in KS
            },
            "neighbor_distance_by_k": {
                str(k): _distribution_summary(
                    aggregate_directed_to_twins(neighbor_distances[:, k - 1]),
                    weights_d0,
                )
                for k in KS
            },
        },
        "candidates": summaries,
        "gates": gates,
        "evidence_boundary": {
            "development_lance_opened": False,
            "public_test_lance_opened": False,
            "optimizer_steps": 0,
            "model_loaded": False,
            "schedule_generated": False,
            "pixels_decoded": False,
            "claim": (
                "D1-0 tests whether the frozen Training pool can stably raise "
                "relative physical conditional energy; it does not show that "
                "native training improves."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": 1,
        "audit_id": summary["audit_id"],
        "inputs": {
            "train_lance": str(train_lance),
            "manifest": str(manifest_path),
            "release_config": str(release_config),
        },
        "output_dir": str(output_dir),
        "ks": list(KS),
        "main_k": MAIN_K,
        "tau": TAU,
        "neighbor_chunk_size": args.neighbor_chunk_size,
        "selection": {
            "coverage_cells": 64,
            "twins_per_cell": 64,
            "selected_per_cell": 16,
            "natural_weight": 0.5,
            "high_pool_weight": 0.5,
            "stable_tie_break": "ascending_twin_id",
        },
    }
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    catalog_path = output_dir / "per_twin_catalog.jsonl"
    _json_dump(config_path, config)
    _json_dump(summary_path, summary)

    with catalog_path.open("x", encoding="utf-8") as stream:
        e50_set = set(e50.tolist())
        r50_set = set(r50.tolist())
        for twin in range(twin_count):
            directed = [2 * twin, 2 * twin + 1]
            row = {
                "twin_id": twin,
                "pair_ids": [arrays.pair_ids[index] for index in directed],
                "pair_indices": directed,
                "orientation_bin": int(orientation[twin]),
                "speed_bin": int(speed_bin[twin]),
                "goal_distance_bin": int(goal_bin[twin]),
                "coverage_cell": int(cells[twin]),
                "query_speed": float(speed[twin]),
                "goal_distance": float(goal_distance[twin]),
                "conditional_energy_physical": float(conditional[twin]),
                "gap_energy": float(4.0 * conditional[twin]),
                "gap_rms_px": float(gap[twin]),
                "future_gap_direction_px": arrays.gap_directed[directed].tolist(),
                "future_gap_min_px": float(np.min(arrays.gap_directed[directed])),
                "history_gap_mean_px": float(history_gap[twin]),
                "history_gap_min_px": float(history_gap_min[twin]),
                "background_future_variation": {
                    str(k): float(background[k][twin]) for k in KS
                },
                "relative_conditional_score": {
                    str(k): float(score[k][twin]) for k in KS
                },
                "selected": {
                    "D1-E50": twin in e50_set,
                    "D1-R50": twin in r50_set,
                },
                "action_leverage": "not_identifiable_query_action_zero",
                "neighbors": {
                    direction: {
                        "pair_indices_k128": neighbors[index].tolist(),
                        "distances_k128": neighbor_distances[index].tolist(),
                    }
                    for direction, index in zip(("forward", "reverse"), directed, strict=True)
                },
            }
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    receipt = {
        "schema_version": 1,
        "audit_id": summary["audit_id"],
        "status": summary["status"],
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "input_sha256": {
            "manifest": file_sha256(manifest_path),
            "release_config": file_sha256(release_config),
            "train_lance_directory": train_lance_sha256,
            "audit_script": file_sha256(Path(__file__).resolve()),
        },
        "output_sha256": {
            "config.json": file_sha256(config_path),
            "summary.json": file_sha256(summary_path),
            "per_twin_catalog.jsonl": file_sha256(catalog_path),
        },
        "gates": gates,
        "development_lance_opened": False,
        "public_test_lance_opened": False,
        "optimizer_steps": 0,
        "schedule_generated": False,
    }
    receipt_path = output_dir / "receipt.json"
    _json_dump(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "rho_d0": summaries["D0"]["aggregate"]["rho_phys_ratio_of_means"],
                "rho_e50": summaries["D1-E50"]["aggregate"]["rho_phys_ratio_of_means"],
                "rho_r50": summaries["D1-R50"]["aggregate"]["rho_phys_ratio_of_means"],
            },
            indent=2,
        )
    )
    if not gates["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
