#!/usr/bin/env python3
"""Build Contact rollout2 targets with an empirical second action block.

This is the single-factor follow-up to the repeated-query RC-COJA transfer.
The first query block and every released history remain unchanged.  The second
block is a deterministic, selection-free sample of five consecutive actions
from the ordinary PushT training population.  The same block is executed for
the low- and high-friction members of each pair.  No future, contact outcome,
model output, or hidden condition is used to choose a block or retain a row.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import h5py
import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
if str(CONTEXTWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEXTWORLD_ROOT))

from contextworld.evaluation import pusht_contact_friction_h3 as contact  # noqa: E402


DEFAULT_MANIFEST = CONTEXTWORLD_ROOT / (
    "artifacts/synthesis/pusht_contact_friction_h3_release_v3/manifest.json"
)
DEFAULT_PUSHT_H5 = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "lewm-pusht/pusht_expert_train.h5"
)
ROW_ORDER = "template_then_low_friction_then_high_friction"
ROWS_PER_TEMPLATE = 2
RAW_STEPS_PER_BLOCK = 5
ACTION_CATALOG_SEED = 20260827


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _empirical_action_catalog(
    dataset: Path,
    *,
    template_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Select one contiguous five-action block from each of N episodes."""

    with h5py.File(dataset, "r") as handle:
        required = ("action", "ep_len", "ep_offset")
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"PushT action source is missing {missing}")
        episode_lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        episode_offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        if len(episode_lengths) < template_count:
            raise RuntimeError("Not enough original PushT episodes")
        if np.any(episode_lengths[:template_count] < RAW_STEPS_PER_BLOCK):
            raise RuntimeError("Original PushT episode is shorter than one block")
        generator = np.random.default_rng(int(seed))
        episodes = np.arange(template_count, dtype=np.int64)
        local_starts = np.asarray(
            [
                generator.integers(
                    0,
                    int(episode_lengths[index]) - RAW_STEPS_PER_BLOCK + 1,
                )
                for index in episodes
            ],
            dtype=np.int64,
        )
        absolute_starts = episode_offsets[episodes] + local_starts
        actions = np.stack(
            [
                np.asarray(
                    handle["action"][
                        start : start + RAW_STEPS_PER_BLOCK
                    ],
                    dtype=np.float32,
                )
                for start in absolute_starts
            ]
        )
        source_shape = tuple(handle["action"].shape)
        source_dtype = str(handle["action"].dtype)
    if actions.shape != (template_count, RAW_STEPS_PER_BLOCK, 2):
        raise RuntimeError(f"Unexpected empirical action shape {actions.shape}")
    return actions, episodes, absolute_starts, {
        "source_action_shape": list(source_shape),
        "source_action_dtype": source_dtype,
        "dataset_size_bytes": dataset.stat().st_size,
        "catalog_seed": int(seed),
        "selection": (
            "one deterministic contiguous five-step block from each of the "
            "first template_count original PushT episodes"
        ),
    }


def _simulate(
    template: contact.ContactFrictionTemplate,
    *,
    mode: str,
    second_actions: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, int, bool]:
    env, _ = contact.make_contact_friction_env(
        template, mode=mode, resolution=resolution
    )
    contacts = 0
    bounds: list[dict[str, list[float]]] = []
    try:
        for action in np.asarray(template.history_actions, dtype=np.float64):
            env.step(action)
        observed = contact.body_snapshot(env)
        expected = np.asarray(template.canonical_query_snapshot, dtype=np.float64)
        deviation = float(
            np.max(np.abs(contact._snapshot_delta(observed, expected)))
        )
        if deviation > contact.STRICT_QUERY_FULL_STATE_TOLERANCE:
            raise RuntimeError(
                f"query identity changed for {template.template_id}: {deviation}"
            )
        for action in np.asarray(template.query_actions, dtype=np.float64):
            contacts += contact._step_and_count_agent_block_contacts(env, action)
            bounds.append(contact._body_shape_bounds(env))
        for action in np.asarray(second_actions, dtype=np.float64):
            contacts += contact._step_and_count_agent_block_contacts(env, action)
            bounds.append(contact._body_shape_bounds(env))
        future2 = np.asarray(env.render(), dtype=np.uint8).copy()
    finally:
        env.close()
    inside = all(contact._bounds_inside_playfield(value) for value in bounds)
    return future2, contacts, bool(inside)


def _chunked(values: list[Any], size: int) -> Iterable[tuple[int, list[Any]]]:
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def _build_chunk(
    task: tuple[int, list[dict[str, Any]], np.ndarray, int]
) -> tuple[int, np.ndarray, dict[str, Any]]:
    start, rows, second_blocks, resolution = task
    images: list[np.ndarray] = []
    hidden_gaps: list[float] = []
    contacts: list[int] = []
    inside: list[bool] = []
    degenerate = 0
    for row, second_actions in zip(rows, second_blocks, strict=True):
        template = contact.ContactFrictionTemplate(**row["template"])
        per_mode: list[np.ndarray] = []
        for mode in contact.ENDPOINT_MODES:
            future2, count, within = _simulate(
                template,
                mode=mode,
                second_actions=second_actions,
                resolution=resolution,
            )
            images.append(future2)
            per_mode.append(future2)
            contacts.append(count)
            inside.append(within)
        degenerate += int(np.array_equal(per_mode[0], per_mode[1]))
        hidden_gaps.append(
            float(
                np.mean(
                    np.abs(
                        per_mode[0].astype(np.int16)
                        - per_mode[1].astype(np.int16)
                    )
                )
            )
        )
    return start, np.stack(images), {
        "hidden_degenerate": degenerate,
        "hidden_gaps": hidden_gaps,
        "contacts": contacts,
        "inside": inside,
    }


def _build(
    *,
    manifest_path: Path,
    dataset: Path,
    template_count: int,
    resolution: int,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = manifest["splits"]["train"]["pairs"]
    if template_count <= 0 or len(all_rows) < template_count:
        raise RuntimeError("Requested Contact template population is unavailable")
    rows = all_rows[:template_count]
    second_actions, episodes, starts, action_source = _empirical_action_catalog(
        dataset,
        template_count=template_count,
        seed=ACTION_CATALOG_SEED,
    )
    expected_rows = ROWS_PER_TEMPLATE * template_count
    final = np.empty((expected_rows, resolution, resolution, 3), dtype=np.uint8)
    tasks = [
        (
            start,
            chunk,
            second_actions[start : start + len(chunk)],
            resolution,
        )
        for start, chunk in _chunked(rows, chunk_size)
    ]
    metrics: list[dict[str, Any]] = []
    if workers == 1:
        results = map(_build_chunk, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        )
        results = executor.map(_build_chunk, tasks, chunksize=1)
    try:
        completed = 0
        for start, array, metric in results:
            row_start = ROWS_PER_TEMPLATE * start
            final[row_start : row_start + len(array)] = array
            metrics.append(metric)
            completed += len(array) // ROWS_PER_TEMPLATE
            print(
                f"Contact empirical-action rollout2 templates "
                f"{completed}/{template_count}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    hidden_gaps = np.asarray(
        [value for item in metrics for value in item["hidden_gaps"]],
        dtype=np.float64,
    )
    contact_counts = np.asarray(
        [value for item in metrics for value in item["contacts"]],
        dtype=np.int64,
    )
    inside = np.asarray(
        [value for item in metrics for value in item["inside"]], dtype=bool
    )
    template_ids = [
        contact.ContactFrictionTemplate(**row["template"]).template_id
        for row in rows
    ]
    payload = {
        "future2_pixels": torch.from_numpy(final).permute(0, 3, 1, 2),
        "second_action_blocks_raw": torch.from_numpy(second_actions),
        "source_episode_indices": torch.from_numpy(episodes),
        "source_action_start_rows": torch.from_numpy(starts),
        "template_ids": tuple(template_ids),
        "template_count": template_count,
        "condition_pair_count": template_count,
        "row_order": ROW_ORDER,
        "second_action_block": "empirical_original_pusht_contiguous_block",
        "source_manifest_sha256": _sha256(manifest_path),
    }
    receipt = {
        "schema_version": 1,
        "status": "completed_contact_rollout2_empirical_action_targets",
        "builder": str(THIS_SOURCE),
        "builder_sha256": _sha256(THIS_SOURCE),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "original_pusht_action_source": str(dataset),
        "action_source": action_source,
        "template_count": template_count,
        "model_row_count": expected_rows,
        "condition_pair_count": template_count,
        "resolution": resolution,
        "row_order": ROW_ORDER,
        "first_action_block": "released_query_action_block",
        "second_action_block": payload["second_action_block"],
        "second_action_blocks_sha256": _array_sha256(second_actions),
        "source_episode_indices_sha256": _array_sha256(episodes),
        "source_action_start_rows_sha256": _array_sha256(starts),
        "future2_pixels_sha256": _array_sha256(final),
        "template_ids_sha256": hashlib.sha256(
            "\n".join(template_ids).encode("utf-8")
        ).hexdigest(),
        "zero_hidden_future2_gap_count": int(
            sum(item["hidden_degenerate"] for item in metrics)
        ),
        "minimum_hidden_future2_mean_absolute_pixel_gap": float(
            hidden_gaps.min()
        ),
        "mean_rollout_contact_steps": float(contact_counts.mean()),
        "all_rollouts_inside_playfield": bool(inside.all()),
        "all_released_templates_retained": template_count == len(all_rows),
        "same_second_action_for_both_hidden_conditions": True,
        "future_outcome_used_for_selection": False,
        "contact_used_for_selection": False,
        "model_or_planner_output_used": False,
        "hidden_label_at_model_or_loss_boundary": False,
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "inference_compute_added": 0,
        "public_test_opened": False,
    }
    return payload, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PUSHT_H5)
    parser.add_argument("--template-count", type=int, default=8192)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers <= 0 or args.chunk_size <= 0:
        raise ValueError("workers and chunk-size must be positive")
    output = args.output.expanduser().resolve()
    receipt_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload, receipt = _build(
        manifest_path=args.manifest.expanduser().resolve(),
        dataset=args.dataset.expanduser().resolve(),
        template_count=args.template_count,
        resolution=args.resolution,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt["rollout2_targets"] = str(output)
    receipt["rollout2_targets_bytes"] = output.stat().st_size
    receipt["rollout2_targets_sha256"] = _sha256(output)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
