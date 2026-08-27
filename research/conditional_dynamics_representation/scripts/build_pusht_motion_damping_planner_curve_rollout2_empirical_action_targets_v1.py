#!/usr/bin/env python3
"""Build Motion rollout2 targets with ordinary replay action support.

This is the action-support-only counterpart of the released zero-hold
rollout2 builder.  History, first planner-curve action, template prefix, row
order, and hidden-condition pairing are unchanged.  For every template we
draw one deterministic contiguous five-action block from the ordinary PushT
training population and execute that same block for all four first-action
branches and both damping conditions.  No future, contact, model output, or
hidden condition selects an action block or retained example.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (
    build_pusht_contact_friction_rollout2_empirical_action_targets_v1
    as empirical,
)
from research.conditional_dynamics_representation.scripts import (
    build_pusht_motion_damping_planner_curve_rollout2_targets_v1 as zero,
)


THIS_SOURCE = Path(__file__).resolve()
SECOND_ACTION = "empirical_original_pusht_contiguous_block"
STATUS = "completed_motion_rollout2_empirical_action_targets"


def _simulate(
    template: Any,
    *,
    mode: str,
    second_actions: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, int, bool]:
    env, _ = zero.motion.make_motion_damping_env(
        template, mode=mode, resolution=resolution
    )
    contacts = 0
    bounds: list[dict[str, list[float]]] = []
    try:
        for action in np.asarray(template.history_actions, dtype=np.float64):
            env.step(action)
        observed = zero.motion.friction.body_snapshot(env)
        expected = np.asarray(
            template.expected_natural_query_snapshot, dtype=np.float64
        )
        deviation = float(
            np.max(
                np.abs(zero.motion.friction._snapshot_delta(observed, expected))
            )
        )
        if deviation > zero.motion.QUERY_REFERENCE_TOLERANCE:
            raise RuntimeError(
                f"query identity changed for {template.template_id}: {deviation}"
            )
        for action in np.asarray(template.query_actions, dtype=np.float64):
            contacts += zero.motion.friction._step_and_count_agent_block_contacts(
                env, action
            )
            bounds.append(zero.motion.friction._body_shape_bounds(env))
        for action in np.asarray(second_actions, dtype=np.float64):
            contacts += zero.motion.friction._step_and_count_agent_block_contacts(
                env, action
            )
            bounds.append(zero.motion.friction._body_shape_bounds(env))
        future2 = np.asarray(env.render(), dtype=np.uint8).copy()
    finally:
        env.close()
    inside = all(
        zero.motion.friction._bounds_inside_playfield(value) for value in bounds
    )
    return future2, contacts, bool(inside)


def _build_chunk(
    task: tuple[int, list[dict[str, Any]], np.ndarray, int]
) -> tuple[int, np.ndarray, dict[str, Any]]:
    start, rows, second_blocks, resolution = task
    images: list[np.ndarray] = []
    contacts: list[int] = []
    inside: list[bool] = []
    hidden_gaps: list[float] = []
    action_gaps: list[float] = []
    hidden_degenerate = 0
    for local_index, (row, second_actions) in enumerate(
        zip(rows, second_blocks, strict=True)
    ):
        template = zero.motion.MotionDampingTemplate(**row["template"])
        global_index = start + local_index
        expected = "forward" if global_index % 2 == 0 else "reverse"
        if not template.template_id.endswith(expected):
            raise RuntimeError("frozen forward/reverse twin order changed")
        per_action: list[tuple[np.ndarray, np.ndarray]] = []
        for scale in zero.SCALES:
            branch = replace(
                template,
                query_actions=tuple(
                    map(tuple, zero._query_actions(template, scale))
                ),
            )
            per_mode: list[np.ndarray] = []
            for mode in zero.motion.ENDPOINT_MODES:
                future2, count, within = _simulate(
                    branch,
                    mode=mode,
                    second_actions=second_actions,
                    resolution=resolution,
                )
                images.append(future2)
                per_mode.append(future2)
                contacts.append(count)
                inside.append(within)
            hidden_degenerate += int(np.array_equal(per_mode[0], per_mode[1]))
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
            per_action.append((per_mode[0], per_mode[1]))
        for action_index in range(1, len(zero.SCALES)):
            for mode_index in range(2):
                action_gaps.append(
                    float(
                        np.mean(
                            np.abs(
                                per_action[0][mode_index].astype(np.int16)
                                - per_action[action_index][mode_index].astype(
                                    np.int16
                                )
                            )
                        )
                    )
                )
    return start, np.stack(images), {
        "hidden_degenerate": hidden_degenerate,
        "hidden_gaps": hidden_gaps,
        "action_gaps": action_gaps,
        "contacts": contacts,
        "inside": inside,
    }


def _build(
    *,
    manifest_path: Path,
    source_overlay: Path,
    dataset: Path,
    template_count: int,
    resolution: int,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_receipt_path = source_overlay.with_suffix(source_overlay.suffix + ".json")
    if not source_overlay.is_file() or not source_receipt_path.is_file():
        raise FileNotFoundError("source planner-curve overlay is incomplete")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    source_sha = zero._sha256(source_overlay)
    checks = {
        "source_overlay_self_hash": source_receipt.get("overlay_sha256")
        == source_sha,
        "source_manifest_exact": source_receipt.get("source_manifest_sha256")
        == zero._sha256(manifest_path),
        "source_row_order_exact": source_receipt.get("row_order")
        == "template_then_action_branch_then_damping_mode",
        "source_action_branches_exact": source_receipt.get("action_branches")
        == list(zero.ACTION_BRANCHES),
        "template_prefix_available": int(
            source_receipt.get("template_count", -1)
        )
        >= template_count,
        "template_count_positive_even": template_count > 0
        and template_count % 2 == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"empirical rollout2 source contract failed: {checks}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["splits"]["train"]["pairs"][:template_count]
    if len(rows) != template_count:
        raise RuntimeError("manifest does not contain requested template prefix")
    second_actions, episodes, starts, action_source = (
        empirical._empirical_action_catalog(
            dataset,
            template_count=template_count,
            seed=empirical.ACTION_CATALOG_SEED,
        )
    )

    expected_rows = zero.ROWS_PER_TEMPLATE * template_count
    final = np.empty((expected_rows, resolution, resolution, 3), dtype=np.uint8)
    tasks = [
        (
            start,
            chunk,
            second_actions[start : start + len(chunk)],
            resolution,
        )
        for start, chunk in zero._chunked(rows, chunk_size)
    ]
    metrics: list[dict[str, Any]] = []
    executor = None
    if workers == 1:
        results = map(_build_chunk, tasks)
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        )
        results = executor.map(_build_chunk, tasks, chunksize=1)
    try:
        completed = 0
        for start, array, metric in results:
            row_start = zero.ROWS_PER_TEMPLATE * start
            final[row_start : row_start + len(array)] = array
            metrics.append(metric)
            completed += len(array) // zero.ROWS_PER_TEMPLATE
            print(
                f"Motion empirical-action rollout2 templates "
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
    action_gaps = np.asarray(
        [value for item in metrics for value in item["action_gaps"]],
        dtype=np.float64,
    )
    contacts = np.asarray(
        [value for item in metrics for value in item["contacts"]],
        dtype=np.int64,
    )
    inside = np.asarray(
        [value for item in metrics for value in item["inside"]], dtype=bool
    )
    template_ids = [
        zero.motion.MotionDampingTemplate(**row["template"]).template_id
        for row in rows
    ]
    payload = {
        "future2_pixels": torch.from_numpy(final).permute(0, 3, 1, 2),
        "second_action_blocks_raw": torch.from_numpy(second_actions),
        "source_episode_indices": torch.from_numpy(episodes),
        "source_action_start_rows": torch.from_numpy(starts),
        "template_ids": tuple(template_ids),
        "template_count": template_count,
        "condition_pair_count": len(zero.SCALES) * template_count,
        "row_order": "template_then_action_branch_then_damping_mode",
        "action_branches": zero.ACTION_BRANCHES,
        "second_action_block": SECOND_ACTION,
        "source_overlay_sha256": source_sha,
    }
    receipt = {
        "schema_version": 1,
        "status": STATUS,
        "builder": str(THIS_SOURCE),
        "builder_sha256": zero._sha256(THIS_SOURCE),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": zero._sha256(manifest_path),
        "source_overlay": str(source_overlay),
        "source_overlay_sha256": source_sha,
        "source_overlay_receipt_sha256": zero._sha256(source_receipt_path),
        "original_pusht_action_source": str(dataset),
        "action_source": action_source,
        "template_count": template_count,
        "model_row_count": expected_rows,
        "condition_pair_count": len(zero.SCALES) * template_count,
        "resolution": resolution,
        "row_order": payload["row_order"],
        "action_branches": list(zero.ACTION_BRANCHES),
        "first_action_block": "same fixed planner curve as source overlay",
        "second_action_block": SECOND_ACTION,
        "second_action_blocks_sha256": zero._array_sha256(second_actions),
        "source_episode_indices_sha256": zero._array_sha256(episodes),
        "source_action_start_rows_sha256": zero._array_sha256(starts),
        "future2_pixels_sha256": zero._array_sha256(final),
        "template_ids_sha256": hashlib.sha256(
            "\n".join(template_ids).encode("utf-8")
        ).hexdigest(),
        "zero_hidden_future2_gap_count": int(
            sum(item["hidden_degenerate"] for item in metrics)
        ),
        "minimum_hidden_future2_mean_absolute_pixel_gap": float(
            hidden_gaps.min()
        ),
        "minimum_action_future2_mean_absolute_pixel_gap": float(
            action_gaps.min()
        ),
        "mean_rollout_contact_steps": float(contacts.mean()),
        "all_rollouts_inside_playfield": bool(inside.all()),
        "same_second_action_across_first_action_branches": True,
        "same_second_action_across_hidden_conditions": True,
        "future_outcome_used_for_selection": False,
        "contact_used_for_selection": False,
        "model_or_planner_output_used": False,
        "hidden_label_at_model_or_loss_boundary": False,
        "teacher_free": True,
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "inference_compute_added": 0,
        "source_checks": checks,
        "public_test_opened": False,
    }
    return payload, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=zero.DEFAULT_MANIFEST)
    parser.add_argument("--source-overlay", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=empirical.DEFAULT_PUSHT_H5)
    parser.add_argument("--template-count", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
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
        source_overlay=args.source_overlay.expanduser().resolve(),
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
    receipt["rollout2_targets_sha256"] = zero._sha256(output)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
