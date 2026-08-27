#!/usr/bin/env python3
"""Build the second rollout target for the fixed 2-history x 4-action curve.

The existing planner-curve overlay ends at the first queried future.  This
builder replays exactly the same history and first action block, applies the
deployment evaluator's zero-action hold for one more block, and stores only
that second future image.  Pairing, actions, and model-visible inputs are
unchanged; no model, hidden label, contact outcome, or future outcome selects
an example.
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
from typing import Any, Iterable

import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
if str(CONTEXTWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEXTWORLD_ROOT))

from contextworld.evaluation import pusht_motion_damping_h3 as motion  # noqa: E402


DEFAULT_MANIFEST = CONTEXTWORLD_ROOT / (
    "artifacts/synthesis/pusht_motion_damping_h3_release_v4/manifest.json"
)
SCALES = (0.0, 0.25, 0.625, 1.0)
ACTION_BRANCHES = (
    "planner_scale_0p000",
    "planner_scale_0p250",
    "planner_scale_0p625",
    "planner_scale_1p000",
)
ROWS_PER_TEMPLATE = 8
LEAD_SECONDS = 1.0


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


def _base_action(template: motion.MotionDampingTemplate) -> np.ndarray:
    snapshot = np.asarray(
        template.expected_natural_query_snapshot, dtype=np.float64
    )
    agent = snapshot[0:2]
    intercept = snapshot[6:8] + LEAD_SECONDS * snapshot[8:10]
    return np.clip((intercept - agent) / 100.0, -1.0, 1.0)


def _query_actions(
    template: motion.MotionDampingTemplate, scale: float
) -> np.ndarray:
    action = np.clip(float(scale) * _base_action(template), -1.0, 1.0)
    return np.repeat(action[None], motion.ACTION_BLOCK, axis=0)


def _simulate_rollout2(
    template: motion.MotionDampingTemplate,
    *,
    mode: str,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Return exact first/second future pixels from one continuous simulator."""

    env, _ = motion.make_motion_damping_env(
        template, mode=mode, resolution=resolution
    )
    contacts = 0
    bounds: list[dict[str, list[float]]] = []
    try:
        for action in np.asarray(template.history_actions, dtype=np.float64):
            env.step(action)
        observed_query = motion.friction.body_snapshot(env)
        expected_query = np.asarray(
            template.expected_natural_query_snapshot, dtype=np.float64
        )
        deviation = float(
            np.max(
                np.abs(
                    motion.friction._snapshot_delta(
                        observed_query, expected_query
                    )
                )
            )
        )
        if deviation > motion.QUERY_REFERENCE_TOLERANCE:
            raise RuntimeError(
                f"query identity changed for {template.template_id}: {deviation}"
            )
        for action in np.asarray(template.query_actions, dtype=np.float64):
            contacts += motion.friction._step_and_count_agent_block_contacts(
                env, action
            )
            bounds.append(motion.friction._body_shape_bounds(env))
        future1 = np.asarray(env.render(), dtype=np.uint8).copy()
        zero = np.zeros(2, dtype=np.float64)
        for _ in range(motion.ACTION_BLOCK):
            contacts += motion.friction._step_and_count_agent_block_contacts(
                env, zero
            )
            bounds.append(motion.friction._body_shape_bounds(env))
        future2 = np.asarray(env.render(), dtype=np.uint8).copy()
    finally:
        env.close()
    inside = all(
        motion.friction._bounds_inside_playfield(value) for value in bounds
    )
    return future1, future2, contacts, bool(inside)


def _chunked(values: list[Any], size: int) -> Iterable[tuple[int, list[Any]]]:
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def _build_chunk(
    task: tuple[int, list[dict[str, Any]], int]
) -> tuple[int, np.ndarray, dict[str, Any]]:
    start, rows, resolution = task
    images: list[np.ndarray] = []
    first_images: list[np.ndarray] = []
    contacts: list[int] = []
    inside: list[bool] = []
    hidden_degenerate = 0
    action_gaps: list[float] = []
    hidden_gaps: list[float] = []
    for local_index, row in enumerate(rows):
        template = motion.MotionDampingTemplate(**row["template"])
        global_index = start + local_index
        expected = "forward" if global_index % 2 == 0 else "reverse"
        if not template.template_id.endswith(expected):
            raise RuntimeError("frozen forward/reverse twin order changed")
        per_action: list[tuple[np.ndarray, np.ndarray]] = []
        for scale in SCALES:
            branch = replace(
                template,
                query_actions=tuple(map(tuple, _query_actions(template, scale))),
            )
            per_mode: list[np.ndarray] = []
            for mode in motion.ENDPOINT_MODES:
                future1, future2, count, within = _simulate_rollout2(
                    branch, mode=mode, resolution=resolution
                )
                first_images.append(future1)
                images.append(future2)
                contacts.append(count)
                inside.append(within)
                per_mode.append(future2)
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
        for action_index in range(1, len(SCALES)):
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
    return (
        start,
        np.stack(images),
        {
            "future1_sha256": _array_sha256(np.stack(first_images)),
            "hidden_degenerate": hidden_degenerate,
            "hidden_gaps": hidden_gaps,
            "action_gaps": action_gaps,
            "contacts": contacts,
            "inside": inside,
        },
    )


def _build(
    *,
    manifest_path: Path,
    source_overlay: Path,
    template_count: int,
    resolution: int,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_receipt_path = source_overlay.with_suffix(source_overlay.suffix + ".json")
    if not source_overlay.is_file() or not source_receipt_path.is_file():
        raise FileNotFoundError("source planner-curve overlay is incomplete")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    source_sha = _sha256(source_overlay)
    checks = {
        "source_overlay_self_hash": source_receipt.get("overlay_sha256") == source_sha,
        "source_manifest_exact": source_receipt.get("source_manifest_sha256")
        == _sha256(manifest_path),
        "source_row_order_exact": source_receipt.get("row_order")
        == "template_then_action_branch_then_damping_mode",
        "source_action_branches_exact": source_receipt.get("action_branches")
        == list(ACTION_BRANCHES),
        "template_prefix_available": int(source_receipt.get("template_count", -1))
        >= template_count,
        "template_count_positive_even": template_count > 0 and template_count % 2 == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"rollout2 source contract failed: {checks}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["splits"]["train"]["pairs"][:template_count]
    if len(rows) != template_count:
        raise RuntimeError("manifest does not contain requested template prefix")

    expected_rows = ROWS_PER_TEMPLATE * template_count
    final = np.empty((expected_rows, resolution, resolution, 3), dtype=np.uint8)
    metrics: list[dict[str, Any]] = []
    tasks = [
        (start, chunk, resolution) for start, chunk in _chunked(rows, chunk_size)
    ]
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
            row_start = ROWS_PER_TEMPLATE * start
            final[row_start : row_start + len(array)] = array
            metrics.append(metric)
            completed += len(array) // ROWS_PER_TEMPLATE
            print(
                f"rollout2 templates {completed}/{template_count}", flush=True
            )
    finally:
        if workers != 1:
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
        motion.MotionDampingTemplate(**row["template"]).template_id for row in rows
    ]
    payload = {
        "future2_pixels": torch.from_numpy(final).permute(0, 3, 1, 2),
        "template_ids": tuple(template_ids),
        "template_count": template_count,
        "condition_pair_count": len(SCALES) * template_count,
        "row_order": "template_then_action_branch_then_damping_mode",
        "action_branches": ACTION_BRANCHES,
        "second_action_block": "five_zero_actions",
        "source_overlay_sha256": source_sha,
    }
    receipt = {
        "schema_version": 1,
        "status": "completed_rollout2_targets",
        "builder": str(THIS_SOURCE),
        "builder_sha256": _sha256(THIS_SOURCE),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_overlay": str(source_overlay),
        "source_overlay_sha256": source_sha,
        "source_overlay_receipt_sha256": _sha256(source_receipt_path),
        "template_count": template_count,
        "model_row_count": expected_rows,
        "condition_pair_count": len(SCALES) * template_count,
        "resolution": resolution,
        "row_order": payload["row_order"],
        "action_branches": list(ACTION_BRANCHES),
        "first_action_block": "same fixed planner curve as source overlay",
        "second_action_block": payload["second_action_block"],
        "future2_pixels_sha256": _array_sha256(final),
        "template_ids_sha256": hashlib.sha256(
            "\n".join(template_ids).encode()
        ).hexdigest(),
        "zero_hidden_future2_gap_count": int(
            sum(item["hidden_degenerate"] for item in metrics)
        ),
        "minimum_hidden_future2_mean_absolute_pixel_gap": float(hidden_gaps.min()),
        "minimum_action_future2_mean_absolute_pixel_gap": float(action_gaps.min()),
        "mean_rollout_contact_steps": float(contacts.mean()),
        "all_rollouts_inside_playfield": bool(inside.all()),
        "hidden_label_used": False,
        "future_outcome_used_for_selection": False,
        "contact_used_for_selection": False,
        "planner_model_or_output_used": False,
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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-overlay", type=Path, required=True)
    parser.add_argument("--template-count", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers <= 0 or args.chunk_size <= 0:
        raise ValueError("workers and chunk-size must be positive")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError(output)
    payload, receipt = _build(
        manifest_path=args.manifest.expanduser().resolve(),
        source_overlay=args.source_overlay.expanduser().resolve(),
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
    receipt_path = output.with_suffix(output.suffix + ".json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
