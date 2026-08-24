#!/usr/bin/env python3
"""Build a small real-future history x action training overlay.

For every frozen Motion-Damping training template this builder renders the
existing zero-action branch and one deterministic nonzero action branch under
both damping modes.  The two action branches share the exact same causal
history and query; all four futures come from the simulator.  Only RGB and
actions are stored for model consumption.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
if str(CONTEXTWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEXTWORLD_ROOT))

from contextworld.evaluation.pusht_motion_damping_h3 import (  # noqa: E402
    ENDPOINT_MODES,
    MotionDampingTemplate,
    simulate_motion_damping_clip,
)


DEFAULT_MANIFEST = CONTEXTWORLD_ROOT / (
    "artifacts/synthesis/pusht_motion_damping_h3_release_v4/manifest.json"
)
DEFAULT_TEMPLATE_COUNT = 256
DEFAULT_RESOLUTION = 224
ACTION_BRANCHES = ("observed_zero", "query_velocity_unit")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alternate_query_actions(template: MotionDampingTemplate) -> np.ndarray:
    """Use one unit action aligned with the hidden-invariant query velocity."""

    velocity = np.asarray(
        template.expected_natural_query_snapshot[8:10], dtype=np.float64
    )
    norm = float(np.linalg.norm(velocity))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("query block velocity must be finite and nonzero")
    action = velocity / norm
    result = np.repeat(action[None, :], 5, axis=0)
    if (
        result.shape != (5, 2)
        or not np.all(np.isfinite(result))
        or float(np.max(np.abs(result))) > 1.0 + 1.0e-12
    ):
        raise RuntimeError("alternate query action left the valid action box")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _build(
    *,
    manifest_path: Path,
    template_count: int,
    resolution: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["splits"]["train"]["pairs"]
    if template_count <= 0 or template_count % 2:
        raise ValueError("template_count must be positive and even")
    if template_count > len(rows):
        raise ValueError("template_count exceeds the frozen training catalog")

    pixels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    template_ids: list[str] = []
    action_future_gaps: list[float] = []
    hidden_future_gaps: list[float] = []
    maximum_query_prefix_difference = 0
    maximum_alternate_query_contacts = 0

    for index, row in enumerate(rows[:template_count]):
        template = MotionDampingTemplate(**row["template"])
        expected_direction = "forward" if index % 2 == 0 else "reverse"
        if not template.template_id.endswith(expected_direction):
            raise RuntimeError("frozen forward/reverse twin order changed")
        alternate = alternate_query_actions(template)
        variants = (
            template,
            replace(
                template,
                query_actions=tuple(map(tuple, alternate.tolist())),
            ),
        )
        rollout: dict[tuple[int, str], dict[str, Any]] = {}
        for action_index, branch_template in enumerate(variants):
            for mode in ENDPOINT_MODES:
                item = simulate_motion_damping_clip(
                    branch_template,
                    mode=mode,
                    resolution=resolution,
                )
                rollout[(action_index, mode)] = item
                pixels.append(
                    np.asarray(item["model_pixels"], dtype=np.uint8)
                )
                actions.append(
                    np.asarray(item["action_blocks"], dtype=np.float32)
                )
                if action_index == 1:
                    maximum_alternate_query_contacts = max(
                        maximum_alternate_query_contacts,
                        int(item["query_contact_steps"]),
                    )
        template_ids.append(template.template_id)

        for mode in ENDPOINT_MODES:
            observed = rollout[(0, mode)]
            counterfactual = rollout[(1, mode)]
            prefix_difference = int(
                np.max(
                    np.abs(
                        observed["model_pixels"][:3].astype(np.int16)
                        - counterfactual["model_pixels"][:3].astype(np.int16)
                    )
                )
            )
            maximum_query_prefix_difference = max(
                maximum_query_prefix_difference, prefix_difference
            )
            action_future_gaps.append(
                float(
                    np.mean(
                        np.abs(
                            observed["model_pixels"][3].astype(np.int16)
                            - counterfactual["model_pixels"][3].astype(np.int16)
                        )
                    )
                )
            )
        for action_index in range(2):
            low = rollout[(action_index, ENDPOINT_MODES[0])]
            high = rollout[(action_index, ENDPOINT_MODES[1])]
            if not np.array_equal(
                low["model_pixels"][2], high["model_pixels"][2]
            ):
                raise RuntimeError("hidden modes changed the common query")
            if not np.array_equal(
                low["action_blocks"], high["action_blocks"]
            ):
                raise RuntimeError("hidden modes received different actions")
            hidden_future_gaps.append(
                float(
                    np.mean(
                        np.abs(
                            low["model_pixels"][3].astype(np.int16)
                            - high["model_pixels"][3].astype(np.int16)
                        )
                    )
                )
            )

        if (index + 1) % 32 == 0 or index + 1 == template_count:
            print(
                f"cartesian templates {index + 1}/{template_count}",
                flush=True,
            )

    pixel_array = np.stack(pixels)
    action_array = np.stack(actions)
    expected_rows = 4 * template_count
    if pixel_array.shape != (
        expected_rows,
        4,
        resolution,
        resolution,
        3,
    ):
        raise RuntimeError(f"unexpected pixel array shape {pixel_array.shape}")
    if action_array.shape != (expected_rows, 4, 5, 2):
        raise RuntimeError(f"unexpected action array shape {action_array.shape}")
    if maximum_query_prefix_difference != 0:
        raise RuntimeError("action branches changed history or query pixels")
    if min(action_future_gaps) <= 0.0 or min(hidden_future_gaps) <= 0.0:
        raise RuntimeError("a real action or hidden future failed to separate")

    payload = {
        "pixels": torch.from_numpy(pixel_array).permute(0, 1, 4, 2, 3),
        "raw_action_blocks": torch.from_numpy(action_array),
        "template_ids": tuple(template_ids),
        "template_count": template_count,
        "condition_pair_count": 2 * template_count,
        "row_order": "template_then_action_branch_then_damping_mode",
        "action_branches": ACTION_BRANCHES,
    }
    receipt = {
        "schema_version": 1,
        "status": "completed_real_future_cartesian_overlay",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "template_count": template_count,
        "condition_pair_count": 2 * template_count,
        "model_row_count": expected_rows,
        "resolution": resolution,
        "row_order": payload["row_order"],
        "action_branches": list(ACTION_BRANCHES),
        "model_visible_fields": ["pixels", "actions"],
        "hidden_labels_stored": False,
        "pair_metadata_at_model_boundary": False,
        "maximum_history_or_query_pixel_difference_across_actions": (
            maximum_query_prefix_difference
        ),
        "minimum_action_future_mean_absolute_pixel_gap": float(
            min(action_future_gaps)
        ),
        "minimum_hidden_future_mean_absolute_pixel_gap": float(
            min(hidden_future_gaps)
        ),
        "maximum_alternate_query_contact_steps": (
            maximum_alternate_query_contacts
        ),
        "pixels_sha256": _array_sha256(pixel_array),
        "raw_action_blocks_sha256": _array_sha256(action_array),
        "template_ids_sha256": hashlib.sha256(
            "\n".join(template_ids).encode("utf-8")
        ).hexdigest(),
        "public_test_opened": False,
    }
    return payload, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template-count", type=int, default=DEFAULT_TEMPLATE_COUNT)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    output = Path(os.path.abspath(args.output.expanduser()))
    receipt_path = output.with_suffix(output.suffix + ".json")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite cartesian overlay")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload, receipt = _build(
        manifest_path=manifest,
        template_count=int(args.template_count),
        resolution=int(args.resolution),
    )
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt["overlay"] = str(output)
    receipt["overlay_bytes"] = output.stat().st_size
    receipt["overlay_sha256"] = _sha256(output)
    receipt["builder"] = str(THIS_SOURCE)
    receipt["builder_sha256"] = _sha256(THIS_SOURCE)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
