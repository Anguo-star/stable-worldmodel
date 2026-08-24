#!/usr/bin/env python3
"""Recover conditional pairs from visible ``(query RGB, action)`` only.

This is a zero-training qualification of the remaining privilege boundary in
the replay Cartesian data.  The proposed grouping key excludes history,
future, damping labels, pair ids, template ids and row order.  If every key
has exactly two rows, those rows are a matched history intervention at the
same visible query and action.  The explicit row-order groups are consulted
only after mining to audit equivalence; they are not inputs to the miner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import torch


THIS_SOURCE = Path(__file__).resolve()
OVERLAY_SHA256 = (
    "f991f81ba19a84350dee7df543ff6093a96f13f6227c1b1a1f135a15fbbfd79f"
)
EXPECTED_TEMPLATE_COUNT = 2048
EXPECTED_ROW_COUNT = 8192
EXPECTED_PAIR_COUNT = 4096
KEY_SCHEMA = "sha256(query_rgb_uint8_bytes || raw_action_blocks_float_bytes)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visible_condition_key(
    query_rgb: torch.Tensor,
    raw_action_blocks: torch.Tensor,
) -> bytes:
    if query_rgb.dtype != torch.uint8 or query_rgb.shape != (3, 224, 224):
        raise ValueError("visible query key requires uint8 RGB 3x224x224")
    if (
        not raw_action_blocks.is_floating_point()
        or raw_action_blocks.shape != (4, 5, 2)
        or not bool(torch.isfinite(raw_action_blocks).all())
    ):
        raise ValueError("visible action key requires finite float 4x5x2")
    digest = hashlib.sha256()
    digest.update(query_rgb.contiguous().numpy().tobytes())
    digest.update(raw_action_blocks.contiguous().numpy().tobytes())
    return digest.digest()


def mine_visible_condition_groups(
    pixels: torch.Tensor,
    raw_action_blocks: torch.Tensor,
) -> tuple[torch.Tensor, list[bytes]]:
    if (
        pixels.dtype != torch.uint8
        or pixels.ndim != 5
        or pixels.shape[1:] != (4, 3, 224, 224)
    ):
        raise ValueError("pixels must be uint8 [N,4,3,224,224]")
    if (
        not raw_action_blocks.is_floating_point()
        or raw_action_blocks.shape != (pixels.shape[0], 4, 5, 2)
    ):
        raise ValueError("raw actions must be float [N,4,5,2]")
    buckets: dict[bytes, list[int]] = {}
    for row in range(int(pixels.shape[0])):
        key = visible_condition_key(
            pixels[row, 2],
            raw_action_blocks[row],
        )
        buckets.setdefault(key, []).append(row)
    bad = sorted(
        (key.hex(), len(rows))
        for key, rows in buckets.items()
        if len(rows) != 2
    )
    if bad:
        raise RuntimeError(
            "visible-condition keys are not uniquely binary: "
            + json.dumps(bad[:8])
        )
    ordered = sorted(buckets.items(), key=lambda item: tuple(item[1]))
    groups = torch.tensor([rows for _, rows in ordered], dtype=torch.long)
    return groups, [key for key, _ in ordered]


def _group_mapping_sha256(groups: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for left, right in groups.tolist():
        digest.update(struct.pack("<qq", int(left), int(right)))
    return digest.hexdigest()


def _key_set_sha256(keys: list[bytes]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key)
    return digest.hexdigest()


def analyze(overlay: Path) -> dict[str, Any]:
    overlay = overlay.expanduser().resolve()
    if not overlay.is_file() or _sha256(overlay) != OVERLAY_SHA256:
        raise RuntimeError("visible-condition overlay identity changed")
    payload = torch.load(
        overlay,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    pixels = payload["pixels"]
    raw_actions = payload["raw_action_blocks"]
    template_count = int(payload["template_count"])
    if (
        template_count != EXPECTED_TEMPLATE_COUNT
        or int(pixels.shape[0]) != EXPECTED_ROW_COUNT
        or payload["row_order"]
        != "template_then_action_branch_then_damping_mode"
    ):
        raise RuntimeError("visible-condition overlay shape contract changed")

    mined, keys = mine_visible_condition_groups(pixels, raw_actions)
    explicit = torch.tensor(
        [
            (4 * template + 2 * branch, 4 * template + 2 * branch + 1)
            for template in range(template_count)
            for branch in range(2)
        ],
        dtype=torch.long,
    )
    mined_set = {tuple(rows) for rows in mined.tolist()}
    explicit_set = {tuple(rows) for rows in explicit.tolist()}
    histories_differ = all(
        not torch.equal(pixels[left, :2], pixels[right, :2])
        for left, right in mined.tolist()
    )
    futures_differ = all(
        not torch.equal(pixels[left, 3], pixels[right, 3])
        for left, right in mined.tolist()
    )
    checks = {
        "row_count_exact": int(pixels.shape[0]) == EXPECTED_ROW_COUNT,
        "visible_key_count_exact": len(keys) == EXPECTED_PAIR_COUNT,
        "every_visible_key_binary": mined.shape == (EXPECTED_PAIR_COUNT, 2),
        "mined_pair_set_equals_explicit_pair_set": mined_set == explicit_set,
        "every_mined_pair_has_distinct_history": histories_differ,
        "every_mined_pair_has_distinct_future": futures_differ,
        "history_excluded_from_key": True,
        "future_excluded_from_key": True,
        "hidden_label_pair_id_template_id_and_row_order_excluded_from_key": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"visible-condition mining failed: {checks}")
    return {
        "schema_version": 1,
        "status": "completed_exact_visible_condition_pair_recovery",
        "overlay": str(overlay),
        "overlay_sha256": OVERLAY_SHA256,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
        "key_schema": KEY_SCHEMA,
        "key_inputs": ["current_query_rgb", "complete_raw_action_blocks"],
        "key_excludes": [
            "history_rgb",
            "future_rgb",
            "damping_or_hidden_mode",
            "pair_id",
            "template_id",
            "row_order",
        ],
        "row_count": int(pixels.shape[0]),
        "visible_key_count": len(keys),
        "pair_count": int(mined.shape[0]),
        "group_width": int(mined.shape[1]),
        "visible_key_set_sha256": _key_set_sha256(keys),
        "mined_group_mapping_sha256": _group_mapping_sha256(mined),
        "explicit_group_mapping_sha256": _group_mapping_sha256(explicit),
        "checks": checks,
        "conclusion": (
            "For this complete replay Cartesian asset, privileged pair "
            "annotations are redundant: the exact 4096 binary intervention "
            "groups are a deterministic function of visible (Q,A). The "
            "remaining assumption is matched conditional-overlap data, not "
            "access to a hidden dynamics label."
        ),
        "claim_boundary": {
            "training_steps": 0,
            "checkpoint_opened": False,
            "ordinary_unmatched_replay_pair_recovery_claimed": False,
            "approximate_query_matching_tested": False,
            "conditional_overlap_still_required": True,
            "general_label_free_method_claimed": False,
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = analyze(args.overlay)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
