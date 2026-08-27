#!/usr/bin/env python3
"""Matched native/COJA training on a 2-history x 4-action planner curve.

The deployed LeWM, optimizer, 64/64 row mixture, and canonical conditional
relation are unchanged.  The only new variable is that every hidden query now
contains four real action branches (0, 0.25, 0.625, and 1.0 on the observable
planner ray) instead of one action branch or one action pair.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_full_release_replay_cartesian_absolute_single_stage_v1
    as full,
)


ACTION_BRANCHES = [
    "planner_scale_0p000",
    "planner_scale_0p250",
    "planner_scale_0p625",
    "planner_scale_1p000",
]
ACTION_SOURCE = "fixed_deployment_planner_proposal_curve"
CANDIDATE_PREFIX = (
    "pusht_motion_damping_full_release_planner_curve_cartesian_"
    "absolute_single_stage"
)
ROWS_PER_TEMPLATE = 8
ROWS_PER_TWIN = 16
ACTION_BRANCH_COUNT = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PlannerCurveTwinBatchStream:
    """Yield complete forward/reverse twins with all four action branches."""

    def __init__(self, pair_count: int, *, batch_size: int, seed: int) -> None:
        if pair_count <= 0 or pair_count % ACTION_BRANCH_COUNT:
            raise ValueError("condition pair_count must contain four actions")
        template_count = pair_count // ACTION_BRANCH_COUNT
        if template_count % 2:
            raise ValueError("templates must contain forward/reverse twins")
        if batch_size <= 0 or batch_size % ROWS_PER_TWIN:
            raise ValueError("hidden batch must contain complete 16-row twins")
        self.template_count = template_count
        self.twin_count = template_count // 2
        self.twins_per_batch = batch_size // ROWS_PER_TWIN
        if self.twin_count % self.twins_per_batch:
            raise ValueError("twin_count must divide evenly across batches")
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            order = torch.randperm(self.twin_count, generator=self.generator)
            for start in range(0, self.twin_count, self.twins_per_batch):
                twins = order[start : start + self.twins_per_batch]
                templates = torch.stack(
                    [2 * twins, 2 * twins + 1], dim=1
                ).reshape(-1)
                offsets = torch.arange(ROWS_PER_TEMPLATE)
                yield (
                    ROWS_PER_TEMPLATE * templates[:, None] + offsets[None, :]
                ).reshape(-1)


def _validate_grid(
    pixels: torch.Tensor,
    raw_action_blocks: torch.Tensor,
    *,
    template_count: int,
    expected_degenerate_action_futures: int,
    expected_degenerate_hidden_futures: int,
) -> dict[str, bool]:
    expected_rows = ROWS_PER_TEMPLATE * int(template_count)
    if pixels.dtype != torch.uint8 or pixels.shape != (
        expected_rows,
        4,
        3,
        224,
        224,
    ):
        raise ValueError(f"unexpected planner-curve pixels {tuple(pixels.shape)}")
    if not raw_action_blocks.is_floating_point() or raw_action_blocks.shape != (
        expected_rows,
        4,
        5,
        2,
    ):
        raise ValueError(
            f"unexpected planner-curve actions {tuple(raw_action_blocks.shape)}"
        )
    if not bool(torch.isfinite(raw_action_blocks).all()):
        raise FloatingPointError("planner-curve actions contain nonfinite values")

    checks = {
        "history_query_exact_across_action_branches": True,
        "query_exact_across_hidden_modes": True,
        "support_actions_exact_across_action_branches": True,
        "query_actions_distinct_across_action_branches": True,
        "actions_exact_across_hidden_modes": True,
        "history_differs_across_hidden_modes": True,
    }
    degenerate = 0
    hidden_degenerate = 0
    for template_index in range(template_count):
        start = ROWS_PER_TEMPLATE * template_index
        zero_by_mode = (start, start + 1)
        query_actions: list[torch.Tensor] = []
        for action_index in range(ACTION_BRANCH_COUNT):
            low = start + 2 * action_index
            high = low + 1
            query_actions.append(raw_action_blocks[low, 2])
            checks["query_exact_across_hidden_modes"] &= bool(
                torch.equal(pixels[low, 2], pixels[high, 2])
            )
            checks["actions_exact_across_hidden_modes"] &= bool(
                torch.equal(raw_action_blocks[low], raw_action_blocks[high])
            )
            checks["history_differs_across_hidden_modes"] &= bool(
                not torch.equal(pixels[low, 1], pixels[high, 1])
            )
            hidden_degenerate += int(
                torch.equal(pixels[low, 3], pixels[high, 3])
            )
            if action_index:
                for mode_index, zero in enumerate(zero_by_mode):
                    row = low + mode_index
                    checks["history_query_exact_across_action_branches"] &= bool(
                        torch.equal(pixels[zero, :3], pixels[row, :3])
                    )
                    checks["support_actions_exact_across_action_branches"] &= bool(
                        torch.equal(raw_action_blocks[zero, :2], raw_action_blocks[row, :2])
                        and torch.equal(raw_action_blocks[zero, 3:], raw_action_blocks[row, 3:])
                    )
                    degenerate += int(torch.equal(pixels[zero, 3], pixels[row, 3]))
        for left in range(ACTION_BRANCH_COUNT):
            for right in range(left + 1, ACTION_BRANCH_COUNT):
                checks["query_actions_distinct_across_action_branches"] &= bool(
                    not torch.equal(query_actions[left], query_actions[right])
                )
    checks["degenerate_action_future_count_exact"] = (
        degenerate == int(expected_degenerate_action_futures)
    )
    checks["degenerate_hidden_future_count_exact"] = (
        hidden_degenerate == int(expected_degenerate_hidden_futures)
    )
    if not all(checks.values()):
        raise RuntimeError(f"planner-curve grid contract failed: {checks}")
    return checks


def _validate_receipt(receipt: dict[str, Any]) -> dict[str, bool]:
    count = int(receipt.get("template_count", -1))
    hard = receipt.get("planner_curve_support_hard_checks", {})
    checks = {
        "template_count_registered": count in full.ALLOWED_TEMPLATE_COUNTS,
        "row_count_exact": int(receipt.get("model_row_count", -1)) == 8 * count,
        "condition_pair_count_exact": int(
            receipt.get("condition_pair_count", -1)
        )
        == 4 * count,
        "action_branches_exact": receipt.get("action_branches") == ACTION_BRANCHES,
        "action_source_exact": receipt.get("action_source") == ACTION_SOURCE,
        "scale_grid_exact": receipt.get("scale_grid") == [0.0, 0.25, 0.625, 1.0],
        "teacher_free": receipt.get("teacher_free") is True,
        "model_boundary_pixels_actions_only": (
            receipt.get("hidden_labels_stored") is False
            and receipt.get("pair_metadata_at_model_boundary") is False
        ),
        "selection_is_label_teacher_and_outcome_free": (
            receipt.get("planner_model_or_output_used") is False
            and receipt.get("hidden_label_used") is False
            and receipt.get("future_outcome_used_for_selection") is False
            and receipt.get("contact_used_for_selection") is False
        ),
        "prefix_exact": int(
            receipt.get("maximum_history_or_query_pixel_difference_across_actions", -1)
        )
        == 0,
        "support_hard_checks_all_true": bool(hard)
        and all(bool(value) for value in hard.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "planner-curve overlay contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return checks


def _compact_support(
    receipt_path: Path, receipt: dict[str, Any]
) -> dict[str, Any]:
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "action_source": receipt["action_source"],
        "action_rule": receipt["action_rule"],
        "action_branches": receipt["action_branches"],
        "lead_seconds": receipt["lead_seconds"],
        "scale_grid": receipt["scale_grid"],
        "scale_counts": receipt["scale_counts"],
        "teacher_free": receipt["teacher_free"],
        "planner_model_or_output_used": receipt["planner_model_or_output_used"],
        "hidden_label_used": receipt["hidden_label_used"],
        "future_outcome_used_for_selection": receipt[
            "future_outcome_used_for_selection"
        ],
        "contact_used_for_selection": receipt["contact_used_for_selection"],
        "reported_physical_covariates": {
            "note": "reported after fixed assignment; never selection gates",
            "query_contact_rate": receipt["query_contact_rate"],
            "mean_query_contact_steps": receipt["mean_query_contact_steps"],
            "hidden_action_interaction_nonzero_rate": receipt[
                "hidden_action_interaction_nonzero_rate"
            ],
            "median_hidden_action_interaction_norm": receipt[
                "median_hidden_action_interaction_norm"
            ],
        },
        "hard_checks": receipt["planner_curve_support_hard_checks"],
    }


def _bind_overlay(overlay: Path) -> dict[str, Any]:
    overlay = overlay.expanduser().resolve()
    receipt_path = overlay.with_suffix(overlay.suffix + ".json")
    if not overlay.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing overlay or receipt: {overlay}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("overlay_sha256") != _sha256(overlay):
        raise RuntimeError("planner-curve overlay hash changed")
    checks = _validate_receipt(receipt)
    template_count = int(receipt["template_count"])
    condition_pair_count = ACTION_BRANCH_COUNT * template_count

    replay = full.replay
    replay.OVERLAY_SHA256 = str(receipt["overlay_sha256"])
    replay.OVERLAY_RECEIPT_SHA256 = _sha256(receipt_path)
    replay.OVERLAY_TEMPLATE_COUNT = template_count
    replay.OVERLAY_CONDITION_PAIR_COUNT = condition_pair_count
    replay.ACTION_BRANCHES = ACTION_BRANCHES
    replay.ACTION_SOURCE = ACTION_SOURCE
    replay.validate_receipt = _validate_receipt
    replay.compact_replay_support = _compact_support
    return {
        "overlay": str(overlay),
        "overlay_sha256": str(receipt["overlay_sha256"]),
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "checks": checks,
        "template_count": template_count,
        "condition_pair_count": condition_pair_count,
        "zero_action_future_gap_count": int(receipt["zero_action_future_gap_count"]),
        "zero_hidden_future_gap_count": int(receipt["zero_hidden_future_gap_count"]),
        "planner_curve_support": _compact_support(receipt_path, receipt),
    }


def _rewrite_multiaction_receipts(output: Path) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    contract = payload["result"]["motion_cartesian_action_pair_contract"]
    grouping = payload["result"]["batch"]["motion_damping_twin_grouping"]
    grouping.update(
        {
            "condition_rows_per_group": ROWS_PER_TWIN,
            "condition_pairs_per_group": 8,
            "action_branches_per_condition": ACTION_BRANCH_COUNT,
            "real_future_cartesian_grid": True,
        }
    )
    contract["cartesian_training_overlay"].update(
        {
            "rows_per_forward_reverse_twin": ROWS_PER_TWIN,
            "condition_pair_count": 4
            * int(contract["cartesian_training_overlay"]["template_count"]),
            "action_branches": ACTION_BRANCHES,
        }
    )
    contract["objective"] = (
        "native_mse_on_real_2_history_x_4_action_grid + 0.09*SIGReg + "
        "aux_weight*(canonical_history_response + canonical_assignment_0p5)"
    )
    contract["checks"].pop("eight_row_twin_grouping", None)
    contract["checks"]["sixteen_row_twin_grouping"] = (
        grouping["condition_rows_per_group"] == ROWS_PER_TWIN
        and grouping["condition_pairs_per_group"] == 8
        and grouping["action_branches_per_condition"] == ACTION_BRANCH_COUNT
    )
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_sha256 = _sha256(report)
    sidecar = output / "motion_cartesian_action_pair_v1.json"
    if sidecar.is_file():
        sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
        sidecar_value["training_report_sha256"] = report_sha256
        sidecar.write_text(
            json.dumps(sidecar_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    addendum = output / "full_release_replay_cartesian_method_v1.json"
    if addendum.is_file():
        value = json.loads(addendum.read_text(encoding="utf-8"))
        value["candidate"] = value["candidate"].replace(
            "replay_cartesian", "planner_curve_cartesian"
        )
        value["source"] = str(THIS_SOURCE)
        value["source_sha256"] = _sha256(THIS_SOURCE)
        value["training_report_sha256"] = report_sha256
        value["factorial_cell"]["action_coverage"] = (
            "same_query_four_point_deployment_planner_curve"
        )
        value["planner_curve_support"] = value["overlay"].get(
            "planner_curve_support"
        )
        addendum.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = full._arguments(effective)
    cartesian = full.replay.base
    original = {
        "source": full.THIS_SOURCE,
        "prefix": full.CANDIDATE_PREFIX,
        "bind": full._bind_overlay,
        "validate_receipt": full.replay.validate_receipt,
        "compact": full.replay.compact_replay_support,
        "rows_per_template": cartesian.ROWS_PER_TEMPLATE,
        "rows_per_twin": cartesian.ROWS_PER_TWIN,
        "stream": cartesian.CartesianTwinBatchStream,
        "validate_grid": cartesian.validate_cartesian_grid,
    }
    receipt_path = args.cartesian_overlay.expanduser().resolve().with_suffix(
        args.cartesian_overlay.suffix + ".json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_degenerate = int(receipt["zero_action_future_gap_count"])
    expected_hidden_degenerate = int(receipt["zero_hidden_future_gap_count"])

    def validate_grid(
        pixels: torch.Tensor,
        raw_action_blocks: torch.Tensor,
        *,
        template_count: int,
    ) -> dict[str, bool]:
        return _validate_grid(
            pixels,
            raw_action_blocks,
            template_count=template_count,
            expected_degenerate_action_futures=expected_degenerate,
            expected_degenerate_hidden_futures=expected_hidden_degenerate,
        )

    full.THIS_SOURCE = THIS_SOURCE
    full.CANDIDATE_PREFIX = CANDIDATE_PREFIX
    full._bind_overlay = _bind_overlay
    cartesian.ROWS_PER_TEMPLATE = ROWS_PER_TEMPLATE
    cartesian.ROWS_PER_TWIN = ROWS_PER_TWIN
    cartesian.CartesianTwinBatchStream = PlannerCurveTwinBatchStream
    cartesian.validate_cartesian_grid = validate_grid
    try:
        status = full.main(effective)
        if status == 0 and "--dry-run" not in effective:
            _rewrite_multiaction_receipts(args.output.expanduser().resolve())
        return status
    finally:
        full.THIS_SOURCE = original["source"]
        full.CANDIDATE_PREFIX = original["prefix"]
        full._bind_overlay = original["bind"]
        full.replay.validate_receipt = original["validate_receipt"]
        full.replay.compact_replay_support = original["compact"]
        cartesian.ROWS_PER_TEMPLATE = original["rows_per_template"]
        cartesian.ROWS_PER_TWIN = original["rows_per_twin"]
        cartesian.CartesianTwinBatchStream = original["stream"]
        cartesian.validate_cartesian_grid = original["validate_grid"]


if __name__ == "__main__":
    raise SystemExit(main())
