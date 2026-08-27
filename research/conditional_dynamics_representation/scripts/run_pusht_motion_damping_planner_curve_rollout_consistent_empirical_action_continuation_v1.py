#!/usr/bin/env python3
"""Motion RC-COJA with empirical continuation actions and no model change."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_planner_curve_rollout_consistent_continuation_v1
    as base,
)


EXPECTED_STATUS = "completed_motion_rollout2_empirical_action_targets"
SECOND_ACTION = "empirical_original_pusht_contiguous_block"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contract(
    path: Path, *, source_overlay: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    receipt_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("Motion empirical rollout2 targets are missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_overlay = source_overlay.expanduser().resolve()
    checks = {
        "targets_self_hash": receipt.get("rollout2_targets_sha256")
        == _sha256(path),
        "source_overlay_exact": receipt.get("source_overlay_sha256")
        == _sha256(source_overlay),
        "status_exact": receipt.get("status") == EXPECTED_STATUS,
        "resolution_exact": int(receipt.get("resolution", -1)) == 224,
        "template_count_positive_even": int(
            receipt.get("template_count", -1)
        )
        > 0
        and int(receipt["template_count"]) % 2 == 0,
        "row_count_exact": int(receipt.get("model_row_count", -1))
        == 8 * int(receipt.get("template_count", -1)),
        "pair_count_exact": int(receipt.get("condition_pair_count", -1))
        == 4 * int(receipt.get("template_count", -1)),
        "row_order_exact": receipt.get("row_order")
        == "template_then_action_branch_then_damping_mode",
        "actions_exact": receipt.get("action_branches")
        == base.planner.ACTION_BRANCHES,
        "second_action_empirical": receipt.get("second_action_block")
        == SECOND_ACTION,
        "same_action_across_branches": receipt.get(
            "same_second_action_across_first_action_branches"
        )
        is True,
        "same_action_across_hidden_conditions": receipt.get(
            "same_second_action_across_hidden_conditions"
        )
        is True,
        "selection_free": receipt.get("future_outcome_used_for_selection")
        is False
        and receipt.get("contact_used_for_selection") is False
        and receipt.get("model_or_planner_output_used") is False,
        "model_boundary_label_free": receipt.get(
            "hidden_label_at_model_or_loss_boundary"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Motion empirical target contract failed: {checks}")
    return (
        {
            "path": str(path),
            "sha256": _sha256(path),
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "template_count": int(receipt["template_count"]),
            "row_count": int(receipt["model_row_count"]),
            "condition_pair_count": int(receipt["condition_pair_count"]),
            "zero_hidden_future2_gap_count": int(
                receipt["zero_hidden_future2_gap_count"]
            ),
            "second_action_block": SECOND_ACTION,
            "checks": checks,
        },
        receipt,
    )


def _install_data(
    trainer: Any,
    *,
    native_install: Callable[..., dict[str, Any]],
    overlay: Path,
    rollout2: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    installed = native_install(trainer, overlay=overlay)
    native_split = trainer.trainer._training_split

    def training_split(*args: Any, **kwargs: Any):
        split = native_split(*args, **kwargs)
        payload = torch.load(
            Path(rollout2["path"]), map_location="cpu", weights_only=True
        )
        future2 = payload.get("future2_pixels")
        raw_second = payload.get("second_action_blocks_raw")
        row_count = int(rollout2["row_count"])
        template_count = int(rollout2["template_count"])
        checks = {
            "payload_template_count": int(payload.get("template_count", -1))
            == template_count,
            "payload_pair_count": int(
                payload.get("condition_pair_count", -1)
            )
            == int(rollout2["condition_pair_count"]),
            "payload_row_order": payload.get("row_order")
            == "template_then_action_branch_then_damping_mode",
            "payload_action_branches": list(
                payload.get("action_branches", ())
            )
            == list(base.planner.ACTION_BRANCHES),
            "payload_second_action": payload.get("second_action_block")
            == SECOND_ACTION,
            "future2_shape": isinstance(future2, torch.Tensor)
            and future2.dtype == torch.uint8
            and tuple(future2.shape) == (row_count, 3, 224, 224),
            "second_action_shape": isinstance(raw_second, torch.Tensor)
            and tuple(raw_second.shape) == (template_count, 5, 2),
            "source_prefix_available": split.pixels.size(0) >= row_count,
            "source_pixels_shape": tuple(split.pixels.shape[1:])
            == (4, 3, 224, 224),
            "source_actions_shape": tuple(split.action.shape[1:]) == (4, 10),
            "action_stats_available": kwargs.get("action_stats") is not None,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"Motion empirical materialization failed: {checks}"
            )
        normalized_second = (
            trainer.trainer.mixed.pilot.normalize_action_blocks(
                raw_second.float().reshape(template_count, 1, 10),
                kwargs["action_stats"],
            )
            .repeat_interleave(8, dim=0)[:, 0]
        )
        actions4 = split.action[:row_count].clone()
        actions4[:, 3] = normalized_second
        state["pixels4"] = split.pixels[:row_count]
        state["actions4"] = actions4
        state["future2"] = future2
        state["stream"] = iter(
            base.planner.PlannerCurveTwinBatchStream(
                int(rollout2["condition_pair_count"]),
                batch_size=base.RC_BATCH_SIZE,
                seed=base.RC_SEED,
            )
        )
        state["materialization_checks"] = checks
        return split

    trainer.trainer._training_split = training_split
    return installed


_NATIVE_REWRITE: Callable[..., None] = base._rewrite_report


def _rewrite_report(
    output: Path,
    *,
    args: Any,
    rollout2: dict[str, Any],
    rollout_state: dict[str, Any],
) -> None:
    _NATIVE_REWRITE(
        output,
        args=args,
        rollout2=rollout2,
        rollout_state=rollout_state,
    )
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    contract = payload["result"]["rollout_consistent_coja_contract"]
    contract["checks"]["second_action_empirical"] = True
    contract["objective"]["second_action_block"] = SECOND_ACTION
    contract["rollout2_targets"] = rollout2
    payload["provenance"]["method"]["candidate"] = (
        "pusht_motion_damping_planner_curve_rollout_consistent_"
        "empirical_action_rho0.25_continuation_v1"
    )
    payload["provenance"]["method"]["source"] = str(THIS_SOURCE)
    payload["provenance"]["method"]["source_sha256"] = _sha256(THIS_SOURCE)
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = output / "rollout_consistent_coja_v1.json"
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["candidate"] = payload["provenance"]["method"]["candidate"]
    sidecar_payload["second_action_block"] = SECOND_ACTION
    sidecar_payload["source"] = str(THIS_SOURCE)
    sidecar_payload["source_sha256"] = _sha256(THIS_SOURCE)
    sidecar_payload["training_report_sha256"] = _sha256(report)
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    original = {
        "source": base.THIS_SOURCE,
        "load": base._load_rollout2_contract,
        "data": base._install_rollout_data,
        "rewrite": base._rewrite_report,
    }
    base.THIS_SOURCE = THIS_SOURCE
    base._load_rollout2_contract = _load_contract
    base._install_rollout_data = _install_data
    base._rewrite_report = _rewrite_report
    try:
        return base.main(argv)
    finally:
        base.THIS_SOURCE = original["source"]
        base._load_rollout2_contract = original["load"]
        base._install_rollout_data = original["data"]
        base._rewrite_report = original["rewrite"]


if __name__ == "__main__":
    raise SystemExit(main())
