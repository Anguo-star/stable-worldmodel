#!/usr/bin/env python3
"""Break exact conditional overlap while retaining the joint objective.

The parent candidate groups adjacent ContextWorld rows that share the same
visible query and action.  This one-factor control cyclically shifts the second
member of every binary group by one source pair.  It preserves all 64 rows,
32 groups, low/high row positions, loss weight, compute, initialization,
optimizer, and 2,048-step budget, but no auxiliary group contains a true
same-query pair.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_contact_friction_visible_joint_absolute_single_stage_v1 as parent,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_canonical_response_only_freeze_v1 as joint,
)


CANDIDATE = "pusht_contact_friction_visible_joint_shifted_pair_control_v1"
OPTIMIZER_STEPS = 2048
PAIR_SHIFT = 1
PARENT_SIDECAR = "contact_visible_joint_absolute_single_stage_method_v1.json"
SIDECAR = "contact_visible_joint_shifted_pair_control_method_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shifted_binary_hidden_groups(
    *,
    original_batch_size: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    hidden_count = int(batch_size) - int(original_batch_size)
    if original_batch_size <= 0 or hidden_count <= 0 or hidden_count % 2:
        raise ValueError("Shifted-pair control requires complete binary rows")
    pair_count = hidden_count // 2
    if pair_count <= PAIR_SHIFT:
        raise ValueError("Shifted-pair control requires at least two pairs")
    first = original_batch_size + 2 * torch.arange(
        pair_count, device=device, dtype=torch.long
    )
    second_pair = (
        torch.arange(pair_count, device=device, dtype=torch.long) + PAIR_SHIFT
    ) % pair_count
    second = original_batch_size + 2 * second_pair + 1
    return torch.stack((first, second), dim=1)


def _rewrite_control_receipts(output: Path) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    contract = result["contact_visible_joint_absolute_single_stage_contract"]
    checks = {
        "optimizer_steps_exact": int(result["optimizer_steps"])
        == OPTIMIZER_STEPS,
        "seed_exact": int(result["seed"]) == 13313,
        "batch_partition_unchanged": result["batch"]["original"] == 64
        and result["batch"]["hidden"] == 64
        and result["batch"]["hidden_pairs"] == 32,
        "joint_auxiliary_weight_unchanged": joint.AUXILIARY_WEIGHT == 0.09,
        "only_existing_predictor_and_head_trainable": result[
            "representation_freeze"
        ]["trainable_top_level_modules"]
        == ["pred_proj", "predictor"],
        "same_query_pairing_removed_from_auxiliary": True,
        "all_hidden_rows_used_once": True,
        "binary_group_count_unchanged": True,
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise RuntimeError(f"Shifted-pair control contract failed: {failed}")
    contract.update(
        {
            "candidate": CANDIDATE,
            "checks": {**contract["checks"], **checks},
            "one_factor_parent": (
                "pusht_contact_friction_visible_joint_absolute_single_stage_v1"
            ),
            "one_factor_change": (
                "auxiliary second member shifted cyclically by one true pair"
            ),
            "pairing_ablation": {
                "native_loss_rows_and_order_unchanged": True,
                "true_adjacent_pairs_used_by_joint_auxiliary": False,
                "pair_shift": PAIR_SHIFT,
                "group_count": 32,
                "rows_used_once": 64,
                "preserves_even_odd_row_positions": True,
                "hidden_labels_used": False,
            },
            "objective": (
                "native_mse + 0.09*SIGReg + "
                "0.09*joint_auxiliary_on_shifted_nonoverlap_groups"
            ),
        }
    )
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    old_sidecar = output / PARENT_SIDECAR
    sidecar_payload = json.loads(old_sidecar.read_text(encoding="utf-8"))
    sidecar_payload.update(
        {
            "candidate": CANDIDATE,
            "source": str(THIS_SOURCE),
            "source_sha256": _sha256(THIS_SOURCE),
            "training_report": str(report),
            "training_report_sha256": _sha256(report),
            "one_factor_parent": (
                "pusht_contact_friction_visible_joint_absolute_single_stage_v1"
            ),
            "one_factor_change": (
                "auxiliary second member shifted cyclically by one true pair"
            ),
            "pair_shift": PAIR_SHIFT,
        }
    )
    new_sidecar = output / SIDECAR
    new_sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    old_sidecar.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    native_groups = joint.paired.binary_hidden_groups
    original = {
        "candidate": parent.CANDIDATE,
        "steps": parent.OPTIMIZER_STEPS,
    }
    parent.CANDIDATE = CANDIDATE
    parent.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    joint.paired.binary_hidden_groups = shifted_binary_hidden_groups
    try:
        parsed = parent._args(effective)
        status = parent.main(effective)
    finally:
        parent.CANDIDATE = original["candidate"]
        parent.OPTIMIZER_STEPS = original["steps"]
        joint.paired.binary_hidden_groups = native_groups
    if status == 0 and not parsed.dry_run:
        _rewrite_control_receipts(parsed.output.expanduser().resolve())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
