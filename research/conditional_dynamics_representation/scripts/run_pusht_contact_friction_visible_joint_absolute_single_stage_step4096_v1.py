#!/usr/bin/env python3
"""Budget-extension probe for the parameter-free Contact joint relation.

This is the exact center-free, absolute-coordinate single-stage candidate run
from the same published PushT checkpoint for 4,096 optimizer steps.  The sole
purpose is to distinguish continued learning from a 2,048-step capacity
plateau; no model, loss, data, optimizer, seed, or inference behavior changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_contact_friction_visible_joint_absolute_single_stage_v1 as parent,
)


CANDIDATE = (
    "pusht_contact_friction_visible_joint_absolute_single_stage_step4096_v1"
)
OPTIMIZER_STEPS = 4096
PARENT_SIDECAR = "contact_visible_joint_absolute_single_stage_method_v1.json"
SIDECAR = (
    "contact_visible_joint_absolute_single_stage_step4096_method_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_budget_receipts(output: Path) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    contract = payload["result"][
        "contact_visible_joint_absolute_single_stage_contract"
    ]
    contract.update(
        {
            "one_factor_parent": (
                "pusht_contact_friction_visible_joint_absolute_single_stage_v1"
            ),
            "one_factor_change": "optimizer steps 2048 -> 4096",
            "budget_extension_only": True,
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
            "budget_extension_only": True,
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
    original = {
        "candidate": parent.CANDIDATE,
        "steps": parent.OPTIMIZER_STEPS,
    }
    parent.CANDIDATE = CANDIDATE
    parent.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    try:
        parsed = parent._args(effective)
        status = parent.main(effective)
    finally:
        parent.CANDIDATE = original["candidate"]
        parent.OPTIMIZER_STEPS = original["steps"]
    if status == 0 and not parsed.dry_run:
        _rewrite_budget_receipts(parsed.output.expanduser().resolve())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
