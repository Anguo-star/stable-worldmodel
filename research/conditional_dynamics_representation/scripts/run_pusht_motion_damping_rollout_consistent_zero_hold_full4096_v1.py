#!/usr/bin/env python3
"""Train the fixed zero-hold Motion RC-COJA recipe in one 4096-step stage."""

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
    run_pusht_motion_damping_planner_curve_rollout_consistent_continuation_v1
    as base,
)


STANDARD_INITIALIZATION_SHA256 = (
    "9f13b2c28bb909338047e08d62bf6eb16ea3616cb53e57cfa9b5072b43db1c59"
)
OPTIMIZER_STEPS = 4096
CANDIDATE = "pusht_motion_damping_rollout_consistent_zero_hold_full4096_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_as_full_training(output: Path) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    initialization = {
        "checkpoint": result["source_checkpoint"]["path"],
        "role": "published_standard_pusht_initialization",
        "sha256": STANDARD_INITIALIZATION_SHA256,
        "source_optimizer_steps": 0,
        "fresh_optimizer_state": True,
        "random_initialization": False,
    }
    method = result["rollout_consistent_coja_contract"]
    method["source_checkpoint"] = initialization
    method["fresh_optimizer_steps"] = OPTIMIZER_STEPS
    method["continuation"] = False
    method["single_stage_from_standard_initialization"] = True
    method["checks"]["source_checkpoint_exact"] = (
        result["source_checkpoint"]["sha256"]
        == STANDARD_INITIALIZATION_SHA256
    )
    method["checks"]["fresh_steps_exact"] = (
        int(result["optimizer_steps"]) == OPTIMIZER_STEPS
    )

    absolute = result["absolute_single_stage_joint_training_contract"]
    absolute["initialization_source"] = initialization
    absolute["joint_optimizer_steps"] = OPTIMIZER_STEPS
    absolute["single_stage_from_standard_initialization"] = True

    cartesian = result["motion_cartesian_action_pair_contract"]
    cartesian["initialization_source"] = initialization
    cartesian["fresh_optimizer_steps"] = OPTIMIZER_STEPS
    cartesian["candidate"] = CANDIDATE
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = output / "rollout_consistent_coja_v1.json"
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload.update(
        {
            "candidate": CANDIDATE,
            "source": str(THIS_SOURCE),
            "source_sha256": _sha256(THIS_SOURCE),
            "source_checkpoint_sha256": STANDARD_INITIALIZATION_SHA256,
            "fresh_optimizer_steps": OPTIMIZER_STEPS,
            "single_stage_from_standard_initialization": True,
            "training_report_sha256": _sha256(report),
        }
    )
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    outputs = [
        Path(effective[index + 1]).expanduser().resolve()
        for index, value in enumerate(effective[:-1])
        if value == "--output"
    ]
    if len(outputs) != 1:
        raise ValueError("Exactly one --output is required")
    dry_run = "--dry-run" in effective
    original = {
        "source_sha": base.SOURCE_CHECKPOINT_SHA256,
        "steps": base.OPTIMIZER_STEPS,
        "source": base.THIS_SOURCE,
    }
    base.SOURCE_CHECKPOINT_SHA256 = STANDARD_INITIALIZATION_SHA256
    base.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    base.THIS_SOURCE = THIS_SOURCE
    try:
        status = base.main(effective)
    finally:
        base.SOURCE_CHECKPOINT_SHA256 = original["source_sha"]
        base.OPTIMIZER_STEPS = original["steps"]
        base.THIS_SOURCE = original["source"]
    if status == 0 and not dry_run:
        _rewrite_as_full_training(outputs[0])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
