#!/usr/bin/env python3
"""One-factor Contact test of paired absolute-future calibration.

This arm is identical to the center-free visible-joint Contact candidate except
that its training-only paired auxiliary also fits the normalized common center.
It reuses the already defined canonical exact-future objective.  Data, model,
initialization, optimizer, seed, budget, auxiliary weight, parameter count, and
inference path are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    canonical_margin_exact_future_v1 as exact,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_contact_friction_visible_joint_absolute_single_stage_v1 as parent,
)


CANDIDATE = "pusht_contact_friction_visible_joint_exact_future_single_stage_v1"
PARENT_SIDECAR = "contact_visible_joint_absolute_single_stage_method_v1.json"
SIDECAR = "contact_visible_joint_exact_future_single_stage_method_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_future_auxiliary(
    prediction: Any,
    target: Any,
    groups: Any,
) -> dict[str, Any]:
    result = exact.canonical_margin_exact_future(prediction, target, groups)
    return {
        **result,
        "response_loss": result["ccrm_normalized_error_by_group"].mean(),
        "direct_common_center_mse_included": True,
    }


def _rename_components(components: dict[str, Any]) -> dict[str, Any]:
    updated = dict(components)
    updated["canonical_exact_future_loss"] = updated.pop(
        "canonical_response_only_loss"
    )
    updated["included_direct_common_center_mse"] = updated.pop(
        "excluded_direct_common_center_mse"
    )
    return updated


def _rewrite_exact_receipts(output: Path) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    parent_contract = result.pop(
        "contact_visible_joint_absolute_single_stage_contract"
    )
    parent_contract["objective"] = (
        "native_mse + 0.09*SIGReg + "
        "0.09*(pair_normalized_exact_future + canonical_assignment_0p5)"
    )
    parent_contract["first_loss_components"] = _rename_components(
        parent_contract["first_loss_components"]
    )
    parent_contract["last_loss_components"] = _rename_components(
        parent_contract["last_loss_components"]
    )
    parent_contract.update(
        {
            "one_factor_parent": parent.CANDIDATE,
            "one_factor_change": (
                "include normalized paired common-center MSE already present "
                "in canonical exact-future decomposition"
            ),
            "direct_normalized_common_center_mse_included": True,
        }
    )
    result["contact_visible_joint_exact_future_single_stage_contract"] = (
        parent_contract
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
            "one_factor_parent": parent.CANDIDATE,
            "direct_normalized_common_center_mse_included": True,
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
    parsed = parent._args(effective)
    original = parent.joint.objective.canonical_response_only
    parent.joint.objective.canonical_response_only = _exact_future_auxiliary
    try:
        status = parent.main(effective)
    finally:
        parent.joint.objective.canonical_response_only = original
    if status == 0 and not parsed.dry_run:
        _rewrite_exact_receipts(parsed.output.expanduser().resolve())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
