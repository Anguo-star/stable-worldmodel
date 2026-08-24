#!/usr/bin/env python3
"""Run the fixed visible-joint recipe on existing Contact action coverage.

This changes only the hidden training population to ContextWorld's already
built nine-scale Contact action-coverage v2 asset.  Model, initialization,
absolute coordinates, objective, seed and 2,048-step budget are inherited
unchanged from the fixed-action single-stage arm.  The asset has 2,048 rather
than 8,192 training pairs, so a negative result is not a clean action-support
falsification; a positive transfer despite that reduction is informative.
"""

from __future__ import annotations

import argparse
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
    build_pusht_contact_friction_action_coverage_current_dev_hybrid_v1
    as hybrid,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_contact_friction_visible_joint_absolute_single_stage_v1 as base,
)


CANDIDATE = "pusht_contact_friction_visible_joint_action_coverage_v1"
TRAIN_PAIR_COUNT = 2048
ACTION_HASH_COUNT = 2048
QUERY_ACTION_SCALE_GRID = [
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.375,
    1.5,
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_contract(argv: Sequence[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root", type=Path, required=True)
    args, _ = parser.parse_known_args(list(argv))
    root = args.data_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    receipt_path = root / "receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise RuntimeError("Contact hybrid data root is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    train = manifest["splits"]["train"]
    checks = {
        "protocol_exact": (
            manifest["protocol"]
            == hybrid.PROTOCOL
        ),
        "hybrid_receipt_manifest_exact": (
            receipt["manifest_sha256"] == _sha256(manifest_path)
            and all(receipt["checks"].values())
        ),
        "action_train_source_exact": (
            manifest["hybrid_sources"]["train"]["manifest_sha256"]
            == hybrid.ACTION_MANIFEST_SHA256
        ),
        "current_development_source_exact": (
            manifest["hybrid_sources"]
            ["loader_validation_and_validation"]["manifest_sha256"]
            == hybrid.CURRENT_MANIFEST_SHA256
        ),
        "train_pair_count_exact": (
            int(manifest["pair_counts"]["train"]) == TRAIN_PAIR_COUNT
        ),
        "action_hash_count_exact": (
            int(train["action_hash_count"]) == ACTION_HASH_COUNT
        ),
        "scale_grid_exact": (
            list(train["query_action_scale_grid"])
            == QUERY_ACTION_SCALE_GRID
        ),
        "builder_checks_passed": train["passed"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Contact action coverage contract failed: {checks}")
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "train_pair_count": TRAIN_PAIR_COUNT,
        "action_hash_count": ACTION_HASH_COUNT,
        "query_action_scale_grid": QUERY_ACTION_SCALE_GRID,
        "checks": checks,
        "comparison_boundary": (
            "action coverage v2 has 2048 train pairs whereas the current "
            "fixed-action v3 population has 8192; negative effects are "
            "not attributable to action coverage alone"
        ),
    }


def _rewrite(output: Path, *, data: dict[str, Any]) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    contract = payload["result"][
        "contact_visible_joint_absolute_single_stage_contract"
    ]
    contract["training_data"] = data
    contract["checks"]["action_coverage_data_contract"] = all(
        data["checks"].values()
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
    sidecar = output / (
        "contact_visible_joint_absolute_single_stage_method_v1.json"
    )
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata.update(
        {
            "candidate": CANDIDATE,
            "source": str(THIS_SOURCE),
            "source_sha256": _sha256(THIS_SOURCE),
            "training_report_sha256": _sha256(report),
            "training_data": data,
        }
    )
    sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    data = _data_contract(effective)
    original = {"candidate": base.CANDIDATE, "source": base.THIS_SOURCE}
    base.CANDIDATE = CANDIDATE
    base.THIS_SOURCE = THIS_SOURCE
    try:
        status = base.main(effective)
        if "--dry-run" not in effective:
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--output", type=Path, required=True)
            args, _ = parser.parse_known_args(effective)
            _rewrite(args.output.expanduser().resolve(), data=data)
        return status
    finally:
        base.CANDIDATE = original["candidate"]
        base.THIS_SOURCE = original["source"]


if __name__ == "__main__":
    raise SystemExit(main())
