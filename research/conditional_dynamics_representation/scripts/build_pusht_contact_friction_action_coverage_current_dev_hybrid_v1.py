#!/usr/bin/env python3
"""Bind existing Contact action-coverage train to current frozen Development."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
ACTION_ROOT = CONTEXTWORLD_ROOT / (
    "artifacts/synthesis/pusht_contact_friction_h3_action_coverage_v2"
)
CURRENT_ROOT = CONTEXTWORLD_ROOT / (
    "artifacts/synthesis/pusht_contact_friction_h3_release_v3"
)
ACTION_MANIFEST_SHA256 = (
    "f7c7f7408ca9a9dcf43da782c6f21fbf7b1d81d6a6f75e41fe31d5b997505861"
)
CURRENT_MANIFEST_SHA256 = (
    "cbb9b1a1c030a3c66ea8acbf25c5e1a302f1c43907beeadcdc9d8bd1e989f3d5"
)
PROTOCOL = (
    "pusht_contact_friction_action_coverage_v2_"
    "current_development_hybrid_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path) -> dict:
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    action_manifest_path = ACTION_ROOT / "manifest.json"
    current_manifest_path = CURRENT_ROOT / "manifest.json"
    if (
        _sha256(action_manifest_path) != ACTION_MANIFEST_SHA256
        or _sha256(current_manifest_path) != CURRENT_MANIFEST_SHA256
    ):
        raise RuntimeError("Contact source manifest identity changed")
    action = json.loads(action_manifest_path.read_text(encoding="utf-8"))
    current = json.loads(current_manifest_path.read_text(encoding="utf-8"))

    manifest = copy.deepcopy(action)
    manifest["protocol"] = PROTOCOL
    manifest["pair_counts"]["loader_validation"] = current["pair_counts"][
        "loader_validation"
    ]
    manifest["pair_counts"]["validation"] = current["pair_counts"][
        "validation"
    ]
    for split in ("loader_validation", "validation"):
        manifest["splits"][split] = copy.deepcopy(current["splits"][split])
    loader = manifest["splits"]["loader_validation"]
    loader["frozen_split_reuse"] = {
        "passed": True,
        "pair_identity_preserved": True,
        "model_visible_bytes_preserved": True,
        "source_manifest_sha256": CURRENT_MANIFEST_SHA256,
        "source_table_sha256": loader["table_sha256"],
        "destination_table_sha256": loader["table_sha256"],
    }
    manifest["hybrid_sources"] = {
        "train": {
            "root": str(ACTION_ROOT),
            "manifest_sha256": ACTION_MANIFEST_SHA256,
            "table_sha256": manifest["splits"]["train"]["table_sha256"],
        },
        "loader_validation_and_validation": {
            "root": str(CURRENT_ROOT),
            "manifest_sha256": CURRENT_MANIFEST_SHA256,
            "loader_validation_table_sha256": loader["table_sha256"],
        },
    }

    output.mkdir(parents=True)
    links = {
        "train.lance": ACTION_ROOT / "train.lance",
        "loader_validation.lance": CURRENT_ROOT / "loader_validation.lance",
        "validation.lance": CURRENT_ROOT / "validation.lance",
    }
    for name, target in links.items():
        os.symlink(target, output / name, target_is_directory=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checks = {
        "action_train_manifest_exact": True,
        "current_development_manifest_exact": True,
        "train_pair_count_2048": manifest["pair_counts"]["train"] == 2048,
        "development_pair_count_256": (
            manifest["pair_counts"]["loader_validation"] == 256
        ),
        "current_development_table_exact": (
            loader["table_sha256"]
            == current["splits"]["loader_validation"]["table_sha256"]
        ),
        "all_table_links_resolve": all(
            (output / name).resolve().exists() for name in links
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Contact hybrid build failed: {checks}")
    receipt = {
        "schema_version": 1,
        "status": "completed_action_train_current_development_hybrid",
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
        "root": str(output),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "protocol": PROTOCOL,
        "checks": checks,
        "claim_boundary": {
            "training_rows_rebuilt": False,
            "development_rows_rebuilt": False,
            "public_test_opened": False,
            "train_pair_count_differs_from_current_fixed_action_arm": True,
        },
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
