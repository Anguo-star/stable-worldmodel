#!/usr/bin/env python3
"""Train the single contact-support Cartesian Motion falsification.

Relative to the completed zero-contact legacy-scale candidate, only the real
counterfactual action branch changes: it points from the query agent toward
the block at frozen amplitude 0.45 and contacts the block in every simulator
rollout.  Source, templates, sampler, objective, optimizer, budget, model and
inference are unchanged.
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
    run_pusht_motion_damping_cartesian_action_pair_v1 as base,
)


CANDIDATE = "pusht_motion_damping_contact_cartesian_action_pair_v1"
OVERLAY_SHA256 = "73df58e521a28b467ffb731be23c1cb2e7623c6620c66e75224d31f6e80e833c"
OVERLAY_RECEIPT_SHA256 = "8bc3f3f471cadcb6dedb6b3344c2709e66e3cb4a3832d88007b195da90df8b39"
OVERLAY_TEMPLATE_COUNT = 2048
OVERLAY_CONDITION_PAIR_COUNT = 4096
ACTION_BRANCHES = ["observed_zero", "toward_block_0p45"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overlay_receipt(argv: Sequence[str]) -> tuple[Path, dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cartesian-overlay", type=Path, required=True)
    args, _ = parser.parse_known_args(list(argv))
    overlay = args.cartesian_overlay.expanduser().resolve()
    receipt_path = overlay.with_suffix(overlay.suffix + ".json")
    if (
        not receipt_path.is_file()
        or _sha256(receipt_path) != OVERLAY_RECEIPT_SHA256
    ):
        raise RuntimeError("contact Cartesian overlay receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = {
        "overlay_sha_exact": receipt.get("overlay_sha256") == OVERLAY_SHA256,
        "template_count_exact": (
            int(receipt.get("template_count", -1)) == OVERLAY_TEMPLATE_COUNT
        ),
        "action_branches_exact": receipt.get("action_branches") == ACTION_BRANCHES,
        "amplitude_exact": float(receipt.get("action_amplitude", -1.0)) == 0.45,
        "contact_hard_checks_pass": all(
            receipt.get("contact_support_hard_checks", {}).values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"contact Cartesian receipt contract failed: {checks}")
    return receipt_path, receipt


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    receipt_path, receipt = _overlay_receipt(effective)
    native_rewrite = base._rewrite_report

    def rewrite_report(
        output: Path,
        *,
        freeze_state: dict[str, Any],
        cartesian_state: dict[str, Any],
        overlay: Path,
    ) -> Path:
        report = native_rewrite(
            output,
            freeze_state=freeze_state,
            cartesian_state=cartesian_state,
            overlay=overlay,
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        contract = payload["result"][
            "motion_cartesian_action_pair_contract"
        ]["cartesian_training_overlay"]
        contract["action_branches"] = ACTION_BRANCHES
        contract["contact_support"] = {
            "receipt": str(receipt_path),
            "receipt_sha256": OVERLAY_RECEIPT_SHA256,
            "action_rule": receipt["action_rule"],
            "action_amplitude": receipt["action_amplitude"],
            "minimum_alternate_query_contact_steps": receipt[
                "minimum_alternate_query_contact_steps"
            ],
            "mean_alternate_query_contact_steps": receipt[
                "mean_alternate_query_contact_steps"
            ],
            "all_alternate_model_bounds_inside_playfield": receipt[
                "all_alternate_model_bounds_inside_playfield"
            ],
            "minimum_hidden_action_interaction_norm": receipt[
                "minimum_hidden_action_interaction_norm"
            ],
            "median_hidden_action_interaction_norm": receipt[
                "median_hidden_action_interaction_norm"
            ],
            "hard_checks": receipt["contact_support_hard_checks"],
        }
        report.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

    base.THIS_SOURCE = THIS_SOURCE
    base.CANDIDATE = CANDIDATE
    base.OVERLAY_SHA256 = OVERLAY_SHA256
    base.OVERLAY_TEMPLATE_COUNT = OVERLAY_TEMPLATE_COUNT
    base.OVERLAY_CONDITION_PAIR_COUNT = OVERLAY_CONDITION_PAIR_COUNT
    base._rewrite_report = rewrite_report
    return base.main(effective)


if __name__ == "__main__":
    raise SystemExit(main())
