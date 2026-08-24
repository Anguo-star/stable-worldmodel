#!/usr/bin/env python3
"""Train the empirical-replay-support Cartesian Motion falsification.

Relative to the completed contact-support legacy-scale candidate, only the real
counterfactual action branch changes: the query action is one verbatim five-step
block replayed from the original PushT expert action column, so the alternate
action marginal is the *empirical replay marginal* of that dataset and
explicitly not the planner/CEM proposal distribution.  Source checkpoint,
templates, sampler, objective, optimizer, budget, model and inference are
unchanged.
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


CANDIDATE = "pusht_motion_damping_replay_cartesian_action_pair_v1"

# Frozen identities of the completed 2,048-template empirical-replay overlay.
UNRESOLVED_SHA_SENTINEL = "FILL_AFTER_FULL_BUILD"
OVERLAY_SHA256 = (
    "f991f81ba19a84350dee7df543ff6093a96f13f6227c1b1a1f135a15fbbfd79f"
)
OVERLAY_RECEIPT_SHA256 = (
    "a87627e910b24d7821ab3d2353f0b38971269497be7bdc8838eccfa5c51f9f7b"
)

OVERLAY_TEMPLATE_COUNT = 2048
OVERLAY_CONDITION_PAIR_COUNT = 4096
SELECTED_BLOCK_COUNT = 1024
REPLAY_SAMPLING_SEED = 20260824
REPLAY_BLOCK_LENGTH = 5
ANGULAR_SECTOR_COUNT = 8
ACTION_BRANCHES = ["observed_zero", "empirical_replay_5step"]
ACTION_SOURCE = (
    "empirical_replay_marginal_of_original_pusht_expert_train_action_column"
)
NOT_A_PLANNER_NOTE = (
    "verbatim dataset action blocks; this is the empirical replay marginal, "
    "explicitly not the exact planner/CEM proposal distribution"
)
PHYSICAL_COVARIATE_NOTE = (
    "contact steps, playfield bounds and hidden-by-action interaction norms "
    "are reported outcomes of applying the unconditional replay marginal to "
    "new query states; they are covariates, never selection gates"
)

REQUIRED_HARD_CHECKS = (
    "all_action_components_finite_and_legal",
    "all_eight_angular_sectors_nonempty",
    "exact_history_and_query_prefix_equality",
    "forward_reverse_twin_reuse_exact",
    "selection_without_replacement_exact",
)
REPLAY_IDENTITY_FIELDS = (
    "builder",
    "builder_sha256",
    "pixels_sha256",
    "raw_action_blocks_sha256",
    "source_manifest_sha256",
    "template_ids_sha256",
)
REPORTED_COVARIATE_FIELDS = (
    "all_replay_model_bounds_inside_playfield",
    "interaction_definition",
    "maximum_replay_query_contact_steps",
    "mean_replay_query_contact_steps",
    "median_hidden_action_interaction_norm",
    "minimum_hidden_action_interaction_norm",
    "minimum_replay_query_contact_steps",
    "replay_model_bounds_inside_count",
    "replay_model_bounds_inside_fraction",
    "replay_model_bounds_total_count",
)
COMPACT_RECEIPT_FIELDS = (
    "action_component_maximum",
    "action_component_minimum",
    "action_distribution_note",
    "action_rule",
    "action_source",
    "all_replay_action_components_finite_and_legal",
    "angular_sector_counts",
    "angular_sector_definition",
    "circular_resultant_length",
    "eligible_block_count",
    "every_replay_block_used_by_exactly_two_twin_templates",
    "fraction_turns_above_quarter_pi",
    "maximum_history_or_query_pixel_difference_across_actions",
    "mean_absolute_wrapped_turn_angle",
    "per_sequence_action_rms_quantiles",
    "per_step_action_norm_quantiles",
    "physical_outcomes_used_for_selection",
    "replay_action_column_sha256",
    "replay_action_column_shape",
    "replay_block_alignment",
    "replay_block_assignment_count",
    "replay_block_length",
    "replay_h5",
    "replay_h5_bytes",
    "replay_sampling_seed",
    "selected_block_count",
    "selected_block_start_indices_sha256",
    "selected_unique_block_count",
    "teacher_free",
    "turn_sample_count",
    "zero_norm_step_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_resolved_identity() -> None:
    """Refuse to do anything while either identity digest is a sentinel."""

    unresolved = sorted(
        name
        for name, value in (
            ("OVERLAY_SHA256", OVERLAY_SHA256),
            ("OVERLAY_RECEIPT_SHA256", OVERLAY_RECEIPT_SHA256),
        )
        if value == UNRESOLVED_SHA_SENTINEL
    )
    if unresolved:
        raise RuntimeError(
            "replay Cartesian overlay identity is unresolved: "
            f"{unresolved} still hold {UNRESOLVED_SHA_SENTINEL!r}; fill both "
            "constants from the completed full overlay build before any "
            "dry-run or training"
        )


def _int_field(receipt: dict[str, Any], key: str) -> int:
    try:
        return int(receipt[key])
    except (KeyError, TypeError, ValueError):
        return -1


def validate_receipt(receipt: dict[str, Any]) -> dict[str, bool]:
    """Data-identity contract for the empirical-replay overlay receipt.

    Contact steps, playfield bounds and interaction norms are required to be
    *present* so the report can quote them, but they never gate the run: the
    replay marginal is unconditional by construction.
    """

    hard_checks = receipt.get("replay_support_hard_checks")
    hard_checks = hard_checks if isinstance(hard_checks, dict) else {}
    sectors = receipt.get("angular_sector_counts")
    sectors = list(sectors) if isinstance(sectors, (list, tuple)) else []
    selected = _int_field(receipt, "selected_block_count")
    checks = {
        "overlay_sha_exact": receipt.get("overlay_sha256") == OVERLAY_SHA256,
        "template_count_exact": (
            _int_field(receipt, "template_count") == OVERLAY_TEMPLATE_COUNT
        ),
        "condition_pair_count_exact": (
            _int_field(receipt, "condition_pair_count")
            == OVERLAY_CONDITION_PAIR_COUNT
        ),
        "action_branches_exact": receipt.get("action_branches") == ACTION_BRANCHES,
        "action_source_is_empirical_replay_marginal": (
            receipt.get("action_source") == ACTION_SOURCE
        ),
        "action_distribution_disclaims_planner": (
            receipt.get("action_distribution_note") == NOT_A_PLANNER_NOTE
        ),
        "teacher_free": receipt.get("teacher_free") is True,
        "sampling_seed_exact": (
            _int_field(receipt, "replay_sampling_seed") == REPLAY_SAMPLING_SEED
        ),
        "block_length_exact": (
            _int_field(receipt, "replay_block_length") == REPLAY_BLOCK_LENGTH
        ),
        "eligible_exceeds_selected": (
            _int_field(receipt, "eligible_block_count") > selected > 0
        ),
        "selected_unique_count_exact": (
            _int_field(receipt, "selected_unique_block_count")
            == SELECTED_BLOCK_COUNT
            and selected == SELECTED_BLOCK_COUNT
        ),
        "forward_reverse_twin_reuse_exact": (
            receipt.get("every_replay_block_used_by_exactly_two_twin_templates")
            is True
            and _int_field(receipt, "replay_block_assignment_count")
            == OVERLAY_TEMPLATE_COUNT
        ),
        "prefix_pixel_difference_zero": (
            _int_field(
                receipt,
                "maximum_history_or_query_pixel_difference_across_actions",
            )
            == 0
        ),
        "action_components_finite_and_legal": (
            receipt.get("all_replay_action_components_finite_and_legal") is True
        ),
        "eight_angular_sectors_nonempty": (
            len(sectors) == ANGULAR_SECTOR_COUNT
            and all(int(count) > 0 for count in sectors)
        ),
        "physical_outcomes_not_used_for_selection": (
            receipt.get("physical_outcomes_used_for_selection") is False
        ),
        "all_replay_support_hard_checks": (
            tuple(sorted(hard_checks)) == REQUIRED_HARD_CHECKS
            and all(bool(value) for value in hard_checks.values())
        ),
        "compact_receipt_fields_present": all(
            key in receipt
            for group in (
                COMPACT_RECEIPT_FIELDS,
                REPLAY_IDENTITY_FIELDS,
                REPORTED_COVARIATE_FIELDS,
            )
            for key in group
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"replay Cartesian receipt contract failed: {failed}")
    return checks


def compact_replay_support(
    receipt_path: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    support: dict[str, Any] = {
        "receipt": str(receipt_path),
        "receipt_sha256": OVERLAY_RECEIPT_SHA256,
        "hard_checks": dict(receipt["replay_support_hard_checks"]),
        "identity": {key: receipt[key] for key in REPLAY_IDENTITY_FIELDS},
        "reported_physical_covariates": {
            "note": PHYSICAL_COVARIATE_NOTE,
            **{key: receipt[key] for key in REPORTED_COVARIATE_FIELDS},
        },
    }
    support.update({key: receipt[key] for key in COMPACT_RECEIPT_FIELDS})
    return support


def _overlay_receipt(argv: Sequence[str]) -> tuple[Path, dict[str, Any]]:
    _require_resolved_identity()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cartesian-overlay", type=Path, required=True)
    args, _ = parser.parse_known_args(list(argv))
    overlay = args.cartesian_overlay.expanduser().resolve()
    receipt_path = overlay.with_suffix(overlay.suffix + ".json")
    if not receipt_path.is_file() or _sha256(receipt_path) != OVERLAY_RECEIPT_SHA256:
        raise RuntimeError("replay Cartesian overlay receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_receipt(receipt)
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
        contract["replay_support"] = compact_replay_support(receipt_path, receipt)
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
    # The base runner keeps writing its historical sidecar name
    # ``motion_cartesian_action_pair_v1.json``; because it reads CANDIDATE /
    # THIS_SOURCE / OVERLAY_SHA256 as module globals at write time, that one
    # file already carries this candidate's provenance.  Emitting a second
    # replay-named sidecar would duplicate a training output, so we do not.
    return base.main(effective)


if __name__ == "__main__":
    raise SystemExit(main())
