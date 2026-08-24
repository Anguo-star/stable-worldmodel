from __future__ import annotations

import hashlib
import json

import pytest

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_replay_cartesian_action_pair_v1 as subject,
)


RESOLVED_OVERLAY_SHA = "a" * 64
RESOLVED_RECEIPT_SHA = "b" * 64


def _fake_receipt(**overrides):
    receipt = {
        "overlay_sha256": RESOLVED_OVERLAY_SHA,
        "template_count": 2048,
        "condition_pair_count": 4096,
        "action_branches": ["observed_zero", "empirical_replay_5step"],
        "action_source": subject.ACTION_SOURCE,
        "action_distribution_note": subject.NOT_A_PLANNER_NOTE,
        "action_rule": "verbatim_5step_action_block_from_original_h5",
        "teacher_free": True,
        "replay_sampling_seed": 20260824,
        "replay_block_length": 5,
        "replay_block_alignment": "episode_relative_multiple_of_5",
        "eligible_block_count": 7331,
        "selected_block_count": 1024,
        "selected_unique_block_count": 1024,
        "selected_block_start_indices_sha256": "c" * 64,
        "replay_block_assignment_count": 2048,
        "every_replay_block_used_by_exactly_two_twin_templates": True,
        "maximum_history_or_query_pixel_difference_across_actions": 0,
        "all_replay_action_components_finite_and_legal": True,
        "angular_sector_counts": [610, 655, 640, 601, 633, 662, 648, 671],
        "angular_sector_definition": "atan2 eight pi/4 sectors",
        "action_component_minimum": -0.98,
        "action_component_maximum": 0.99,
        "per_step_action_norm_quantiles": {"q000": 0.0, "q100": 1.3},
        "per_sequence_action_rms_quantiles": {"q000": 0.1, "q100": 1.2},
        "circular_resultant_length": 0.031,
        "mean_absolute_wrapped_turn_angle": 0.42,
        "fraction_turns_above_quarter_pi": 0.37,
        "turn_sample_count": 3901,
        "zero_norm_step_count": 12,
        "physical_outcomes_used_for_selection": False,
        "replay_h5": "/data/pusht_expert_train.h5",
        "replay_h5_bytes": 4900000000,
        "replay_action_column_sha256": "d" * 64,
        "replay_action_column_shape": [206214, 2],
        "builder": "/repo/build_replay_overlay.py",
        "builder_sha256": "e" * 64,
        "pixels_sha256": "f" * 64,
        "raw_action_blocks_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "template_ids_sha256": "3" * 64,
        "minimum_replay_query_contact_steps": 0,
        "mean_replay_query_contact_steps": 1.84,
        "maximum_replay_query_contact_steps": 5,
        "all_replay_model_bounds_inside_playfield": False,
        "replay_model_bounds_inside_count": 16101,
        "replay_model_bounds_total_count": 16384,
        "replay_model_bounds_inside_fraction": 0.9827,
        "minimum_hidden_action_interaction_norm": 0.0,
        "median_hidden_action_interaction_norm": 3.21,
        "interaction_definition": "norm(delta_a1 - delta_a0)",
        "replay_support_hard_checks": {
            "all_action_components_finite_and_legal": True,
            "all_eight_angular_sectors_nonempty": True,
            "exact_history_and_query_prefix_equality": True,
            "forward_reverse_twin_reuse_exact": True,
            "selection_without_replacement_exact": True,
        },
    }
    receipt.update(overrides)
    return receipt


@pytest.fixture
def resolved(monkeypatch):
    monkeypatch.setattr(subject, "OVERLAY_SHA256", RESOLVED_OVERLAY_SHA)
    monkeypatch.setattr(subject, "OVERLAY_RECEIPT_SHA256", RESOLVED_RECEIPT_SHA)


@pytest.fixture
def isolated_base(monkeypatch):
    """Keep ``main``'s writes to the shared base module inside one test."""

    for name in (
        "THIS_SOURCE",
        "CANDIDATE",
        "OVERLAY_SHA256",
        "OVERLAY_TEMPLATE_COUNT",
        "OVERLAY_CONDITION_PAIR_COUNT",
        "_rewrite_report",
    ):
        monkeypatch.setattr(subject.base, name, getattr(subject.base, name))
    return subject.base


# --------------------------------------------------------------------------
# frozen identity of the candidate
# --------------------------------------------------------------------------


def test_replay_candidate_matches_legacy_scale_contact_arm():
    assert subject.CANDIDATE == "pusht_motion_damping_replay_cartesian_action_pair_v1"
    assert subject.OVERLAY_TEMPLATE_COUNT == 2048
    assert subject.OVERLAY_CONDITION_PAIR_COUNT == 4096
    assert subject.SELECTED_BLOCK_COUNT == 1024
    assert subject.REPLAY_SAMPLING_SEED == 20260824
    assert subject.ACTION_BRANCHES == [
        "observed_zero",
        "empirical_replay_5step",
    ]


def test_base_module_is_the_shared_cartesian_runner():
    assert subject.base.__name__.endswith(
        "run_pusht_motion_damping_cartesian_action_pair_v1"
    )
    assert subject.base.OPTIMIZER_STEPS == 1024
    assert subject.base.ROWS_PER_TWIN == 8


# --------------------------------------------------------------------------
# sentinel refusal
# --------------------------------------------------------------------------


def test_sha_constants_are_frozen_after_the_full_build():
    assert subject.OVERLAY_SHA256 == (
        "f991f81ba19a84350dee7df543ff6093a96f13f6227c1b1a1f135a15fbbfd79f"
    )
    assert subject.OVERLAY_RECEIPT_SHA256 == (
        "a87627e910b24d7821ab3d2353f0b38971269497be7bdc8838eccfa5c51f9f7b"
    )
    assert subject.UNRESOLVED_SHA_SENTINEL == "FILL_AFTER_FULL_BUILD"


def test_require_resolved_identity_refuses_both_sentinels(monkeypatch):
    monkeypatch.setattr(
        subject, "OVERLAY_SHA256", subject.UNRESOLVED_SHA_SENTINEL
    )
    monkeypatch.setattr(
        subject, "OVERLAY_RECEIPT_SHA256", subject.UNRESOLVED_SHA_SENTINEL
    )
    with pytest.raises(RuntimeError) as error:
        subject._require_resolved_identity()
    message = str(error.value)
    assert "OVERLAY_SHA256" in message
    assert "OVERLAY_RECEIPT_SHA256" in message


@pytest.mark.parametrize(
    "resolved_name",
    ["OVERLAY_SHA256", "OVERLAY_RECEIPT_SHA256"],
)
def test_require_resolved_identity_refuses_one_remaining_sentinel(
    monkeypatch, resolved_name
):
    monkeypatch.setattr(
        subject, "OVERLAY_SHA256", subject.UNRESOLVED_SHA_SENTINEL
    )
    monkeypatch.setattr(
        subject, "OVERLAY_RECEIPT_SHA256", subject.UNRESOLVED_SHA_SENTINEL
    )
    monkeypatch.setattr(subject, resolved_name, "9" * 64)
    with pytest.raises(RuntimeError):
        subject._require_resolved_identity()


def test_require_resolved_identity_accepts_two_filled_digests(resolved):
    subject._require_resolved_identity()


def test_main_refuses_while_sentinel_remains(monkeypatch, tmp_path):
    def explode(*_args, **_kwargs):
        raise AssertionError("training must not start on a sentinel")

    monkeypatch.setattr(subject.base, "main", explode)
    monkeypatch.setattr(
        subject, "OVERLAY_SHA256", subject.UNRESOLVED_SHA_SENTINEL
    )
    monkeypatch.setattr(
        subject, "OVERLAY_RECEIPT_SHA256", subject.UNRESOLVED_SHA_SENTINEL
    )
    with pytest.raises(RuntimeError, match="unresolved"):
        subject.main(["--cartesian-overlay", str(tmp_path / "overlay.pt")])


def test_overlay_receipt_refuses_before_touching_the_filesystem(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        subject, "OVERLAY_SHA256", subject.UNRESOLVED_SHA_SENTINEL
    )
    monkeypatch.setattr(
        subject, "OVERLAY_RECEIPT_SHA256", subject.UNRESOLVED_SHA_SENTINEL
    )
    missing = tmp_path / "absent.pt"
    with pytest.raises(RuntimeError, match="unresolved"):
        subject._overlay_receipt(["--cartesian-overlay", str(missing)])


# --------------------------------------------------------------------------
# receipt identity contract
# --------------------------------------------------------------------------


def test_validate_receipt_accepts_the_full_contract(resolved):
    checks = subject.validate_receipt(_fake_receipt())
    assert all(checks.values())
    assert checks["eligible_exceeds_selected"] is True
    assert checks["selected_unique_count_exact"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"overlay_sha256": "0" * 64},
        {"template_count": 256},
        {"condition_pair_count": 512},
        {"action_branches": ["observed_zero", "toward_block_0p45"]},
        {"action_source": "cem_planner_proposal_distribution"},
        {"action_distribution_note": "sampled from the deployed planner"},
        {"teacher_free": False},
        {"replay_sampling_seed": 20260825},
        {"replay_block_length": 4},
        {"selected_unique_block_count": 1023},
        {"selected_block_count": 512, "selected_unique_block_count": 512},
        {"every_replay_block_used_by_exactly_two_twin_templates": False},
        {"replay_block_assignment_count": 1024},
        {"maximum_history_or_query_pixel_difference_across_actions": 1},
        {"all_replay_action_components_finite_and_legal": False},
        {"physical_outcomes_used_for_selection": True},
    ],
)
def test_validate_receipt_rejects_each_identity_violation(resolved, overrides):
    with pytest.raises(RuntimeError, match="receipt contract failed"):
        subject.validate_receipt(_fake_receipt(**overrides))


def test_eligible_count_must_strictly_exceed_selected(resolved):
    with pytest.raises(RuntimeError, match="eligible_exceeds_selected"):
        subject.validate_receipt(_fake_receipt(eligible_block_count=1024))


def test_all_eight_sectors_must_be_nonempty(resolved):
    with pytest.raises(RuntimeError, match="eight_angular_sectors_nonempty"):
        subject.validate_receipt(
            _fake_receipt(angular_sector_counts=[610, 655, 640, 0, 633, 662, 648, 671])
        )


def test_sector_list_must_have_eight_entries(resolved):
    with pytest.raises(RuntimeError, match="eight_angular_sectors_nonempty"):
        subject.validate_receipt(_fake_receipt(angular_sector_counts=[1, 2, 3, 4]))


@pytest.mark.parametrize("check_name", subject.REQUIRED_HARD_CHECKS)
def test_every_replay_support_hard_check_must_be_true(resolved, check_name):
    checks = dict(_fake_receipt()["replay_support_hard_checks"])
    checks[check_name] = False
    with pytest.raises(RuntimeError, match="all_replay_support_hard_checks"):
        subject.validate_receipt(_fake_receipt(replay_support_hard_checks=checks))


def test_all_five_hard_checks_must_be_present(resolved):
    checks = dict(_fake_receipt()["replay_support_hard_checks"])
    del checks["exact_history_and_query_prefix_equality"]
    with pytest.raises(RuntimeError, match="all_replay_support_hard_checks"):
        subject.validate_receipt(_fake_receipt(replay_support_hard_checks=checks))


def test_hard_check_set_is_exactly_the_five_builder_gates():
    assert len(subject.REQUIRED_HARD_CHECKS) == 5
    assert subject.REQUIRED_HARD_CHECKS == tuple(
        sorted(subject.REQUIRED_HARD_CHECKS)
    )


@pytest.mark.parametrize(
    "field",
    list(subject.REPLAY_IDENTITY_FIELDS) + list(subject.COMPACT_RECEIPT_FIELDS),
)
def test_missing_identity_or_compact_field_is_rejected(resolved, field):
    receipt = _fake_receipt()
    del receipt[field]
    with pytest.raises(RuntimeError, match="receipt contract failed"):
        subject.validate_receipt(receipt)


def test_non_integer_counts_are_rejected_not_crashed(resolved):
    with pytest.raises(RuntimeError, match="receipt contract failed"):
        subject.validate_receipt(_fake_receipt(template_count="many"))


# --------------------------------------------------------------------------
# physical covariates are reported, never gates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"minimum_replay_query_contact_steps": 0},
        {"mean_replay_query_contact_steps": 0.0},
        {"maximum_replay_query_contact_steps": 0},
        {"all_replay_model_bounds_inside_playfield": False},
        {"replay_model_bounds_inside_fraction": 0.5},
        {"replay_model_bounds_inside_count": 8192},
        {"minimum_hidden_action_interaction_norm": 0.0},
        {"median_hidden_action_interaction_norm": 0.0},
    ],
)
def test_physical_covariates_do_not_gate_validation(resolved, overrides):
    checks = subject.validate_receipt(_fake_receipt(**overrides))
    assert all(checks.values())


def test_no_check_name_mentions_contact_bounds_or_interaction(resolved):
    checks = subject.validate_receipt(_fake_receipt())
    for name in checks:
        assert "contact" not in name
        assert "bounds" not in name
        assert "playfield" not in name
        assert "interaction" not in name


def test_missing_covariate_field_is_still_required_for_reporting(resolved):
    receipt = _fake_receipt()
    del receipt["mean_replay_query_contact_steps"]
    with pytest.raises(RuntimeError, match="compact_receipt_fields_present"):
        subject.validate_receipt(receipt)


# --------------------------------------------------------------------------
# compact receipt attached to the report
# --------------------------------------------------------------------------


def test_compact_replay_support_carries_identity_and_covariates(resolved, tmp_path):
    receipt_path = tmp_path / "overlay.pt.json"
    support = subject.compact_replay_support(receipt_path, _fake_receipt())
    assert support["receipt"] == str(receipt_path)
    assert support["receipt_sha256"] == RESOLVED_RECEIPT_SHA
    assert support["action_source"] == subject.ACTION_SOURCE
    assert "not the exact planner/CEM" in support["action_distribution_note"]
    assert support["identity"].keys() == set(subject.REPLAY_IDENTITY_FIELDS)
    covariates = support["reported_physical_covariates"]
    assert "covariates, never selection gates" in covariates["note"]
    for field in subject.REPORTED_COVARIATE_FIELDS:
        assert field in covariates
    assert set(subject.COMPACT_RECEIPT_FIELDS) <= support.keys()
    assert len(support["hard_checks"]) == 5


def test_compact_replay_support_is_json_serializable(resolved, tmp_path):
    support = subject.compact_replay_support(
        tmp_path / "overlay.pt.json", _fake_receipt()
    )
    assert json.loads(json.dumps(support, sort_keys=True)) == support


# --------------------------------------------------------------------------
# receipt file handling and installation into the base runner
# --------------------------------------------------------------------------


def _write_receipt(tmp_path, monkeypatch, receipt):
    overlay = tmp_path / "overlay.pt"
    overlay.write_bytes(b"not-a-real-overlay")
    receipt_path = tmp_path / "overlay.pt.json"
    body = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    receipt_path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    monkeypatch.setattr(subject, "OVERLAY_RECEIPT_SHA256", digest)
    monkeypatch.setattr(subject, "OVERLAY_SHA256", RESOLVED_OVERLAY_SHA)
    return overlay, receipt_path, digest


def test_overlay_receipt_reads_and_validates_a_temporary_receipt(
    tmp_path, monkeypatch
):
    overlay, receipt_path, _ = _write_receipt(tmp_path, monkeypatch, _fake_receipt())
    found_path, receipt = subject._overlay_receipt(
        ["--cartesian-overlay", str(overlay)]
    )
    assert found_path == receipt_path
    assert receipt["selected_unique_block_count"] == 1024


def test_overlay_receipt_rejects_a_mutated_receipt_file(tmp_path, monkeypatch):
    overlay, receipt_path, _ = _write_receipt(tmp_path, monkeypatch, _fake_receipt())
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt changed"):
        subject._overlay_receipt(["--cartesian-overlay", str(overlay)])


def test_overlay_receipt_rejects_a_missing_receipt_file(tmp_path, monkeypatch, resolved):
    overlay = tmp_path / "overlay.pt"
    overlay.write_bytes(b"overlay")
    with pytest.raises(RuntimeError, match="receipt changed"):
        subject._overlay_receipt(["--cartesian-overlay", str(overlay)])


def test_main_installs_replay_identity_on_the_base_runner(
    tmp_path, monkeypatch, resolved, isolated_base
):
    monkeypatch.setattr(
        subject,
        "_overlay_receipt",
        lambda _argv: (tmp_path / "overlay.pt.json", _fake_receipt()),
    )
    monkeypatch.setattr(subject.base, "main", lambda _argv: 23)
    assert subject.main(["--cartesian-overlay", str(tmp_path / "overlay.pt")]) == 23
    assert subject.base.CANDIDATE == subject.CANDIDATE
    assert subject.base.OVERLAY_SHA256 == RESOLVED_OVERLAY_SHA
    assert subject.base.OVERLAY_TEMPLATE_COUNT == 2048
    assert subject.base.OVERLAY_CONDITION_PAIR_COUNT == 4096
    assert subject.base.THIS_SOURCE == subject.THIS_SOURCE


def test_main_rewrites_only_branches_and_replay_support(
    tmp_path, monkeypatch, resolved, isolated_base
):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "provenance": {"method": {"candidate": "native"}},
                "result": {
                    "motion_cartesian_action_pair_contract": {
                        "objective": "native_mse_on_real_2x2_history_action_grid",
                        "checks": {"overlay_sha_exact": True},
                        "cartesian_training_overlay": {
                            "action_branches": [
                                "observed_zero",
                                "query_velocity_unit",
                            ],
                            "template_count": 2048,
                            "training_only_frozen_teacher": False,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        subject,
        "_overlay_receipt",
        lambda _argv: (tmp_path / "overlay.pt.json", _fake_receipt()),
    )
    monkeypatch.setattr(
        subject.base, "_rewrite_report", lambda _output, **_kwargs: report_path
    )

    captured = {}

    def fake_main(_argv):
        captured["report"] = subject.base._rewrite_report(
            tmp_path,
            freeze_state={},
            cartesian_state={},
            overlay=tmp_path / "overlay.pt",
        )
        return 0

    monkeypatch.setattr(subject.base, "main", fake_main)
    assert subject.main(["--cartesian-overlay", str(tmp_path / "overlay.pt")]) == 0

    payload = json.loads(captured["report"].read_text(encoding="utf-8"))
    contract = payload["result"]["motion_cartesian_action_pair_contract"]
    overlay = contract["cartesian_training_overlay"]
    assert overlay["action_branches"] == [
        "observed_zero",
        "empirical_replay_5step",
    ]
    assert overlay["template_count"] == 2048
    assert overlay["training_only_frozen_teacher"] is False
    assert contract["objective"] == "native_mse_on_real_2x2_history_action_grid"
    support = overlay["replay_support"]
    assert support["teacher_free"] is True
    assert support["replay_sampling_seed"] == 20260824
    assert support["identity"]["pixels_sha256"] == "f" * 64
    assert support["reported_physical_covariates"][
        "all_replay_model_bounds_inside_playfield"
    ] is False
