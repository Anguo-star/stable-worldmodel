#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import types

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/summarize_icl_native_reversal_mechanism_v1.py"
)
_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/icl_native_reversal_mechanism_v1.yaml"
)


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("native_reversal_summary_v1", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()


def test_static_scope_is_read_only_training_only():
    assert mod.MODEL_LOADED is False
    assert mod.OPTIMIZER_STEPS == 0
    assert mod.CHECKPOINT_WRITTEN is False
    assert mod.DEVELOPMENT_OPENED is False
    assert mod.PUBLIC_TEST_OPENED is False
    assert "training_only" in mod.CLAIM_SCOPE


@pytest.mark.parametrize("token", ["development", "public", "test", "validation", "val", "sealed"])
def test_training_guard_rejects_eval_tokens(token):
    with pytest.raises(ValueError, match="Training-only"):
        mod.training_only_path_guard(Path("/tmp") / token / "artifact.json")


def test_training_guard_accepts_training_mechanism_path():
    mod.training_only_path_guard(Path("/repo/artifacts/training/mechanism/result.json"))


def test_exclusive_mkdir_refuses_overwrite(tmp_path):
    target = tmp_path / "out"
    mod.exclusive_mkdir(target)
    with pytest.raises(FileExistsError):
        mod.exclusive_mkdir(target)


def test_module_group_l2_flat_and_nested():
    assert mod.module_group_l2({"a": 3.0, "b": 4.0}) == 5.0
    assert mod.module_group_l2({"a": {"l2_norm": 3.0}, "b": {"l2_norm": 4.0}}) == 5.0


def test_module_group_l2_rejects_negative():
    with pytest.raises(RuntimeError, match="invalid gradient norm"):
        mod.module_group_l2({"bad": -1.0})


def test_aggregate_scalar():
    assert mod.aggregate_scalar([1.0, 2.0, 3.0]) == {
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
    }


def test_selected_file_set_digest_is_order_invariant(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    first = mod.selected_file_set_digest(tmp_path, [a, b])
    second = mod.selected_file_set_digest(tmp_path, [b, a])
    assert first == second
    expected = hashlib.sha256()
    for path in (a, b):
        expected.update(path.name.encode())
        expected.update(b"\0")
        expected.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        expected.update(b"\0")
    assert first == expected.hexdigest()


def test_no_eval_access_scope_contract():
    mod._validate_no_eval_access(
        {
            "probe_split": "train",
            "development_data_opened": False,
            "public_test_opened": False,
            "development_open_attempts": 0,
        },
        label="synthetic",
    )
    with pytest.raises(RuntimeError):
        mod._validate_no_eval_access(
            {"probe_split": "train", "public_test_opened": True},
            label="synthetic",
        )


def test_config_claim_boundary_and_scope():
    config = mod._load_config(_CONFIG)
    assert config["scope"]["split"] == "train"
    assert config["claim_boundary"]["universal_data_only_cause_not_claimed"] is True
    assert config["claim_boundary"]["coja_remains_positive_control"] is True
    assert config["action_delay"]["arms"] == {
        "lewm_native": "a0",
        "lewm_pldm_objective": "a3",
        "pldm_native": "a4",
    }


def test_real_frozen_inputs_check_only_contract():
    summary, rows, inputs = mod.run(_CONFIG)
    assert summary["status"] == "passed_training_only_native_reversal_mechanism_summary"
    assert summary["action_delay"]["gate"]["result"] == "model_side_objective_route_separation_supported"
    assert summary["action_strength"]["gate"]["result"] == "reverse_confirmation_not_established"
    assert summary["synthesis"]["universal_low_pixel_or_data_only_root_cause"] == "rejected"
    assert summary["synthesis"]["coja_status"].startswith("retained")
    assert len(rows) == 3 * 3 * 6 + 2
    assert len(inputs) == 1 + 18 + 9 + 2


def test_action_delay_contrast_has_expected_direction():
    summary, _, _ = mod.run(_CONFIG)
    contrast = summary["action_delay"]["contrasts"]
    assert contrast["common_probe_and_step0_latents_exact"] is True
    assert contrast["same_logical_batches_within_seed"] is True
    assert (
        contrast["lewm_pldm_objective_vs_lewm_native_step0_total_gradient_l2_ratio"]
        > 2.0
    )
    assert (
        contrast[
            "lewm_pldm_objective_vs_lewm_native_step0_local_response_improvement_magnitude_ratio"
        ]
        > 10.0
    )
    assert (
        contrast[
            "lewm_pldm_objective_vs_pldm_native_step0_response_derivative_relative_difference"
        ]
        < 1.0e-3
    )
    assert contrast["step256_all_arms_signed_gain_still_negative"] is True


def test_frozen_outcome_labels_are_opposite_reversals():
    summary, _, _ = mod.run(_CONFIG)
    assert summary["frozen_outcomes"] == {
        "action_delay_lewm_native": "negative",
        "action_delay_pldm_native": "positive",
        "action_strength_lewm_native": "positive",
        "action_strength_pldm_native": "negative",
    }
