from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.conditional_dynamics_representation.scripts import (
    audit_icl_root_cause_reversal_matrix_v1 as matrix,
)


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict:
    scoreboard = {
        "component_results": [
            {
                "component_id": "task_a",
                "method_name": "LeWM control",
                "icl_ability": {
                    "result": "FAIL",
                    "primary_metric": {"mean": 0.4},
                },
            }
        ]
    }
    behavior = {"status": "completed", "public_test_opened": False}
    scoreboard_sha = _write(tmp_path / "context" / "scoreboard.json", scoreboard)
    behavior_sha = _write(tmp_path / "stable" / "behavior.json", behavior)
    return {
        "schema_version": 1,
        "analysis_id": matrix.ANALYSIS_ID,
        "status": "frozen_discovery_matrix",
        "authority": {
            "optimizer_steps_authorized": 0,
            "public_test_raw_access_authorized": False,
            "public_test_rerun_authorized": False,
            "checkpoint_mutation_authorized": False,
        },
        "comparability": {
            "cross_family_absolute_metric_comparison": "forbidden",
            "universal_numeric_threshold": "forbidden",
            "allowed_basis": [
                "within_cell_relative_delta",
                "paired_uncertainty",
                "intervention_direction_consistency",
            ],
        },
        "support_contract": {
            "primary_behavior_evidence": "on_support_correct_vs_swapped_g_swap",
            "removed_history": "auxiliary_off_support_only",
        },
        "required_matrix_roles": ["negative_cell"],
        "sources": [
            {
                "source_id": "scoreboard",
                "repository": "ContextWorld",
                "path": "scoreboard.json",
                "sha256": scoreboard_sha,
                "kind": "historical_published_outcome",
                "root_cause_eligible": False,
            },
            {
                "source_id": "behavior",
                "repository": "stable-worldmodel",
                "path": "behavior.json",
                "sha256": behavior_sha,
                "kind": "development_on_support",
                "root_cause_eligible": True,
            },
        ],
        "cells": [
            {
                "cell_id": "task_a_lewm",
                "task": "task_a",
                "model_family": "LeWM",
                "recipe": "native",
                "matrix_role": "negative_cell",
                "observed_icl_outcome": "negative",
                "evidence_tier": "historical_frozen_outcome",
                "binding_cause_status": "unresolved",
                "outcome_check": {
                    "type": "scoreboard_row",
                    "source": "scoreboard",
                    "component_id": "task_a",
                    "method_contains": "lewm",
                    "expected_result": "FAIL",
                    "expected_metric_mean": 0.4,
                },
                "layers": {
                    "data": {"status": "missing", "sources": []},
                    "representation": {"status": "missing", "sources": []},
                    "optimization": {"status": "missing", "sources": []},
                    "behavior": {
                        "status": "available",
                        "sources": ["behavior"],
                        "primary": True,
                    },
                },
            }
        ],
    }


def test_valid_matrix_verifies_outcome_and_reports_missing_layers(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    report = matrix.validate_and_inventory(
        config,
        root=tmp_path / "stable",
        contextworld_root=tmp_path / "context",
    )
    assert report["status"] == "passed_read_only_matrix_inventory"
    assert report["cells"][0]["missing_layers"] == [
        "data",
        "representation",
        "optimization",
    ]
    assert report["claim_status"]["effective_conditional_visibility_general_mechanism"] == (
        "evidence_inventory_only_pending_matched_interventions"
    )


def test_hash_guard_is_fail_closed(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["sources"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="sha256 changed"):
        matrix.validate_and_inventory(
            config,
            root=tmp_path / "stable",
            contextworld_root=tmp_path / "context",
        )


def test_public_raw_source_is_rejected(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    source = config["sources"][0]
    source["path"] = "public_test/raw/scoreboard.json"
    source["sha256"] = _write(
        tmp_path / "context" / source["path"],
        {"component_results": []},
    )
    with pytest.raises(ValueError, match="raw Public Test"):
        matrix.validate_and_inventory(
            config,
            root=tmp_path / "stable",
            contextworld_root=tmp_path / "context",
        )


def test_removed_history_cannot_enter_any_layer(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    removed_path = tmp_path / "stable" / "removed.json"
    removed_sha = _write(removed_path, {"support": "off"})
    config["sources"].append(
        {
            "source_id": "removed",
            "repository": "stable-worldmodel",
            "path": "removed.json",
            "sha256": removed_sha,
            "kind": "off_support_auxiliary",
            "root_cause_eligible": False,
        }
    )
    config["cells"][0]["layers"]["behavior"] = {
        "status": "available",
        "sources": ["removed"],
        "primary": True,
    }
    with pytest.raises(ValueError, match="outcome-only source entered a root-cause layer"):
        matrix.validate_and_inventory(
            config,
            root=tmp_path / "stable",
            contextworld_root=tmp_path / "context",
        )


def test_primary_behavior_can_add_a_development_intervention(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    intervention_path = tmp_path / "stable" / "intervention.json"
    intervention_sha = _write(intervention_path, {"status": "completed"})
    config["sources"].append(
        {
            "source_id": "intervention",
            "repository": "stable-worldmodel",
            "path": "intervention.json",
            "sha256": intervention_sha,
            "kind": "development_intervention",
            "root_cause_eligible": True,
        }
    )
    config["cells"][0]["layers"]["behavior"]["sources"].append("intervention")
    report = matrix.validate_and_inventory(
        config,
        root=tmp_path / "stable",
        contextworld_root=tmp_path / "context",
    )
    assert report["cells"][0]["layers"]["behavior"]["sources"] == [
        "behavior",
        "intervention",
    ]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("cross_family_absolute_metric_comparison", "allowed", "must be forbidden"),
        ("universal_numeric_threshold", "required", "must be forbidden"),
    ],
)
def test_cross_family_absolute_and_universal_threshold_are_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    config = _fixture(tmp_path)
    config["comparability"][field] = value
    with pytest.raises(ValueError, match=match):
        matrix.validate_and_inventory(
            config,
            root=tmp_path / "stable",
            contextworld_root=tmp_path / "context",
        )


def test_historical_scoreboard_cannot_be_root_cause_evidence(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["sources"][0]["root_cause_eligible"] = True
    with pytest.raises(ValueError, match="conflicts with source kind"):
        matrix.validate_and_inventory(
            config,
            root=tmp_path / "stable",
            contextworld_root=tmp_path / "context",
        )


def test_historical_outcome_cannot_enter_a_root_cause_layer(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["cells"][0]["layers"]["data"] = {
        "status": "partial",
        "sources": ["scoreboard"],
    }
    with pytest.raises(ValueError, match="outcome-only source"):
        matrix.validate_and_inventory(
            config,
            root=tmp_path / "stable",
            contextworld_root=tmp_path / "context",
        )


def test_field_outcome_check_is_supported(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    payload_path = tmp_path / "context" / "development.yaml"
    payload_path.write_text("result:\n  passed: false\n  score: 0.5\n", encoding="utf-8")
    config["sources"].append(
        {
            "source_id": "development_label",
            "repository": "ContextWorld",
            "path": "development.yaml",
            "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
            "kind": "configuration_provenance",
            "root_cause_eligible": False,
        }
    )
    config["cells"][0]["outcome_check"] = {
        "type": "fields",
        "source": "development_label",
        "checks": [
            {"path": ["result", "passed"], "expected": False},
            {"path": ["result", "score"], "expected": 0.5, "tolerance": 1.0e-12},
        ],
    }
    report = matrix.validate_and_inventory(
        config,
        root=tmp_path / "stable",
        contextworld_root=tmp_path / "context",
    )
    assert report["cells"][0]["outcome_verification"]["type"] == "fields"


def test_input_is_not_mutated(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    before = copy.deepcopy(config)
    matrix.validate_and_inventory(
        config,
        root=tmp_path / "stable",
        contextworld_root=tmp_path / "context",
    )
    assert config == before
