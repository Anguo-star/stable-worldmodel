from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_sigreg_replay_rule_switch_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "sigreg_replay_rule_switch_validation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_correction_copies_only_preexisting_decision_fields(
    tmp_path: Path,
) -> None:
    replay = {
        "schema_version": 2,
        "status": "old",
        "protocol_role": "old_role",
        "adapter": {"implementation": "StableWorldModelLeWMAdapter"},
        "comparison": {"required_results": {"model": [3072]}},
        "training_provenance": {"passage_formal_by_model": {"model": {}}},
        "metrics": {"old": True},
        "gates": {"decision_contract": "all_histories_strict_v1"},
        "artifacts": {"output_root": "old", "catalog": "catalog.json"},
    }
    source = {
        "decision_protocol": {
            "name": "informative_history_rule_switch_v2",
        },
        "metrics": {
            "primary": "native_mse",
            "no_crossing_attempt_role": "auxiliary_default_tendency_only",
        },
        "gates": {
            "decision_contract": "informative_history_rule_switch_v2",
            "minimum_same_history_two_target_accuracy_exclusive": 0.5,
        },
    }

    corrected = MODULE.build_corrected_validation(
        replay,
        source,
        replay_path=tmp_path / "replay.yaml",
        rule_switch_source_path=tmp_path / "source.yaml",
    )

    assert corrected["status"] == MODULE.CORRECTIVE_STATUS
    assert corrected["decision_protocol"] == source["decision_protocol"]
    assert corrected["metrics"] == source["metrics"]
    assert corrected["gates"] == source["gates"]
    assert corrected["adapter"] == replay["adapter"]
    assert corrected["comparison"] == replay["comparison"]
    assert corrected["training_provenance"] == replay["training_provenance"]
    assert corrected["artifacts"]["catalog"] == "catalog.json"
    assert (
        corrected["protocol_correction"]["decision_source"][
            "frozen_before_any_replay50_checkpoint"
        ]
        is True
    )


def test_correction_rejects_non_rule_switch_source(tmp_path: Path) -> None:
    replay = {"artifacts": {}}
    source = {
        "decision_protocol": {},
        "metrics": {},
        "gates": {"decision_contract": "all_histories_strict_v1"},
    }

    try:
        MODULE.build_corrected_validation(
            replay,
            source,
            replay_path=tmp_path / "replay.yaml",
            rule_switch_source_path=tmp_path / "source.yaml",
        )
    except ValueError as error:
        assert "rule-switch-v2" in str(error)
    else:
        raise AssertionError("Expected a non-rule-switch source to fail")
