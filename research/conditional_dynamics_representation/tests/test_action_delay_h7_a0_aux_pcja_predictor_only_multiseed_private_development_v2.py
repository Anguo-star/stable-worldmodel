from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
BUILDER = ROOT / (
    "scripts/build_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
EVALUATOR = ROOT / (
    "scripts/eval_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
FREEZER = ROOT / (
    "scripts/freeze_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
TERMINAL_FREEZER = ROOT / (
    "scripts/freeze_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "terminal_v2.py"
)
PROTOCOL = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_protocol_freeze.json"
)
IMPLEMENTATION = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_implementation_freeze.json"
)
TERMINAL = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_terminal_identity_addendum_v1.json"
)
RELEASE = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2"
)


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


builder = _load(BUILDER, "_test_predictor_only_multiseed_v2_builder")


def test_selector_is_fresh_and_covers_fifteen_sources() -> None:
    result = builder.selector_preflight(
        builder.effective_config(), verify_protocol=False
    )
    assert result["status"] == "passed"
    assert result["queries"] == 300
    assert len(result["overlaps"]) == 15
    assert result["frozen_preflight"] == {
        "source_count": 15,
        "prior_private_release_count": 10,
        "reset_state_census_count": 24128,
        "simulator_seed_census_count": 15460,
        "metadata_census_sha256": (
            "e496f197ff08d54d359248a854bc5fb694febaf8cec4e7e0dfc763192ba7a562"
        ),
        "selector_rows_sha256": (
            "76313d8aee42b766ce1083d26a0eea4e8b9bee6923312b54767c22c8f2252e6f"
        ),
        "selected_reset_state_list_sha256": (
            "855001e020a7023abfc554025c7b17e8cc650ef2ec90da7f5662f94327f1f635"
        ),
        "selected_simulator_seed_list_sha256": (
            "6169f611ddfadb6b84a325b456148c21305f6ddfab02db4f7ca5f24287a22000"
        ),
        "rejection_ledger_sha256": (
            "899ecf1301f2d5eafbc4efed5d8310edfb1ec115841e36ac300376c586887709"
        ),
        "rejection_events": 511,
        "rejection_reason_counts": {
            "frozen_metadata_census": 479,
            "already_selected": 32,
        },
        "final_reset_state_overlap_count": 0,
        "final_simulator_seed_overlap_count": 0,
        "candidate_output_used": False,
    }


def test_builder_has_full_checkpoint_blind_release_path() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    imports = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imports
    assert "torch.load" not in source
    assert "def build_release" in source
    assert "def _content_gated_source_identity" in source
    assert "def validate_release_identity" in source
    assert "--audit-existing" in source


def test_content_gate_executes_all_forty_five_files() -> None:
    result = builder.source_rebind_preflight(builder.effective_config())
    assert result["status"] == "passed"
    assert result["contextworld_head_gated"] is False
    assert result["contextworld_executable_file_count"] == 24
    assert result["overlap_source_count"] == 8
    assert result["renderer_dependency_count"] == 2
    assert result["scoring_runtime_dependency_count"] == 11
    assert result["adjudicated_source_count"] == 4
    assert result["unadjudicated_differences"] == 0


def test_freezer_and_evaluator_own_the_three_cell_contract() -> None:
    freezer = _load(FREEZER, "_test_predictor_only_multiseed_v2_freezer")
    if not RELEASE.exists() and not TERMINAL.exists():
        expected = freezer.expected_protocol()
        assert expected["content_sha256"] == builder.canonical_sha256(
            {key: value for key, value in expected.items() if key != "content_sha256"}
        )
    source = EVALUATOR.read_text(encoding="utf-8")
    for candidate_id in builder.AUTHORIZED_IDS:
        assert candidate_id in source
    names = {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"_score_candidate", "_finalize", "_conclusion"} <= names
    assert "cross_seed_pooling_performed" in source
    assert "cross_seed_averaging_or_rescue_used" in source
    assert TERMINAL_FREEZER.is_file()


def test_frozen_files_and_release_validate_when_present() -> None:
    if PROTOCOL.exists():
        assert builder.validate_protocol_freeze()["release_id"] == builder.EXPECTED_RELEASE_ID
    if IMPLEMENTATION.exists():
        assert (
            builder.validate_implementation_freeze()["release_id"]
            == builder.EXPECTED_RELEASE_ID
        )
    if RELEASE.exists():
        assert builder.validate_release_identity()["passed"] is True
    if TERMINAL.exists():
        assert (
            builder.validate_terminal_addendum()["release_id"]
            == builder.EXPECTED_RELEASE_ID
        )


def test_terminal_freezer_waits_for_release() -> None:
    if RELEASE.exists():
        freezer = _load(
            TERMINAL_FREEZER, "_test_predictor_only_multiseed_v2_terminal_freezer"
        )
        payload = freezer.expected_payload()
        assert [row["candidate_id"] for row in payload["cells"]] == list(
            builder.AUTHORIZED_IDS
        )
    else:
        pytest.skip("terminal identity is intentionally post-release")
