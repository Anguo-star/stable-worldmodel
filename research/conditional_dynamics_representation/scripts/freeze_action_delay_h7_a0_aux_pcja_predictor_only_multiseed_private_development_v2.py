#!/usr/bin/env python3
"""Create the checkpoint-blind protocol and implementation freezes for v2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
BUILDER = THIS_SOURCE.with_name(
    "build_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
CONFIG = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_development_v2.yaml"
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
RELEASE_ROOT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2"
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _builder() -> Any:
    name = "_predictor_only_multiseed_v2_freezer_builder"
    specification = importlib.util.spec_from_file_location(name, BUILDER)
    require(
        specification is not None and specification.loader is not None,
        "cannot import builder",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def expected_protocol() -> dict[str, Any]:
    builder = _builder()
    raw = builder.load_config()
    merged = builder.effective_config(raw)
    builder.validate_config(raw)
    builder.validate_binding_chain(merged)
    require(not RELEASE_ROOT.exists(), "protocol must precede release construction")
    require(not TERMINAL.exists(), "protocol must precede terminal identity")
    selection = builder.selector_preflight(merged, verify_protocol=False)
    source = builder.source_rebind_preflight(merged)
    content = {
        "schema_version": 1,
        "status": (
            "frozen_checkpoint_blind_before_multiseed_release_build_claim_or_score"
        ),
        "release_id": builder.EXPECTED_RELEASE_ID,
        "preregistration": {
            "path": CONFIG.relative_to(REPO_ROOT).as_posix(),
            "sha256": builder.file_sha256(CONFIG),
        },
        "contract_sections_addendum": {
            "path": builder.CONTRACT_SECTIONS_CONFIG.relative_to(REPO_ROOT).as_posix(),
            "sha256": builder.file_sha256(builder.CONTRACT_SECTIONS_CONFIG),
        },
        "build_source_gate_addendum": {
            "path": builder.BUILD_SOURCE_GATE_CONFIG.relative_to(REPO_ROOT).as_posix(),
            "sha256": builder.file_sha256(builder.BUILD_SOURCE_GATE_CONFIG),
            "receipt_content_sha256": builder.BUILD_SOURCE_GATE_RECEIPT_CONTENT_SHA256,
        },
        "frozen_contract_hashes": {
            section: builder.canonical_sha256(merged[section])
            for section in builder.v1._contract_sections(merged)
        },
        "selector_preflight": selection["frozen_preflight"],
        "source_rebind_preflight": source,
        "candidate_terminal_boundary": {
            "candidate_ids": list(builder.AUTHORIZED_IDS),
            "training_seeds": [4096, 5120, 3072],
            "terminal_optimizer_step": builder.TERMINAL_STEP,
            "terminal_file_hashes_recorded": False,
            "checkpoint_bytes_opened": False,
            "model_loaded_or_prediction_computed": False,
        },
        "exclusion_census": {
            "source_count": builder.EXPECTED_SOURCE_COUNT,
            "prior_private_release_count": builder.EXPECTED_PRIOR_PRIVATE_COUNT,
            "explicit_private_bindings": builder.prior_release_bindings(),
            "full_overlap_audit_required": True,
        },
        "scope": {
            "release_built": False,
            "exclusive_claim_committed": False,
            "candidate_checkpoint_deserialized_or_model_loaded": False,
            "prediction_latent_or_score_computed": False,
            "public_or_test_results_opened": False,
            "terminal_addendum_present_at_freeze": False,
        },
    }
    return {**content, "content_sha256": builder.canonical_sha256(content)}


def expected_implementation() -> dict[str, Any]:
    builder = _builder()
    require(PROTOCOL.is_file() and not PROTOCOL.is_symlink(), "protocol freeze absent")
    protocol = _read_json(PROTOCOL)
    require(protocol == expected_protocol(), "protocol freeze differs")
    require(not RELEASE_ROOT.exists(), "implementation must precede release")
    require(not TERMINAL.exists(), "implementation must precede terminal identity")
    content = {
        "schema_version": 1,
        "status": (
            "frozen_checkpoint_blind_implementation_before_multiseed_release_"
            "build_claim_or_score"
        ),
        "release_id": builder.EXPECTED_RELEASE_ID,
        "config_sha256": builder.file_sha256(CONFIG),
        "protocol_freeze_sha256": builder.file_sha256(PROTOCOL),
        "protocol_freeze_content_sha256": protocol["content_sha256"],
        "implementation_sources": builder._implementation_sources(),
        "build_contract": "fresh_generation_only_no_recovery_input",
        "terminal_identity_addendum": {
            "path": TERMINAL.relative_to(REPO_ROOT).as_posix(),
            "present_at_freeze": False,
            "release_build_requires_absence": True,
            "hash_only_freeze_required_before_claim": True,
        },
        "observation_boundary": {
            "release_assets_built_before_freeze": False,
            "candidate_checkpoint_deserialized_or_model_loaded": False,
            "prediction_latent_or_score_computed": False,
            "public_or_test_results_opened": False,
        },
    }
    return {**content, "content_sha256": builder.canonical_sha256(content)}


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short freeze write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_or_check(
    path: Path, expected: Mapping[str, Any], *, check: bool
) -> str:
    if path.exists():
        metadata = os.lstat(path)
        require(
            stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
            f"unsafe freeze: {path}",
        )
        require(_read_json(path) == expected, f"existing freeze differs: {path}")
        return "validated_existing"
    require(not check, f"freeze absent: {path}")
    _write_exclusive(path, expected)
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("protocol", "implementation", "all"), default="all"
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    actions: dict[str, str] = {}
    if arguments.phase in {"protocol", "all"}:
        actions["protocol"] = _publish_or_check(
            PROTOCOL, expected_protocol(), check=arguments.check
        )
    if arguments.phase in {"implementation", "all"}:
        actions["implementation"] = _publish_or_check(
            IMPLEMENTATION, expected_implementation(), check=arguments.check
        )
    print(json.dumps({"status": "passed", "actions": actions}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
