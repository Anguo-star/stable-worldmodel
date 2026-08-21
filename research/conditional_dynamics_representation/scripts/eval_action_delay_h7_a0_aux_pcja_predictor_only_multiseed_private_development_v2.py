#!/usr/bin/env python3
"""Consume the one-use three-cell predictor-only multi-seed release v2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
PARENT = THIS_SOURCE.with_name(
    "eval_action_delay_h7_a0_aux_pcja_native_sampler_private_development_v1.py"
)
CONFIG = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_development_v2.yaml"
)
PROTOCOL = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_protocol_freeze.json"
)
TERMINAL_MANIFEST = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_terminal_identity_addendum_v1.json"
)
IMPLEMENTATION_FREEZE = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_implementation_freeze.json"
)
BUILDER = THIS_SOURCE.with_name(
    "build_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
FREEZER = THIS_SOURCE.with_name(
    "freeze_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
TERMINAL_FREEZER = THIS_SOURCE.with_name(
    "freeze_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "terminal_v2.py"
)
FOCUSED_TEST = ROOT / (
    "tests/test_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
OUTPUT_ROOT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2"
)

RELEASE_ID = "action-delay-h7-a0-aux-pcja-predictor-only-multiseed-private-v2"
AUTHORIZED_IDS = (
    "action_delay_h7_a0_aux_pcja_predictor_only_s4096_v1",
    "action_delay_h7_a0_aux_pcja_predictor_only_s5120_v1",
    "historical-predictor-only-PCJA-s3072-step1024",
)
CALIBRATION_ID = AUTHORIZED_IDS[2]
REPLICATION_IDS = AUTHORIZED_IDS[:2]
EVAL_SEEDS = (124, 125, 126, 127, 128, 129)
BOOTSTRAP_SEED = 2_026_081_902
EXPECTED_SOURCE_COUNT = 15
EXPECTED_PRIOR_PRIVATE_COUNT = 10
EXPECTED_QUERIES = 300

WORK_ROOT = OUTPUT_ROOT / ".private_batch_in_progress"
CLAIM_PATH = WORK_ROOT / "exclusive_consumption_claim.json"
STAGING_ROOT = WORK_ROOT / "candidate_staging"
RESULT_ROOT = OUTPUT_ROOT / "candidate_results"
RECEIPT_PATH = OUTPUT_ROOT / "consumption_receipt.json"


def _load(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


base = _load(PARENT, "_predictor_only_multiseed_v2_eval_parent")


def _config_cells(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {row["candidate_id"]: dict(row) for row in config["candidates"]}
    calibration = dict(config["positive_calibration"])
    calibration["candidate_id"] = calibration["control_id"]
    rows[CALIBRATION_ID] = calibration
    base.require(tuple(rows) == AUTHORIZED_IDS, "configured cell population changed")
    return rows


def _validate_static_contract(builder: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    config = builder.load_config()
    protocol = builder.validate_protocol_freeze(config)
    builder.validate_config(config)
    cells = _config_cells(config)
    base.require(config["release_id"] == RELEASE_ID, "release id changed")
    base.require(
        tuple(config["selector"]["eval_seeds"]) == EVAL_SEEDS
        and int(config["selector"]["bootstrap_seed"]) == BOOTSTRAP_SEED,
        "evaluation seeds changed",
    )
    base.require(
        all(int(row["optimizer_step"]) == 1024 for row in cells.values()),
        "terminal budget changed",
    )
    base.require(
        config["consumption_contract"]["expected_models"] == 3
        and config["decision_contract"]["cross_seed_pooling_performed"] is False
        and config["decision_contract"]["cross_seed_averaging_or_rescue_allowed"]
        is False,
        "three-cell decision contract changed",
    )
    base.require(
        config["exclusion_contract"]["source_count"] == EXPECTED_SOURCE_COUNT
        and config["exclusion_contract"]["prior_private_release_count"]
        == EXPECTED_PRIOR_PRIVATE_COUNT,
        "exclusion census changed",
    )
    gates = config["scoring"]["gates"]
    base.require(
        gates["target_latent_physical_group_separation_required"] is True
        and float(gates["physical_group_macro_accuracy_minimum"]) == base.GATE_MACRO
        and float(gates["minimum_physical_group_accuracy_minimum"])
        == base.GATE_MINIMUM_GROUP
        and float(gates["paired_query_bootstrap_95_percent_lower_bound_minimum"])
        == base.GATE_BOOTSTRAP_LOWER,
        "frozen gates changed",
    )
    return config, {
        "config": base._stable_ref(CONFIG, purpose="multiseed v2 preregistration"),
        "protocol": base._stable_ref(PROTOCOL, purpose="multiseed v2 protocol"),
        "protocol_content_sha256": protocol["content_sha256"],
    }


def _audit_terminal_identities(
    builder: Any, config: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = builder.validate_terminal_addendum(config)
    observed, manifest_ref = base._stable_json(
        TERMINAL_MANIFEST,
        purpose="multiseed v2 terminal identity",
        self_hash="content",
    )
    base.require(observed == manifest, "terminal manifest changed across validation")
    configured = _config_cells(config)
    cells: dict[str, dict[str, Any]] = {}
    for row in manifest["cells"]:
        candidate_id = row["candidate_id"]
        base.require(
            candidate_id in configured and candidate_id not in cells,
            "terminal candidate repeated",
        )
        refs = {
            role: base._stable_ref(
                base._data_path(artifact["path"], purpose=f"{candidate_id} {role}"),
                purpose=f"{candidate_id} {role}",
                expected_sha256=artifact["sha256"],
                expected_bytes=artifact["bytes"],
            )
            for role, artifact in sorted(row["artifacts"].items())
        }
        base.require(
            {"checkpoint", "checkpoint_config"} <= set(refs),
            f"terminal artifacts incomplete: {candidate_id}",
        )
        cells[candidate_id] = {
            "candidate_id": candidate_id,
            "role": row["role"],
            "variant": row["variant"],
            "model_family": row["model_family"],
            "objective": row["objective"],
            "sampler": row["sampler"],
            "seed": int(row["seed"]),
            "optimizer_step": int(row["optimizer_step"]),
            "checkpoint": refs["checkpoint"]["path"],
            "checkpoint_sha256": refs["checkpoint"]["sha256"],
            "checkpoint_config": refs["checkpoint_config"]["path"],
            "checkpoint_config_sha256": refs["checkpoint_config"]["sha256"],
            "terminal_artifacts": refs,
            "checkpoint_deserialized_or_model_loaded": False,
            "prediction_latent_or_score_computed": False,
        }
    base.require(tuple(cells) == AUTHORIZED_IDS, "terminal cell missing or reordered")
    return {
        "manifest": {**manifest_ref, "content_sha256": manifest["content_sha256"]},
        "cells": cells,
        "training_identity": manifest["training_identity"],
        "checkpoint_deserialized_or_model_loaded": False,
        "prediction_latent_or_score_computed": False,
    }


def _preclaim_release_audit(builder: Any, *, required: bool) -> dict[str, Any] | None:
    identity_path = OUTPUT_ROOT / "release_identity.json"
    if not identity_path.exists():
        base.require(not required, "frozen multiseed v2 release is absent")
        return None
    validated = builder.validate_release_identity(OUTPUT_ROOT)
    identity, identity_ref = base._stable_json(
        identity_path, purpose="multiseed v2 release identity", self_hash="identity"
    )
    base.require(identity == validated["release_identity"], "release identity changed")
    base.require(
        identity.get("schema_version") == 8
        and identity.get("status")
        == "frozen_before_predictor_only_multiseed_model_open_inference_or_scoring",
        "release identity status changed",
    )
    base.require(
        identity.get("terminal_identity_addendum_absent_at_release_build") is True
        and identity.get("authorized_candidate_ids") == list(AUTHORIZED_IDS),
        "release was not checkpoint blind",
    )
    catalog, catalog_ref = base._stable_json(
        OUTPUT_ROOT / "catalog.json", purpose="multiseed v2 catalog"
    )
    base.require(
        catalog.get("release_id") == RELEASE_ID
        and len(catalog.get("queries", ())) == EXPECTED_QUERIES,
        "release catalog changed",
    )
    return {
        "identity": {**identity_ref, "identity_sha256": identity["identity_sha256"]},
        "catalog": catalog_ref,
        "release_identity": identity,
    }


def _dependency_audit(builder: Any, *, require_implementation: bool) -> dict[str, Any]:
    sources = {
        "builder": BUILDER,
        "evaluator": THIS_SOURCE,
        "freezer": FREEZER,
        "terminal_freezer": TERMINAL_FREEZER,
        "focused_test": FOCUSED_TEST,
        "parent_evaluator": PARENT,
        "v4_scoring_kernel": base.V4_EVALUATOR,
        "v4_release_helpers": base.V4_BUILDER,
        "owned_adapter": base.BASE_EVALUATOR,
    }
    refs = {name: base._stable_ref(path, purpose=name) for name, path in sources.items()}
    if IMPLEMENTATION_FREEZE.exists():
        implementation = builder.validate_implementation_freeze()
        payload, reference = base._stable_json(
            IMPLEMENTATION_FREEZE,
            purpose="multiseed v2 implementation freeze",
            self_hash="content",
        )
        base.require(payload == implementation, "implementation freeze changed")
        refs["implementation_freeze"] = {
            **reference,
            "content_sha256": payload["content_sha256"],
        }
    else:
        base.require(not require_implementation, "implementation freeze is absent")
    return refs


def _install() -> Any:
    assignments = {
        "CONFIG": CONFIG,
        "PROTOCOL": PROTOCOL,
        "TERMINAL_MANIFEST": TERMINAL_MANIFEST,
        "IMPLEMENTATION_FREEZE": IMPLEMENTATION_FREEZE,
        "BUILDER": BUILDER,
        "FREEZER": FREEZER,
        "FOCUSED_TEST": FOCUSED_TEST,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "RELEASE_ID": RELEASE_ID,
        "CANDIDATE_ID": AUTHORIZED_IDS[0],
        "AUTHORIZED_IDS": AUTHORIZED_IDS,
        "EVAL_SEEDS": EVAL_SEEDS,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "EXPECTED_SOURCE_COUNT": EXPECTED_SOURCE_COUNT,
        "EXPECTED_PRIOR_PRIVATE_COUNT": EXPECTED_PRIOR_PRIVATE_COUNT,
        "EXPECTED_QUERIES": EXPECTED_QUERIES,
        "WORK_ROOT": WORK_ROOT,
        "CLAIM_PATH": CLAIM_PATH,
        "STAGING_ROOT": STAGING_ROOT,
        "RESULT_ROOT": RESULT_ROOT,
        "RECEIPT_PATH": RECEIPT_PATH,
    }
    for name, value in assignments.items():
        setattr(base, name, value)
    base._validate_static_contract = _validate_static_contract
    base._audit_terminal_identities = _audit_terminal_identities
    base._preclaim_release_audit = _preclaim_release_audit
    base._dependency_audit = _dependency_audit
    return base


@contextmanager
def _candidate_scope(candidate_id: str) -> Iterator[Any]:
    module = _install()
    base.require(candidate_id in AUTHORIZED_IDS, f"unauthorized cell: {candidate_id}")
    previous = module.CANDIDATE_ID
    module.CANDIDATE_ID = candidate_id
    try:
        yield module
    finally:
        module.CANDIDATE_ID = previous


def _score_candidate(
    candidate_id: str, *, device: str, batch_size: int
) -> dict[str, Any]:
    with _candidate_scope(candidate_id) as module:
        return module._score_candidate(device=device, batch_size=batch_size)


def _load_staged(
    candidate_id: str,
    *,
    claim: Mapping[str, Any],
    release_identity_sha256: str,
) -> dict[str, Any]:
    with _candidate_scope(candidate_id) as module:
        return module._load_staged(
            claim=claim, release_identity_sha256=release_identity_sha256
        )


def _conclusion(calibration: bool, confirmations: Mapping[str, bool]) -> str:
    if not calibration:
        return "invalid_release_or_adapter_no_candidate_conclusion"
    if all(confirmations.values()):
        return "both_confirmation_seeds_pass_all_four_gates"
    return "no_cross_seed_replication_claim"


def _finalize() -> dict[str, Any]:
    module = _install()
    if RECEIPT_PATH.exists():
        receipt, _ = module._stable_json(
            RECEIPT_PATH, purpose="multiseed v2 consumption receipt", self_hash="content"
        )
        module.require(
            receipt.get("status") == "consumed_by_complete_calibration_plus_two_seed_batch"
            and receipt.get("release_id") == RELEASE_ID,
            "conflicting consumption receipt",
        )
        return receipt
    claim, context, _scorer, _helpers, _adapter = module._postclaim_context()
    release_identity_sha256 = context["release"]["release_identity"]["identity_sha256"]
    staged = {
        candidate_id: _load_staged(
            candidate_id,
            claim=claim,
            release_identity_sha256=release_identity_sha256,
        )
        for candidate_id in AUTHORIZED_IDS
    }
    passed = {candidate_id: bool(row["gate"]["passed"]) for candidate_id, row in staged.items()}
    published_refs: dict[str, dict[str, Any]] = {}
    for candidate_id, row in staged.items():
        publication = dict(row)
        publication["status"] = "published_but_uncommitted_until_consumption_receipt"
        destination = RESULT_ROOT / f"{candidate_id}.json"
        published = module._publish_or_validate(destination, publication)
        reference = module._stable_ref(destination, purpose=f"published {candidate_id}")
        published_refs[candidate_id] = {
            **reference,
            "path": str(destination.relative_to(REPO_ROOT)),
            "content_sha256": published["content_sha256"],
            "four_gate_passed": passed[candidate_id],
        }
    module.require(
        module._preclaim_audit(require_release=True, require_implementation=True)[
            "audit_sha256"
        ]
        == claim["preclaim_audit_sha256"],
        "frozen identities changed before receipt commit",
    )
    confirmations = {candidate_id: passed[candidate_id] for candidate_id in REPLICATION_IDS}
    receipt = {
        "schema_version": 1,
        "status": "consumed_by_complete_calibration_plus_two_seed_batch",
        "release_id": RELEASE_ID,
        "claim_content_sha256": claim["content_sha256"],
        "release_identity_sha256": release_identity_sha256,
        "terminal_manifest_file_sha256": claim["terminal_manifest_file_sha256"],
        "terminal_manifest_content_sha256": claim[
            "terminal_manifest_content_sha256"
        ],
        "published_results": published_refs,
        "decision": {
            "calibration_four_gate_passed": passed[CALIBRATION_ID],
            "confirmation_four_gate_passed": confirmations,
            "replication_passed": passed[CALIBRATION_ID]
            and all(confirmations.values()),
            "cross_seed_pooling_performed": False,
            "cross_seed_averaging_or_rescue_used": False,
            "both_seeds_reported_regardless_of_outcome": True,
            "conclusion": _conclusion(passed[CALIBRATION_ID], confirmations),
            "comparative_public_or_cem_claim_authorized": False,
        },
        "scope": {
            "models_scored": list(AUTHORIZED_IDS),
            "expected_models": 3,
            "public_test_or_sealed_content_opened": False,
            "online_environment_calls": 0,
            "candidate_additions_after_consumption": "forbidden",
            "receipt_is_unique_completion_commit_marker_and_published_last": True,
        },
    }
    return module._exclusive_json(RECEIPT_PATH, receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--claim", action="store_true")
    mode.add_argument("--candidate-id", choices=AUTHORIZED_IDS)
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    module = _install()
    module.require(arguments.batch_size > 0, "batch size must be positive")
    if arguments.audit_only:
        audit = module._preclaim_audit(
            require_release=True, require_implementation=True
        )
        result = {
            "status": "preclaim_hash_and_json_structure_audit_passed",
            "release_id": RELEASE_ID,
            "authorized_candidate_ids": list(AUTHORIZED_IDS),
            "audit_sha256": audit["audit_sha256"],
            "checkpoint_deserialized_or_model_loaded": False,
            "prediction_latent_or_score_computed": False,
            "public_test_or_sealed_content_opened": False,
        }
    elif arguments.claim:
        result = module._claim()
    elif arguments.candidate_id:
        result = _score_candidate(
            arguments.candidate_id,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
    else:
        result = _finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
