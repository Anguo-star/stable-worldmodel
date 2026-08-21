#!/usr/bin/env python3
"""Build the one-use multi-seed Development release v2 for the predictor-only line.

This is a THIN OVERLAY over the frozen predictor-only v1 builder.  It changes
exactly four things and nothing else:

  1. the release identity and output root,
  2. the selector seeds (fresh eval seeds 124-129, catalog 2026081901,
     bootstrap 2026081902, disjoint from every prior release),
  3. the census, which now also excludes the consumed predictor-only v1 release
     (14 sources -> 15, 9 prior private releases -> 10),
  4. the authorized candidate set, which is THREE cells instead of one: the two
     replication seeds 4096 and 5120 plus the seed-3072 positive calibration.

Selection, generation, scoring, and gating are inherited unchanged.  Any
behavioural change to those invalidates the preregistration, so this module
must not reimplement them.

The builder is checkpoint blind.  It never opens a candidate checkpoint and
never computes a prediction or a score; candidate hashes enter only through the
preregistration, which pins them, and through a later create-once terminal
identity addendum.

Per AUDIT_CONTRACT.md:
  - section 1: content invariants are gated, positions are recorded.  The census
    counts declared in the preregistration are RE-DERIVED here and compared, so
    a wrong number in the YAML fails the build instead of being adopted.
  - section 3: shared constants are read from their authority (the parent
    builder module) rather than restated, so they cannot drift.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import yaml

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"

PARENT_BUILDER = THIS_SOURCE.with_name(
    "build_action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1.py"
)
CONFIG = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_development_v2.yaml"
)
AUDIT_CONTRACT = ROOT / "AUDIT_CONTRACT.md"
PROTOCOL_FREEZE = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_protocol_freeze.json"
)
IMPLEMENTATION_FREEZE = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_implementation_freeze.json"
)
TERMINAL_MANIFEST = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_terminal_identity_addendum_v1.json"
)
FREEZER = THIS_SOURCE.with_name(
    "freeze_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
TERMINAL_FREEZER = THIS_SOURCE.with_name(
    "freeze_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "terminal_v2.py"
)
EVALUATOR = THIS_SOURCE.with_name(
    "eval_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
FOCUSED_TEST = ROOT / (
    "tests/test_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)

EXPECTED_RELEASE_ID = "action-delay-h7-a0-aux-pcja-predictor-only-multiseed-private-v2"
EXPECTED_OUTPUT_ROOT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2"
)

EXPECTED_EVAL_SEEDS = (124, 125, 126, 127, 128, 129)
EXPECTED_CATALOG_SEED = 2_026_081_901
EXPECTED_BOOTSTRAP_SEED = 2_026_081_902
EXPECTED_QUERIES = 300

# One more consumed private release than v1: the v1 release itself, now spent.
EXPECTED_SOURCE_COUNT = 15
EXPECTED_PRIOR_PRIVATE_COUNT = 10

AUTHORIZED_IDS = (
    "action_delay_h7_a0_aux_pcja_predictor_only_s4096_v1",
    "action_delay_h7_a0_aux_pcja_predictor_only_s5120_v1",
    "historical-predictor-only-PCJA-s3072-step1024",
)
CALIBRATION_ID = "historical-predictor-only-PCJA-s3072-step1024"
REPLICATION_SEEDS = (4096, 5120)
CALIBRATION_SEED = 3072
TERMINAL_STEP = 1024

# The predictor-only v1 release, now a consumed source that must be excluded.
PRIOR_RELEASE_ID = "action-delay-h7-a0-aux-pcja-predictor-only-private-v1"
PRIOR_RELEASE_ROOT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1"
)
PRIOR_RELEASE_BINDING = {
    "catalog": (
        PRIOR_RELEASE_ROOT / "catalog.json",
        "a1d0f7584cf3b658cb3f4707f8ef63c68615f3b78dc0e42aed4854c45a49fa30",
    ),
    "release_identity": (
        PRIOR_RELEASE_ROOT / "release_identity.json",
        "5396900d5d181ba5ac3086fd2c967af1908b0dada9a759b07e362479428c8cf7",
    ),
    "consumption_receipt": (
        PRIOR_RELEASE_ROOT / "consumption_receipt.json",
        "d8d6f6ca4834e21e78a64639f4acd231ac0935fbeea42a92e237177a1097b5e7",
    ),
}
PRIOR_CONFIG = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1.yaml"
)
PRIOR_CONFIG_SHA256 = (
    "f57f2de07db43ce6f33499eb042d74a73213b49e4a09cda55884fe3ad887ec6d"
)
PRIOR_PROTOCOL = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1_"
    "protocol_freeze.json"
)
PRIOR_PROTOCOL_SHA256 = (
    "94488799f2582bcd31c581317059178f54a0ff5e3296baf17e3b00d8b9d097fa"
)

# The v3 source-rebind receipt: the closure is gated by content, HEAD is recorded.
V3_RECEIPT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1_"
    "source_rebind_addendum_v3/source_rebind_receipt.json"
)
V3_RECEIPT_CONTENT_SHA256 = (
    "eefbc58f4fb06fe899ac37122bcb98d785fc9c3ec517fdfe4b44f8d13d96439f"
)

# ---------------------------------------------------------------------------
# The two append-only addenda this release depends on.
#
# Neither the frozen v1 files nor the frozen v2 preregistration may be edited.
# The v2 preregistration was drafted without three sections the frozen builder
# chain requires, and the builder's own source-identity check gates on git HEAD
# ahead of its per-file hash loop.  Both are resolved append-only.
# ---------------------------------------------------------------------------

BUILD_SOURCE_GATE_CONFIG = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_build_source_gate_addendum_v1.yaml"
)
BUILD_SOURCE_GATE_RECEIPT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_build_source_gate_addendum_v1/build_source_gate_receipt.json"
)
BUILD_SOURCE_GATE_RECEIPT_CONTENT_SHA256 = (
    "f485cdc32c2a4035ea48ed4ed80801f120238dd2206dd63187095b8fb43da5e6"
)
CONTRACT_SECTIONS_CONFIG = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_contract_sections_addendum_v1.yaml"
)

# The binding chain advances by exactly one link: v1 becomes the inherited
# parent and the latest consumed release.  Derived from disk, pinned here.
PARENT_CONFIG_SHA256_V2 = (
    "f57f2de07db43ce6f33499eb042d74a73213b49e4a09cda55884fe3ad887ec6d"
)
PARENT_PROTOCOL_SHA256_V2 = (
    "94488799f2582bcd31c581317059178f54a0ff5e3296baf17e3b00d8b9d097fa"
)


def _load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


v1 = _load_module(PARENT_BUILDER, "_predictor_only_multiseed_v2_parent")

# Section 3 of the audit contract: take shared machinery from its authority.
ContractError = v1.parent.ContractError
require = v1.require
canonical_sha256 = v1.canonical_sha256
file_sha256 = v1.file_sha256
_json = v1._json
_regular_exact = v1._regular_exact
POST_FREEZE_DYNAMIC_ROOTS = v1.POST_FREEZE_DYNAMIC_ROOTS
_without_hash = v1._without_hash
_static_manifest = v1._static_manifest
_V1_PATCHED_V4 = v1._patched_v4
_V1_PRIOR_RELEASE_BINDINGS = v1.prior_release_bindings


def _binding_payload(
    binding: Mapping[str, tuple[Path, str]],
) -> dict[str, dict[str, str]]:
    return {
        role: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": digest,
        }
        for role, (path, digest) in sorted(binding.items())
    }


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"YAML mapping required: {path}")
    return value


def load_contract_sections_addendum() -> dict[str, Any]:
    """The append-only addendum supplying the three missing sections."""

    value = yaml.safe_load(
        _regular_exact(
            CONTRACT_SECTIONS_CONFIG, None, "contract-sections addendum"
        ).read_text(encoding="utf-8")
    )
    require(isinstance(value, dict), "contract-sections addendum must be a mapping")
    require(
        value.get("status") == "frozen_before_any_v2_release_build_claim_or_score"
        and value.get("overwrite") == "forbidden",
        "contract-sections addendum status changed",
    )
    closed = value.get("authority", {}).get("what_remains_closed", {})
    require(closed.get("any_new_training") is True, "training must remain closed")
    for key in (
        "evaluation_or_scoring_authorized",
        "checkpoint_may_be_opened",
        "one_use_consumption_rule_relaxed",
        "spent_v1_release_may_be_reconsumed",
        "candidate_set_may_change",
        "seeds_may_change",
        "gate_thresholds_may_change",
        "terminal_budget_may_change",
        "cross_seed_pooling_averaging_or_rescue",
        "selective_reporting",
        "rejecting_the_pairing_method_family",
    ):
        require(closed.get(key) is False, f"contract-sections authority widened: {key}")
    for section in ("training_contract", "runtime_rebind", "exclusion_contract_binding_chain"):
        require(isinstance(value.get(section), dict), f"addendum lacks {section}")
    return value


def effective_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The preregistration plus the append-only sections, merged in memory.

    Neither file on disk is modified.  The merge refuses to overwrite anything
    the preregistration already declares, so the addendum can only ADD.
    """

    config = load_config() if config is None else dict(config)
    addendum = load_contract_sections_addendum()
    merged = copy.deepcopy(dict(config))

    for section in ("training_contract", "runtime_rebind"):
        require(
            section not in merged,
            f"the preregistration already declares {section}; the addendum must only add",
        )
        merged[section] = copy.deepcopy(addendum[section])

    exclusion = merged.setdefault("exclusion_contract", {})
    require(
        "binding_chain" not in exclusion,
        "the preregistration already declares a binding chain",
    )
    exclusion["binding_chain"] = copy.deepcopy(
        addendum["exclusion_contract_binding_chain"]
    )

    # The addendum must not have moved anything the preregistration pinned.
    require(
        merged["release_id"] == EXPECTED_RELEASE_ID
        and tuple(merged["selector"]["eval_seeds"]) == EXPECTED_EVAL_SEEDS
        and int(merged["scoring"]["bootstrap_seed"]) == EXPECTED_BOOTSTRAP_SEED,
        "the merged config no longer matches the preregistration",
    )
    require(
        merged["training_contract"]["terminal_optimizer_step"] == TERMINAL_STEP
        and merged["training_contract"]["auxiliary_pcja_gradient_allowed"]
        == ["predictor", "pred_proj"]
        and merged["training_contract"]["auxiliary_pcja_gradient_blocked"]
        == ["encoder", "projector", "action_encoder"],
        "the addendum changed the training identity",
    )
    return merged


def validate_binding_chain(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The v2 binding chain: v1 is both the inherited parent and latest consumed.

    The frozen v1 builder's own _validate_config_binding pins v1's parent
    (cutoff512) and v1's nine-release list, so it cannot validate this release.
    This is the v2 equivalent, advancing the chain by exactly one link.
    """

    config = effective_config() if config is None else dict(config)
    chain = config.get("exclusion_contract", {}).get("binding_chain", {})

    expected_parent_config = {
        "path": PRIOR_CONFIG.relative_to(REPO_ROOT).as_posix(),
        "sha256": PARENT_CONFIG_SHA256_V2,
    }
    expected_parent_protocol = {
        "path": PRIOR_PROTOCOL.relative_to(REPO_ROOT).as_posix(),
        "sha256": PARENT_PROTOCOL_SHA256_V2,
    }
    expected_latest = {
        "release_id": PRIOR_RELEASE_ID,
        "catalog_sha256": PRIOR_RELEASE_BINDING["catalog"][1],
        "release_identity_sha256": PRIOR_RELEASE_BINDING["release_identity"][1],
        "consumption_receipt_sha256": PRIOR_RELEASE_BINDING["consumption_receipt"][1],
    }
    consumed = list(
        config.get("exclusion_contract", {}).get("consumed_private_releases", ())
    )
    checks = {
        "parent_config": chain.get("inherited_config") == expected_parent_config,
        "parent_protocol": chain.get("inherited_protocol_freeze")
        == expected_parent_protocol,
        "latest": chain.get("latest_consumed_release") == expected_latest,
        # Files on disk must still be what the chain says they are.
        "parent_files": file_sha256(PRIOR_CONFIG) == PARENT_CONFIG_SHA256_V2
        and file_sha256(PRIOR_PROTOCOL) == PARENT_PROTOCOL_SHA256_V2,
        "release_list": len(consumed) == EXPECTED_PRIOR_PRIVATE_COUNT
        and PRIOR_RELEASE_ID in consumed,
    }
    require(all(checks.values()), f"v2 binding chain changed: {checks}")
    return {
        "passed": True,
        "inherited_parent": PRIOR_RELEASE_ID,
        "latest_consumed_release": PRIOR_RELEASE_ID,
        "prior_private_release_count": EXPECTED_PRIOR_PRIVATE_COUNT,
        "source_count": EXPECTED_SOURCE_COUNT,
    }


def validate_build_source_gate() -> dict[str, Any]:
    """The build-time source gate is content, not position.

    The builder chain's verify_source_identity compares git HEAD to a pinned
    commit BEFORE its own per-file hash loop, so once HEAD moves the content
    check never runs.  The build-source-gate addendum demotes that comparison
    to a recorded fact; this reads its published receipt.
    """

    receipt = _json(
        _regular_exact(BUILD_SOURCE_GATE_RECEIPT, None, "build-source-gate receipt")
    )
    require(
        receipt.get("content_sha256") == BUILD_SOURCE_GATE_RECEIPT_CONTENT_SHA256,
        "build-source-gate receipt changed",
    )
    require(
        receipt.get("status")
        == "passed_build_source_gate_before_any_v2_release_build_claim_or_score",
        "build-source-gate receipt status changed",
    )
    gate = receipt.get("build_source_gate", {})
    require(
        gate.get("passed") is True
        and int(gate.get("total_file_count", -1)) == 45
        and int(gate.get("files_byte_identical", -1)) == 41
        and int(gate.get("unadjudicated_differences", -1)) == 0
        and gate.get("contextworld_head_gated") is False,
        "the build source gate is no longer the audited one",
    )
    return {
        "gate": gate.get("gate"),
        "total_file_count": gate.get("total_file_count"),
        "files_byte_identical": gate.get("files_byte_identical"),
        "unadjudicated_differences": 0,
        "head_recorded": gate.get("contextworld_head_recorded"),
        "head_gated": False,
    }


def validate_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Check the preregistration against this module before anything is built.

    Everything asserted here is a CONTENT invariant.  Census counts are
    re-derived from the consumed-release list rather than trusted, so a wrong
    number in the YAML fails closed.
    """

    config = load_config() if config is None else dict(config)

    require(config.get("release_id") == EXPECTED_RELEASE_ID, "release id changed")
    require(
        config.get("status")
        == "frozen_before_any_v2_release_build_claim_or_score",
        "preregistration status changed",
    )
    require(config.get("overwrite") == "forbidden", "overwrite protection removed")

    selector = config.get("selector", {})
    require(
        tuple(selector.get("eval_seeds", ())) == EXPECTED_EVAL_SEEDS,
        "eval seeds changed",
    )
    require(
        int(selector.get("catalog_seed", -1)) == EXPECTED_CATALOG_SEED
        and int(selector.get("bootstrap_seed", -1)) == EXPECTED_BOOTSTRAP_SEED,
        "selector seeds changed",
    )
    require(
        int(selector.get("expected_queries", -1)) == EXPECTED_QUERIES,
        "query count changed",
    )

    # Freshness is a content property: no eval seed may repeat a prior release.
    freshness = selector.get("freshness_evidence", {})
    prior_seeds = set(freshness.get("eval_seeds_used_by_all_prior_releases", ()))
    require(
        not (set(EXPECTED_EVAL_SEEDS) & prior_seeds),
        "eval seeds collide with a prior release",
    )
    require(
        EXPECTED_BOOTSTRAP_SEED
        not in set(freshness.get("bootstrap_seeds_used_by_all_prior_releases", ())),
        "bootstrap seed reused",
    )
    require(
        EXPECTED_CATALOG_SEED
        not in set(freshness.get("catalog_seeds_used_by_all_prior_releases", ())),
        "catalog seed reused",
    )

    # Three cells: two replication seeds plus the calibration.
    candidates = list(config.get("candidates", ()))
    calibration = config.get("positive_calibration", {})
    require(len(candidates) == 2, "replication candidate count changed")
    require(
        tuple(int(row["seed"]) for row in candidates) == REPLICATION_SEEDS,
        "replication seeds changed",
    )
    require(
        int(calibration.get("seed", -1)) == CALIBRATION_SEED,
        "calibration seed changed",
    )
    require(
        calibration.get("control_id") == CALIBRATION_ID,
        "calibration identity changed",
    )
    require(
        calibration.get("same_new_release_four_gate_pass_required") is True,
        "calibration gate requirement removed",
    )
    for row in (*candidates, calibration):
        require(
            int(row.get("optimizer_step", -1)) == TERMINAL_STEP,
            "terminal budget changed",
        )
    declared = tuple(row["candidate_id"] for row in candidates) + (CALIBRATION_ID,)
    require(declared == AUTHORIZED_IDS, "authorized candidate set changed")

    # Census counts are DERIVED, not adopted.
    exclusion = config.get("exclusion_contract", {})
    consumed = list(exclusion.get("consumed_private_releases", ()))
    require(
        PRIOR_RELEASE_ID in consumed,
        "the spent v1 release is not excluded by the new release",
    )
    require(
        len(consumed) == EXPECTED_PRIOR_PRIVATE_COUNT
        and int(exclusion.get("prior_private_release_count", -1))
        == EXPECTED_PRIOR_PRIVATE_COUNT,
        f"prior private release count must be {EXPECTED_PRIOR_PRIVATE_COUNT}, "
        f"declared {exclusion.get('prior_private_release_count')}, "
        f"listed {len(consumed)}",
    )
    require(
        int(exclusion.get("source_count", -1)) == EXPECTED_SOURCE_COUNT,
        f"source count must be {EXPECTED_SOURCE_COUNT}",
    )

    # Scoring must be inherited verbatim from v1.
    prior = yaml.safe_load(PRIOR_CONFIG.read_text(encoding="utf-8"))
    scoring = config.get("scoring", {})
    require(scoring.get("gates") == prior["scoring"]["gates"], "gate thresholds changed")
    for key in (
        "physical_group",
        "prediction_rule",
        "tie_break",
        "aggregation",
        "paired_bootstrap_resamples",
    ):
        require(scoring.get(key) == prior["scoring"][key], f"scoring rule changed: {key}")
    require(
        config.get("generation") == prior["generation"], "generation contract changed"
    )
    require(
        int(scoring.get("bootstrap_seed", -1)) == EXPECTED_BOOTSTRAP_SEED,
        "scoring bootstrap seed disagrees with the selector",
    )

    # Boundaries that make the numbers mean anything.
    decision = config.get("decision_contract", {})
    require(
        decision.get("cross_seed_pooling_performed") is False
        and decision.get("cross_seed_averaging_or_rescue_allowed") is False,
        "per-seed independence weakened",
    )
    require(
        decision.get("replication_pass")
        == "both_confirmation_seeds_individually_pass_all_four_gates",
        "replication rule changed",
    )
    require(
        decision.get("both_seeds_reported_regardless_of_outcome") is True,
        "selective reporting allowed",
    )
    require(
        decision.get("public_claim_authorized") is False
        and decision.get("cem_claim_authorized") is False,
        "claim boundary widened",
    )
    consumption = config.get("consumption_contract", {})
    require(int(consumption.get("expected_models", -1)) == 3, "scoring batch changed")
    require(
        consumption.get("candidate_additions_after_consumption") == "forbidden",
        "one-use contract weakened",
    )
    authority = config.get("authority", {}).get("does_not_authorize", {})
    for key in (
        "any_new_training",
        "reopening_or_re_consuming_the_v1_release",
        "editing_any_v1_file_receipt_or_result",
        "public_or_test_access",
        "cross_seed_pooling_averaging_or_rescue",
        "rejecting_the_pairing_method_family",
    ):
        require(authority.get(key) is True, f"authority widened: {key}")

    scope = config.get("scope", {})
    require(scope.get("one_use") is True, "release is no longer one-use")
    require(
        scope.get("release_built_checkpoint_blind_before_terminal_identity") is True,
        "checkpoint blindness dropped",
    )
    require(
        scope.get("preclaim_prediction_latent_or_score_forbidden") is True,
        "preclaim scoring allowed",
    )
    return config


def validate_prior_release_is_spent() -> dict[str, Any]:
    """The v1 release must be present, unmodified, and consumed.

    This release inherits v1's exclusion metadata, so v1 has to be exactly the
    artifact the preregistration thinks it is.  It is read, never written.
    """

    for role, (path, digest) in PRIOR_RELEASE_BINDING.items():
        _regular_exact(path, digest, f"consumed predictor-only v1 {role}")

    catalog = _json(PRIOR_RELEASE_BINDING["catalog"][0])
    identity = _json(PRIOR_RELEASE_BINDING["release_identity"][0])
    receipt = _json(PRIOR_RELEASE_BINDING["consumption_receipt"][0])

    require(catalog.get("release_id") == PRIOR_RELEASE_ID, "v1 catalog identity changed")
    require(identity.get("release_id") == PRIOR_RELEASE_ID, "v1 identity changed")
    require(
        receipt.get("release_id") == PRIOR_RELEASE_ID
        and "consumed" in str(receipt.get("status", "")),
        "v1 is not consumed; this release assumes it is spent",
    )
    require(
        receipt.get("scope", {}).get("candidate_additions_after_consumption")
        == "forbidden",
        "v1 remains open to additions",
    )
    require(
        len(catalog.get("queries", ())) == EXPECTED_QUERIES,
        "v1 catalog query count changed",
    )
    _regular_exact(PRIOR_CONFIG, PRIOR_CONFIG_SHA256, "v1 preregistration")
    _regular_exact(PRIOR_PROTOCOL, PRIOR_PROTOCOL_SHA256, "v1 protocol freeze")
    return {
        "release_id": PRIOR_RELEASE_ID,
        "status": receipt.get("status"),
        "mutated_by_this_build": False,
        "queries": len(catalog["queries"]),
    }


def validate_source_closure() -> dict[str, Any]:
    """The scoring source closure is gated by content; HEAD is recorded only."""

    receipt = _json(_regular_exact(V3_RECEIPT, None, "v3 source-rebind receipt"))
    require(
        receipt.get("content_sha256") == V3_RECEIPT_CONTENT_SHA256,
        "v3 source-rebind receipt changed",
    )
    require(
        receipt.get("status")
        == "passed_content_closure_rebind_before_any_new_release_claim_or_score",
        "v3 receipt status changed",
    )
    closure = receipt.get("source_rebind", {})
    require(
        int(closure.get("total_file_count", -1)) == 45
        and int(closure.get("files_byte_identical", -1)) == 41
        and not closure.get("unadjudicated_differences"),
        "the scoring source closure is no longer the audited one",
    )
    return {
        "gate": closure.get("gate"),
        "total_file_count": closure.get("total_file_count"),
        "files_byte_identical": closure.get("files_byte_identical"),
        "adjudicated_differences": len(closure.get("adjudicated_differences", ())),
        "unadjudicated_differences": 0,
        "head_recorded": closure.get("head_before"),
        "head_gated": False,
    }


def preflight(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate everything that can be validated before a single asset exists."""

    config = validate_config(config)
    merged = effective_config(config)
    binding = validate_binding_chain(merged)
    build_gate = validate_build_source_gate()
    prior = validate_prior_release_is_spent()
    closure = validate_source_closure()
    require(AUDIT_CONTRACT.is_file(), "audit contract missing")
    require(
        not EXPECTED_OUTPUT_ROOT.exists(),
        f"refusing to overwrite an existing release: {EXPECTED_OUTPUT_ROOT}",
    )
    # The frozen builder chain's own section census must now be satisfiable.
    sections = v1._contract_sections(merged)
    missing = [name for name in sections if name not in merged]
    require(not missing, f"contract sections still missing after merge: {missing}")
    v1.parent._validate_runtime_rebind(merged)
    return {
        "status": "preflight_passed_nothing_built",
        "release_id": EXPECTED_RELEASE_ID,
        "config_sha256": file_sha256(CONFIG),
        "contract_sections_addendum_sha256": file_sha256(CONTRACT_SECTIONS_CONFIG),
        "build_source_gate_addendum_sha256": file_sha256(BUILD_SOURCE_GATE_CONFIG),
        "contract_sections_present": list(sections),
        "runtime_rebind_validated_by_frozen_chain": True,
        "binding_chain": binding,
        "build_source_gate": build_gate,
        "eval_seeds": list(EXPECTED_EVAL_SEEDS),
        "catalog_seed": EXPECTED_CATALOG_SEED,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
        "expected_queries": EXPECTED_QUERIES,
        "source_count": EXPECTED_SOURCE_COUNT,
        "prior_private_release_count": EXPECTED_PRIOR_PRIVATE_COUNT,
        "authorized_candidate_ids": list(AUTHORIZED_IDS),
        "expected_models": 3,
        "spent_v1_release": prior,
        "source_closure": closure,
        "release_root_exists": False,
        "release_assets_generated": 0,
        "candidate_checkpoint_deserialized_or_model_loaded": False,
        "prediction_latent_or_score_computed": False,
    }


def prior_release_bindings() -> dict[str, dict[str, dict[str, str]]]:
    """Advance the inherited nine-release binding set by the spent v1 release."""

    inherited = _V1_PRIOR_RELEASE_BINDINGS()
    require(PRIOR_RELEASE_ID not in inherited, "spent v1 release already inherited")
    bindings = {
        **inherited,
        PRIOR_RELEASE_ID: _binding_payload(PRIOR_RELEASE_BINDING),
    }
    # The legacy mapping stores the five releases added after the base private
    # builder; the other five are bound by that base config itself.
    require(len(bindings) == 5, "explicit private-release binding census changed")
    return bindings


def _prior_metadata(selector: Any) -> tuple[set[str], set[str]]:
    for role, (path, digest) in PRIOR_RELEASE_BINDING.items():
        _regular_exact(path, digest, f"spent predictor-only v1 {role}")
    catalog = _json(PRIOR_RELEASE_BINDING["catalog"][0])
    require(
        catalog.get("release_id") == PRIOR_RELEASE_ID
        and len(catalog.get("queries", ())) == EXPECTED_QUERIES,
        "spent predictor-only v1 catalog changed",
    )
    reset = {
        selector.reset_state_sha256(row["template"]["reset_state"])
        for row in catalog["queries"]
    }
    simulator = {
        str(int(row["template"]["simulator_seed"])) for row in catalog["queries"]
    }
    require(
        len(reset) == len(simulator) == EXPECTED_QUERIES,
        "spent predictor-only v1 metadata repeats",
    )
    return reset, simulator


def metadata_census() -> dict[str, Any]:
    inherited = v1.metadata_census()
    selector = v1.parent.parent.parent._selector_module()
    reset, simulator = _prior_metadata(selector)
    sources = dict(inherited["sources"])
    require(PRIOR_RELEASE_ID not in sources, "spent v1 source repeats")
    sources[PRIOR_RELEASE_ID] = (reset, simulator)
    require(len(sources) == EXPECTED_SOURCE_COUNT, "fifteen-source census incomplete")
    all_reset = set().union(*(value[0] for value in sources.values()))
    all_simulator = set().union(*(value[1] for value in sources.values()))
    summary = {
        name: {
            "reset_state_count": len(value[0]),
            "reset_state_sha256": canonical_sha256(sorted(value[0])),
            "simulator_seed_count": len(value[1]),
            "simulator_seed_sha256": canonical_sha256(sorted(value[1])),
        }
        for name, value in sorted(sources.items())
    }
    identity = {
        "schema_version": 1,
        "protocol": "fifteen_source_checkpoint_blind_metadata_census_v1",
        "source_count": EXPECTED_SOURCE_COUNT,
        "prior_private_release_count": EXPECTED_PRIOR_PRIVATE_COUNT,
        "source_summary": summary,
        "inherited_fourteen_source_identity": inherited["metadata_census_sha256"],
        "tenth_private_release_binding": _binding_payload(PRIOR_RELEASE_BINDING),
        "reset_state_count": len(all_reset),
        "reset_state_sha256": canonical_sha256(sorted(all_reset)),
        "simulator_seed_count": len(all_simulator),
        "simulator_seed_sha256": canonical_sha256(sorted(all_simulator)),
    }
    return {
        "reset_state": all_reset,
        "simulator_seed": all_simulator,
        "sources": sources,
        "identity": identity,
        "metadata_census_sha256": canonical_sha256(identity),
    }


@contextmanager
def _fresh_selector_constants(module: Any) -> Iterator[None]:
    values = {
        "EXPECTED_EVAL_SEEDS": EXPECTED_EVAL_SEEDS,
        "EXPECTED_CATALOG_SEED": EXPECTED_CATALOG_SEED,
        "EXPECTED_BOOTSTRAP_SEED": EXPECTED_BOOTSTRAP_SEED,
        "EXPECTED_QUERIES": EXPECTED_QUERIES,
    }
    previous = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


def selector_preflight(
    config: Mapping[str, Any] | None = None, *, verify_protocol: bool = True
) -> dict[str, Any]:
    config = effective_config() if config is None else dict(config)
    census = metadata_census()
    selector = v1.parent.parent.parent._selector_module()
    with _fresh_selector_constants(selector):
        selected = selector._selector(config, census, exclude_metadata=True)
    overlaps = {
        name: {
            "reset_state_overlap_count": len(
                selected["selected_reset_state"] & value[0]
            ),
            "simulator_seed_overlap_count": len(
                selected["selected_simulator_seed"] & value[1]
            ),
        }
        for name, value in sorted(census["sources"].items())
    }
    require(len(overlaps) == EXPECTED_SOURCE_COUNT, "selector census changed")
    require(
        all(count == 0 for row in overlaps.values() for count in row.values()),
        "selector overlaps a frozen source",
    )
    observed = {
        "source_count": EXPECTED_SOURCE_COUNT,
        "prior_private_release_count": EXPECTED_PRIOR_PRIVATE_COUNT,
        "reset_state_census_count": len(census["reset_state"]),
        "simulator_seed_census_count": len(census["simulator_seed"]),
        "metadata_census_sha256": census["metadata_census_sha256"],
        "selector_rows_sha256": selected["selector_rows_sha256"],
        "selected_reset_state_list_sha256": selected[
            "selected_reset_state_list_sha256"
        ],
        "selected_simulator_seed_list_sha256": selected[
            "selected_simulator_seed_list_sha256"
        ],
        "rejection_ledger_sha256": selected["rejection_ledger_sha256"],
        "rejection_events": len(selected["rejection_ledger"]),
        "rejection_reason_counts": dict(
            Counter(
                reason
                for row in selected["rejection_ledger"]
                for reason in row["reasons"]
            )
        ),
        "final_reset_state_overlap_count": sum(
            row["reset_state_overlap_count"] for row in overlaps.values()
        ),
        "final_simulator_seed_overlap_count": sum(
            row["simulator_seed_overlap_count"] for row in overlaps.values()
        ),
        "candidate_output_used": False,
    }
    if verify_protocol and PROTOCOL_FREEZE.exists():
        protocol = _json(PROTOCOL_FREEZE)
        require(
            protocol.get("selector_preflight") == observed,
            "selector differs from frozen protocol",
        )
    return {
        "status": "passed",
        "release_id": EXPECTED_RELEASE_ID,
        "eval_seeds": list(EXPECTED_EVAL_SEEDS),
        "queries": EXPECTED_QUERIES,
        "rows": selected["rows"],
        "overlaps": overlaps,
        "frozen_preflight": observed,
        "candidate_terminal_artifacts_opened": [],
        "release_assets_generated": 0,
        "model_loaded_or_inference_performed": False,
        "score_computed_or_observed": False,
    }


@contextmanager
def _patched_v4(module: Any) -> Iterator[None]:
    with _V1_PATCHED_V4(module):
        previous = {
            "PRIOR_RELEASES": module.PRIOR_RELEASES,
            "EXPECTED_SOURCES": module.EXPECTED_SOURCES,
            "EXPECTED_RELEASE_ID": module.EXPECTED_RELEASE_ID,
            "EXPECTED_OUTPUT_ROOT": module.EXPECTED_OUTPUT_ROOT,
            "EXPECTED_EVAL_SEEDS": module.EXPECTED_EVAL_SEEDS,
            "EXPECTED_QUERIES": getattr(module, "EXPECTED_QUERIES", None),
        }
        try:
            module.PRIOR_RELEASES = {
                **module.PRIOR_RELEASES,
                PRIOR_RELEASE_ID: v1.parent._legacy_binding(PRIOR_RELEASE_BINDING),
            }
            module.EXPECTED_SOURCES = module.EXPECTED_SOURCES | {PRIOR_RELEASE_ID}
            module.EXPECTED_RELEASE_ID = EXPECTED_RELEASE_ID
            module.EXPECTED_OUTPUT_ROOT = EXPECTED_OUTPUT_ROOT
            module.EXPECTED_EVAL_SEEDS = EXPECTED_EVAL_SEEDS
            if previous["EXPECTED_QUERIES"] is not None:
                module.EXPECTED_QUERIES = EXPECTED_QUERIES
            yield
        finally:
            for name, value in previous.items():
                if value is not None:
                    setattr(module, name, value)


def _effective_config_with_rebind(
    v4: Any, config: Mapping[str, Any]
) -> dict[str, Any]:
    config = dict(config)
    old_bindings = v1.prior_release_bindings
    old_validator = v1.parent._validate_config_binding
    try:
        v1.prior_release_bindings = prior_release_bindings
        v1.parent._validate_config_binding = lambda value: validate_binding_chain(value)
        effective = v1._effective_config_with_rebind(v4, config)
    finally:
        v1.prior_release_bindings = old_bindings
        v1.parent._validate_config_binding = old_validator
    return effective


def _content_gated_source_identity(
    v4: Any, effective: Mapping[str, Any]
) -> dict[str, Any]:
    """Run every inherited byte check while recording, not gating, git HEAD."""

    receipt = _json(BUILD_SOURCE_GATE_RECEIPT)
    gate = receipt["build_source_gate"]
    patched = copy.deepcopy(dict(effective))
    adjudicated = dict(gate.get("adjudicated_observed_sha256", {}))
    require(len(adjudicated) == 4, "adjudicated source set changed")
    for relative, digest in adjudicated.items():
        require(relative in patched["source_hashes"], f"unknown adjudication: {relative}")
        patched["source_hashes"][relative] = digest

    base = v4.base
    contextworld_root = base.resolve_repo_path(
        patched["contextworld_root"], purpose="ContextWorld root"
    ).resolve()
    native_git_head = base._git_head
    native_verify = base.verify_source_identity
    head_before = native_git_head(contextworld_root)

    def recorded_not_gated(repo: Path) -> str:
        if Path(repo).resolve() == contextworld_root:
            return str(patched["contextworld_commit"])
        return native_git_head(repo)

    try:
        base._git_head = recorded_not_gated
        identity = native_verify(patched)
    finally:
        base._git_head = native_git_head
    head_after = native_git_head(contextworld_root)
    require(head_before == head_after, "ContextWorld changed during source audit")
    identity.update(
        {
            "contextworld_commit": head_after,
            "contextworld_commit_recorded_not_gated": True,
            "inherited_contextworld_commit_provenance": patched[
                "contextworld_commit"
            ],
            "build_source_gate_receipt_content_sha256": receipt[
                "content_sha256"
            ],
            "adjudicated_source_count": len(adjudicated),
            "unadjudicated_differences": 0,
        }
    )
    return identity


def source_rebind_preflight(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = effective_config() if config is None else dict(config)
    v4 = v1.parent.parent.parent._v4()
    with _patched_v4(v4):
        identity = _content_gated_source_identity(
            v4, _effective_config_with_rebind(v4, config)
        )
    files = identity["files"]
    return {
        "status": "passed",
        "protocol": "content_gated_45_file_closure_recorded_head_v1",
        "contextworld_head_recorded": identity["contextworld_commit"],
        "contextworld_head_gated": False,
        "contextworld_executable_file_count": len(files["contextworld"]),
        "contextworld_executable_tree_sha256": canonical_sha256(
            files["contextworld"]
        ),
        "overlap_source_count": len(files["overlap_sources"]),
        "overlap_source_tree_sha256": canonical_sha256(files["overlap_sources"]),
        "renderer_dependency_count": len(files["renderer_dependencies"]),
        "renderer_dependency_tree_sha256": canonical_sha256(
            files["renderer_dependencies"]
        ),
        "scoring_runtime_dependency_count": len(
            files["scoring_runtime_dependencies"]
        ),
        "scoring_runtime_dependency_tree_sha256": canonical_sha256(
            files["scoring_runtime_dependencies"]
        ),
        "adjudicated_source_count": identity["adjudicated_source_count"],
        "unadjudicated_differences": 0,
        "candidate_terminal_artifacts_opened": [],
        "prediction_latent_or_score_computed": False,
        "public_or_test_results_opened": False,
    }


def validate_protocol_freeze(
    config: Mapping[str, Any] | None = None, *, check_selector: bool = True
) -> dict[str, Any]:
    raw = load_config() if config is None else dict(config)
    validate_config(raw)
    merged = effective_config(raw)
    validate_binding_chain(merged)
    protocol = _json(_regular_exact(PROTOCOL_FREEZE, None, "protocol freeze"))
    require(
        protocol.get("content_sha256")
        == canonical_sha256(_without_hash(protocol)),
        "protocol self-hash changed",
    )
    require(
        protocol.get("status")
        == "frozen_checkpoint_blind_before_multiseed_release_build_claim_or_score",
        "protocol status changed",
    )
    require(
        protocol.get("preregistration", {}).get("sha256") == file_sha256(CONFIG),
        "protocol/config binding changed",
    )
    require(
        protocol.get("contract_sections_addendum", {}).get("sha256")
        == file_sha256(CONTRACT_SECTIONS_CONFIG),
        "protocol/addendum binding changed",
    )
    for section in v1._contract_sections(merged):
        require(
            protocol["frozen_contract_hashes"][section]
            == canonical_sha256(merged[section]),
            f"frozen section changed: {section}",
        )
    if check_selector:
        require(
            protocol.get("selector_preflight")
            == selector_preflight(merged, verify_protocol=False)["frozen_preflight"],
            "selector freeze changed",
        )
        require(
            protocol.get("source_rebind_preflight")
            == source_rebind_preflight(merged),
            "source-content freeze changed",
        )
    return protocol


def _implementation_sources() -> dict[str, dict[str, str]]:
    paths = {
        "builder": THIS_SOURCE,
        "evaluator": EVALUATOR,
        "freezer": FREEZER,
        "terminal_freezer": TERMINAL_FREEZER,
        "focused_test": FOCUSED_TEST,
        "parent_builder": PARENT_BUILDER,
        "parent_evaluator": THIS_SOURCE.with_name(
            "eval_action_delay_h7_a0_aux_pcja_predictor_only_private_"
            "development_v1.py"
        ),
        "build_source_gate_auditor": THIS_SOURCE.with_name(
            "audit_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_"
            "private_development_v2_build_source_gate_v1.py"
        ),
        "audit_contract": AUDIT_CONTRACT,
    }
    return {
        name: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(_regular_exact(path, None, name)),
        }
        for name, path in paths.items()
    }


def validate_implementation_freeze(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del config
    freeze = _json(
        _regular_exact(IMPLEMENTATION_FREEZE, None, "implementation freeze")
    )
    require(
        freeze.get("content_sha256") == canonical_sha256(_without_hash(freeze)),
        "implementation self-hash changed",
    )
    require(
        freeze.get("config_sha256") == file_sha256(CONFIG),
        "implementation/config binding changed",
    )
    require(
        freeze.get("protocol_freeze_sha256") == file_sha256(PROTOCOL_FREEZE),
        "implementation/protocol binding changed",
    )
    require(
        freeze.get("implementation_sources") == _implementation_sources(),
        "implementation sources changed",
    )
    require(
        freeze.get("build_contract")
        == "fresh_generation_only_no_recovery_input",
        "build contract changed",
    )
    return freeze


def validate_terminal_addendum(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del config
    manifest = _json(
        _regular_exact(TERMINAL_MANIFEST, None, "terminal identity addendum")
    )
    require(
        manifest.get("content_sha256")
        == canonical_sha256(_without_hash(manifest)),
        "terminal manifest self-hash changed",
    )
    require(
        manifest.get("status")
        == "frozen_hash_only_terminal_identity_after_release_build_before_claim_or_score",
        "terminal manifest status changed",
    )
    require(
        manifest.get("release_id") == EXPECTED_RELEASE_ID,
        "terminal release changed",
    )
    require(
        tuple(row.get("candidate_id") for row in manifest.get("cells", ()))
        == AUTHORIZED_IDS,
        "terminal candidate population changed",
    )
    require(
        manifest.get("scope", {}).get("checkpoint_deserialized_or_model_loaded")
        is False
        and manifest.get("scope", {}).get("prediction_latent_or_score_computed")
        is False,
        "terminal manifest crossed the claim boundary",
    )
    return manifest


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value["exclusion_contract"]["consumed_private_release_bindings"] = (
        prior_release_bindings()
    )
    return value


def _generation_assignments(
    generation: Any, rows: Sequence[Mapping[str, Any]]
) -> list[Any]:
    return [
        generation.ActionDelayH7ValidationAssignment(
            query_id=row["query_id"],
            eval_seed=row["eval_seed"],
            evaluation_index=row["evaluation_index"],
            room=row["room"],
            template=generation.LongHistoryDelayTemplate(
                template_id=row["query_id"],
                direction=row["direction"],
                reset_state=tuple(row["reset_state"]),
                goal_state=tuple(row["goal_state"]),
                simulator_seed=row["simulator_seed"],
            ),
        )
        for row in rows
    ]


@contextmanager
def _patched_generation(
    generation: Any, assignments: Sequence[Any]
) -> Iterator[None]:
    old = (
        generation.EVAL_SEEDS,
        generation.QUERY_COUNT,
        generation.select_validation_assignments,
    )
    try:
        generation.EVAL_SEEDS = EXPECTED_EVAL_SEEDS
        generation.QUERY_COUNT = EXPECTED_QUERIES
        generation.select_validation_assignments = lambda _config: list(assignments)
        yield
    finally:
        (
            generation.EVAL_SEEDS,
            generation.QUERY_COUNT,
            generation.select_validation_assignments,
        ) = old


def _finalize_staging(
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    implementation: Mapping[str, Any],
    selector: Mapping[str, Any],
    v4: Any,
    effective: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    raw_catalog: dict[str, Any],
    replay: dict[str, Any],
    staging_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    runtime_config = _runtime_config(config)
    counts = replay.get("counts", {})
    require(replay.get("status") == "passed", "deterministic replay failed")
    require(
        int(counts.get("assets_reopened", -1)) == EXPECTED_QUERIES,
        "asset replay count changed",
    )
    require(
        int(counts.get("physical_rollouts_replayed", -1)) == 3300
        and int(counts.get("failures", -1)) == 0,
        "physical replay failed",
    )
    catalog = v4.enrich_catalog(
        raw_catalog,
        staging_root=staging_root,
        final_root=final_root,
        config=runtime_config,
    )
    overlap = v4.full_overlap_audit(
        runtime_config,
        effective,
        catalog,
        contextworld_root=Path(source_identity["contextworld_root"]),
    )
    comparisons = overlap.get("comparisons")
    require(overlap.get("passed") is True, "full fifteen-source overlap failed")
    require(
        isinstance(comparisons, dict)
        and len(comparisons) == EXPECTED_SOURCE_COUNT
        and all(row.get("passed") is True for row in comparisons.values()),
        "full overlap audit changed",
    )
    overlap.update(
        {
            "schema_version": 8,
            "release_id": EXPECTED_RELEASE_ID,
            "audit_protocol": (
                "full_delay_fifteen_source_role_aware_conditional_disjointness_v1"
            ),
            "required_source_count": EXPECTED_SOURCE_COUNT,
            "direct_training_source_count": 3,
            "consumed_evaluation_source_count": 12,
        }
    )
    v4.base.atomic_write_json(staging_root / "catalog.json", catalog)
    v4.base.atomic_write_json(staging_root / "overlap_audit.json", overlap)
    exclusion = v4.base._load_json(staging_root / "training_exclusion_manifest.json")
    exclusion.update(
        {
            "release_id": EXPECTED_RELEASE_ID,
            "private_catalog_content_manifest_sha256": catalog[
                "content_manifest_sha256"
            ],
        }
    )
    v4.base.atomic_write_json(
        staging_root / "training_exclusion_manifest.json", exclusion
    )
    replay["selector_preflight"] = {
        key: value for key, value in selector.items() if key != "rows"
    }
    replay["publication_build_mode"] = (
        "fresh_generation_and_postfreeze_full_audit_v1"
    )
    v4.base.atomic_write_json(staging_root / "audit_report.json", replay)
    build_report = {
        "schema_version": 8,
        "status": "passed",
        "release_id": EXPECTED_RELEASE_ID,
        "publication_build_mode": "fresh_generation_and_postfreeze_full_audit_v1",
        "source_identity": dict(source_identity),
        "selector_preflight": selector["frozen_preflight"],
        "deterministic_replay_passed_after_current_implementation_freeze": True,
        "fifteen_source_overlap_passed_after_current_implementation_freeze": True,
        "candidate_terminal_artifacts_opened": [],
        "checkpoint_deserialized_or_model_loaded": False,
        "prediction_latent_or_score_computed": False,
        "public_or_test_results_opened": False,
    }
    v4.base.atomic_write_json(staging_root / "build_report.json", build_report)
    payload = _static_manifest(staging_root)
    body = {
        "schema_version": 8,
        "status": "frozen_before_predictor_only_multiseed_model_open_inference_or_scoring",
        "release_id": EXPECTED_RELEASE_ID,
        "publication_build_mode": "fresh_generation_and_postfreeze_full_audit_v1",
        "config_sha256": file_sha256(CONFIG),
        "protocol_freeze_sha256": file_sha256(PROTOCOL_FREEZE),
        "protocol_freeze_content_sha256": protocol["content_sha256"],
        "implementation_freeze_sha256": file_sha256(IMPLEMENTATION_FREEZE),
        "implementation_freeze_content_sha256": implementation["content_sha256"],
        "terminal_identity_addendum_absent_at_release_build": True,
        "terminal_identity_addendum_required_before_claim": True,
        "implementation_sources": _implementation_sources(),
        "selector_preflight": selector["frozen_preflight"],
        "catalog_sha256": file_sha256(staging_root / "catalog.json"),
        "catalog_content_manifest_sha256": catalog["content_manifest_sha256"],
        "build_report_sha256": file_sha256(staging_root / "build_report.json"),
        "audit_report_sha256": file_sha256(staging_root / "audit_report.json"),
        "overlap_audit_sha256": file_sha256(staging_root / "overlap_audit.json"),
        "training_exclusion_manifest_sha256": file_sha256(
            staging_root / "training_exclusion_manifest.json"
        ),
        "full_overlap_source_count": EXPECTED_SOURCE_COUNT,
        "prior_private_release_count": EXPECTED_PRIOR_PRIVATE_COUNT,
        "consumed_private_release_bindings": prior_release_bindings(),
        "authorized_candidate_ids": list(AUTHORIZED_IDS),
        "published_payload_file_count": len(payload),
        "published_payload_bytes": sum(row["bytes"] for row in payload.values()),
        "published_payload_tree_root_sha256": canonical_sha256(payload),
        "release_identity_is_final_commit_marker": True,
        "candidate_terminal_artifacts_opened": [],
        "checkpoint_deserialized_or_model_loaded": False,
        "prediction_latent_or_score_computed": False,
        "public_or_test_results_opened": False,
    }
    identity = {**body, "identity_sha256": canonical_sha256(body)}
    v4.publication._write_exclusive_bytes(
        staging_root / "release_identity.json",
        (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode(),
        mode=0o400,
    )
    published = v4.publication._publish_release_tree_exclusive(
        staging_root, final_root, commit_name="release_identity.json"
    )
    return {
        "status": identity["status"],
        "release_root": str(final_root),
        "release_identity_sha256": identity["identity_sha256"],
        "publication_build_mode": body["publication_build_mode"],
        "publication": published,
        "candidate_terminal_artifacts_opened": [],
    }


def build_release(config: Mapping[str, Any], *, workers: int) -> dict[str, Any]:
    raw = dict(config)
    merged = effective_config(raw)
    protocol = validate_protocol_freeze(raw)
    implementation = validate_implementation_freeze(raw)
    require(not TERMINAL_MANIFEST.exists(), "release must precede terminal identity")
    selector = selector_preflight(merged)
    source_rebind_preflight(merged)
    v4 = v1.parent.parent.parent._v4()
    with _patched_v4(v4):
        effective = _effective_config_with_rebind(v4, merged)
        source_identity = _content_gated_source_identity(v4, effective)
        final_root = EXPECTED_OUTPUT_ROOT.resolve()
        require(not final_root.exists(), f"refusing to overwrite release: {final_root}")
        final_root.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = Path(
            tempfile.mkdtemp(prefix=f".{final_root.name}.", dir=final_root.parent)
        )
        staging_root = staging_parent / "release"
        generation = v4.base._load_contextworld_generation(effective)
        assignments = _generation_assignments(generation, selector["rows"])
        with _patched_generation(generation, assignments):
            upstream = v4.base._upstream_config(effective)
            catalog, _exclusion, _report = generation.build_validation_release(
                config=upstream,
                repo_root=Path(source_identity["contextworld_root"]),
                output_root=staging_root,
                workers=int(workers),
                stable_worldmodel_commit=str(effective["stablewm_commit"]),
            )
            replay = generation.audit_validation_release(
                config=upstream,
                repo_root=Path(source_identity["contextworld_root"]),
                catalog_path=staging_root / "catalog.json",
                replay_physics=True,
            )
        return _finalize_staging(
            config=merged,
            protocol=protocol,
            implementation=implementation,
            selector=selector,
            v4=v4,
            effective=effective,
            source_identity=source_identity,
            raw_catalog=catalog,
            replay=replay,
            staging_root=staging_root,
            final_root=final_root,
        )


def validate_release_identity(
    release_root: Path = EXPECTED_OUTPUT_ROOT,
) -> dict[str, Any]:
    validate_protocol_freeze()
    validate_implementation_freeze()
    root = release_root.resolve()
    require(root == EXPECTED_OUTPUT_ROOT.resolve(), "release root changed")
    identity = _json(_regular_exact(root / "release_identity.json", None, "release identity"))
    require(
        identity.get("identity_sha256")
        == canonical_sha256(_without_hash(identity)),
        "release identity self-hash changed",
    )
    require(
        identity.get("schema_version") == 8
        and identity.get("status")
        == "frozen_before_predictor_only_multiseed_model_open_inference_or_scoring",
        "release status changed",
    )
    require(
        identity.get("release_id") == EXPECTED_RELEASE_ID
        and identity.get("authorized_candidate_ids") == list(AUTHORIZED_IDS),
        "release scope changed",
    )
    require(
        identity.get("terminal_identity_addendum_absent_at_release_build") is True,
        "release was not checkpoint blind",
    )
    require(
        identity.get("full_overlap_source_count") == EXPECTED_SOURCE_COUNT
        and identity.get("prior_private_release_count")
        == EXPECTED_PRIOR_PRIVATE_COUNT,
        "release exclusion census changed",
    )
    require(
        identity.get("consumed_private_release_bindings")
        == prior_release_bindings(),
        "prior release bindings changed",
    )
    payload = _static_manifest(root)
    require(
        canonical_sha256(payload) == identity["published_payload_tree_root_sha256"],
        "release payload tree changed",
    )
    return {
        "passed": True,
        "release_root": str(root),
        "release_identity": identity,
        "queries": EXPECTED_QUERIES,
        "candidate_terminal_artifacts_opened": [],
        "public_or_test_results_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--audit-existing", action="store_true")
    options = parser.parse_args(argv)
    require(
        options.config.resolve() == CONFIG.resolve(), "formal config path changed"
    )
    config = load_config()
    if options.audit_existing:
        result = validate_release_identity()
    elif options.preflight_only:
        result = preflight(config)
        selection = selector_preflight(effective_config(config))
        selection.pop("rows", None)
        result["selector_preflight"] = selection
    else:
        require(options.workers > 0, "workers must be positive")
        result = build_release(config, workers=options.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
