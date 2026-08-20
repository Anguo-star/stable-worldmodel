#!/usr/bin/env python3
"""Audit the build-source gate for the predictor-only multi-seed release v2.

This replaces the builder's git-HEAD equality check with the content invariant
it was standing in for: the 45-file scoring source closure must be byte
identical except for the four files already adjudicated in the v3 addendum,
which must sit at exactly the digests recorded there.

Per AUDIT_CONTRACT.md:
  - section 1: the content check is the gate.  HEAD is read twice, required to
    be stable across the read, and RECORDED.  It is never compared to a
    predetermined value.
  - section 3: the closure is obtained by CALLING the frozen v3 auditor rather
    than by restating its file list here.  Two copies of a file set is a
    future drift point; there is one authority.

This module never opens a candidate checkpoint, never loads a model, never
computes a prediction or a score, and never builds or mutates a release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"

CONFIG = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_build_source_gate_addendum_v1.yaml"
)
V3_AUDITOR = THIS_SOURCE.with_name(
    "audit_action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1_"
    "source_rebind_addendum_v3.py"
)
V3_RECEIPT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1_"
    "source_rebind_addendum_v3/source_rebind_receipt.json"
)
V3_RECEIPT_CONTENT_SHA256 = (
    "eefbc58f4fb06fe899ac37122bcb98d785fc9c3ec517fdfe4b44f8d13d96439f"
)
AUDIT_CONTRACT = ROOT / "AUDIT_CONTRACT.md"
V2_PREREGISTRATION = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_development_v2.yaml"
)
OUTPUT_ROOT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_build_source_gate_addendum_v1"
)
RECEIPT_PATH = OUTPUT_ROOT / "build_source_gate_receipt.json"

EXPECTED_TOTAL_FILES = 45
EXPECTED_BYTE_IDENTICAL = 41
EXPECTED_ADJUDICATED = 4
EXPECTED_CATEGORIES = {
    "contextworld": 24,
    "overlap_sources": 8,
    "renderer_dependencies": 2,
    "scoring_runtime_dependencies": 11,
}

# The four adjudications are INHERITED from v3 and pinned to the digests
# recorded there.  This module does not re-adjudicate them; it refuses if any
# of them has moved.
INHERITED_ADJUDICATED = {
    "contextworld/evaluation/action_delay_h7_score.py":
        "112fb51c72f132bdc5cb2d052cbebf19593a6f69df854e5be52689f608f807d5",
    "contextworld/benchmarks/adapters.py":
        "cc9e758b7081a57251e8cd026e9ac9ff8a17e3f300d52f464bdad871edcf26b2",
    "contextworld/evaluation/__init__.py":
        "f7c8632df364aef88fba8a7b2eee8f2d2e7d6d68b3ef39ba51fee9043252814b",
    "contextworld/synthesis/__init__.py":
        "88213a0c1dd9c74734e61bdb6c7cedb416c9c960f18a39f94399cfff3af1dd84",
}

CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, purpose: str) -> Path:
    metadata = os.lstat(path)
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{purpose} is not a regular non-symlink file: {path}",
    )
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    require(result.returncode == 0, f"cannot read git HEAD: {root}")
    return result.stdout.strip()


def _load_v3_auditor() -> Any:
    """Load the frozen v3 auditor.  It is the authority for the closure."""

    _regular(V3_AUDITOR, "v3 source-rebind auditor")
    specification = importlib.util.spec_from_file_location(
        "_v2_build_source_gate_v3_authority", V3_AUDITOR
    )
    require(
        specification is not None and specification.loader is not None,
        "cannot import the v3 source-rebind auditor",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules["_v2_build_source_gate_v3_authority"] = module
    specification.loader.exec_module(module)
    require(
        hasattr(module, "audit_source_closure"),
        "the v3 auditor no longer exposes audit_source_closure",
    )
    return module


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(_regular(path, "build-source-gate addendum").read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"YAML mapping required: {path}")
    return value


def validate_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything asserted here is a content invariant of the addendum itself."""

    config = load_config() if config is None else dict(config)

    require(
        config.get("analysis_id")
        == "action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
           "development_v2_build_source_gate_addendum_v1",
        "addendum identity changed",
    )
    require(
        config.get("status") == "frozen_before_any_v2_release_build_claim_or_score",
        "addendum status changed",
    )
    require(config.get("overwrite") == "forbidden", "overwrite protection removed")

    gate = config.get("build_source_gate", {})
    require(
        gate.get("require_head_equals_a_pinned_commit") is False
        and gate.get("head_is_recorded_not_gated") is True,
        "the addendum no longer demotes the commit gate",
    )
    require(
        gate.get("require_every_sha256_exact_except_the_four_adjudicated") is True
        and gate.get("adjudicated_exceptions_pinned_by_hash") is True
        and gate.get("unlisted_difference_behaviour") == "invalidate",
        "the content gate was weakened",
    )
    require(
        gate.get("require_head_stable_during_build") is True
        and gate.get("require_stablewm_worktree_clean") is True,
        "stability during the read was dropped",
    )

    # The four adjudications must match this module's pins exactly.
    declared = {
        row["path"]: row["pinned_observed_sha256"]
        for row in config.get("adjudicated_exceptions", {}).get("files", ())
    }
    require(
        declared == INHERITED_ADJUDICATED,
        "the adjudicated exception set in the addendum disagrees with the auditor",
    )

    authority = config.get("authority", {})
    closed = authority.get("what_remains_closed", {})
    for key in (
        "any_new_training",
        "one_use_consumption_rule_relaxed",
        "spent_v1_release_may_be_reconsumed",
        "candidate_set_may_change",
        "gate_thresholds_may_change",
        "cross_seed_pooling_averaging_or_rescue",
        "rejecting_the_pairing_method_family",
        "any_other_frozen_guard_loosened",
    ):
        require(closed.get(key) is False or closed.get(key) is True, f"authority key missing: {key}")
    require(closed.get("any_new_training") is True, "training must remain closed")
    for key in (
        "evaluation_or_scoring_authorized",
        "checkpoint_may_be_opened",
        "one_use_consumption_rule_relaxed",
        "spent_v1_release_may_be_reconsumed",
        "candidate_set_may_change",
        "gate_thresholds_may_change",
        "cross_seed_pooling_averaging_or_rescue",
        "rejecting_the_pairing_method_family",
        "any_other_frozen_guard_loosened",
    ):
        require(closed.get(key) is False, f"authority widened: {key}")

    decision = config.get("decision_contract", {})
    for key in (
        "score_allowed_before_release_identity",
        "exclusive_claim_allowed_before_release_identity",
        "terminal_identity_allowed_before_release_identity",
        "prior_receipt_mutation_allowed",
        "v1_release_mutation_allowed",
    ):
        require(decision.get(key) is False, f"decision boundary widened: {key}")
    require(
        decision.get("if_any_unadjudicated_file_differs")
        == "invalidate_without_build_claim_or_score"
        and decision.get("if_head_moves_during_the_build") == "invalidate",
        "invalidation rule changed",
    )
    return config


def validate_v3_receipt() -> dict[str, Any]:
    """The v3 receipt is the published record this addendum inherits from."""

    receipt = _json(_regular(V3_RECEIPT, "v3 source-rebind receipt"))
    require(
        receipt.get("content_sha256") == V3_RECEIPT_CONTENT_SHA256,
        "v3 source-rebind receipt changed",
    )
    closure = receipt.get("source_rebind", {})
    require(
        int(closure.get("total_file_count", -1)) == EXPECTED_TOTAL_FILES
        and int(closure.get("files_byte_identical", -1)) == EXPECTED_BYTE_IDENTICAL
        and not closure.get("unadjudicated_differences"),
        "the published v3 closure is not the audited one",
    )
    published = {
        row["path"]: row["observed_sha256"]
        for row in closure.get("adjudicated_differences", ())
    }
    require(
        published == INHERITED_ADJUDICATED,
        "the v3 receipt's adjudicated digests disagree with this module's pins",
    )
    return {
        "path": str(V3_RECEIPT.relative_to(REPO_ROOT)),
        "content_sha256": receipt["content_sha256"],
        "sha256": file_sha256(V3_RECEIPT),
    }


def audit_build_source_gate() -> dict[str, Any]:
    """Run the content gate.  HEAD is read, required stable, and recorded."""

    head_before = _git_head(CONTEXTWORLD_ROOT)
    closure = _load_v3_auditor().audit_source_closure()
    head_after = _git_head(CONTEXTWORLD_ROOT)

    require(
        head_before == head_after,
        f"ContextWorld HEAD moved during the audit: {head_before} -> {head_after}",
    )
    require(
        closure.get("head_stable_during_audit") is True,
        "the closure audit observed an unstable tree",
    )
    require(
        int(closure.get("total_file_count", -1)) == EXPECTED_TOTAL_FILES,
        f"closure file count changed: {closure.get('total_file_count')}",
    )
    require(
        int(closure.get("files_byte_identical", -1)) == EXPECTED_BYTE_IDENTICAL,
        f"byte-identical count changed: {closure.get('files_byte_identical')}",
    )
    require(
        not closure.get("unadjudicated_differences"),
        f"unadjudicated closure differences: {closure.get('unadjudicated_differences')}",
    )
    observed = {
        row["path"]: row["observed_sha256"]
        for row in closure.get("adjudicated_differences", ())
    }
    require(
        observed == INHERITED_ADJUDICATED,
        "an adjudicated file moved; the written verdict no longer covers it",
    )
    categories = {
        name: row.get("file_count")
        for name, row in closure.get("categories", {}).items()
    }
    require(categories == EXPECTED_CATEGORIES, f"category census changed: {categories}")
    require(
        not closure.get("stablewm_status_entries"),
        "the StableWM runtime worktree is dirty",
    )
    return {
        "gate": "content_closure_with_pinned_adjudicated_exceptions",
        "passed": True,
        "total_file_count": closure["total_file_count"],
        "files_byte_identical": closure["files_byte_identical"],
        "adjudicated_differences": EXPECTED_ADJUDICATED,
        "unadjudicated_differences": 0,
        "categories": categories,
        "adjudicated_observed_sha256": observed,
        # Recorded, not gated.  No comparison to a predetermined value.
        "contextworld_head_recorded": head_before,
        "contextworld_head_gated": False,
        "contextworld_head_stable_during_audit": True,
        "stablewm_commit_recorded": closure.get("stablewm_commit"),
        "stablewm_worktree_clean": True,
    }


def _preexisting_outputs() -> dict[str, str]:
    """Permitted pre-existing files are pinned by hash, not by emptiness.

    Requiring an empty output directory is itself a position proxy; it was the
    fourth instance of the pattern this addendum exists to fix.
    """

    if not OUTPUT_ROOT.exists():
        return {}
    return {
        path.name: file_sha256(path)
        for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file()
    }


def audit(*, require_receipt_absent: bool = True) -> dict[str, Any]:
    config = validate_config()
    require(AUDIT_CONTRACT.is_file(), "audit contract missing")
    require(V2_PREREGISTRATION.is_file(), "v2 preregistration missing")
    v3 = validate_v3_receipt()
    gate = audit_build_source_gate()
    existing = _preexisting_outputs()
    if require_receipt_absent:
        require(
            RECEIPT_PATH.name not in existing,
            f"receipt already published: {RECEIPT_PATH}",
        )
    return {
        "status": "passed_build_source_gate_before_any_v2_release_build_claim_or_score",
        "release_id": "action-delay-h7-a0-aux-pcja-predictor-only-multiseed-private-v2",
        "addendum": {
            "path": str(CONFIG.relative_to(REPO_ROOT)),
            "sha256": file_sha256(CONFIG),
        },
        "audit_contract": {
            "path": str(AUDIT_CONTRACT.relative_to(REPO_ROOT)),
            "sha256": file_sha256(AUDIT_CONTRACT),
            "version": 1,
        },
        "inherits_from_v3_receipt": v3,
        "auditor": {
            "path": str(THIS_SOURCE.relative_to(REPO_ROOT)),
            "sha256": file_sha256(THIS_SOURCE),
        },
        "closure_authority": {
            "path": str(V3_AUDITOR.relative_to(REPO_ROOT)),
            "sha256": file_sha256(V3_AUDITOR),
            "note": "the closure is obtained by calling this module, not restated here",
        },
        "build_source_gate": gate,
        "preexisting_output_files": existing,
        "boundaries": {
            "training_steps_run": 0,
            "release_assets_generated": 0,
            "release_built_or_mutated": False,
            "checkpoint_deserialized_or_model_loaded": False,
            "prediction_latent_or_score_computed": False,
            "public_test_or_sealed_content_opened": False,
            "frozen_v1_or_recovery_or_attempt_files_edited": False,
            "exclusive_claim_committed": False,
        },
    }


def _publish(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["content_sha256"] = canonical_sha256(
        {k: v for k, v in body.items() if k != "content_sha256"}
    )
    data = (
        json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(RECEIPT_PATH, flags, 0o444)
    except FileExistsError as error:
        raise ContractError(f"refusing to overwrite create-once receipt: {RECEIPT_PATH}") from error
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short write: {RECEIPT_PATH}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="audit only; publish nothing")
    mode.add_argument("--publish", action="store_true", help="audit, then publish the create-once receipt")
    options = parser.parse_args(argv)
    result = audit(require_receipt_absent=True)
    if options.publish:
        result = _publish(result)
        result["receipt_sha256"] = file_sha256(RECEIPT_PATH)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
