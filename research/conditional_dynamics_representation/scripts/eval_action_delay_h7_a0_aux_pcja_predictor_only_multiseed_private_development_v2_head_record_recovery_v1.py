#!/usr/bin/env python3
"""Append-only recovery for the v2 recorded-but-not-gated HEAD defect.

The frozen source closure explicitly declares ContextWorld HEAD to be
informational, but the v2 protocol validator compared the complete preflight
mapping and therefore gated on that recorded field.  This wrapper changes only
that comparison.  Every content hash, count, selector identity, release byte,
terminal identity, implementation source, and scoring path remains governed by
the original evaluator.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
ORIGINAL = THIS_SOURCE.with_name(
    "eval_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
ADDENDUM = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_head_record_recovery_addendum_v1.json"
)
RECOVERED_FIELD = "contextworld_head_recorded"


def _load(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


original = _load(ORIGINAL, "_predictor_only_multiseed_v2_head_recovery_parent")
require = original.base.require
canonical_sha256 = original.base.canonical_sha256


def _file_sha256(path: Path) -> str:
    return original.base._stable_ref(path, purpose=path.name)["sha256"]


def _without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "content_sha256"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON mapping: {path}")
    return value


def _load_addendum() -> dict[str, Any]:
    require(ADDENDUM.is_file(), "HEAD-record recovery addendum is absent")
    payload = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == 1, "recovery schema changed")
    require(
        payload.get("status")
        == "frozen_after_release_and_terminal_before_claim_or_checkpoint_open",
        "recovery boundary changed",
    )
    require(
        payload.get("content_sha256")
        == canonical_sha256(_without_hash(payload)),
        "recovery addendum self-hash changed",
    )
    require(
        payload.get("recovered_field") == RECOVERED_FIELD
        and payload.get("only_allowed_preflight_difference")
        == [RECOVERED_FIELD],
        "recovery scope widened",
    )
    rule = payload.get("validation_rule", {})
    require(
        rule.get("contextworld_head_gated") is False
        and rule.get("all_non_head_fields_exact") is True
        and rule.get("head_stable_during_each_content_audit") is True,
        "recovery validation rule weakened",
    )
    boundary = payload.get("evidence_boundary", {})
    require(
        boundary.get("exclusive_claim_absent_at_freeze") is True
        and boundary.get("checkpoint_deserialized_or_model_loaded") is False
        and boundary.get("prediction_latent_or_score_computed") is False,
        "recovery crossed the claim boundary",
    )
    sources = payload.get("sources", {})
    require(
        sources.get("recovery_evaluator_sha256") == _file_sha256(THIS_SOURCE)
        and sources.get("original_evaluator_sha256") == _file_sha256(ORIGINAL),
        "recovery evaluator source changed",
    )
    require(
        sources.get("builder_sha256") == _file_sha256(original.BUILDER),
        "recovery builder source changed",
    )
    bindings = payload.get("bindings", {})
    protocol = _read_json(original.PROTOCOL)
    implementation = _read_json(original.IMPLEMENTATION_FREEZE)
    terminal = _read_json(original.TERMINAL_MANIFEST)
    release_identity_path = original.OUTPUT_ROOT / "release_identity.json"
    release_identity = _read_json(release_identity_path)
    for name, path, content_key in (
        ("protocol", original.PROTOCOL, "content_sha256"),
        ("implementation", original.IMPLEMENTATION_FREEZE, "content_sha256"),
        ("terminal", original.TERMINAL_MANIFEST, "content_sha256"),
        ("release_identity", release_identity_path, "identity_sha256"),
    ):
        value = {
            "file_sha256": _file_sha256(path),
            "content_sha256": {
                "protocol": protocol,
                "implementation": implementation,
                "terminal": terminal,
                "release_identity": release_identity,
            }[name][content_key],
        }
        require(bindings.get(name) == value, f"recovery binding changed: {name}")
    frozen_source = dict(protocol.get("source_rebind_preflight", {}))
    require(
        frozen_source.get(RECOVERED_FIELD)
        == payload.get("frozen_contextworld_head_recorded"),
        "recovery frozen HEAD does not match the protocol",
    )
    require(
        payload.get("content_closure")
        == {
            key: value
            for key, value in frozen_source.items()
            if key != RECOVERED_FIELD
        },
        "recovery content closure differs from the frozen protocol",
    )
    return payload


def _install_builder_recovery(builder: Any, addendum: Mapping[str, Any]) -> Any:
    if getattr(builder, "_head_record_recovery_v1_installed", False):
        return builder
    native_validate = builder.validate_protocol_freeze

    def recovered_validate_protocol_freeze(
        config: Mapping[str, Any] | None = None,
        *,
        check_selector: bool = True,
    ) -> dict[str, Any]:
        protocol = native_validate(config, check_selector=False)
        if not check_selector:
            return protocol
        raw = builder.load_config() if config is None else dict(config)
        merged = builder.effective_config(raw)
        require(
            protocol.get("selector_preflight")
            == builder.selector_preflight(
                merged, verify_protocol=False
            )["frozen_preflight"],
            "selector freeze changed",
        )
        frozen = dict(protocol.get("source_rebind_preflight", {}))
        current = dict(builder.source_rebind_preflight(merged))
        require(
            frozen.get("contextworld_head_gated") is False
            and current.get("contextworld_head_gated") is False,
            "ContextWorld HEAD unexpectedly became a gate",
        )
        changed = sorted(
            key
            for key in set(frozen) | set(current)
            if frozen.get(key) != current.get(key)
        )
        require(
            set(changed) <= {RECOVERED_FIELD},
            f"source-content closure changed outside recorded HEAD: {changed}",
        )
        require(
            {
                key: value
                for key, value in frozen.items()
                if key != RECOVERED_FIELD
            }
            == {
                key: value
                for key, value in current.items()
                if key != RECOVERED_FIELD
            },
            "non-HEAD source-content freeze changed",
        )
        require(
            frozen.get(RECOVERED_FIELD)
            == addendum["frozen_contextworld_head_recorded"],
            "frozen informational HEAD changed",
        )
        return protocol

    builder.validate_protocol_freeze = recovered_validate_protocol_freeze
    builder._head_record_recovery_v1_installed = True
    return builder


def _install_recovery() -> Any:
    addendum = _load_addendum()
    module = original._install()
    native_load = module._load_module
    native_dependency_audit = original._dependency_audit

    def recovered_load(path: Path, name: str) -> Any:
        loaded = native_load(path, name)
        if Path(path).resolve() == original.BUILDER.resolve():
            return _install_builder_recovery(loaded, addendum)
        return loaded

    def recovered_dependency_audit(
        builder: Any, *, require_implementation: bool
    ) -> dict[str, Any]:
        dependencies = native_dependency_audit(
            _install_builder_recovery(builder, addendum),
            require_implementation=require_implementation,
        )
        dependencies["head_record_recovery_evaluator"] = module._stable_ref(
            THIS_SOURCE, purpose="HEAD-record recovery evaluator"
        )
        dependencies["head_record_recovery_addendum"] = module._stable_ref(
            ADDENDUM, purpose="HEAD-record recovery addendum"
        )
        return dependencies

    module._load_module = recovered_load
    original._dependency_audit = recovered_dependency_audit
    module._dependency_audit = recovered_dependency_audit
    return module


def main(argv: Sequence[str] | None = None) -> int:
    _install_recovery()
    return int(original.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
