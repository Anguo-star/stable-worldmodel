#!/usr/bin/env python3
"""Correct the replay50 Door summary to the pre-existing rule-switch-v2 gate.

The replay protocol was accidentally derived from the older
``all_histories_strict_v1`` Validation configuration.  That contract lets the
rule-ambiguous no-crossing history veto an otherwise correct rule readout.
This materializer changes no checkpoint, query, target, score, or threshold.
It copies the decision fields byte-for-byte from the rule-switch-v2 protocol
frozen on 2026-07-24, before any replay50 checkpoint existed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


REPLAY_VALIDATION_SHA256 = (
    "19c879c1e75e96ae0034092d8a2681f9b1e8ef9048f575d879731adcd4714efc"
)
RULE_SWITCH_SOURCE_SHA256 = (
    "16b5e20c43077c374cfb394a77b2fe56d45068489954021e7545f1e80833faa3"
)
CORRECTIVE_STATUS = (
    "corrective_alignment_to_preexisting_rule_switch_v2_"
    "after_materialization_mismatch"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_yaml(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen protocol hash mismatch for {path}: "
            f"expected={expected_sha256}, observed={observed}"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def build_corrected_validation(
    replay: dict[str, Any],
    rule_switch_source: dict[str, Any],
    *,
    replay_path: Path,
    rule_switch_source_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(replay)
    if (
        rule_switch_source.get("gates", {}).get("decision_contract")
        != "informative_history_rule_switch_v2"
    ):
        raise ValueError("Decision source is not rule-switch-v2")
    config["schema_version"] = 3
    config["status"] = CORRECTIVE_STATUS
    config["protocol_role"] = (
        "sigreg_replay50_rule_switch_v2_falsification_final"
    )
    config["decision_protocol"] = copy.deepcopy(
        rule_switch_source["decision_protocol"]
    )
    config["metrics"] = copy.deepcopy(rule_switch_source["metrics"])
    config["gates"] = copy.deepcopy(rule_switch_source["gates"])
    config["protocol_correction"] = {
        "detected_after_lambda_0p90_scoring": True,
        "incorrect_inherited_contract": "all_histories_strict_v1",
        "correct_contract": "informative_history_rule_switch_v2",
        "replay_validation_source": {
            "path": str(replay_path.resolve()),
            "sha256": REPLAY_VALIDATION_SHA256,
        },
        "decision_source": {
            "path": str(rule_switch_source_path.resolve()),
            "sha256": RULE_SWITCH_SOURCE_SHA256,
            "frozen_date": "2026-07-24",
            "frozen_before_any_replay50_checkpoint": True,
        },
        "changed_fields": [
            "schema_version",
            "status",
            "protocol_role",
            "decision_protocol",
            "metrics",
            "gates",
            "artifacts.output_root",
        ],
        "unchanged": [
            "training_checkpoints",
            "training_reports",
            "catalog_and_payloads",
            "normalizer",
            "model_predictions",
            "encoded_true_futures",
            "raw_score_records",
            "rule_switch_thresholds",
        ],
        "reason": (
            "The no-crossing history has no unique rule label and was already "
            "frozen as auxiliary-only by the project-wide rule-switch-v2 "
            "protocol. It cannot veto the capability question."
        ),
    }
    config["artifacts"]["output_root"] = (
        "artifacts/evaluation/history3/"
        "hidden_passage_sigreg_replay50_rule_switch_v2"
    )
    return config


def _serialized(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )


def _write_exact(path: Path, content: str) -> None:
    if path.exists():
        observed = path.read_text(encoding="utf-8")
        if observed != content:
            raise FileExistsError(
                f"Refusing to overwrite a different generated protocol: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-validation", type=Path, required=True)
    parser.add_argument("--rule-switch-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = _load_frozen_yaml(
        args.replay_validation.resolve(), REPLAY_VALIDATION_SHA256
    )
    rule_switch_source = _load_frozen_yaml(
        args.rule_switch_source.resolve(), RULE_SWITCH_SOURCE_SHA256
    )
    output = args.output.resolve()
    config = build_corrected_validation(
        replay,
        rule_switch_source,
        replay_path=args.replay_validation,
        rule_switch_source_path=args.rule_switch_source,
    )
    _write_exact(output, _serialized(config))
    print(
        yaml.safe_dump(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "decision_contract": config["gates"]["decision_contract"],
            },
            sort_keys=False,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
