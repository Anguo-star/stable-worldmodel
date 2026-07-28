#!/usr/bin/env python3
"""Freeze the adaptive replay50 SIGReg 0.90 falsification extension.

This extension is intentionally labelled adaptive: it is frozen after the
0.09/0.20/0.30 Door and CEM results showed that 0.30 preserves original
planning but does not yet pass every Door stratum.  The extension tests the
strongest remaining simple explanation before any new objective is developed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


TRAINING_BASE_SHA256 = (
    "59908ee2e9eb4a44d4bcbfe65a543667c26f30cac8420282c5d324fb9704e55d"
)
VALIDATION_BASE_SHA256 = (
    "962e573ea049d75513538fa66ed83be02faa0c1790ce871efbee36ab0a137898"
)
MODEL_ID = "H3_PassageReplay50_SIGReg0p90"
WEIGHT = 0.90


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


def build_training_extension(
    base: dict[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.update(
        {
            "benchmark": (
                "tworoom_hidden_passage_history3_"
                "sigreg_replay50_adaptive_extension_v1"
            ),
            "status": "frozen_adaptive_falsification_before_training",
            "adaptive_freeze_date": "2026-07-28",
            "adaptive_rationale": (
                "λ=0.30 retained paired original-domain CEM but only partially "
                "learned Door. Test λ=0.90 before attributing the result to a "
                "structural need for a new objective."
            ),
            "source_replay50_protocol": {
                "path": str(source_path.resolve()),
                "sha256": TRAINING_BASE_SHA256,
            },
        }
    )
    protocol = config["training_protocol"]
    protocol["group_sampling"][MODEL_ID] = {
        "original": 0.5,
        "passage_mixed": 0.5,
    }
    protocol["adaptive_sigreg_falsification_extension"] = {
        "weight": WEIGHT,
        "motivation_observed_before_freeze": {
            "sigreg_0p30_door_formal_passed": False,
            "sigreg_0p30_matching_history_target_accuracy": (
                0.6766666666666666
            ),
            "sigreg_0p30_original_cem_success_rate": 0.93,
            "sigreg_0p30_original_cem_noninferior_to_0p09": True,
        },
        "decision": (
            "if λ=0.90 passes Door and is paired-CEM-noninferior to the "
            "replay-matched λ=0.09 control, stop the new-objective route"
        ),
        "cem_noninferiority_margin_absolute": 0.05,
        "comparison_unit": "paired_frozen_query",
        "confidence": 0.95,
        "adaptive_not_part_of_original_preregistration": True,
    }
    config["models"].append(
        {
            "model_id": MODEL_ID,
            "display_name": (
                "50% 原始 replay + 50% Door mixed-rule，"
                "LeWM SIGReg λ=0.90（自适应反证扩展）"
            ),
            "training_groups": ["original", "passage_mixed"],
            "lewm_sigreg_weight": WEIGHT,
        }
    )
    config["comparison"] = {
        "required_scoring_order": [MODEL_ID],
        "training_seed": 3072,
        "falsification_control_from_frozen_predecessor": (
            "H3_PassageReplay50_SIGReg0p09"
        ),
        "adaptive_candidate": MODEL_ID,
        "only_intended_training_difference_from_replay50_control": (
            "lewm_sigreg_weight"
        ),
    }
    return config


def build_validation_extension(
    base: dict[str, Any],
    *,
    source_path: Path,
    training_extension_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.update(
        {
            "status": (
                "preregistered_before_independent_catalog_generation_and_scoring"
            ),
            "protocol_role": "sigreg_replay50_adaptive_extension_v1",
            "adaptive_status": (
                "frozen_after_0p30_scoring_before_0p90_training_and_scoring"
            ),
            "adaptive_freeze_date": "2026-07-28",
            "source_replay50_validation_protocol": {
                "path": str(source_path.resolve()),
                "sha256": VALIDATION_BASE_SHA256,
            },
        }
    )
    comparison = config["comparison"]
    comparison["required_results"][MODEL_ID] = [3072]
    comparison["checkpoint_training_model_id"][MODEL_ID] = MODEL_ID
    comparison["checkpoint_training_group"][MODEL_ID] = "passage_mixed"
    comparison["attribution_gate"] = {
        "required_all_pass_model_ids": [MODEL_ID],
        "required_passed_training_seeds": 1,
        "role": (
            "adaptive simple-weight falsification; CEM noninferiority is a "
            "separate required gate"
        ),
    }
    prior = copy.deepcopy(
        config["training_provenance"]["passage_formal_by_model"][
            "H3_PassageReplay50_SIGReg0p30"
        ]
    )
    prior["training_benchmark_config"] = str(
        training_extension_path.resolve()
    )
    prior["lewm_sigreg_weight"] = WEIGHT
    config["training_provenance"]["passage_formal_by_model"][MODEL_ID] = prior
    config["falsification_gate"] = {
        "door_requirement": (
            "the λ=0.90 checkpoint must pass every frozen Door Validation-v2 "
            "gate"
        ),
        "original_cem_control": "H3_PassageReplay50_SIGReg0p09",
        "original_cem_noninferiority_margin_absolute": 0.05,
        "confidence": 0.95,
        "stop_new_objective_if_candidate_satisfies_both": True,
        "adaptive_not_part_of_original_preregistration": True,
    }
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
    parser.add_argument("--training-base", type=Path, required=True)
    parser.add_argument("--validation-base", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    training_base = _load_frozen_yaml(
        args.training_base.resolve(), TRAINING_BASE_SHA256
    )
    validation_base = _load_frozen_yaml(
        args.validation_base.resolve(), VALIDATION_BASE_SHA256
    )
    training_output = args.training_output.resolve()
    validation_output = args.validation_output.resolve()
    training = build_training_extension(
        training_base,
        source_path=args.training_base,
    )
    validation = build_validation_extension(
        validation_base,
        source_path=args.validation_base,
        training_extension_path=training_output,
    )
    _write_exact(training_output, _serialized(training))
    _write_exact(validation_output, _serialized(validation))
    print(
        yaml.safe_dump(
            {
                "training_output": str(training_output),
                "training_sha256": file_sha256(training_output),
                "validation_output": str(validation_output),
                "validation_sha256": file_sha256(validation_output),
            },
            sort_keys=False,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
