#!/usr/bin/env python3
"""Materialize the preregistered 50:50 replay SIGReg falsification protocol.

The authoritative Door training and Validation configurations live in the
sibling ContextWorld repository.  This script verifies their exact bytes,
then makes the smallest controlled derivation needed for the falsification
experiment:

* 50% original TwoRoom clips and 50% mixed-rule Door clips;
* one shared initialization, data order, budget, and topology;
* SIGReg weights 0.09, 0.20, and 0.30 as the only intended difference.

The generated files are deterministic and are suitable for committing as
experiment protocols.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


TRAINING_BASE_SHA256 = (
    "af0cb6133a397715108f5d02a51b08d3b82d3acf8c9c04cdda63f3d278b29192"
)
VALIDATION_BASE_SHA256 = (
    "fa6696ca0c433907625c75f8a535d46e808d62628bf2e75b45dcd55d26e78760"
)
TRAINING_SEED = 3072
WEIGHTS = (0.09, 0.20, 0.30)
MODEL_IDS = {
    0.09: "H3_PassageReplay50_SIGReg0p09",
    0.20: "H3_PassageReplay50_SIGReg0p20",
    0.30: "H3_PassageReplay50_SIGReg0p30",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def build_training_protocol(
    base: dict[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.update(
        {
            "schema_version": 2,
            "benchmark": (
                "tworoom_hidden_passage_history3_"
                "sigreg_replay50_falsification_v1"
            ),
            "status": "preregistered_falsification_before_training",
            "preregistered_date": "2026-07-28",
            "question": (
                "在固定 50% 原始 TwoRoom replay、50% Door mixed-rule replay "
                "后，轻度提高 SIGReg 权重是否既能学会 Door 条件规则，又不"
                "降低原始域规划能力？若能，则结构性缺陷假设被证伪。"
            ),
            "source_training_protocol": {
                "path": str(source_path.resolve()),
                "sha256": TRAINING_BASE_SHA256,
            },
        }
    )
    protocol = config["training_protocol"]
    protocol["synthetic_only"] = False
    protocol["paired_training_seeds"] = [TRAINING_SEED]
    protocol["group_sampling"] = {
        MODEL_IDS[weight]: {
            "original": 0.5,
            "passage_mixed": 0.5,
        }
        for weight in WEIGHTS
    }
    protocol["sigreg_falsification_gate"] = {
        "weights": list(WEIGHTS),
        "null_hypothesis": (
            "a modest SIGReg increase solves Door without a practically "
            "meaningful original-domain CEM decrease"
        ),
        "decision": (
            "if any increased weight passes Door and is CEM-noninferior to "
            "the replay-matched 0.09 control, stop the structural-defect and "
            "new-objective route"
        ),
        "cem_noninferiority_margin_absolute": 0.05,
        "comparison_unit": "paired_frozen_query",
        "confidence": 0.95,
    }
    protocol["fairness_contract"] = {
        "equal_across_models": [
            "initialization_checkpoint_bytes",
            "original_train_normalizer_bytes",
            "architecture_and_history_length",
            "training_seed_and_data_order",
            "original_to_synthetic_group_weights",
            "logical_optimizer_draws",
            "optimizer_scheduler_and_encoder_learning_rate",
            "eight_gpu_execution_topology",
        ],
        "intentional_difference": "lewm_sigreg_weight_only",
        "group_sampling": {
            "original": 0.5,
            "passage_mixed": 0.5,
        },
        "per_group_draws_at_formal_budget": {
            "original": 524288,
            "passage_mixed": 524288,
        },
    }
    config["models"] = [
        {
            "model_id": MODEL_IDS[weight],
            "display_name": (
                "50% 原始 replay + 50% Door mixed-rule，"
                f"LeWM SIGReg λ={weight:.2f}"
            ),
            "training_groups": ["original", "passage_mixed"],
            "lewm_sigreg_weight": weight,
        }
        for weight in WEIGHTS
    ]
    config["comparison"] = {
        "required_scoring_order": [MODEL_IDS[weight] for weight in WEIGHTS],
        "training_seed": TRAINING_SEED,
        "falsification_control": MODEL_IDS[0.09],
        "candidate_weights": [MODEL_IDS[0.20], MODEL_IDS[0.30]],
        "only_intended_difference": "lewm_sigreg_weight",
    }
    return config


def build_validation_protocol(
    base: dict[str, Any],
    *,
    source_path: Path,
    training_protocol_path: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.update(
        {
            # Keep the benchmark identity because the sealed catalog was built
            # for this exact Validation-v2 query matrix.
            "benchmark": "tworoom_hidden_passage_history3_validation_v2",
            "status": (
                "preregistered_before_independent_catalog_generation_and_scoring"
            ),
            "date": "2026-07-28",
            "protocol_role": "sigreg_replay50_falsification_v1",
            "source_validation_protocol": {
                "path": str(source_path.resolve()),
                "sha256": VALIDATION_BASE_SHA256,
            },
        }
    )
    model_ids = [MODEL_IDS[weight] for weight in WEIGHTS]
    comparison = config["comparison"]
    comparison["required_results"] = {
        model_id: [TRAINING_SEED] for model_id in model_ids
    }
    comparison["checkpoint_training_model_id"] = {
        model_id: model_id for model_id in model_ids
    }
    comparison["checkpoint_training_group"] = {
        model_id: "passage_mixed" for model_id in model_ids
    }
    comparison["attribution_gate"] = {
        "required_all_pass_model_ids": model_ids,
        "required_passed_training_seeds": 1,
        "role": (
            "report Door pass/fail per weight; the separate CEM "
            "noninferiority gate determines whether the structural claim "
            "survives"
        ),
    }
    training_path = str(training_protocol_path.resolve())
    common_contract = {
        "training_benchmark_config": training_path,
        "profile": "passage_formal",
        "optimizer_steps": 1024,
        "total_logical_draws": 1048576,
        "effective_global_batch": 1024,
        "topology": {
            "world_size": 8,
            "devices": 8,
            "batch_size_per_device": 128,
            "accumulation_steps": 1,
            "execution_topology": "8gpu_x_b128_x_accum1",
        },
        "synthetic_only": False,
        "group_weights": {
            "original": 0.5,
            "passage_mixed": 0.5,
        },
        "primary_synthetic_group": "passage_mixed",
        "frozen_model_modules": [],
        "required_frozen_training_artifacts": {
            "groups": [
                "passage_passable",
                "passage_blocked",
                "passage_mixed",
            ],
            "per_group": ["catalog", "manifest", "synthesis_report"],
            "formal_build_report": "required",
        },
        "validation_exclusion_benchmark": (
            "tworoom_hidden_passage_history3_validation_v2"
        ),
        "initialization_checkpoint": {
            "sha256": (
                "7d141b86cca49145444a69bff89c71ede69e8cf8252bfb933e656c3e2e962b54"
            ),
            "config_sha256": (
                "44b5bde83fbc91634ef84acbdba9a75d3436568ed72266c9a5aa057653f9162b"
            ),
            "role": "model_weight_initialization_only_not_resume",
        },
    }
    config["training_provenance"]["passage_formal_by_model"] = {
        MODEL_IDS[weight]: {
            **copy.deepcopy(common_contract),
            "lewm_sigreg_weight": weight,
        }
        for weight in WEIGHTS
    }
    config["falsification_gate"] = {
        "door_requirement": (
            "the checkpoint must pass every frozen Door Validation-v2 gate"
        ),
        "original_cem_control": MODEL_IDS[0.09],
        "original_cem_noninferiority_margin_absolute": 0.05,
        "confidence": 0.95,
        "stop_new_objective_if_any_candidate_satisfies_both": True,
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
    training = build_training_protocol(
        training_base,
        source_path=args.training_base,
    )
    validation = build_validation_protocol(
        validation_base,
        source_path=args.validation_base,
        training_protocol_path=training_output,
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
