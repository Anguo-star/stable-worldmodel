from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_sigreg_replay_extension.py"
)
SPEC = importlib.util.spec_from_file_location("sigreg_replay_extension", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_training_extension_adds_only_replay_matched_0p90_candidate(
    tmp_path: Path,
) -> None:
    base = {
        "training_protocol": {
            "group_sampling": {
                "H3_PassageReplay50_SIGReg0p09": {
                    "original": 0.5,
                    "passage_mixed": 0.5,
                }
            }
        },
        "models": [
            {
                "model_id": "H3_PassageReplay50_SIGReg0p09",
                "lewm_sigreg_weight": 0.09,
            }
        ],
        "comparison": {},
    }

    extension = MODULE.build_training_extension(
        base, source_path=tmp_path / "base.yaml"
    )

    assert extension["status"] == "frozen_adaptive_falsification_before_training"
    assert extension["training_protocol"]["group_sampling"][MODULE.MODEL_ID] == {
        "original": 0.5,
        "passage_mixed": 0.5,
    }
    candidate = extension["models"][-1]
    assert candidate["model_id"] == MODULE.MODEL_ID
    assert candidate["lewm_sigreg_weight"] == 0.90
    gate = extension["training_protocol"][
        "adaptive_sigreg_falsification_extension"
    ]
    assert gate["adaptive_not_part_of_original_preregistration"] is True


def test_validation_extension_binds_0p90_to_extension_training_config(
    tmp_path: Path,
) -> None:
    base = {
        "comparison": {
            "required_results": {
                "H3_PassageReplay50_SIGReg0p09": [3072],
            },
            "checkpoint_training_model_id": {
                "H3_PassageReplay50_SIGReg0p09": (
                    "H3_PassageReplay50_SIGReg0p09"
                ),
            },
            "checkpoint_training_group": {
                "H3_PassageReplay50_SIGReg0p09": "passage_mixed",
            },
        },
        "training_provenance": {
            "passage_formal_by_model": {
                "H3_PassageReplay50_SIGReg0p30": {
                    "training_benchmark_config": "old.yaml",
                    "lewm_sigreg_weight": 0.30,
                    "group_weights": {
                        "original": 0.5,
                        "passage_mixed": 0.5,
                    },
                }
            }
        },
    }
    training_path = tmp_path / "extension.yaml"

    extension = MODULE.build_validation_extension(
        base,
        source_path=tmp_path / "base.yaml",
        training_extension_path=training_path,
    )

    assert extension["comparison"]["required_results"][MODULE.MODEL_ID] == [3072]
    provenance = extension["training_provenance"]["passage_formal_by_model"][
        MODULE.MODEL_ID
    ]
    assert provenance["training_benchmark_config"] == str(
        training_path.resolve()
    )
    assert provenance["lewm_sigreg_weight"] == 0.90
    assert provenance["group_weights"] == {
        "original": 0.5,
        "passage_mixed": 0.5,
    }
