from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "analyze_lewm_module_swaps.py"
SPEC = importlib.util.spec_from_file_location("module_swaps", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module_swaps = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module_swaps
SPEC.loader.exec_module(module_swaps)


def _state(value: float) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        [
            ("encoder.weight", torch.tensor([value])),
            ("projector.weight", torch.tensor([value])),
            ("projector.running_mean", torch.tensor([value])),
            ("predictor.weight", torch.tensor([value])),
            ("action_encoder.weight", torch.tensor([value])),
            ("pred_proj.weight", torch.tensor([value])),
            ("pred_proj.running_var", torch.tensor([value])),
            ("other_buffer", torch.tensor([value])),
        ]
    )


def test_splice_state_separates_parameters_and_buffers() -> None:
    original = _state(0.0)
    trained = _state(1.0)
    parameter_names = {
        "encoder.weight",
        "projector.weight",
        "predictor.weight",
        "action_encoder.weight",
        "pred_proj.weight",
    }

    hybrid = module_swaps.splice_state(
        original_state=original,
        trained_state=trained,
        parameter_names=parameter_names,
        sources=module_swaps.source_map(
            projector_buffers=module_swaps.TRAINED,
            predictor_parameters=module_swaps.TRAINED,
        ),
    )

    assert hybrid["projector.weight"].item() == 0.0
    assert hybrid["projector.running_mean"].item() == 1.0
    assert hybrid["predictor.weight"].item() == 1.0
    assert hybrid["pred_proj.running_var"].item() == 0.0
    assert hybrid["other_buffer"].item() == 0.0


def test_validate_compatible_states_rejects_shape_mismatch() -> None:
    original = _state(0.0)
    trained = _state(1.0)
    trained["encoder.weight"] = torch.ones(2)

    with pytest.raises(ValueError, match="tensor metadata"):
        module_swaps.validate_compatible_states(original, trained)


def test_source_code_uses_declared_group_order() -> None:
    sources = module_swaps.source_map(
        encoder_parameters=module_swaps.TRAINED,
        projector_buffers=module_swaps.TRAINED,
    )
    assert (
        module_swaps.source_code(
            sources, module_swaps.REPRESENTATION_GROUPS
        )
        == "TOT"
    )
