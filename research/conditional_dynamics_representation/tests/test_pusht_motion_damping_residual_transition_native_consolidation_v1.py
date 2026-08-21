from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_residual_transition_native_consolidation_v1.py"
)


def _load():
    name = "_test_pusht_motion_damping_residual_transition_native_consolidation_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_consolidation_contract_has_no_new_loss_or_parameter_reset():
    module = _load()
    source = Path("/tmp/source.pt")

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))
            self.temporal_input_basis = "absolute"
            self.temporal_output_basis = "absolute"

    model = _Model()

    class _Mixed:
        @staticmethod
        def mixed_prediction_loss(*args, **kwargs):
            return torch.tensor(0.0)

        @staticmethod
        def load_model_for_variant(*args, **kwargs):
            return model, {"sha256": module.SOURCE_CHECKPOINT_SHA256}

        @staticmethod
        def model_config(*args, **kwargs):
            return {"_target_": "test.Model"}

    class _Trainer:
        trainer = type("_Inner", (), {"mixed": _Mixed()})()

    original_sha = module._sha256
    module._sha256 = lambda path: module.SOURCE_CHECKPOINT_SHA256
    try:
        state = module._install_native_consolidation(_Trainer())
        loaded, receipt = _Trainer.trainer.mixed.load_model_for_variant(source)
        config = _Trainer.trainer.mixed.model_config()
    finally:
        module._sha256 = original_sha

    assert loaded is model
    assert model.temporal_input_basis == "causal_transition"
    assert model.temporal_output_basis == "residual"
    assert state["model"]["parameter_reset"] is False
    assert (
        state["_native_loss_object"]
        is _Trainer.trainer.mixed.mixed_prediction_loss
    )
    assert receipt["native_consolidation"]["auxiliary_loss_terms"] == 0
    assert config["temporal_input_basis"] == "causal_transition"
    assert config["temporal_output_basis"] == "residual"


def test_source_checkpoint_is_rejected_before_trainer_loading(tmp_path):
    module = _load()
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the source checkpoint")
    with pytest.raises(RuntimeError, match="source checkpoint changed"):
        module._validate_source_checkpoint(["--checkpoint", str(wrong)])
