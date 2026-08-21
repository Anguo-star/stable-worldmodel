from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from stable_worldmodel.wm.lewm.lewm import LeWM


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_temporal_homotopy_exact_future_v1.py"
)


def _load():
    name = "_test_pusht_motion_damping_temporal_homotopy_exact_future_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class _Predictor(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, embedding, action_embedding):
        return self.projection(embedding + action_embedding)


def _model(dimension: int = 5) -> LeWM:
    return LeWM(
        encoder=nn.Linear(dimension, dimension),
        predictor=_Predictor(dimension),
        action_encoder=nn.Identity(),
        projector=nn.Identity(),
        pred_proj=nn.Linear(dimension, dimension),
    )


def test_alpha_schedule_has_exact_endpoints_and_hold():
    module = _load()
    assert module.alpha_for_optimizer_step(0) == 0.0
    assert module.alpha_for_optimizer_step(512) == 0.5
    assert module.alpha_for_optimizer_step(1024) == 1.0
    assert module.alpha_for_optimizer_step(2048) == 1.0
    with pytest.raises(ValueError, match="nonnegative"):
        module.alpha_for_optimizer_step(-1)


def test_alpha_zero_is_exact_loaded_absolute_predictor():
    module = _load()
    torch.manual_seed(7)
    model = _model()
    embedding = torch.randn(3, 4, 5)
    action = torch.randn(3, 4, 5)
    native_predict = model.predict
    expected = native_predict(embedding, action)

    observed = module.temporal_homotopy_prediction(
        model=model,
        native_predict=native_predict,
        embedding=embedding,
        action_embedding=action,
        alpha=0.0,
    )

    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_alpha_one_is_exact_saved_causal_residual_predictor():
    module = _load()
    torch.manual_seed(11)
    model = _model()
    embedding = torch.randn(3, 4, 5)
    action = torch.randn(3, 4, 5)
    native_predict = model.predict

    observed = module.temporal_homotopy_prediction(
        model=model,
        native_predict=native_predict,
        embedding=embedding,
        action_embedding=action,
        alpha=1.0,
    )
    model.temporal_input_basis = "causal_transition"
    model.temporal_output_basis = "residual"
    expected = native_predict(embedding, action)

    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_homotopy_prediction_does_not_add_or_mutate_parameters():
    module = _load()
    torch.manual_seed(19)
    model = _model()
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    embedding = torch.randn(2, 3, 5)
    action = torch.randn(2, 3, 5)
    native_predict = model.predict

    module.temporal_homotopy_prediction(
        model=model,
        native_predict=native_predict,
        embedding=embedding,
        action_embedding=action,
        alpha=0.37,
    )

    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)


def test_state_hash_accepts_scalar_and_bfloat16_buffers():
    module = _load()
    model = _model()
    model.register_buffer("scalar_counter", torch.tensor(3, dtype=torch.long))
    model.register_buffer(
        "bfloat_marker", torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    )

    first = module._state_sha256(model)
    second = module._state_sha256(model)

    assert first == second
    assert len(first) == 64
