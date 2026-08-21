from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_prefix_aligned_transition_v2.py"
)


def _load():
    name = "_test_pusht_motion_damping_prefix_aligned_transition_v2"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class _RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.temporal_input_basis = "absolute"
        self.calls = []

    def predict(self, embedding, action):
        self.calls.append((embedding.detach().clone(), action.detach().clone()))
        return self.weight * (embedding + action)


def test_each_output_uses_one_leakage_free_aligned_prefix():
    module = _load()
    model = _RecordingModel()
    receipt = module._install_model_predict(model)
    embedding = torch.tensor([[[1.0], [3.0], [8.0]]])
    action = torch.tensor([[[10.0], [20.0], [30.0]]])

    prediction = model.predict(embedding, action)

    assert receipt["learned_parameters_added"] == 0
    assert len(model.calls) == 3
    expected_histories = (
        torch.tensor([[[1.0]]]),
        torch.tensor([[[2.0], [3.0]]]),
        torch.tensor([[[2.0], [5.0], [8.0]]]),
    )
    for length, ((seen_history, seen_action), expected) in enumerate(
        zip(model.calls, expected_histories),
        start=1,
    ):
        torch.testing.assert_close(seen_history, expected)
        torch.testing.assert_close(seen_action, action[:, :length])
    torch.testing.assert_close(
        prediction,
        torch.tensor([[[11.0], [23.0], [38.0]]]),
    )
