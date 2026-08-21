from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_residual_output_basis_v1.py"
)


def _load():
    name = "_test_pusht_motion_damping_residual_output_basis_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 5),
            nn.GELU(),
            nn.Linear(5, 3),
        )


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_proj = _Projection()


def test_zero_initialization_reuses_existing_projection_parameters():
    module = _load()
    model = _Model()
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    receipt = module.zero_initialize_displacement_projection(model)

    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    assert torch.count_nonzero(model.pred_proj.net[-1].weight) == 0
    assert torch.count_nonzero(model.pred_proj.net[-1].bias) == 0
    assert receipt["parameter_count_reinitialized"] == 18


def test_method_args_require_explicit_input_basis():
    module = _load()
    common = [
        "--model", "lewm",
        "--seed", "14321",
        "--optimizer-steps", "1024",
        "--output", "/tmp/residual-output-test",
    ]
    assert module._discovery_args(
        [*common, "--input-basis", "absolute"]
    ).input_basis == "absolute"
    assert module._discovery_args(
        [*common, "--input-basis", "causal_transition"]
    ).input_basis == "causal_transition"
    assert "--input-basis" not in module._trainer_argv(
        [*common, "--input-basis", "absolute"]
    )
