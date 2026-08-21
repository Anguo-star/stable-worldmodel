from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_residual_transition_ccrm_v1.py"
)


def _load():
    name = "_test_pusht_motion_damping_residual_transition_ccrm_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_binary_groups_cover_only_adjacent_hidden_rows():
    module = _load()
    groups = module.binary_hidden_groups(
        original_batch_size=2,
        batch_size=8,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(
        groups,
        torch.tensor([[2, 3], [4, 5], [6, 7]]),
    )


def test_binary_groups_reject_incomplete_pair():
    module = _load()
    try:
        module.binary_hidden_groups(
            original_batch_size=2,
            batch_size=7,
            device=torch.device("cpu"),
        )
    except ValueError as error:
        assert "adjacent" in str(error)
    else:
        raise AssertionError("incomplete hidden pair was accepted")


def test_generic_paired_auxiliary_accepts_one_explicit_positive_weight():
    module = _load()

    class _Mixed:
        load_model_for_variant = staticmethod(lambda *args, **kwargs: (None, {}))
        mixed_prediction_loss = staticmethod(lambda **kwargs: torch.tensor(0.0))

    class _Trainer:
        trainer = type("_Inner", (), {"mixed": _Mixed()})()

    state = module._install_paired_auxiliary(
        _Trainer(),
        objective=lambda *args: {},
        objective_name="test",
        response_metric_key="metric",
        weight=0.03,
    )
    assert state["weight"] == 0.03

    for invalid in (0.0, -0.1, float("inf")):
        try:
            module._install_paired_auxiliary(
                _Trainer(),
                objective=lambda *args: {},
                objective_name="test",
                response_metric_key="metric",
                weight=invalid,
            )
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError(f"invalid weight accepted: {invalid}")
