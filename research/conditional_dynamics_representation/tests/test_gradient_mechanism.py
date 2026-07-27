from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "analyze_lewm_gradient_mechanism.py"
SPEC = importlib.util.spec_from_file_location("gradient_mechanism", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gradient_mechanism = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gradient_mechanism
SPEC.loader.exec_module(gradient_mechanism)


def test_descent_direction_summary_signs_contraction() -> None:
    summary = gradient_mechanism.descent_direction_summary(
        ["encoder.weight"],
        (torch.tensor([2.0]),),
        (torch.tensor([3.0]),),
    )
    encoder = summary["encoder"]
    assert encoder["gradient_dot"] == 6.0
    assert encoder["predicted_distance_change_per_unit_lr"] == -6.0
    assert encoder["descent_cosine"] == -1.0
    assert encoder["effect"] == "contracts_pair_distance"


def test_descent_direction_summary_signs_expansion() -> None:
    summary = gradient_mechanism.descent_direction_summary(
        ["projector.weight"],
        (torch.tensor([2.0]),),
        (torch.tensor([-3.0]),),
    )
    projector = summary["projector"]
    assert projector["predicted_distance_change_per_unit_lr"] == 6.0
    assert projector["descent_cosine"] == 1.0
    assert projector["effect"] == "expands_pair_distance"


def test_descent_direction_summary_handles_disjoint_gradients() -> None:
    summary = gradient_mechanism.descent_direction_summary(
        ["predictor.weight"],
        (None,),
        (torch.tensor([1.0]),),
    )
    assert summary["predictor"]["effect"] == "no_shared_gradient"
    assert summary["predictor"]["descent_cosine"] is None
