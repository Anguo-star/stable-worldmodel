from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_residual_transition_original_only_consolidation_v1.py"
)


def _load():
    name = "_test_pusht_motion_damping_original_only_consolidation_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_original_rows_replace_hidden_rows_without_changing_total_batch():
    module = _load()
    observed = {}

    class _Mixed:
        @staticmethod
        def train_variant(*args, **kwargs):
            observed.update(kwargs)
            return {
                "batch": {
                    "original": kwargs["original_batch_size"],
                    "hidden": kwargs["batch_size"] - kwargs["original_batch_size"],
                    "total": kwargs["batch_size"],
                }
            }

    class _Trainer:
        trainer = type("_Inner", (), {"mixed": _Mixed()})()

    state = module._install_original_only_training(_Trainer())
    result = _Trainer.trainer.mixed.train_variant(
        variant=module.causal.NATIVE_VARIANT,
        batch_size=128,
        original_batch_size=64,
    )
    assert observed["original_batch_size"] == 128
    assert result["batch"] == {"original": 128, "hidden": 0, "total": 128}
    assert state["calls"] == 1
    assert state["batch"] == {
        "total": 128,
        "requested_original": 64,
        "actual_original": 128,
        "actual_hidden": 0,
    }
