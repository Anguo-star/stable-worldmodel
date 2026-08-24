from __future__ import annotations

from pathlib import Path

import pytest

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_v1 as method,
)


def test_validate_args_requires_absolute_1024(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "baseline.ckpt"
    checkpoint.write_bytes(b"baseline")
    monkeypatch.setattr(
        method,
        "_sha256",
        lambda path: method.PUSHT_BASELINE_SHA256
        if Path(path) == checkpoint
        else "different",
    )
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--model",
        "lewm",
        "--seed",
        "14321",
        "--optimizer-steps",
        "1024",
        "--input-basis",
        "absolute",
        "--output",
        str(tmp_path / "out"),
    ]
    assert method._validate_args(argv).input_basis == "absolute"
    with pytest.raises(RuntimeError, match="absolute single-stage"):
        method._validate_args(
            ["causal_transition" if value == "absolute" else value for value in argv]
        )


def test_absolute_wrapper_restores_original_temporal_coordinates() -> None:
    class Model:
        temporal_input_basis = "causal_transition"
        temporal_output_basis = "residual"

    class Mixed:
        @staticmethod
        def load_model_for_variant(*_args, **_kwargs):
            return Model(), {"residual_output_initialization": {"old": True}}

        @staticmethod
        def model_config(*_args, **_kwargs):
            return {
                "temporal_input_basis": "causal_transition",
                "temporal_output_basis": "residual",
            }

    class TrainerModule:
        mixed = Mixed()

    class Trainer:
        trainer = TrainerModule()

    def native_install(_trainer):
        return {"native": True}

    trainer = Trainer()
    state = method._install_absolute_freeze_only(
        trainer,
        native_install=native_install,
    )
    model, receipt = trainer.trainer.mixed.load_model_for_variant(Path("x"))
    assert model.temporal_input_basis == "absolute"
    assert model.temporal_output_basis == "absolute"
    assert receipt["output_projection_reinitialized"] is False
    assert "residual_output_initialization" not in receipt
    assert trainer.trainer.mixed.model_config() == {
        "temporal_input_basis": "absolute",
        "temporal_output_basis": "absolute",
    }
    assert state["absolute_basis"]["output_projection_reinitialized"] is False
