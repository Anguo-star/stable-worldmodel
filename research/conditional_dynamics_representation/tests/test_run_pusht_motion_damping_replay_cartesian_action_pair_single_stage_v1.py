from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_replay_cartesian_action_pair_single_stage_v1 as method,
)


def _argv(checkpoint: Path, output: Path) -> list[str]:
    return [
        "--checkpoint",
        str(checkpoint),
        "--model",
        "lewm",
        "--seed",
        "14321",
        "--optimizer-steps",
        "3072",
        "--input-basis",
        "causal_transition",
        "--output",
        str(output),
    ]


def test_validate_args_binds_single_stage_boundary(
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
    args = method._validate_args(_argv(checkpoint, tmp_path / "out"))
    assert args.optimizer_steps == 3072
    assert args.seed == 14321
    with pytest.raises(RuntimeError, match="single-stage replay-pair contract"):
        method._validate_args(
            [
                value if value != "3072" else "1024"
                for value in _argv(checkpoint, tmp_path / "out2")
            ]
        )


def test_main_installs_residual_before_freeze_and_restores_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_args = argparse.Namespace(dry_run=True, output=tmp_path / "out")
    monkeypatch.setattr(method, "_validate_args", lambda _argv: fake_args)
    calls: list[str] = []

    def native_freeze(_trainer):
        calls.append("freeze")
        return {"freeze": True}

    def residual_install(_trainer, *, input_basis: str):
        calls.append(f"residual:{input_basis}")
        return {"model": {"initialization": {"weight_exactly_zero": True}}}

    monkeypatch.setattr(method.canonical, "_install_freeze_only", native_freeze)
    monkeypatch.setattr(
        method.residual,
        "_install_residual_output",
        residual_install,
    )
    original_source = method.canonical.SOURCE_CHECKPOINT_SHA256
    original_steps = method.canonical.OPTIMIZER_STEPS
    original_candidate = method.parent.CANDIDATE
    original_parent_source = method.parent.THIS_SOURCE

    def fake_parent_main(_argv):
        assert (
            method.canonical.SOURCE_CHECKPOINT_SHA256
            == method.PUSHT_BASELINE_SHA256
        )
        assert method.canonical.OPTIMIZER_STEPS == 3072
        assert method.cartesian.OPTIMIZER_STEPS == 3072
        assert method.parent.CANDIDATE == method.CANDIDATE
        state = method.canonical._install_freeze_only(object())
        assert state == {"freeze": True}
        return 0

    monkeypatch.setattr(method.parent, "main", fake_parent_main)
    assert method.main([]) == 0
    assert calls == ["residual:causal_transition", "freeze"]
    assert method.canonical.SOURCE_CHECKPOINT_SHA256 == original_source
    assert method.canonical.OPTIMIZER_STEPS == original_steps
    assert method.canonical._install_freeze_only is native_freeze
    assert method.parent.CANDIDATE == original_candidate
    assert method.parent.THIS_SOURCE == original_parent_source


def test_rewrite_report_records_only_removed_motion_stage(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    report = output / "training_report.json"
    report.write_text(
        json.dumps(
            {
                "provenance": {},
                "result": {
                    "source_checkpoint": {
                        "path": "baseline.ckpt",
                        "sha256": method.PUSHT_BASELINE_SHA256,
                    },
                    "optimizer_steps": 3072,
                    "motion_cartesian_action_pair_contract": {
                        "continuation_source": {
                            "checkpoint": "baseline.ckpt",
                            "sha256": method.PUSHT_BASELINE_SHA256,
                            "source_optimizer_steps": 2048,
                        },
                        "fresh_optimizer_steps": 3072,
                        "training_only_frozen_teacher": False,
                        "checks": {
                            "saved_model_parameter_count_unchanged": True
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = output / method.EXPECTED_SIDE_CAR
    sidecar.write_text("{}\n", encoding="utf-8")
    residual_state = {
        "model": {
            "initialization": {
                "weight_exactly_zero": True,
                "bias_exactly_zero": True,
            },
            "parameter_count_before": 10,
            "parameter_count_after": 10,
            "input_basis": "causal_transition",
            "output_basis": "residual",
        }
    }
    method._rewrite_single_stage_report(
        output,
        residual_state=residual_state,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    contract = payload["result"]["single_stage_joint_training_contract"]
    assert contract["joint_optimizer_steps"] == 3072
    assert contract["initialization_source"][
        "motion_adaptation_steps_before_joint_training"
    ] == 0
    assert contract["initialization_source"]["random_initialization"] is False
    assert contract["explicit_simulator_matched_pairs_retained"] is True
    assert "source_optimizer_steps" not in contract["initialization_source"]
    rewritten_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert rewritten_sidecar["candidate"] == method.CANDIDATE
    assert rewritten_sidecar["fresh_optimizer_steps"] == 3072
