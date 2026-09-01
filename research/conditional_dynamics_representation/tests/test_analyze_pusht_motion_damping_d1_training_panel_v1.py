from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.conditional_dynamics_representation.scripts import (
    analyze_pusht_motion_damping_d1_training_panel_v1 as panel,
)


def _pool(twins: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.zeros(twins * 4, 4, 2, dtype=torch.float32)
    actions = torch.zeros(twins * 4, 3, dtype=torch.float32)
    for twin in range(twins):
        for pair in range(2):
            left = 4 * twin + 2 * pair
            right = left + 1
            pixels[left, 2] = torch.tensor((twin, pair), dtype=torch.float32)
            pixels[right, 2] = pixels[left, 2]
            pixels[left, :2] = torch.tensor((twin, pair), dtype=torch.float32)
            pixels[right, :2] = torch.tensor((twin, pair + 1), dtype=torch.float32)
            pixels[left, 3] = torch.tensor((twin + pair, 0.0), dtype=torch.float32)
            pixels[right, 3] = torch.tensor((twin + pair + 1, 0.0), dtype=torch.float32)
            actions[left] = torch.tensor((twin, pair, 1.0), dtype=torch.float32)
            actions[right] = actions[left]
    return pixels, actions


def test_training_pool_checks_query_action_and_four_row_twin_identity() -> None:
    pixels, actions = _pool()
    identity = panel.validate_hidden_pool(pixels, actions, expected_twins=2)
    assert identity["pairs"] == 4
    assert identity["rows_per_twin"] == 4
    assert all(identity["checks"].values())


@pytest.mark.parametrize(
    "mutator",
    [
        lambda pixels, actions: pixels.__setitem__((0, 2), torch.tensor([9.0, 9.0])),
        lambda pixels, actions: actions.__setitem__((1, 0), 7.0),
        lambda pixels, actions: pixels.__setitem__((1, slice(0, 2)), pixels[0, :2]),
        lambda pixels, actions: pixels.__setitem__((1, 3), pixels[0, 3]),
    ],
)
def test_training_pool_identity_fails_closed(mutator) -> None:
    pixels, actions = _pool()
    mutator(pixels, actions)
    with pytest.raises(RuntimeError):
        panel.validate_hidden_pool(pixels, actions, expected_twins=2)


def test_paired_records_report_g_swap_gain_alignment_and_nre(monkeypatch) -> None:
    monkeypatch.setattr(panel, "EXPECTED_PAIRS", 4)
    monkeypatch.setattr(panel, "EXPECTED_TWINS", 2)
    targets = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
            [[2.0, 1.0], [3.0, 1.0]],
            [[-1.0, 0.0], [-1.0, 3.0]],
        ]
    )
    predictions = targets.clone()
    summary, records = panel.paired_records(predictions, targets)
    assert len(records) == 4
    assert records[0]["pair_id"] == "pmd-train-000000-forward"
    assert records[1]["direction"] == "reverse"
    assert summary["g_swap"]["positive_fraction"] == pytest.approx(1.0)
    assert summary["response_geometry"]["gain"] == pytest.approx(1.0)
    assert summary["response_geometry"]["alignment"] == pytest.approx(1.0)
    assert summary["response_geometry"]["normalized_response_error"] == pytest.approx(0.0)


def test_paired_delta_bootstrap_and_sign_flip_are_reproducible() -> None:
    left = np.asarray([0.0, 1.0, 2.0, 3.0])
    right = left + 0.5
    first = panel.paired_delta_bootstrap(left, right, seed=19, resamples=300)
    second = panel.paired_delta_bootstrap(left, right, seed=19, resamples=300)
    assert first == second
    assert first["pair_count"] == 4
    assert first["observed_delta_mean"] == pytest.approx(0.5)
    assert first["bootstrap"]["ci_95"] == pytest.approx([0.5, 0.5])


def test_scope_gate_rejects_development_or_public_reads() -> None:
    assert all(
        panel.validate_scope_counts(
            {
                "development_scorer": 0,
                "public_scorer": 0,
                "release_loader": 0,
                "non_training_benchmark_reads": 0,
            }
        ).values()
    )
    with pytest.raises(RuntimeError):
        panel.validate_scope_counts(
            {
                "development_scorer": 1,
                "public_scorer": 0,
                "release_loader": 0,
                "non_training_benchmark_reads": 0,
            }
        )
    with pytest.raises(RuntimeError):
        panel.validate_scope_counts(
            {
                "development_scorer": 0,
                "public_scorer": 1,
                "release_loader": 0,
                "non_training_benchmark_reads": 0,
            }
        )


def test_state_hash_changes_when_a_parameter_changes() -> None:
    model = torch.nn.Linear(2, 2)
    before = panel._module_state_sha256(model)
    with torch.no_grad():
        model.weight[0, 0] += 1.0
    assert panel._module_state_sha256(model) != before


def test_d1_report_schedule_consumed_gate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(panel.d1_runner, "EXPECTED_BATCHES", 4)
    schedule_hash = {"schedule.jsonl": "schedule", "config.json": "config"}
    report = tmp_path / "training_report.json"
    report.write_text(
        __import__("json").dumps(
            {
                "provenance": {"data": {"train_pairs": 8192}},
                "result": {
                    "batch": {"hidden_pairs": 32},
                    "d1_ms50_schedule_contract": {
                        "candidate_id": "D1-MS50",
                        "optimizer_steps": 4,
                        "public_test_opened": False,
                        "schedule_sha256": schedule_hash,
                        "checks": {
                            "full_schedule_consumed_once": True,
                            "schedule_sha256_exact": True,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    observed = panel.validate_d1_report_schedule_consumed(
        report, expected_schedule_sha256=schedule_hash
    )
    assert observed["checks"]["full_schedule_consumed_once"] is True


def test_evaluate_model_is_eval_only_and_encodes_all_rows(monkeypatch) -> None:
    monkeypatch.setattr(panel, "EXPECTED_HIDDEN_ROWS", 4)
    monkeypatch.setattr(panel, "EXPECTED_PAIRS", 2)
    pixels, actions = _pool(twins=1)

    class FakeModel(torch.nn.Module):
        def encode(self, batch):
            base = batch["pixels"][:, 0, 0].reshape(-1, 1)
            emb = base.repeat(1, 4).reshape(-1, 4, 1)
            return {"emb": emb, "act_emb": torch.zeros_like(emb)}

        def predict(self, history, action):
            return history

    class Pilot:
        @staticmethod
        def preprocess_pixels(value, device):
            return value.to(device)

    result = panel.evaluate_model(
        model=FakeModel(),
        mixed=SimpleNamespace(pilot=Pilot()),
        hidden=SimpleNamespace(pixels=pixels, action=actions),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert result["encoded_rows"] == 4
    assert result["predictions"].shape == (2, 2, 1)
    assert result["eval_mode"] is True
    assert result["state_sha256_before"] == result["state_sha256_after"]
