from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.conditional_dynamics_representation.scripts import (
    analyze_pusht_motion_damping_d1_removed_history_v1 as audit,
)


def _arrays(pair_count: int = 4) -> SimpleNamespace:
    faster = np.zeros((pair_count, 4, 2, 2, 3), dtype=np.uint8)
    slower = np.zeros_like(faster)
    actions = np.zeros((pair_count, 4, 5, 2), dtype=np.float32)
    for index in range(pair_count):
        faster[index, :3] = index + 1
        slower[index, :3] = index + 20
        faster[index, 2] = slower[index, 2] = index + 7
        faster[index, 3] = index + 40
        slower[index, 3] = index + 80
        actions[index] = index + 0.25
    return SimpleNamespace(
        pair_count=pair_count,
        pair_ids=tuple(f"dev-{index}" for index in range(pair_count)),
        faster_decay_pixels=faster,
        no_extra_decay_pixels=slower,
        raw_action_blocks=actions,
    )


def test_paired_query_action_identity_and_removed_history() -> None:
    arrays = _arrays()
    identity = audit.validate_paired_modes(
        arrays.faster_decay_pixels,
        arrays.no_extra_decay_pixels,
        arrays.raw_action_blocks,
        arrays.raw_action_blocks.copy(),
        arrays.pair_ids,
        expected_pairs=4,
    )
    assert identity["checks"]["query_rgb_exact"] is True
    assert identity["checks"]["paired_actions_exact"] is True
    removed = audit.build_removed_history(arrays.faster_decay_pixels)
    assert removed.shape[1] == 3
    assert np.array_equal(removed[:, 0], arrays.faster_decay_pixels[:, 2])
    assert np.array_equal(removed[:, 0], removed[:, 2])


@pytest.mark.parametrize("which", ["query", "action", "history", "future"])
def test_pair_identity_fails_closed(which: str) -> None:
    arrays = _arrays()
    faster = arrays.faster_decay_pixels.copy()
    slower = arrays.no_extra_decay_pixels.copy()
    right_actions = arrays.raw_action_blocks.copy()
    if which == "query":
        slower[0, 2, 0, 0, 0] += 1
    elif which == "action":
        right_actions[0, 0, 0, 0] += 1
    elif which == "history":
        slower[0, :3] = faster[0, :3]
    else:
        slower[0, 3] = faster[0, 3]
    with pytest.raises(RuntimeError):
        audit.validate_paired_modes(
            faster,
            slower,
            arrays.raw_action_blocks,
            right_actions,
            arrays.pair_ids,
            expected_pairs=4,
        )


def test_removed_history_metrics_keep_records_and_response_statistics() -> None:
    pair_count = 4
    targets = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]],
         [[1.0, 1.0], [2.0, 1.0]], [[-1.0, 0.0], [-1.0, 3.0]]]
    )
    correct = targets.clone()
    removed = torch.zeros_like(targets)
    summary, records = audit.removed_history_metrics(
        correct, removed, targets, [f"dev-{index}" for index in range(pair_count)], expected_pairs=pair_count
    )
    assert len(records) == pair_count
    assert records[0]["pair_id"] == "dev-0"
    assert summary["pair_count"] == pair_count
    assert summary["target_mse"]["mean_correct"] == pytest.approx(0.0)
    assert summary["target_mse"]["mean_removed"] > 0.0
    assert summary["target_mse"]["mean_removed_minus_correct"] > 0.0
    assert all(row["removed_target_mse_increase"] > 0.0 for row in records)
    assert summary["removed_response"]["gain"]["aggregate"] == pytest.approx(0.0)
    assert summary["removed_response"]["nre"]["aggregate"] == pytest.approx(1.0)


def test_d1_minus_d0_bootstrap_and_sign_flip_are_reproducible() -> None:
    d0 = np.arange(4, dtype=np.float64)
    d1 = d0 + 0.5
    first = audit.paired_delta_bootstrap(d0, d1, seed=17, resamples=200)
    second = audit.paired_delta_bootstrap(d0, d1, seed=17, resamples=200)
    assert first == second
    assert first["difference"] == "D1_minus_D0"
    assert first["observed_delta_mean"] == pytest.approx(0.5)
    assert first["sign_flip"]["two_sided_monte_carlo_p"] > 0.0


def test_source_and_scope_guards() -> None:
    assert all(audit.validate_source_guards().values())
    with pytest.raises(RuntimeError):
        audit.validate_source_guards("adapter = MotionDampingICLEvalDataset(path)")
    assert all(audit.validate_scope_counts().values())
    with pytest.raises(RuntimeError):
        audit.validate_scope_counts({"public_reads": 1})
    with pytest.raises(RuntimeError):
        audit.validate_scope_counts({"optimizer_steps": 1})


def test_eval_adapter_hash_and_all_rows(monkeypatch) -> None:
    monkeypatch.setattr(audit, "EXPECTED_DEVELOPMENT_PAIRS", 2)
    arrays = _arrays(pair_count=2)

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []
            self.model = torch.nn.Linear(1, 1)

        def frozen_state_hash(self) -> str:
            return audit._sha256(__file__)

        def rollout_latents(self, pixels, actions, *, batch_size):
            self.calls.append(("rollout", len(pixels)))
            values = np.asarray(pixels[:, 0, 0, 0, 0], dtype=np.float32)
            return values[:, None, None]

        def encode_pixels(self, pixels, *, batch_size):
            self.calls.append(("target", len(pixels)))
            values = np.asarray(pixels[:, 0, 0, 0, 0] if pixels.ndim == 5 else pixels[:, 0, 0, 0], dtype=np.float32)
            if values.ndim > 1:
                values = values[:, 0]
            return values[:, None]

    adapter = FakeAdapter()
    result = audit.evaluate_adapter(adapter, arrays, batch_size=1, label="synthetic")
    assert result["condition_count"] == 4
    assert result["correct_history_rows"] == 4
    assert result["removed_history_rows"] == 4
    assert result["target_rows"] == 4
    assert result["correct_predictions"][:, :, 0].tolist() == [[1.0, 20.0], [2.0, 21.0]]
    assert result["targets"][:, :, 0].tolist() == [[40.0, 80.0], [41.0, 81.0]]
    assert result["removed_predictions"][:, :, 0].tolist() == [[7.0, 7.0], [8.0, 8.0]]
    assert result["removed_history_manipulation_check"]["passed"] is True
    assert result["state_sha256_before"] == result["state_sha256_after"]
    assert [call[0] for call in adapter.calls] == ["rollout", "rollout", "target"]


def test_exclusive_writer_rejects_overwrite(tmp_path) -> None:
    target = tmp_path / "result.json"
    audit.write_exclusive(target, {"ok": True})
    with pytest.raises(RuntimeError):
        audit.write_exclusive(target, {"ok": False})
