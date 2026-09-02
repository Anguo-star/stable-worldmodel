from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.conditional_dynamics_representation.scripts import (
    analyze_pusht_motion_damping_root_cause_zero_step_v1 as audit,
)


def test_decision_requires_rel50_latent_and_gradient_visibility() -> None:
    latent = {
        "comparisons": {
            "REL50_vs_HASH50": {
                str(scale): {"rho_lat_absolute_delta": 0.01}
                for scale in audit.LATENT_NEIGHBOUR_SCALES
            },
            "HASH50_vs_D0": {
                str(scale): {"rho_lat_absolute_delta": 0.0}
                for scale in audit.LATENT_NEIGHBOUR_SCALES
            },
        }
    }
    gradient = {
        "batch_count_per_arm": audit.AUDIT_BATCH_COUNT,
        "train_mode": {"paired_rng_before_and_after_each_arm": True},
        "eval_mode": {"paired_rng_before_and_after_each_arm": True},
        "comparisons": {
            "REL50_vs_HASH50": {
                "all": {
                    "response_mean_gradient_norm_relative_delta": 0.1,
                    "response_mean_gradient_norm_difference": 0.1,
                    "response_snr16_difference": 0.1,
                },
                "paired_batch_gradient_norm": {
                    "bootstrap": {"interval": {"lower": 0.01, "upper": 0.2}}
                },
            },
            "HASH50_vs_D0": {
                "all": {
                    "response_mean_gradient_norm_relative_delta": -0.1,
                    "response_mean_gradient_norm_difference": -0.1,
                    "response_snr16_difference": -0.1,
                },
                "paired_batch_gradient_norm": {
                    "bootstrap": {"interval": {"lower": -0.2, "upper": 0.01}}
                },
            },
        },
        "state_restoration": {"all": True},
    }
    report = audit._decision(latent, gradient)
    assert report["status"] == "passed_go_for_motion_short_training"
    assert all(report["checks"].values())

    for scale in audit.LATENT_NEIGHBOUR_SCALES:
        latent["comparisons"]["HASH50_vs_D0"][str(scale)]["rho_lat_absolute_delta"] = 0.1
    gradient["comparisons"]["HASH50_vs_D0"]["all"][
        "response_mean_gradient_norm_difference"
    ] = 0.2
    gradient["comparisons"]["HASH50_vs_D0"]["all"]["response_snr16_difference"] = 0.2
    gradient["comparisons"]["HASH50_vs_D0"]["paired_batch_gradient_norm"][
        "bootstrap"
    ]["interval"]["lower"] = 0.01
    failed = audit._decision(latent, gradient)
    assert failed["status"] == "failed_no_go"
    assert failed["checks"]["hash50_not_same_or_larger_improvement_as_rel50"] is False


def test_decision_rejects_legacy_fixture_without_paired_uncertainty() -> None:
    latent = {
        "comparison": {
            str(scale): {"rho_lat_absolute_delta": 0.01}
            for scale in audit.LATENT_NEIGHBOUR_SCALES
        }
    }
    gradient = {
        "batch_count_per_arm": audit.AUDIT_BATCH_COUNT,
        "comparison": {
            "all": {
                "response_mean_gradient_norm_relative_delta": 0.1,
                "response_snr16_relative_delta": 0.0,
            }
        },
        "state_restoration": {"all": True},
    }
    assert audit._decision(latent, gradient)["status"] == "failed_no_go"


def test_schedule_cell_multiset_check_uses_twin_counts_not_cell_indices(monkeypatch) -> None:
    monkeypatch.setattr(audit, "EXPECTED_TWINS", 256)
    monkeypatch.setattr(audit, "EXPECTED_SCHEDULE_BATCHES", 24)
    monkeypatch.setattr(audit, "TWINS_PER_BATCH", 16)
    monkeypatch.setattr(audit, "AUDIT_SCHEDULE_INDICES", (0,))
    cells = np.repeat(np.arange(64), 4)
    base = np.tile(np.asarray([1, 1, 2, 2]), 64)
    different = base.copy()
    different[-4:] = [1, 1, 1, 3]

    def arm(counts: np.ndarray) -> dict[str, object]:
        return {
            "records": [
                {"schedule_index": index, "twin_ids": list(range(16))}
                for index in range(24)
            ],
            "counts": counts,
        }

    schedule_set = {
        "D0": arm(base.copy()),
        "REL50": arm(base),
        "ABS50": arm(base.copy()),
        "HASH50": arm(different),
    }
    with pytest.raises(RuntimeError, match="schedule invariant failed"):
        audit.validate_schedule_invariants(schedule_set, cells=cells)


def test_weighted_latent_metrics_uses_supplied_graph_and_four_arms(monkeypatch) -> None:
    monkeypatch.setattr(audit, "EXPECTED_TWINS", 2)
    monkeypatch.setattr(audit, "EXPECTED_PAIRS", 4)
    monkeypatch.setattr(audit, "LATENT_NEIGHBOUR_SCALES", (1,))
    query = torch.zeros(8, 1)
    future = torch.tensor(
        [[0.0], [2.0], [1.0], [3.0], [0.0], [4.0], [1.0], [5.0]]
    )
    neighbours = np.asarray([[2], [3], [0], [1]], dtype=np.int64)
    weights = {
        arm: np.asarray([0.5, 0.5])
        for arm in audit.ARMS
    }
    weights["REL50"] = np.asarray([0.25, 0.75])
    report = audit.weighted_latent_metrics(query, future, neighbours, weights)
    assert set(report["arms"]) == set(audit.ARMS)
    assert report["comparisons"]["REL50_vs_HASH50"]["1"][
        "weighted_C_difference"
    ] != 0.0
    assert report["physical_neighbour_graph_reused"] is True


def test_device_accumulator_reports_batch_coherence_and_direction_dispersion() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    accumulator = audit.DeviceGradientAccumulator([parameter], ["predictor"])
    accumulator.start_batch()
    accumulator.add((torch.tensor([1.0, 0.0]),))
    accumulator.add((torch.tensor([1.0, 0.0]),))
    accumulator.finish_batch()
    accumulator.start_batch()
    accumulator.add((torch.tensor([-1.0, 0.0]),))
    accumulator.add((torch.tensor([-1.0, 0.0]),))
    accumulator.finish_batch()
    report = accumulator.summary_with_batch_statistics(snr_batch_sizes=(2,))
    stats = report["batch_statistics"]["scopes"]["predictor"]
    assert stats["batch_mean_gradient_norm"]["values"] == [1.0, 1.0]
    assert stats["within_batch_coherence"]["values"] == [1.0, 1.0]
    assert stats["between_batch_direction"]["mean_pairwise_cosine"] == pytest.approx(-1.0)
    assert stats["between_batch_direction"]["directional_dispersion"] == pytest.approx(2.0)
    assert report["scopes"]["all"]["coherence"] == 0.0


def _write_schedule_receipt_fixture(root: Path) -> tuple[dict[str, dict[str, Path]], Path]:
    files: dict[str, dict[str, Path]] = {}
    output_hashes: dict[str, str] = {}
    for arm in audit.ARMS:
        arm_dir = root / arm
        arm_dir.mkdir()
        arm_files = {
            "schedule": arm_dir / "schedule.jsonl",
            "multiplicity": arm_dir / "multiplicity.jsonl",
        }
        arm_files["schedule"].write_text("{}\n", encoding="utf-8")
        arm_files["multiplicity"].write_text("{}\n", encoding="utf-8")
        files[arm] = arm_files
        for kind, path in arm_files.items():
            output_hashes[f"{arm}/{kind}.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "passed_go",
                "arms": list(audit.ARMS),
                "schedule_generated": True,
                "optimizer_steps": 0,
                "development_lance_opened": False,
                "public_test_lance_opened": False,
                "pixels_decoded": False,
                "output_sha256": output_hashes,
            }
        ),
        encoding="utf-8",
    )
    return files, receipt


def test_schedule_receipt_rehashes_outputs_and_supports_explicit_sha(tmp_path: Path) -> None:
    fixture_root = tmp_path.parent / "receipt_fixture_root"
    fixture_root.mkdir()
    files, receipt = _write_schedule_receipt_fixture(fixture_root)
    expected = hashlib.sha256(receipt.read_bytes()).hexdigest()
    report = audit.verify_schedule_receipt(
        fixture_root,
        files,
        expected_receipt_sha256=expected,
    )
    assert report["sha256"] == expected
    assert set(report["output_sha256"]) == {
        f"{arm}/{kind}" for arm in audit.ARMS for kind in ("schedule", "multiplicity")
    }
    with pytest.raises(RuntimeError, match="receipt SHA256"):
        audit.verify_schedule_receipt(
            fixture_root,
            files,
            expected_receipt_sha256="0" * 64,
        )


@pytest.mark.parametrize("part", ["development", "public", "test", "validation", "public_test"])
def test_training_only_path_guard_rejects_forbidden_split_tokens(tmp_path: Path, part: str) -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        audit.assert_training_only_path(tmp_path / part / "schedule.jsonl")


def test_parse_args_exposes_explicit_schedule_root_and_receipt_pin(tmp_path: Path) -> None:
    expected = "a" * 64
    args = audit.parse_args(
        [
            "--check-only",
            "--schedule-root",
            str(tmp_path),
            "--expected-schedule-receipt-sha256",
            expected,
        ]
    )
    assert args.schedule_dir == tmp_path
    assert args.expected_schedule_receipt_sha256 == expected
    assert set(args.schedule_files) == set(audit.ARMS)


def test_formal_schedule_schema_loads_all_four_multiplicities() -> None:
    schedule_set = audit.load_schedule_set(audit.DEFAULT_SCHEDULE_DIR)
    assert set(schedule_set["arms"]) == set(audit.ARMS)
    for arm in audit.ARMS:
        counts = schedule_set["arms"][arm]["counts"]
        assert counts.shape == (audit.EXPECTED_TWINS,)
        assert int(counts.sum()) == audit.EXPECTED_SCHEDULE_BATCHES * audit.TWINS_PER_BATCH
    catalog_rows, _ = audit.frozen_audit.load_catalog(audit.DEFAULT_CATALOG)
    checks = audit.validate_schedule_invariants(
        schedule_set["arms"], cells=audit._catalog_cells(catalog_rows)
    )
    assert checks["same_16_preregistered_schedule_strata"] is True
    assert checks["all_invariants_pass"] is True


def test_source_boundary_has_zero_step_and_no_nontraining_table_literals() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert "validation.lance" not in source
    assert "public_test.lance" not in source
    assert "development.lance" not in source
    assert '"optimizer_steps": 0' in source
    assert '"development_opened": False' in source
    assert '"public_test_opened": False' in source
