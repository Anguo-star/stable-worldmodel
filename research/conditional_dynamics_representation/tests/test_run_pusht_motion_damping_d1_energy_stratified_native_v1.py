from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RESEARCH_ROOT / (
    "scripts/run_pusht_motion_damping_d1_energy_stratified_native_v1.py"
)
SPEC = importlib.util.spec_from_file_location("d1_energy_stratified_native_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d1
SPEC.loader.exec_module(d1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_schedule_and_builder_sources_are_exact() -> None:
    frozen = d1.verify_schedule_artifact()
    assert frozen["observed_sha256"] == d1.EXPECTED_SCHEDULE_SHA256
    assert frozen["summary"]["status"] == "passed_go"
    assert frozen["summary"]["gates"]["passed"] is True
    assert _sha(Path(d1.builder.__file__)) == d1.EXPECTED_BUILDER_SHA256


def test_schedule_records_preserve_batch_and_row_order() -> None:
    records = d1.load_schedule_records(d1.DEFAULT_SCHEDULE_DIR / "schedule.jsonl")
    assert len(records) == 8192
    for index in (0, 1, 255, 256, 8191):
        row = records[index]
        assert row["schedule_index"] == index
        assert row["optimizer_step"] == index + 1
        assert row["twin_ids"] == [
            *row["high_twin_ids"],
            *row["natural_twin_ids"],
        ]
        assert row["hidden_row_indices"] == d1.builder.expand_hidden_rows(
            row["twin_ids"]
        )


@pytest.mark.parametrize(
    ("pair_count", "batch_size", "seed", "message"),
    [
        (8190, 64, 14321, "pair_count"),
        (8192, 32, 14321, "batch_size"),
        (8192, 64, 14322, "training seed"),
    ],
)
def test_stream_fails_closed_on_training_identity_drift(
    pair_count: int, batch_size: int, seed: int, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        d1.EnergyStratifiedTwinBatchStream(
            pair_count, batch_size=batch_size, seed=seed
        )


def test_stream_returns_only_int64_row_indices_and_exhausts() -> None:
    d1.STREAM_INSTANCES.clear()
    stream = d1.EnergyStratifiedTwinBatchStream(8192, batch_size=64, seed=14321)
    iterator = iter(stream)
    first = next(iterator)
    assert isinstance(first, torch.Tensor)
    assert first.dtype == torch.int64
    assert first.shape == (64,)
    assert first.tolist() == stream.records[0]["hidden_row_indices"]
    for _ in range(8191):
        next(iterator)
    with pytest.raises(RuntimeError, match="exactly 8,192"):
        next(iterator)
    assert stream.consumed_batches == 8192
    assert stream.exhaustion_checked is True


def test_stream_rejects_a_modified_schedule(tmp_path: Path) -> None:
    schedule = tmp_path / "schedule.jsonl"
    schedule.write_bytes((d1.DEFAULT_SCHEDULE_DIR / "schedule.jsonl").read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="schedule SHA256 changed"):
        d1.EnergyStratifiedTwinBatchStream(
            8192,
            batch_size=64,
            seed=14321,
            schedule_path=schedule,
        )


@pytest.fixture(scope="module")
def preflight() -> dict:
    return d1.cpu_preflight_payload()


def test_cpu_preflight_consumes_d1_and_checks_full_e0_identity(preflight: dict) -> None:
    assert preflight["status"] == "passed_go"
    assert all(preflight["checks"].values())
    assert preflight["d1_consumption"] == {
        "batches": 8192,
        "row_tensor_sha256": (
            "c83fc4c96715daf8463200f834ef614a0ad6813e13ee5ebf15f82a920731ecc0"
        ),
        "multiplicity_min": 16,
        "multiplicity_median": 32.0,
        "multiplicity_max": 48,
        "metadata_at_model_or_loss_boundary": False,
        "stream_output": "torch.int64 row-index tensor only",
    }
    assert preflight["e0_identity"]["batches"] == 8192
    assert preflight["e0_identity"]["each_twin_multiplicity"] == 32
    assert preflight["e0_identity"]["row_tensor_sha256"] == (
        "2c64a7f707c27ed0894545c2762e47ca440b64026c6bd848737588c4b5d04088"
    )
    assert preflight["evidence_boundary"]["gpu_used"] is False
    assert preflight["evidence_boundary"]["model_loaded"] is False
    assert preflight["evidence_boundary"]["optimizer_steps_run"] == 0
    assert preflight["evidence_boundary"]["full_training_authorized"] is False


def test_preflight_report_is_deterministic_and_receipted(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    report_first = d1.write_cpu_preflight(first)
    report_second = d1.write_cpu_preflight(second)
    assert _sha(report_first) == _sha(report_second)
    for directory in (first, second):
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["status"] == "passed_go"
        assert all(receipt["checks"].values())
        assert receipt["output_sha256"]["report.json"] == _sha(
            directory / "report.json"
        )


def test_scheduled_loader_changes_only_the_trainer_stream_class() -> None:
    sentinel = object()
    trainer = SimpleNamespace(CompleteTwinPairedBatchStream=sentinel, retained=17)
    wrapped = d1.scheduled_trainer_loader(lambda: trainer)
    observed = wrapped()
    assert observed is trainer
    assert observed.CompleteTwinPairedBatchStream is d1.EnergyStratifiedTwinBatchStream
    assert observed.retained == 17
    assert d1.CAUSAL is d1.CANONICAL.causal


def test_terminal_report_records_schedule_as_the_only_change(
    tmp_path: Path, preflight: dict
) -> None:
    report = tmp_path / "training_report.json"
    sidecar = tmp_path / d1.SIDECAR
    report.write_text(
        json.dumps(
            {
                "result": {
                    "optimizer_steps": 8192,
                    "seed": 14321,
                    "batch": {"hidden": 64},
                },
                "provenance": {"method": {}},
            }
        ),
        encoding="utf-8",
    )
    sidecar.write_text(json.dumps({"parent": "native-control"}), encoding="utf-8")
    original_instances = list(d1.STREAM_INSTANCES)
    fake_stream = SimpleNamespace(
        consumed_batches=8192,
        schedule_path=d1.DEFAULT_SCHEDULE_DIR / "schedule.jsonl",
    )
    try:
        d1.STREAM_INSTANCES[:] = [fake_stream]
        d1._rewrite_training_report(tmp_path, copy.deepcopy(preflight))
    finally:
        d1.STREAM_INSTANCES[:] = original_instances
    payload = json.loads(report.read_text())
    contract = payload["result"]["d1_ms50_schedule_contract"]
    assert all(contract["checks"].values())
    assert contract["single_change"] == "hidden twin exposure schedule"
    assert contract["model_or_loss_changed"] is False
    assert contract["learned_parameters_added"] == 0
    assert contract["loss_terms_added"] == 0
    assert contract["score_or_arm_metadata_at_model_or_loss_boundary"] is False
    updated_sidecar = json.loads(sidecar.read_text())
    assert updated_sidecar["parent"] == "native-control"
    assert updated_sidecar["schedule_batches_consumed"] == 8192
    assert updated_sidecar["model_or_loss_changed"] is False


def test_terminal_output_parser_does_not_reenter_parent_budget_gate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "completed"
    observed = d1._training_output_path(
        [
            "--checkpoint",
            "/frozen/checkpoint.pt",
            "--optimizer-steps",
            "8192",
            "--output",
            str(output),
        ]
    )
    assert observed == output.resolve()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "PARENT.absolute._validate_args(training_argv)" not in source


def test_preflight_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        d1.write_cpu_preflight(output)


def test_source_keeps_split_names_inside_the_fail_closed_builder_guard() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "lance.dataset" not in source
    assert "optimizer.step" not in source
    assert '"development_lance_opened": False' in source
    assert '"public_test_lance_opened": False' in source
    assert '"metadata_at_model_or_loss_boundary": False' in source
