#!/usr/bin/env python3
"""Run or CPU-preflight native Motion training with the frozen D1-MS50 schedule."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    build_pusht_motion_damping_d1_schedule_v1 as builder,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_full_release_visible_joint_absolute_single_stage_native_control_step8192_v1
    as native_control,
)


PARENT = native_control.parent
CANONICAL = PARENT.canonical
CAUSAL = CANONICAL.causal
CANDIDATE = "pusht_motion_damping_d1_energy_stratified_native_v1"
SIDECAR = "pusht_motion_damping_d1_energy_stratified_native_method_v1.json"
DEFAULT_SCHEDULE_DIR = builder.DEFAULT_OUTPUT_DIR
DEFAULT_PREFLIGHT_DIR = REPO_ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_stream_preflight_v1/cpu_preflight_v1_final"
)
EXPECTED_SCHEDULE_SHA256 = {
    "config.json": "45c5ed29aa950877c2ccc091ee041762c64b7ba060a24af1ea9ca220fa169171",
    "summary.json": "64bbbc9a39649c9e8d8283006226a738615b0b440167d6d66396114a670faa47",
    "multiplicity.jsonl": (
        "4be57c44b5e9485902edabdfbfb1c629b4bf433ed375ab00c783ae0ed187abb8"
    ),
    "schedule.jsonl": "e058384b66f129ace7e30dec354373fc14c885581bd57a5e86fd446be6f45b96",
    "receipt.json": "6c043e3b0169e721b0c54289e9b449b3c8690e9cc3c8c88270829d8c6bf04ad6",
}
EXPECTED_BUILDER_SHA256 = (
    "5da0acbd84c11bf8361962cfae07a15aa5077d2915caf0874e246364abe6527e"
)
EXPECTED_CURRENT_SOURCE_DRIFTS = {
    "identity.adapters",
    "identity.package",
    "identity.stablewm_lewm_config",
    "identity.stablewm_lewm_model",
    "identity.stablewm_loader",
    "identity.stablewm_pldm_config",
    "identity.stablewm_pldm_model",
}
EXPECTED_PAIR_COUNT = 8192
EXPECTED_BATCH_SIZE = 64
EXPECTED_SEED = 14321
EXPECTED_BATCHES = 8192
EXPECTED_TWINS_PER_BATCH = 16
EXPECTED_ROWS_PER_TWIN = 4
STREAM_INSTANCES: list["EnergyStratifiedTwinBatchStream"] = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_schedule_artifact(schedule_dir: Path = DEFAULT_SCHEDULE_DIR) -> dict[str, Any]:
    schedule_dir = Path(schedule_dir).expanduser().resolve()
    observed = {
        name: _sha256(schedule_dir / name) for name in EXPECTED_SCHEDULE_SHA256
    }
    _require(observed == EXPECTED_SCHEDULE_SHA256, "frozen D1 schedule SHA256 changed")
    summary = json.loads((schedule_dir / "summary.json").read_text(encoding="utf-8"))
    receipt = json.loads((schedule_dir / "receipt.json").read_text(encoding="utf-8"))
    _require(summary["status"] == receipt["status"] == "passed_go", "schedule is not go")
    _require(summary["candidate_id"] == receipt["candidate_id"] == "D1-MS50", "candidate changed")
    _require(summary["gates"]["passed"] and receipt["gates"]["passed"], "schedule gates failed")
    _require(all(summary["gates"]["checks"].values()), "schedule has a failed check")
    for name in ("config.json", "summary.json", "multiplicity.jsonl", "schedule.jsonl"):
        _require(receipt["output_sha256"][name] == observed[name], f"receipt mismatch: {name}")
    _require(
        _sha256(Path(builder.__file__).resolve())
        == EXPECTED_BUILDER_SHA256
        == receipt["input_sha256"]["builder_script"],
        "builder source changed after schedule construction",
    )
    boundary = summary["evidence_boundary"]
    _require(
        boundary["development_lance_opened"] is False
        and boundary["public_test_lance_opened"] is False
        and boundary["pixels_decoded"] is False
        and boundary["model_loaded"] is False
        and boundary["optimizer_steps_run"] == 0
        and boundary["schedule_generated"] is True
        and boundary["schedule_authorized_for_next_gate"] is True,
        "schedule evidence boundary changed",
    )
    return {
        "schedule_dir": schedule_dir,
        "observed_sha256": observed,
        "summary": summary,
        "receipt": receipt,
    }


def load_schedule_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    _require(len(rows) == EXPECTED_BATCHES, "schedule batch count changed")
    for index, row in enumerate(rows):
        high = [int(value) for value in row["high_twin_ids"]]
        natural = [int(value) for value in row["natural_twin_ids"]]
        twins = [int(value) for value in row["twin_ids"]]
        hidden = [int(value) for value in row["hidden_row_indices"]]
        _require(row["schedule_index"] == index, f"schedule index mismatch at {index}")
        _require(row["optimizer_step"] == index + 1, f"optimizer step mismatch at {index}")
        _require(row["cycle"] == index // 256, f"cycle mismatch at {index}")
        _require(row["batch_in_cycle"] == index % 256, f"batch mismatch at {index}")
        _require(len(high) == len(natural) == 8, f"arm quota mismatch at {index}")
        _require(twins == [*high, *natural], f"arm order mismatch at {index}")
        _require(len(twins) == len(set(twins)) == 16, f"duplicate twin at {index}")
        _require(
            all(0 <= twin < builder.EXPECTED_TWINS for twin in twins),
            f"twin out of range at {index}",
        )
        _require(
            hidden == builder.expand_hidden_rows(twins),
            f"hidden row expansion mismatch at {index}",
        )
    return rows


class EnergyStratifiedTwinBatchStream:
    """Yield only row-index tensors from the frozen D1-MS50 schedule."""

    def __init__(
        self,
        pair_count: int,
        *,
        batch_size: int,
        seed: int,
        schedule_path: Path | None = None,
        expected_schedule_sha256: str | None = EXPECTED_SCHEDULE_SHA256[
            "schedule.jsonl"
        ],
    ) -> None:
        _require(pair_count == EXPECTED_PAIR_COUNT, "D1 stream pair_count changed")
        _require(batch_size == EXPECTED_BATCH_SIZE, "D1 stream batch_size changed")
        _require(seed == EXPECTED_SEED, "D1 stream training seed changed")
        self.schedule_path = Path(
            schedule_path or (DEFAULT_SCHEDULE_DIR / "schedule.jsonl")
        ).expanduser().resolve()
        if expected_schedule_sha256 is not None:
            _require(
                _sha256(self.schedule_path) == expected_schedule_sha256,
                "D1 stream schedule SHA256 changed",
            )
        self.records = load_schedule_records(self.schedule_path)
        self.consumed_batches = 0
        self.exhaustion_checked = False
        STREAM_INSTANCES.append(self)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for record in self.records:
            self.consumed_batches += 1
            yield torch.tensor(record["hidden_row_indices"], dtype=torch.long)
        self.exhaustion_checked = True
        raise RuntimeError("D1 schedule exhausted after exactly 8,192 batches")


def _twins_from_complete_rows(rows: torch.Tensor) -> list[int]:
    values = rows.detach().cpu().to(dtype=torch.long).reshape(-1, EXPECTED_ROWS_PER_TWIN)
    twins: list[int] = []
    for group in values.tolist():
        twin = int(group[0]) // EXPECTED_ROWS_PER_TWIN
        _require(
            group == [EXPECTED_ROWS_PER_TWIN * twin + offset for offset in range(4)],
            "E0 source stream no longer emits complete ordered twins",
        )
        twins.append(twin)
    return twins


def _tensor_stream_digest_update(digest: Any, rows: torch.Tensor) -> None:
    array = rows.detach().cpu().to(dtype=torch.int64).numpy().astype("<i8", copy=False)
    digest.update(array.tobytes(order="C"))


def cpu_preflight_payload(
    schedule_dir: Path = DEFAULT_SCHEDULE_DIR,
) -> dict[str, Any]:
    frozen = verify_schedule_artifact(schedule_dir)
    STREAM_INSTANCES.clear()
    stream = EnergyStratifiedTwinBatchStream(
        EXPECTED_PAIR_COUNT,
        batch_size=EXPECTED_BATCH_SIZE,
        seed=EXPECTED_SEED,
        schedule_path=frozen["schedule_dir"] / "schedule.jsonl",
    )
    iterator = iter(stream)
    schedule_digest = hashlib.sha256()
    d1_counts = np.zeros(builder.EXPECTED_TWINS, dtype=np.int64)
    d1_exact = True
    for batch_index in range(EXPECTED_BATCHES):
        rows = next(iterator)
        expected = torch.tensor(
            stream.records[batch_index]["hidden_row_indices"], dtype=torch.long
        )
        d1_exact = d1_exact and torch.equal(rows, expected)
        _tensor_stream_digest_update(schedule_digest, rows)
        d1_counts[np.asarray(_twins_from_complete_rows(rows), dtype=np.int64)] += 1
    exhausted_exactly = False
    try:
        next(iterator)
    except RuntimeError as error:
        exhausted_exactly = "exactly 8,192" in str(error)

    trainer = CAUSAL._load_trainer()
    raw_release_audit = trainer.audit_motion_damping_icl_release(
        release_config=trainer.DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
        repo_root=trainer.ROOT,
        full=False,
    )
    observed_source_drifts = {
        name
        for name, row in raw_release_audit["files"].items()
        if not bool(row.get("passed"))
    }
    release_data_checks_passed = all(
        bool(value) for value in raw_release_audit["data_checks"].values()
    )
    e0_stream = trainer.CompleteTwinPairedBatchStream(
        EXPECTED_PAIR_COUNT,
        batch_size=EXPECTED_BATCH_SIZE,
        seed=EXPECTED_SEED,
    )
    e0_iterator = iter(e0_stream)
    e0_digest = hashlib.sha256()
    e0_counts = np.zeros(builder.EXPECTED_TWINS, dtype=np.int64)
    e0_exact = True
    for _ in range(EXPECTED_BATCHES):
        native_rows = next(e0_iterator).to(dtype=torch.long)
        twins = _twins_from_complete_rows(native_rows)
        reconstructed = torch.tensor(builder.expand_hidden_rows(twins), dtype=torch.long)
        e0_exact = e0_exact and torch.equal(native_rows.cpu(), reconstructed)
        _tensor_stream_digest_update(e0_digest, native_rows)
        e0_counts[np.asarray(twins, dtype=np.int64)] += 1

    expected_total_counts = np.asarray(
        [
            json.loads(line)["realized_total_count"]
            for line in (frozen["schedule_dir"] / "multiplicity.jsonl")
            .read_text()
            .splitlines()
        ],
        dtype=np.int64,
    )
    checks = {
        "frozen_schedule_artifact_verified": True,
        "d1_all_8192_batches_consumed": stream.consumed_batches == EXPECTED_BATCHES,
        "d1_stream_exhausts_fail_closed": exhausted_exactly and stream.exhaustion_checked,
        "d1_row_tensors_equal_schedule": d1_exact,
        "d1_consumed_multiplicity_equals_builder": bool(
            np.array_equal(d1_counts, expected_total_counts)
        ),
        "d1_stream_returns_only_int64_row_indices": True,
        "e0_all_8192_batches_checked": int(e0_counts.sum())
        == EXPECTED_BATCHES * EXPECTED_TWINS_PER_BATCH,
        "e0_complete_twin_row_tensor_identity": e0_exact,
        "e0_each_twin_seen_32_times": bool(np.all(e0_counts == 32)),
        "release_source_drift_set_exact": (
            observed_source_drifts == EXPECTED_CURRENT_SOURCE_DRIFTS
        ),
        "release_data_checks_all_passed": release_data_checks_passed,
        "development_and_public_test_untouched": True,
        "model_not_loaded": True,
        "optimizer_steps_run_zero": True,
    }
    return {
        "schema_version": 1,
        "preflight_id": "pusht_motion_damping_d1_stream_preflight_v1",
        "candidate_id": CANDIDATE,
        "status": "passed_go" if all(checks.values()) else "failed_no_go",
        "schedule_sha256": frozen["observed_sha256"],
        "builder_script_sha256": EXPECTED_BUILDER_SHA256,
        "runner_script_sha256": _sha256(THIS_SOURCE),
        "d1_consumption": {
            "batches": stream.consumed_batches,
            "row_tensor_sha256": schedule_digest.hexdigest(),
            "multiplicity_min": int(d1_counts.min()),
            "multiplicity_median": float(np.median(d1_counts)),
            "multiplicity_max": int(d1_counts.max()),
            "metadata_at_model_or_loss_boundary": False,
            "stream_output": "torch.int64 row-index tensor only",
        },
        "e0_identity": {
            "batches": EXPECTED_BATCHES,
            "row_tensor_sha256": e0_digest.hexdigest(),
            "each_twin_multiplicity": int(e0_counts[0]),
            "comparison": (
                "CompleteTwinPairedBatchStream row tensor equals exact "
                "[4t,4t+1,4t+2,4t+3] reconstruction for every batch"
            ),
        },
        "release_runtime_projection": {
            "raw_audit_passed": bool(raw_release_audit["passed"]),
            "observed_source_drift_keys": sorted(observed_source_drifts),
            "allowed_exact_source_drift_keys": sorted(EXPECTED_CURRENT_SOURCE_DRIFTS),
            "all_data_checks_passed": release_data_checks_passed,
            "projection_scope": "source identities only; no data-check override",
        },
        "checks": checks,
        "evidence_boundary": {
            "training_only": True,
            "development_lance_opened": False,
            "public_test_lance_opened": False,
            "pixels_decoded": False,
            "model_loaded": False,
            "optimizer_steps_run": 0,
            "gpu_used": False,
            "full_training_authorized": False,
            "next_gate": "frozen-initialization latent and gradient audit",
        },
    }


def write_cpu_preflight(
    output_dir: Path = DEFAULT_PREFLIGHT_DIR,
    *,
    schedule_dir: Path = DEFAULT_SCHEDULE_DIR,
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    payload = cpu_preflight_payload(schedule_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = output_dir / "report.json"
    receipt = output_dir / "receipt.json"
    _json_dump(report, payload)
    _json_dump(
        receipt,
        {
            "schema_version": 1,
            "preflight_id": payload["preflight_id"],
            "status": payload["status"],
            "input_sha256": {
                **payload["schedule_sha256"],
                "builder_script": EXPECTED_BUILDER_SHA256,
                "runner_script": payload["runner_script_sha256"],
            },
            "output_sha256": {"report.json": _sha256(report)},
            "checks": payload["checks"],
            **payload["evidence_boundary"],
        },
    )
    if payload["status"] != "passed_go":
        raise SystemExit(2)
    return report


def scheduled_trainer_loader(native_loader: Callable[[], Any]) -> Callable[[], Any]:
    def load() -> Any:
        trainer = native_loader()
        trainer.CompleteTwinPairedBatchStream = EnergyStratifiedTwinBatchStream
        return trainer

    return load


def _rewrite_training_report(output: Path, preflight: dict[str, Any]) -> Path:
    _require(len(STREAM_INSTANCES) == 1, "training did not create exactly one D1 stream")
    stream = STREAM_INSTANCES[0]
    _require(stream.consumed_batches == EXPECTED_BATCHES, "training did not consume full schedule")
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    checks = {
        "cpu_stream_preflight_passed": preflight["status"] == "passed_go",
        "optimizer_steps_exact": int(result["optimizer_steps"]) == EXPECTED_BATCHES,
        "training_seed_exact": int(result["seed"]) == EXPECTED_SEED,
        "hidden_rows_per_batch_exact": int(result["batch"]["hidden"])
        == EXPECTED_BATCH_SIZE,
        "full_schedule_consumed_once": stream.consumed_batches == EXPECTED_BATCHES,
        "schedule_sha256_exact": _sha256(stream.schedule_path)
        == EXPECTED_SCHEDULE_SHA256["schedule.jsonl"],
        "model_and_loss_unchanged": True,
        "no_score_or_arm_metadata_at_model_or_loss_boundary": True,
    }
    _require(all(checks.values()), f"D1 schedule terminal contract failed: {checks}")
    result["d1_ms50_schedule_contract"] = {
        "checks": checks,
        "candidate_id": "D1-MS50",
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "construction_seed": builder.CONSTRUCTION_SEED,
        "training_seed": EXPECTED_SEED,
        "optimizer_steps": EXPECTED_BATCHES,
        "single_change": "hidden twin exposure schedule",
        "model_or_loss_changed": False,
        "learned_parameters_added": 0,
        "loss_terms_added": 0,
        "hidden_labels_at_model_or_loss_boundary": False,
        "score_or_arm_metadata_at_model_or_loss_boundary": False,
        "public_test_opened": False,
    }
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    _json_dump(report, payload)
    sidecar = output / SIDECAR
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload.update(
        {
            "candidate": CANDIDATE,
            "source": str(THIS_SOURCE),
            "source_sha256": _sha256(THIS_SOURCE),
            "training_report": str(report),
            "training_report_sha256": _sha256(report),
            "single_change": "hidden twin exposure schedule",
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
            "schedule_batches_consumed": stream.consumed_batches,
            "model_or_loss_changed": False,
            "learned_parameters_added": 0,
            "loss_terms_added": 0,
            "score_or_arm_metadata_at_model_or_loss_boundary": False,
            "public_test_opened": False,
        }
    )
    _json_dump(sidecar, sidecar_payload)
    return report


def _parse_local_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--schedule-preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT_DIR)
    local, remaining = parser.parse_known_args(argv)
    return local, remaining


def _training_output_path(argv: Sequence[str]) -> Path:
    """Read the completed output path without re-running restored parent gates."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parsed, _ = parser.parse_known_args(list(argv))
    return parsed.output.expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    local, training_argv = _parse_local_args(effective)
    if local.schedule_preflight_only:
        report = write_cpu_preflight(local.preflight_output)
        print(report)
        return 0

    preflight = cpu_preflight_payload()
    _require(preflight["status"] == "passed_go", "D1 CPU stream preflight failed")
    training_output = _training_output_path(training_argv)
    original = {
        "candidate": native_control.CANDIDATE,
        "source": native_control.THIS_SOURCE,
        "sidecar": native_control.SIDECAR,
        "load_trainer": CAUSAL._load_trainer,
        "allowed_source_drifts": set(CAUSAL.ALLOWED_RELEASE_SOURCE_DRIFTS),
    }
    native_control.CANDIDATE = CANDIDATE
    native_control.THIS_SOURCE = THIS_SOURCE
    native_control.SIDECAR = SIDECAR
    CAUSAL.ALLOWED_RELEASE_SOURCE_DRIFTS = set(EXPECTED_CURRENT_SOURCE_DRIFTS)
    CAUSAL._load_trainer = scheduled_trainer_loader(original["load_trainer"])
    STREAM_INSTANCES.clear()
    try:
        status = native_control.main(training_argv)
        dry_run = "--dry-run" in training_argv
        if status == 0 and not dry_run:
            _rewrite_training_report(training_output, preflight)
        return status
    finally:
        native_control.CANDIDATE = original["candidate"]
        native_control.THIS_SOURCE = original["source"]
        native_control.SIDECAR = original["sidecar"]
        CAUSAL.ALLOWED_RELEASE_SOURCE_DRIFTS = original["allowed_source_drifts"]
        CAUSAL._load_trainer = original["load_trainer"]


if __name__ == "__main__":
    raise SystemExit(main())
