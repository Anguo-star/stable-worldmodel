#!/usr/bin/env python3
"""Run the frozen, Training-only Motion four-arm root-cause zero-step audit.

The audit compares the pre-registered ``D0``, ``REL50``, ``ABS50`` and
``HASH50`` exposure schedules at one frozen LeWM initialization.  It takes no
optimizer steps and never opens a non-Training split.  The schedule artifact is
an explicit input: its receipt and every declared schedule/multiplicity hash
are checked before a model is loaded.

Two forward/gradient passes are reported.  ``train`` preserves the native
BatchNorm semantics of ``pred_proj``; ``eval`` is a control for batch-coupling.
Neither pass is a training result or an ICL endpoint evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
ROOT = THIS_SOURCE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    analyze_pusht_motion_damping_d1_latent_gradient_zero_step_v1 as frozen_audit,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    conditional_signal_metrics as signal_metrics,
)


ANALYSIS_ID = "pusht_motion_damping_root_cause_zero_step_v1"
ANALYSIS_VARIANT = "analysis_motion_root_cause_zero_step_v1"
ARMS = ("D0", "REL50", "ABS50", "HASH50")
EXPOSURE_ARMS = ("REL50", "ABS50", "HASH50")
PAIRED_COMPARISONS = (
    ("REL50", "HASH50"),
    ("REL50", "ABS50"),
    ("HASH50", "D0"),
)
ALLOWED_PARAMETER_GROUPS = frozen_audit.ALLOWED_PARAMETER_GROUPS
EXPECTED_TWINS = frozen_audit.EXPECTED_TWINS
EXPECTED_PAIRS = frozen_audit.EXPECTED_PAIRS
EXPECTED_HIDDEN_ROWS = frozen_audit.EXPECTED_HIDDEN_ROWS
ORIGINAL_ROWS = frozen_audit.ORIGINAL_ROWS
HIDDEN_ROWS_PER_BATCH = frozen_audit.HIDDEN_ROWS_PER_BATCH
TWINS_PER_BATCH = frozen_audit.TWINS_PER_BATCH
AUDIT_BATCH_COUNT = frozen_audit.AUDIT_BATCH_COUNT
AUDIT_SCHEDULE_INDICES = frozen_audit.AUDIT_SCHEDULE_INDICES
LATENT_NEIGHBOUR_SCALES = frozen_audit.LATENT_NEIGHBOUR_SCALES
LATENT_ENCODE_BATCH_SIZE = frozen_audit.LATENT_ENCODE_BATCH_SIZE
EXPECTED_CHECKPOINT_SHA256 = frozen_audit.EXPECTED_CHECKPOINT_SHA256
EXPECTED_MODEL_STATE_SHA256 = frozen_audit.EXPECTED_MODEL_STATE_SHA256
EXPECTED_V1_CATALOG_SHA256 = frozen_audit.EXPECTED_V1_CATALOG_SHA256
EXPECTED_D1_RUNNER_SHA256 = frozen_audit.EXPECTED_D1_RUNNER_SHA256
EXPECTED_SCHEDULE_BATCHES = frozen_audit.d1_runner.EXPECTED_BATCHES
EXPECTED_SCHEDULE_SEED = frozen_audit.d1_runner.EXPECTED_SEED
BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 20260901

DEFAULT_CHECKPOINT = frozen_audit.DEFAULT_CHECKPOINT
DEFAULT_CATALOG = frozen_audit.DEFAULT_CATALOG
DEFAULT_ORIGINAL_H5 = frozen_audit.native_gradient.DEFAULT_ORIGINAL_H5
DEFAULT_ORIGINAL_LANCE = frozen_audit.native_gradient.DEFAULT_ORIGINAL_LANCE
DEFAULT_SCHEDULE_DIR = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_root_cause_comparator_schedules_v1/"
    "comparator_schedules_v1_final"
)
DEFAULT_OUTPUT = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_root_cause_zero_step_v1/initialization_16batch_v1/"
    "report.json"
)

FORBIDDEN_SPLIT_TOKENS = frozenset(
    {"development", "public", "test", "validation"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = assert_training_only_path(path)
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _tokens(path: Path) -> set[str]:
    return {
        token
        for part in Path(path).parts
        for token in re.split(r"[^a-z0-9]+", part.lower())
        if token
    }


def assert_training_only_path(
    path: Path,
    *,
    expected_name: str | None = None,
) -> Path:
    """Reject non-Training paths before inspecting their contents."""

    resolved = Path(path).expanduser().resolve(strict=False)
    _require(
        _tokens(resolved).isdisjoint(FORBIDDEN_SPLIT_TOKENS),
        f"forbidden non-Training input path: {resolved}",
    )
    if expected_name is not None:
        _require(resolved.name == expected_name, f"expected {expected_name}: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    path = assert_training_only_path(path)
    _require(path.is_file(), f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON input is not an object: {path}")
    return value


def _normalise_key(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _flatten_hash_map(value: Any, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_hash_map(nested, child))
    elif isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        result[prefix] = value.lower()
    return result


def _file_hash_from_receipt(
    hashes: Mapping[str, str],
    *,
    path: Path,
    root: Path,
    arm: str,
    kind: str,
) -> tuple[str, str]:
    relative = _normalise_key(path.relative_to(root).as_posix())
    exact_keys = {
        relative,
        f"{arm.lower()}/{kind.lower()}",
        f"{arm.lower()}_{kind.lower()}",
        f"{kind.lower()}_{arm.lower()}",
    }
    exact = [value for key, value in hashes.items() if _normalise_key(key) in exact_keys]
    if len(exact) == 1:
        return exact[0], "exact"
    suffix = f"/{arm.lower()}/{kind.lower()}"
    candidates = [
        (key, value)
        for key, value in hashes.items()
        if _normalise_key(key).endswith(suffix)
        or _normalise_key(key).endswith(f"/{kind.lower()}")
        and arm.lower() in _normalise_key(key)
    ]
    _require(
        len(candidates) == 1,
        f"schedule receipt has no unique output hash for {arm}/{kind}: "
        f"{sorted(hashes)}",
    )
    return candidates[0][1], candidates[0][0]


def _verify_source_hash_nodes(value: Any, *, source_records: list[dict[str, Any]]) -> None:
    """Verify receipt nodes that bind a source path to a SHA256."""

    if isinstance(value, Mapping):
        raw_hash = value.get("sha256")
        if isinstance(raw_hash, str) and re.fullmatch(r"[0-9a-fA-F]{64}", raw_hash):
            for key in ("path", "source_path", "file", "source", "builder_path"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.endswith((".py", ".json", ".jsonl")):
                    candidate_path = Path(candidate)
                    if not candidate_path.is_absolute():
                        candidate_path = ROOT / candidate_path
                    path = assert_training_only_path(candidate_path)
                    _require(path.is_file(), f"receipt source is missing: {path}")
                    observed = _sha256(path)
                    _require(
                        observed == raw_hash.lower(),
                        f"receipt source SHA256 changed: {path}",
                    )
                    source_records.append(
                        {"path": str(path), "sha256": observed, "verified": True}
                    )
                    break
        for nested in value.values():
            _verify_source_hash_nodes(nested, source_records=source_records)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _verify_source_hash_nodes(nested, source_records=source_records)


def _verify_explicit_source_binding(
    receipt: Mapping[str, Any], *, source_records: list[dict[str, Any]]
) -> None:
    """Verify top-level ``*_source_path``/``*_source_sha256`` receipt pairs."""

    for hash_key, raw_hash in receipt.items():
        key = str(hash_key).lower()
        if not ("source" in key or "builder" in key) or not key.endswith(("sha256", "sha")):
            continue
        if not isinstance(raw_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", raw_hash):
            continue
        path_candidates = (
            hash_key.removesuffix("_sha256") + "_path",
            hash_key.removesuffix("_sha") + "_path",
            "builder_source_path",
            "source_path",
        )
        source_path = next(
            (receipt.get(candidate) for candidate in path_candidates if isinstance(receipt.get(candidate), str)),
            None,
        )
        if source_path is None:
            continue
        source_candidate = Path(source_path)
        if not source_candidate.is_absolute():
            source_candidate = ROOT / source_candidate
        path = assert_training_only_path(source_candidate)
        _require(path.is_file(), f"receipt source is missing: {path}")
        observed = _sha256(path)
        _require(observed == raw_hash.lower(), f"receipt source SHA256 changed: {path}")
        source_records.append({"path": str(path), "sha256": observed, "verified": True})


def verify_schedule_receipt(
    schedule_dir: Path,
    files: Mapping[str, Mapping[str, Path]],
    *,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify receipt-declared hashes and zero-step/split boundary fields."""

    schedule_dir = assert_training_only_path(schedule_dir)
    receipt_path = assert_training_only_path(schedule_dir / "receipt.json", expected_name="receipt.json")
    _require(receipt_path.is_file(), f"missing schedule receipt: {receipt_path}")
    receipt_sha256 = _sha256(receipt_path)
    if expected_receipt_sha256 is not None:
        _require(
            bool(re.fullmatch(r"[0-9a-fA-F]{64}", expected_receipt_sha256))
            and receipt_sha256 == expected_receipt_sha256.lower(),
            "schedule receipt SHA256 does not match the explicit expectation",
        )
    receipt = _read_json(receipt_path)
    raw_output_hashes = receipt.get("output_sha256")
    _require(isinstance(raw_output_hashes, Mapping), "schedule receipt lacks output_sha256")
    output_hashes = _flatten_hash_map(raw_output_hashes)
    observed_outputs: dict[str, str] = {}
    for arm in ARMS:
        for kind, path in files[arm].items():
            path = assert_training_only_path(path)
            _require(path.is_file(), f"missing {arm} {kind}: {path}")
            declared, key = _file_hash_from_receipt(
                output_hashes,
                path=path,
                root=schedule_dir,
                arm=arm,
                kind=kind,
            )
            observed = _sha256(path)
            _require(observed == declared, f"schedule receipt output hash changed: {key}")
            observed_outputs[f"{arm}/{kind}"] = observed

    for name in (
        "development_lance_opened",
        "public_test_lance_opened",
        "pixels_decoded",
    ):
        if name in receipt:
            _require(receipt[name] is False, f"schedule receipt boundary changed: {name}")
    for name in ("optimizer_steps", "optimizer_steps_run"):
        if name in receipt:
            _require(int(receipt[name]) == 0, f"schedule receipt boundary changed: {name}")
    if "schedule_generated" in receipt:
        _require(receipt["schedule_generated"] is True, "schedule receipt is not generated")
    if "arms" in receipt:
        _require(set(receipt["arms"]) == set(ARMS), "schedule receipt arm set changed")

    source_records: list[dict[str, Any]] = []
    _verify_source_hash_nodes(receipt, source_records=source_records)
    _verify_explicit_source_binding(receipt, source_records=source_records)
    for container_name in ("input_sha256", "source_sha256", "builder_source_sha256"):
        value = receipt.get(container_name)
        if isinstance(value, str):
            _require(
                bool(re.fullmatch(r"[0-9a-fA-F]{64}", value)),
                f"invalid schedule receipt hash: {container_name}",
            )
        elif isinstance(value, Mapping):
            for key, nested in _flatten_hash_map(value).items():
                if "source" in key.lower() or "builder" in key.lower():
                    _require(
                        bool(re.fullmatch(r"[0-9a-fA-F]{64}", nested)),
                        f"invalid schedule source hash: {key}",
                    )

    boundary = {
        name: receipt.get(name)
        for name in (
            "development_lance_opened",
            "public_test_lance_opened",
            "pixels_decoded",
            "model_loaded",
            "optimizer_steps",
            "optimizer_steps_run",
            "schedule_generated",
        )
        if name in receipt
    }
    return {
        "path": str(receipt_path),
        "sha256": receipt_sha256,
        "status": receipt.get("status"),
        "output_sha256": observed_outputs,
        "source_hashes_verified": source_records,
        "boundary": boundary,
    }


def _candidate_schedule_paths(root: Path, arm: str, kind: str) -> list[Path]:
    lower = arm.lower()
    return [
        root / arm / f"{kind}.jsonl",
        root / lower / f"{kind}.jsonl",
        root / "arms" / arm / f"{kind}.jsonl",
        root / "arms" / lower / f"{kind}.jsonl",
        root / f"{arm}_{kind}.jsonl",
        root / f"{lower}_{kind}.jsonl",
        root / f"{kind}_{arm}.jsonl",
        root / f"{kind}_{lower}.jsonl",
    ]


def resolve_schedule_files(
    schedule_dir: Path,
    *,
    explicit: Mapping[str, Mapping[str, Path | None]] | None = None,
) -> dict[str, dict[str, Path]]:
    schedule_dir = assert_training_only_path(schedule_dir)
    result: dict[str, dict[str, Path]] = {}
    for arm in ARMS:
        result[arm] = {}
        for kind in ("schedule", "multiplicity"):
            supplied = None if explicit is None else explicit.get(arm, {}).get(kind)
            if supplied is not None:
                path = assert_training_only_path(Path(supplied))
            else:
                candidates = _candidate_schedule_paths(schedule_dir, arm, kind)
                path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
                path = assert_training_only_path(path)
            result[arm][kind] = path
    return result


def validate_twin_rows(rows: torch.Tensor, *, expected_twins: int = EXPECTED_TWINS) -> list[int]:
    """Validate complete, ordered four-row twin groups and return twin IDs."""

    values = rows.detach().cpu().to(dtype=torch.long).reshape(-1, 4)
    _require(values.shape == (TWINS_PER_BATCH, 4), "hidden batch shape changed")
    twins: list[int] = []
    for group in values.tolist():
        twin = int(group[0]) // 4
        _require(group == [4 * twin + offset for offset in range(4)], "split twin")
        _require(0 <= twin < expected_twins, "twin id out of range")
        twins.append(twin)
    _require(len(set(twins)) == TWINS_PER_BATCH, "duplicate twin within batch")
    return twins


def _row_value(row: Mapping[str, Any], names: Sequence[str], *, label: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise RuntimeError(f"schedule row lacks {label}: {sorted(row)}")


def load_schedule_records(
    path: Path,
    *,
    arm: str = "D0",
    expected_batches: int = EXPECTED_SCHEDULE_BATCHES,
) -> list[dict[str, Any]]:
    path = assert_training_only_path(path)
    _require(path.is_file(), f"missing schedule: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(bool(line.strip()), f"blank schedule row at line {line_number}")
            value = json.loads(line)
            _require(isinstance(value, Mapping), f"schedule row {line_number} is not an object")
            row = dict(value)
            index = int(row.get("schedule_index", row.get("batch_index", line_number - 1)))
            _require(index == line_number - 1, f"schedule index mismatch at {line_number}")
            hidden = _row_value(
                row,
                ("hidden_row_indices", "row_indices", "hidden_rows", "indices"),
                label="hidden row indices",
            )
            rows = torch.tensor(hidden, dtype=torch.long)
            twins = validate_twin_rows(rows)
            if "twin_ids" in row:
                _require([int(value) for value in row["twin_ids"]] == twins, "schedule twin order changed")
            if "optimizer_step" in row:
                _require(int(row["optimizer_step"]) == index + 1, "schedule optimizer-step metadata changed")
            if "cycle" in row:
                _require(int(row["cycle"]) == index // 256, "schedule cycle metadata changed")
            if "batch_in_cycle" in row:
                _require(int(row["batch_in_cycle"]) == index % 256, "schedule batch metadata changed")
            records.append(
                {
                    "schedule_index": index,
                    "hidden_row_indices": [int(value) for value in rows.tolist()],
                    "twin_ids": twins,
                    "arm": arm,
                }
            )
    _require(len(records) == expected_batches, f"{arm} schedule batch count changed")
    return records


def load_multiplicity(
    path: Path,
    *,
    arm: str = "D0",
    expected_twins: int = EXPECTED_TWINS,
    expected_total: int | None = None,
) -> np.ndarray:
    path = assert_training_only_path(path)
    _require(path.is_file(), f"missing multiplicity: {path}")
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(bool(line.strip()), f"blank multiplicity row at line {line_number}")
            value = json.loads(line)
            _require(isinstance(value, Mapping), f"multiplicity row {line_number} is not an object")
            rows.append(value)
    _require(len(rows) == expected_twins, f"{arm} multiplicity twin count changed")
    counts = np.empty(expected_twins, dtype=np.int64)
    for twin, row in enumerate(rows):
        _require(int(row.get("twin_id", twin)) == twin, f"{arm} multiplicity twin order changed")
        raw = _row_value(
            row,
            (
                "realized_total_count",
                "realized_full_count",
                "realized_count",
                "exposure_count",
                "multiplicity",
                "count",
            ),
            label="realized count",
        )
        count = int(raw)
        _require(count >= 0, f"{arm} multiplicity is negative")
        counts[twin] = count
    _require(bool(np.all(counts > 0)), f"{arm} has zero-support twins")
    if expected_total is not None:
        _require(int(counts.sum()) == expected_total, f"{arm} multiplicity total changed")
    return counts


def _stream_digest(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        values = np.asarray(record["hidden_row_indices"], dtype="<i8")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _catalog_cells(catalog_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    cells = np.asarray([int(row["coverage_cell"]) for row in catalog_rows], dtype=np.int64)
    _require(cells.shape == (EXPECTED_TWINS,), "catalog coverage shape changed")
    _require(np.array_equal(np.unique(cells), np.arange(64)), "catalog coverage cells changed")
    _require(np.all(np.bincount(cells, minlength=64) == 64), "catalog coverage balance changed")
    return cells


def validate_schedule_invariants(
    schedule_set: Mapping[str, Mapping[str, Any]],
    *,
    cells: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate shared 16-strata budget and matched comparator multiplicities."""

    if "arms" in schedule_set and isinstance(schedule_set["arms"], Mapping):
        schedule_set = schedule_set["arms"]  # type: ignore[assignment]
    _require(set(schedule_set) == set(ARMS), "schedule arm set changed")
    checks: dict[str, bool] = {}
    reference_indices: list[int] | None = None
    for arm in ARMS:
        records = schedule_set[arm]["records"]
        counts = np.asarray(schedule_set[arm]["counts"], dtype=np.int64)
        checks[f"{arm.lower()}_schedule_batch_count"] = len(records) == EXPECTED_SCHEDULE_BATCHES
        checks[f"{arm.lower()}_multiplicity_positive"] = bool(np.all(counts > 0))
        checks[f"{arm.lower()}_multiplicity_total"] = int(counts.sum()) == EXPECTED_SCHEDULE_BATCHES * TWINS_PER_BATCH
        indices = [int(row["schedule_index"]) for row in records]
        if reference_indices is None:
            reference_indices = indices
        checks[f"{arm.lower()}_schedule_index_identity"] = indices == reference_indices
        checks[f"{arm.lower()}_audit_indices_present"] = all(
            index < len(records) for index in AUDIT_SCHEDULE_INDICES
        )
        for index in AUDIT_SCHEDULE_INDICES:
            row = records[index]
            checks[f"{arm.lower()}_batch_{index}_twin_integrity"] = (
                len(row["twin_ids"]) == TWINS_PER_BATCH
                and len(set(row["twin_ids"])) == TWINS_PER_BATCH
            )

    if cells is not None:
        cells = np.asarray(cells, dtype=np.int64)
        _require(cells.shape == (EXPECTED_TWINS,), "schedule cell vector shape changed")
        for arm in ARMS:
            counts = np.asarray(schedule_set[arm]["counts"], dtype=np.int64)
            checks[f"{arm.lower()}_cell_mass_balance"] = all(
                int(counts[cells == cell].sum()) == int(counts[cells == 0].sum())
                for cell in range(64)
            )
        for arm in EXPOSURE_ARMS[1:]:
            left = np.asarray(schedule_set["REL50"]["counts"])
            right = np.asarray(schedule_set[arm]["counts"])
            checks[f"{arm.lower()}_within_cell_weight_multiset"] = all(
                np.array_equal(np.sort(left[cells == cell]), np.sort(right[cells == cell]))
                for cell in range(64)
            )
    else:
        for arm in EXPOSURE_ARMS[1:]:
            checks[f"{arm.lower()}_weight_multiset"] = np.array_equal(
                np.sort(np.asarray(schedule_set["REL50"]["counts"])),
                np.sort(np.asarray(schedule_set[arm]["counts"])),
            )
    checks["same_16_preregistered_schedule_strata"] = (
        reference_indices is not None
        and [reference_indices[index] for index in AUDIT_SCHEDULE_INDICES]
        == list(AUDIT_SCHEDULE_INDICES)
    )
    checks["all_invariants_pass"] = all(checks.values())
    _require(checks["all_invariants_pass"], f"schedule invariant failed: {checks}")
    return checks


def load_schedule_set(
    schedule_dir: Path = DEFAULT_SCHEDULE_DIR,
    *,
    explicit: Mapping[str, Mapping[str, Path | None]] | None = None,
    expected_receipt_sha256: str | None = None,
    expected_batches: int = EXPECTED_SCHEDULE_BATCHES,
) -> dict[str, Any]:
    schedule_dir = assert_training_only_path(schedule_dir)
    files = resolve_schedule_files(schedule_dir, explicit=explicit)
    receipt = verify_schedule_receipt(
        schedule_dir,
        files,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    schedule_set: dict[str, Any] = {}
    total = expected_batches * TWINS_PER_BATCH
    for arm in ARMS:
        records = load_schedule_records(
            files[arm]["schedule"], arm=arm, expected_batches=expected_batches
        )
        counts = load_multiplicity(
            files[arm]["multiplicity"],
            arm=arm,
            expected_twins=EXPECTED_TWINS,
            expected_total=total,
        )
        schedule_set[arm] = {
            "arm": arm,
            "schedule_path": files[arm]["schedule"],
            "multiplicity_path": files[arm]["multiplicity"],
            "schedule_sha256": _sha256(files[arm]["schedule"]),
            "multiplicity_sha256": _sha256(files[arm]["multiplicity"]),
            "records": records,
            "audit_records": [records[index] for index in AUDIT_SCHEDULE_INDICES],
            "counts": counts,
            "weights": counts.astype(np.float64) / float(counts.sum()),
            "stream_sha256": _stream_digest(records),
        }
    _require(
        all(
            schedule_set[arm]["audit_records"][batch]["schedule_index"]
            == AUDIT_SCHEDULE_INDICES[batch]
            for arm in ARMS
            for batch in range(AUDIT_BATCH_COUNT)
        ),
        "audit schedule strata are not the frozen indices",
    )
    return {
        "schedule_dir": schedule_dir,
        "receipt": receipt,
        "arms": schedule_set,
        "expected_batches": expected_batches,
    }


def verify_schedule_artifact(
    schedule_dir: Path = DEFAULT_SCHEDULE_DIR,
    *,
    schedule_files: Mapping[str, Mapping[str, Path | None]] | None = None,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper exposing the schedule receipt audit directly."""

    return load_schedule_set(
        schedule_dir,
        explicit=schedule_files,
        expected_receipt_sha256=expected_receipt_sha256,
    )


def static_identity(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    catalog: Path = DEFAULT_CATALOG,
    schedule_dir: Path = DEFAULT_SCHEDULE_DIR,
    original_h5: Path | None = DEFAULT_ORIGINAL_H5,
    original_lance: Path | None = DEFAULT_ORIGINAL_LANCE,
    schedule_files: Mapping[str, Mapping[str, Path | None]] | None = None,
    expected_schedule_receipt_sha256: str | None = None,
    schedule_root: Path | None = None,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Perform all static checks without loading a model or decoding pixels."""

    if schedule_root is not None:
        schedule_dir = schedule_root
    if expected_receipt_sha256 is not None:
        expected_schedule_receipt_sha256 = expected_receipt_sha256
    checkpoint = assert_training_only_path(checkpoint)
    catalog = assert_training_only_path(catalog, expected_name="per_twin_catalog.jsonl")
    _require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
    _require(_sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256, "checkpoint changed")
    _require(catalog.is_file(), f"missing frozen catalog: {catalog}")
    _require(_sha256(catalog) == EXPECTED_V1_CATALOG_SHA256, "catalog changed")
    _require(
        _sha256(Path(frozen_audit.d1_runner.__file__).resolve()) == EXPECTED_D1_RUNNER_SHA256,
        "frozen D1 runner changed",
    )
    catalog_rows, neighbours = frozen_audit.load_catalog(catalog)
    schedules = load_schedule_set(
        schedule_dir,
        explicit=schedule_files,
        expected_receipt_sha256=expected_schedule_receipt_sha256,
    )
    cells = _catalog_cells(catalog_rows)
    schedule_checks = validate_schedule_invariants(schedules["arms"], cells=cells)
    optional_inputs = {}
    for name, value in (("original_h5", original_h5), ("original_lance", original_lance)):
        if value is not None:
            optional_inputs[name] = str(assert_training_only_path(value))
    return {
        "checkpoint": {"path": str(checkpoint), "sha256": EXPECTED_CHECKPOINT_SHA256},
        "catalog": {
            "path": str(catalog),
            "sha256": EXPECTED_V1_CATALOG_SHA256,
            "directed_neighbour_shape": list(neighbours.shape),
        },
        "schedule": {
            "dir": str(schedules["schedule_dir"]),
            "receipt": schedules["receipt"],
            "arms": {
                arm: {
                    "schedule_path": str(schedules["arms"][arm]["schedule_path"]),
                    "multiplicity_path": str(schedules["arms"][arm]["multiplicity_path"]),
                    "schedule_sha256": schedules["arms"][arm]["schedule_sha256"],
                    "multiplicity_sha256": schedules["arms"][arm]["multiplicity_sha256"],
                    "stream_sha256": schedules["arms"][arm]["stream_sha256"],
                }
                for arm in ARMS
            },
            "audit_schedule_indices": list(AUDIT_SCHEDULE_INDICES),
            "checks": schedule_checks,
        },
        "optional_training_inputs": optional_inputs,
        "authority": {
            "training_only": True,
            "optimizer_steps": 0,
            "development_opened": False,
            "public_test_opened": False,
            "model_loaded": False,
            "pixels_decoded": False,
            "full_training_authorized": False,
        },
    }


def weighted_latent_metrics(
    query_latents: torch.Tensor,
    future_latents: torch.Tensor,
    neighbours: np.ndarray,
    weights: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute weighted C/B/rho on the frozen physical neighbour graph."""

    query = query_latents.detach().cpu().float().reshape(EXPECTED_PAIRS, 2, -1)
    future = future_latents.detach().cpu().float().reshape(EXPECTED_PAIRS, 2, -1)
    _require(bool(torch.isfinite(query).all()), "non-finite query latent")
    _require(bool(torch.isfinite(future).all()), "non-finite future latent")
    _require(np.asarray(neighbours).shape == (EXPECTED_PAIRS, max(LATENT_NEIGHBOUR_SCALES)), "neighbour graph shape changed")
    query_identity_error = float((query[:, 0] - query[:, 1]).abs().max())
    directed = future.mean(dim=1) - query.mean(dim=1)
    conditional = 0.25 * (future[:, 1] - future[:, 0]).square().mean(dim=1)
    twin_conditional = conditional.reshape(EXPECTED_TWINS, 2).mean(dim=1).double()
    background_by_k: dict[int, torch.Tensor] = {}
    neighbour_tensor = torch.from_numpy(np.asarray(neighbours, dtype=np.int64))
    for scale in LATENT_NEIGHBOUR_SCALES:
        values = torch.empty(EXPECTED_PAIRS, dtype=torch.float64)
        for start in range(0, EXPECTED_PAIRS, 128):
            stop = min(start + 128, EXPECTED_PAIRS)
            local = directed[neighbour_tensor[start:stop, :scale]]
            centered = local - local.mean(dim=1, keepdim=True)
            values[start:stop] = centered.double().square().mean(dim=(1, 2))
        background_by_k[scale] = values.reshape(EXPECTED_TWINS, 2).mean(dim=1)

    arms: dict[str, Any] = {}
    for arm in ARMS:
        weight = torch.from_numpy(np.asarray(weights[arm], dtype=np.float64))
        _require(weight.shape == (EXPECTED_TWINS,), f"{arm} weight shape changed")
        _require(bool(torch.isfinite(weight).all()) and bool((weight > 0).all()), f"{arm} invalid weights")
        _require(abs(float(weight.sum()) - 1.0) <= 1.0e-12, f"{arm} weights do not sum to one")
        weighted_c = float((weight * twin_conditional).sum())
        by_k: dict[str, Any] = {}
        for scale in LATENT_NEIGHBOUR_SCALES:
            weighted_b = float((weight * background_by_k[scale]).sum())
            rho = _safe_ratio(weighted_c, weighted_c + weighted_b)
            _require(rho is not None, f"{arm} latent rho denominator is zero")
            by_k[str(scale)] = {
                "weighted_C": weighted_c,
                "weighted_B": weighted_b,
                "weighted_conditional_energy": weighted_c,
                "weighted_background_energy": weighted_b,
                "rho_lat": rho,
            }
        arms[arm] = {"weighted_C": weighted_c, "by_k": by_k}

    comparisons: dict[str, Any] = {}
    for left, right in PAIRED_COMPARISONS:
        key = f"{left}_vs_{right}"
        comparisons[key] = {}
        for scale in LATENT_NEIGHBOUR_SCALES:
            left_value = arms[left]["by_k"][str(scale)]
            right_value = arms[right]["by_k"][str(scale)]
            comparisons[key][str(scale)] = {
                "rho_lat_absolute_delta": left_value["rho_lat"] - right_value["rho_lat"],
                "weighted_C_difference": left_value["weighted_C"] - right_value["weighted_C"],
                "weighted_B_difference": left_value["weighted_B"] - right_value["weighted_B"],
            }
    return {
        "representation": "frozen initialization LeWM image embedding, eval encoder mode",
        "physical_neighbour_graph_reused": True,
        "query_mode_identity_max_abs_error": query_identity_error,
        "arms": arms,
        "comparisons": comparisons,
        "comparison": comparisons,
    }


def _group(name: str) -> str:
    return name.split(".", 1)[0]


def _scope_vector(values: Sequence[torch.Tensor | None], groups: Sequence[str], scope: str) -> torch.Tensor:
    chunks = [
        value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        for value, group in zip(values, groups, strict=True)
        if value is not None and (scope == "all" or group == scope)
    ]
    if not chunks:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(chunks)


class DeviceGradientAccumulator(frozen_audit.DeviceGradientAccumulator):
    """Frozen accumulator plus bounded per-batch geometry summaries.

    Full per-sample parameter gradients are never retained.  A temporary batch
    sum and one normalized direction sum per parameter scope are sufficient for
    the requested within-batch coherence and between-batch dispersion.
    """

    def __init__(self, parameters: Sequence[torch.nn.Parameter], parameter_groups: Sequence[str]) -> None:
        super().__init__(parameters, parameter_groups)
        self._batch_sums = [torch.zeros_like(value, dtype=torch.float32) for value in parameters]
        self._batch_sample_norm_sum = {scope: 0.0 for scope in self.scope_names}
        self._batch_count = 0
        self._batch_rows: dict[str, list[dict[str, float | None]]] = {
            scope: [] for scope in self.scope_names
        }
        self._direction_sums: dict[str, torch.Tensor | None] = {
            scope: None for scope in self.scope_names
        }

    def start_batch(self) -> None:
        for value in self._batch_sums:
            value.zero_()
        self._batch_sample_norm_sum = {scope: 0.0 for scope in self.scope_names}
        self._batch_count = 0

    def add(self, gradients: Sequence[torch.Tensor | None]) -> None:
        super().add(gradients)
        squared = {scope: 0.0 for scope in self.scope_names}
        for destination, value, group in zip(
            self._batch_sums, gradients, self.parameter_groups, strict=True
        ):
            if value is None:
                continue
            converted = value.detach().float()
            destination.add_(converted)
            amount = float(converted.square().sum())
            squared["all"] += amount
            squared[group] += amount
        for scope, amount in squared.items():
            self._batch_sample_norm_sum[scope] += math.sqrt(max(amount, 0.0))
        self._batch_count += 1

    def finish_batch(self) -> None:
        _require(self._batch_count > 0, "cannot finish empty gradient batch")
        for scope in self.scope_names:
            vector = _scope_vector(self._batch_sums, self.parameter_groups, scope)
            mean_vector = vector / float(self._batch_count)
            mean_norm = float(torch.linalg.vector_norm(mean_vector)) if mean_vector.numel() else 0.0
            mean_sample_norm = self._batch_sample_norm_sum[scope] / self._batch_count
            coherence = _safe_ratio(mean_norm, mean_sample_norm)
            self._batch_rows[scope].append(
                {
                    "batch_mean_gradient_norm": mean_norm,
                    "mean_sample_gradient_norm": mean_sample_norm,
                    "within_batch_coherence": coherence,
                }
            )
            if mean_norm > 0.0:
                direction = mean_vector / mean_norm
                prior = self._direction_sums[scope]
                self._direction_sums[scope] = direction if prior is None else prior + direction
        self.start_batch()

    @staticmethod
    def _distribution(values: Sequence[float | None]) -> dict[str, Any]:
        finite = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
        _require(bool(np.isfinite(finite).all()), "non-finite batch statistic")
        if finite.size == 0:
            return {"count": 0, "values": [], "mean": None, "minimum": None, "median": None, "maximum": None}
        return {
            "count": int(finite.size),
            "values": [float(value) for value in finite.tolist()],
            "mean": float(finite.mean()),
            "minimum": float(finite.min()),
            "median": float(np.median(finite)),
            "maximum": float(finite.max()),
        }

    def batch_statistics(self) -> dict[str, Any]:
        result: dict[str, Any] = {"batch_count": len(self._batch_rows["all"]), "scopes": {}}
        count = len(self._batch_rows["all"])
        for scope in self.scope_names:
            rows = self._batch_rows[scope]
            direction_sum = self._direction_sums[scope]
            direction_result: dict[str, Any] = {
                "nonzero_batch_count": int(sum(row["batch_mean_gradient_norm"] > 0.0 for row in rows)),
                "mean_pairwise_cosine": None,
                "directional_dispersion": None,
                "definition": "1 minus the mean pairwise cosine of nonzero batch-mean directions",
            }
            nonzero = direction_result["nonzero_batch_count"]
            if direction_sum is not None and nonzero >= 2:
                resultant_squared = float(direction_sum.square().sum())
                mean_cosine = _safe_ratio(resultant_squared - nonzero, nonzero * (nonzero - 1))
                direction_result["mean_pairwise_cosine"] = mean_cosine
                direction_result["directional_dispersion"] = None if mean_cosine is None else 1.0 - mean_cosine
            result["scopes"][scope] = {
                "batch_mean_gradient_norm": self._distribution(
                    [row["batch_mean_gradient_norm"] for row in rows]
                ),
                "within_batch_coherence": self._distribution(
                    [row["within_batch_coherence"] for row in rows]
                ),
                "between_batch_direction": direction_result,
                "batch_count": count,
            }
        return result

    def summary_with_batch_statistics(self, *, snr_batch_sizes: Sequence[int]) -> dict[str, Any]:
        summary = super().summary(snr_batch_sizes=snr_batch_sizes)
        for scope, values in summary["scopes"].items():
            mean_squared = values["mean_gradient_norm"] ** 2
            second_moment = mean_squared + values["rms_noise"] ** 2
            values["coherence"] = _safe_ratio(mean_squared, second_moment)
            values["cancellation_ratio"] = _safe_ratio(
                values["mean_gradient_norm"] * self.count,
                values["mean_sample_gradient_norm"] * self.count,
            )
        summary["batch_statistics"] = self.batch_statistics()
        return summary


def _gradient_tuple(scalar: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        torch.autograd.grad(scalar, parameters, retain_graph=True, allow_unused=True)
    )


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, Any]:
    parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    return {
        "values": {name: parameter.detach().cpu().clone() for name, parameter in parameters},
        "gradients": {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in parameters
        },
        "hash": frozen_audit.native_gradient.replay.live_ccrm.parameter_value_sha256(model),
    }


def _restore_parameter_snapshot(model: torch.nn.Module, snapshot: Mapping[str, Any]) -> None:
    by_name = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in snapshot["values"].items():
            _require(name in by_name, f"parameter set changed: {name}")
            by_name[name].copy_(value)
    for name, value in snapshot["gradients"].items():
        by_name[name].grad = None if value is None else value.detach().clone().to(by_name[name].device)


def _gradient_strength(
    response: Mapping[str, Any],
    nonconditional: Mapping[str, Any],
    relation: Mapping[str, Any],
) -> dict[str, Any]:
    strength: dict[str, Any] = {}
    for scope in ("all", *ALLOWED_PARAMETER_GROUPS):
        response_norm = response["scopes"][scope]["mean_gradient_norm"]
        nonconditional_norm = nonconditional["scopes"][scope]["mean_gradient_norm"]
        strength[scope] = {
            "weighted_response_mean_gradient_norm": response_norm,
            "weighted_nonconditional_mean_gradient_norm": nonconditional_norm,
            "response_gradient_norm": response_norm,
            "nonconditional_gradient_norm": nonconditional_norm,
            "response_to_nonconditional_norm_ratio": _safe_ratio(response_norm, nonconditional_norm),
            "response_nonconditional_cosine": relation[scope]["cosine"],
            "response_cluster_snr16": response["scopes"][scope]["snr_by_batch_size"].get("16"),
        }
    return strength


def _component_batch_means(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    components = signal_metrics.paired_signal_components(prediction, target)
    return {
        name: float(values.detach().double().mean())
        for name, values in components.items()
        if name in (
            "g_swap",
            "cross_energy",
            "target_delta_energy",
            "prediction_delta_energy",
            "response_loss",
            "correct_loss",
            "swapped_loss",
        )
    }


def _mode_gradient_audit(
    *,
    mode_name: str,
    model: torch.nn.Module,
    mixed: Any,
    hidden: Any,
    anchor: Mapping[str, torch.Tensor],
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    replay = frozen_audit.native_gradient.replay
    modes_before = replay.exact_audit._module_modes(model)
    buffers_before = replay.live_ccrm._buffer_snapshot(model)
    parameters_before = _parameter_snapshot(model)
    rng_before = replay.gradient_core._rng_snapshot()
    if mode_name == "train":
        model.train()
    elif mode_name == "eval":
        model.eval()
    else:
        raise ValueError(f"unknown model mode: {mode_name}")
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and _group(name) in ALLOWED_PARAMETER_GROUPS
    ]
    _require(bool(named_parameters), "no predictor parameters selected")
    parameters = [parameter for _, parameter in named_parameters]
    parameter_groups = [_group(name) for name, _ in named_parameters]
    _require(set(parameter_groups) == set(ALLOWED_PARAMETER_GROUPS), "parameter route changed")
    _require(
        sum(parameter.numel() for parameter in parameters)
        == frozen_audit.native_gradient.EXPECTED_TRAINABLE_PARAMETER_COUNT,
        "trainable parameter count changed",
    )
    accumulators = {
        arm: {
            "response": DeviceGradientAccumulator(parameters, parameter_groups),
            "nonconditional": DeviceGradientAccumulator(parameters, parameter_groups),
        }
        for arm in ARMS
    }
    predictions: dict[str, list[torch.Tensor]] = {arm: [] for arm in ARMS}
    targets: dict[str, list[torch.Tensor]] = {arm: [] for arm in ARMS}
    component_batches: dict[str, dict[str, list[float]]] = {
        arm: {name: [] for name in ("g_swap", "cross_energy", "target_delta_energy", "prediction_delta_energy", "response_loss", "correct_loss", "swapped_loss")}
        for arm in ARMS
    }
    rng_checks: list[bool] = []
    try:
        for batch_index in range(AUDIT_BATCH_COUNT):
            batch_buffers = replay.live_ccrm._buffer_snapshot(model)
            batch_rng = replay.gradient_core._rng_snapshot()
            d0_before: dict[str, torch.Tensor] | None = None
            d0_after: dict[str, torch.Tensor] | None = None
            for arm in ARMS:
                replay.gradient_core._restore_rng(batch_rng)
                replay.exact_audit._restore_buffers(model, batch_buffers)
                before = replay.gradient_core._rng_snapshot()
                accumulators[arm]["response"].start_batch()
                accumulators[arm]["nonconditional"].start_batch()
                batch = frozen_audit._build_batch(
                    hidden,
                    anchor,
                    torch.tensor(rows_by_arm[arm][batch_index]["hidden_row_indices"], dtype=torch.long),
                )
                prediction, target, _ = frozen_audit._arm_forward_and_gradients(
                    batch=batch,
                    model=model,
                    mixed=mixed,
                    device=device,
                    parameters=parameters,
                    response_accumulator=accumulators[arm]["response"],
                    nonconditional_accumulator=accumulators[arm]["nonconditional"],
                )
                accumulators[arm]["response"].finish_batch()
                accumulators[arm]["nonconditional"].finish_batch()
                predictions[arm].append(prediction)
                targets[arm].append(target)
                components = _component_batch_means(prediction, target)
                for name, value in components.items():
                    component_batches[arm][name].append(value)
                after = replay.gradient_core._rng_snapshot()
                if arm == "D0":
                    d0_before, d0_after = before, after
                else:
                    _require(d0_before is not None and d0_after is not None, "D0 RNG witness missing")
                    rng_checks.append(
                        replay.gradient_core._rng_equal(d0_before, before)
                        and replay.gradient_core._rng_equal(d0_after, after)
                    )
                replay.exact_audit._restore_buffers(model, batch_buffers)

        summaries: dict[str, Any] = {}
        for arm in ARMS:
            response = accumulators[arm]["response"].summary_with_batch_statistics(
                snr_batch_sizes=(16, 256)
            )
            nonconditional = accumulators[arm]["nonconditional"].summary_with_batch_statistics(
                snr_batch_sizes=(1, 16)
            )
            relation = frozen_audit.gradient_relation(
                accumulators[arm]["response"], accumulators[arm]["nonconditional"]
            )
            predictions_joined = torch.cat(predictions[arm], dim=0)
            targets_joined = torch.cat(targets[arm], dim=0)
            summaries[arm] = {
                "response_twin_population": response,
                "nonconditional_batch_population": nonconditional,
                "gradient_strength": _gradient_strength(response, nonconditional, relation),
                "paired_output_signal": signal_metrics.paired_signal_summary(
                    predictions_joined, targets_joined, batch_sizes=(16, 256)
                ),
                "loss_means": {
                    name: float(np.mean(component_batches[arm][name]))
                    for name in ("correct_loss", "response_loss", "g_swap")
                },
                "batch_component_means": component_batches[arm],
            }
        comparisons: dict[str, Any] = {}
        cross_arm = {
            (left, right): frozen_audit.gradient_relation(
                accumulators[left]["response"], accumulators[right]["response"]
            )
            for left, right in PAIRED_COMPARISONS
        }
        for left, right in PAIRED_COMPARISONS:
            key = f"{left}_vs_{right}"
            comparisons[key] = {}
            for scope in ("all", *ALLOWED_PARAMETER_GROUPS):
                left_strength = summaries[left]["gradient_strength"][scope]
                right_strength = summaries[right]["gradient_strength"][scope]
                left_response = summaries[left]["response_twin_population"]["scopes"][scope]
                right_response = summaries[right]["response_twin_population"]["scopes"][scope]
                comparisons[key][scope] = {
                    "response_mean_gradient_norm_difference": left_strength["response_gradient_norm"] - right_strength["response_gradient_norm"],
                    "response_mean_gradient_norm_relative_delta": _safe_ratio(
                        left_strength["response_gradient_norm"] - right_strength["response_gradient_norm"],
                        right_strength["response_gradient_norm"],
                    ),
                    "response_to_nonconditional_ratio_difference": (
                        None
                        if left_strength["response_to_nonconditional_norm_ratio"] is None
                        or right_strength["response_to_nonconditional_norm_ratio"] is None
                        else left_strength["response_to_nonconditional_norm_ratio"]
                        - right_strength["response_to_nonconditional_norm_ratio"]
                    ),
                    "response_snr16_difference": (
                        None
                        if left_response["snr_by_batch_size"].get("16") is None
                        or right_response["snr_by_batch_size"].get("16") is None
                        else left_response["snr_by_batch_size"]["16"]
                        - right_response["snr_by_batch_size"]["16"]
                    ),
                    "response_coherence_difference": (
                        None
                        if left_response.get("coherence") is None
                        or right_response.get("coherence") is None
                        else left_response["coherence"] - right_response["coherence"]
                    ),
                    "response_gradient_cosine": cross_arm[(left, right)][scope]["cosine"],
                }
            comparisons[key]["paired_batch_gradient_norm"] = paired_bootstrap_difference(
                summaries[left]["response_twin_population"]["batch_statistics"]["scopes"]["all"]["batch_mean_gradient_norm"]["values"],
                summaries[right]["response_twin_population"]["batch_statistics"]["scopes"]["all"]["batch_mean_gradient_norm"]["values"],
                seed=BOOTSTRAP_SEED + len(comparisons),
            )
            comparisons[key]["paired_output"] = paired_output_comparison(
                component_batches[left], component_batches[right]
            )
        # Compute the restoration witness only after restoring all mutable
        # model state.  The finally block below repeats this operation for
        # exceptional exits.
        _restore_parameter_snapshot(model, parameters_before)
        replay.exact_audit._restore_buffers(model, buffers_before)
        replay.exact_audit._restore_module_modes(model, modes_before)
        replay.gradient_core._restore_rng(rng_before)
        state_restoration = {
            "parameters_unchanged": replay.live_ccrm.parameter_value_sha256(model)
            == parameters_before["hash"],
            "parameter_grad_slots_restored": all(
                (parameter.grad is None) == (parameters_before["gradients"][name] is None)
                and (
                    parameter.grad is None
                    or torch.equal(
                        parameter.grad.detach().cpu(),
                        parameters_before["gradients"][name].detach().cpu(),
                    )
                )
                for name, parameter in model.named_parameters()
            ),
            "buffers_restored": replay.live_ccrm._buffers_equal(
                replay.live_ccrm._buffer_snapshot(model), buffers_before
            ),
            "module_modes_restored": replay.exact_audit._module_modes(model) == modes_before,
            "rng_restored": replay.gradient_core._rng_equal(
                rng_before, replay.gradient_core._rng_snapshot()
            ),
        }
        return {
            "mode": mode_name,
            "batch_count_per_arm": AUDIT_BATCH_COUNT,
            "twin_gradient_units_per_arm": AUDIT_BATCH_COUNT * TWINS_PER_BATCH,
            "same_original_anchor": True,
            "paired_rng_before_and_after_each_arm": bool(rng_checks) and all(rng_checks),
            "arms": summaries,
            "comparisons": comparisons,
            "comparison": comparisons,
            "state_restoration": state_restoration,
        }
    finally:
        _restore_parameter_snapshot(model, parameters_before)
        replay.exact_audit._restore_buffers(model, buffers_before)
        replay.exact_audit._restore_module_modes(model, modes_before)
        replay.gradient_core._restore_rng(rng_before)


def _finite_distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, "paired population is empty")
    _require(bool(np.isfinite(array).all()), "paired population contains non-finite values")
    return {
        "count": int(array.size),
        "values": [float(value) for value in array.tolist()],
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap paired batch differences with a deterministic seed."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    _require(left_array.shape == right_array.shape and left_array.ndim == 1, "paired arrays are not aligned")
    _require(left_array.size > 0 and bool(np.isfinite(left_array).all()) and bool(np.isfinite(right_array).all()), "invalid paired arrays")
    differences = left_array - right_array
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, differences.size, size=(int(resamples), differences.size))
    means = differences[indices].mean(axis=1)
    return {
        "difference_definition": "left minus right, paired by frozen schedule index",
        "differences": _finite_distribution(differences),
        "mean_difference": float(differences.mean()),
        "bootstrap": {
            "resamples": int(resamples),
            "seed": int(seed),
            "confidence": 0.95,
            "interval": {"lower": float(np.quantile(means, 0.025)), "upper": float(np.quantile(means, 0.975))},
        },
    }


def paired_output_comparison(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return {
        name: paired_bootstrap_difference(
            left[name], right[name], seed=BOOTSTRAP_SEED + index
        )
        for index, name in enumerate(
            (
                "g_swap",
                "cross_energy",
                "target_delta_energy",
                "prediction_delta_energy",
                "response_loss",
                "correct_loss",
                "swapped_loss",
            )
        )
    }


def _gradient_visibility(comparison: Mapping[str, Any], *, scope: str = "all") -> bool:
    value = comparison.get(scope, comparison) if isinstance(comparison, Mapping) else {}
    if not isinstance(value, Mapping):
        return False
    norm_candidates = (
        value.get("response_mean_gradient_norm_relative_delta"),
        value.get("response_mean_gradient_norm_difference"),
    )
    geometry_candidates = (
        value.get("response_snr16_difference"),
        value.get("response_to_nonconditional_ratio_difference"),
        value.get("response_coherence_difference"),
    )
    paired = comparison.get("paired_batch_gradient_norm", {})
    try:
        paired_lower = float(paired["bootstrap"]["interval"]["lower"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        any(candidate is not None and float(candidate) > 0.0 for candidate in norm_candidates)
        and any(
            candidate is not None and float(candidate) > 0.0
            for candidate in geometry_candidates
        )
        and paired_lower > 0.0
    )


def _gradient_primary_difference(comparison: Mapping[str, Any] | None) -> float | None:
    if not isinstance(comparison, Mapping):
        return None
    value = comparison.get("all", comparison)
    if not isinstance(value, Mapping):
        return None
    raw = value.get("response_mean_gradient_norm_difference")
    return None if raw is None else float(raw)


def _latent_delta(latent: Mapping[str, Any], left: str, right: str, scale: int) -> float | None:
    comparisons = latent.get("comparisons", latent.get("comparison", {}))
    direct = comparisons.get(f"{left}_vs_{right}") if isinstance(comparisons, Mapping) else None
    if isinstance(direct, Mapping) and str(scale) in direct:
        entry = direct[str(scale)]
        value = entry.get("rho_lat_absolute_delta") if isinstance(entry, Mapping) else entry
        return None if value is None else float(value)
    if left == "REL50" and right == "HASH50" and isinstance(comparisons, Mapping):
        legacy = comparisons.get(str(scale))
        if isinstance(legacy, Mapping) and legacy.get("rho_lat_absolute_delta") is not None:
            return float(legacy["rho_lat_absolute_delta"])
    arms = latent.get("arms", {})
    try:
        return float(
            arms[left]["by_k"][str(scale)]["rho_lat"]
            - arms[right]["by_k"][str(scale)]["rho_lat"]
        )
    except (KeyError, TypeError):
        return None


def _comparison_for(
    gradient: Mapping[str, Any],
    left: str,
    right: str,
) -> Mapping[str, Any] | None:
    key = f"{left}_vs_{right}"
    if key in gradient:
        value = gradient[key]
        return value if isinstance(value, Mapping) else None
    for container_name in ("comparisons", "comparison", "train_mode", "train"):
        container = gradient.get(container_name)
        if isinstance(container, Mapping):
            value = container.get(key)
            if isinstance(value, Mapping):
                return value
            nested = container.get("comparisons")
            if isinstance(nested, Mapping) and isinstance(nested.get(key), Mapping):
                return nested[key]
    # A small compatibility path makes the decision function useful with the
    # earlier two-arm fixture shape; formal four-arm runs always take the
    # named comparison path above.
    legacy = gradient.get("comparison")
    if isinstance(legacy, Mapping) and isinstance(legacy.get("all"), Mapping):
        return legacy
    return None


def _decision(
    latent: Mapping[str, Any],
    gradient: Mapping[str, Any],
    *,
    state_restoration: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    rho_deltas = {
        str(scale): _latent_delta(latent, "REL50", "HASH50", scale)
        for scale in LATENT_NEIGHBOUR_SCALES
    }
    rho_checks = {
        f"rho_lat_k{scale}_rel50_above_hash50": (
            rho_deltas[str(scale)] is not None and rho_deltas[str(scale)] > 0.0
        )
        for scale in LATENT_NEIGHBOUR_SCALES
    }
    rel_gradient = _comparison_for(gradient, "REL50", "HASH50")
    hash_gradient = _comparison_for(gradient, "HASH50", "D0")
    rel_visible = bool(rel_gradient is not None and _gradient_visibility(rel_gradient))
    hash_visible = bool(hash_gradient is not None and _gradient_visibility(hash_gradient))
    hash_rho_deltas = {
        str(scale): _latent_delta(latent, "HASH50", "D0", scale)
        for scale in LATENT_NEIGHBOUR_SCALES
    }
    hash_rho_same_or_larger = all(
        hash_rho_deltas[str(scale)] is not None
        and rho_deltas[str(scale)] is not None
        and hash_rho_deltas[str(scale)] > 0.0
        and hash_rho_deltas[str(scale)] >= rho_deltas[str(scale)]
        for scale in LATENT_NEIGHBOUR_SCALES
    )
    rel_primary = _gradient_primary_difference(rel_gradient)
    hash_primary = _gradient_primary_difference(hash_gradient)
    hash_gradient_same_or_larger = bool(
        hash_visible
        and rel_primary is not None
        and hash_primary is not None
        and hash_primary >= rel_primary
    )
    restoration = state_restoration
    if restoration is None:
        candidate = gradient.get("state_restoration")
        if isinstance(candidate, Mapping):
            restoration = {str(key): bool(value) for key, value in candidate.items()}
    restoration_passed = bool(restoration) and all(restoration.values())
    batch_count = gradient.get("batch_count_per_arm")
    if batch_count is None:
        train = gradient.get("train_mode", gradient.get("train", {}))
        batch_count = train.get("batch_count_per_arm") if isinstance(train, Mapping) else None
    budget_passed = int(batch_count) == AUDIT_BATCH_COUNT if batch_count is not None else False
    train_mode = gradient.get("train_mode", gradient.get("train", {}))
    eval_mode = gradient.get("eval_mode", gradient.get("eval", {}))
    rng_passed = bool(
        isinstance(train_mode, Mapping)
        and train_mode.get("paired_rng_before_and_after_each_arm") is True
        and isinstance(eval_mode, Mapping)
        and eval_mode.get("paired_rng_before_and_after_each_arm") is True
    )
    checks = {
        **rho_checks,
        "rel50_paired_response_gradient_visibility_above_hash50": rel_visible,
        "hash50_not_same_or_larger_improvement_as_rel50": not (
            hash_gradient_same_or_larger and hash_rho_same_or_larger
        ),
        "same_16_batch_and_256_twin_budget": budget_passed,
        "paired_rng_equal_in_train_and_eval_modes": rng_passed,
        "state_restoration_passed": restoration_passed,
    }
    return {
        "status": "passed_go_for_motion_short_training" if all(checks.values()) else "failed_no_go",
        "checks": checks,
        "rho_lat_deltas_rel50_minus_hash50": rho_deltas,
        "rho_lat_deltas_hash50_minus_d0": hash_rho_deltas,
        "placebo_comparison": {
            "hash_rho_same_or_larger_than_rel_contrast": hash_rho_same_or_larger,
            "hash_gradient_same_or_larger_than_rel_contrast": hash_gradient_same_or_larger,
            "rel_response_gradient_norm_difference": rel_primary,
            "hash_response_gradient_norm_difference": hash_primary,
        },
        "rule": (
            "Only REL50 versus HASH50 can open the Motion short-training gate. "
            "All frozen physical scales must improve in rho_lat; the paired batch-gradient "
            "95% bootstrap lower bound, mean response-gradient norm, and at least one of "
            "SNR, response/nonconditional ratio or coherence must improve in native train "
            "mode. HASH50 must not show a same-or-larger D0-relative latent and gradient "
            "effect; paired RNG, budget and state restoration must pass. "
            "ABS50 is descriptive mechanism evidence and is not ranked by this gate."
        ),
        "claim_boundary": "initialization mechanism evidence only; no ICL success claim",
    }


def _materialize_training_inputs(
    *,
    motion: Any,
    original_h5: Path,
    original_lance: Path,
    schedule_set: Mapping[str, Any],
) -> tuple[Any, dict[str, torch.Tensor], dict[str, Any]]:
    trainer = motion.trainer
    mixed = trainer.mixed
    original_h5 = assert_training_only_path(original_h5)
    original_lance = assert_training_only_path(original_lance)
    _require(original_h5.is_file(), f"missing original Training action statistics: {original_h5}")
    _require(original_lance.is_dir(), f"missing original Training table: {original_lance}")
    action_stats = trainer.ACTION_STATS_LOADER(original_h5)
    hidden = trainer._training_split(
        frozen_audit.native_gradient.replay.replay.TRAIN_TABLE,
        expected_pairs=EXPECTED_PAIRS,
        action_stats=action_stats,
    )
    _require(hidden.pair_count == EXPECTED_PAIRS, "hidden pair count changed")
    _require(hidden.pixels.shape[0] == EXPECTED_HIDDEN_ROWS, "hidden row count changed")
    _, original_loader = mixed.original_loader(
        original_lance,
        batch_size=ORIGINAL_ROWS,
        seed=EXPECTED_SCHEDULE_SEED,
        num_workers=0,
    )
    original = next(iter(original_loader))
    original_actions = mixed.pilot.normalize_action_blocks(
        torch.nan_to_num(original["action"].float(), 0.0), action_stats
    )
    anchor = {
        "pixels": original["pixels"].detach().cpu().contiguous(),
        "actions": original_actions.detach().cpu().contiguous(),
    }
    audit_rows = {
        arm: [
            torch.tensor(row["hidden_row_indices"], dtype=torch.long)
            for row in schedule_set["arms"][arm]["audit_records"]
        ]
        for arm in ARMS
    }
    for arm in ARMS:
        _require(len(audit_rows[arm]) == AUDIT_BATCH_COUNT, f"{arm} audit rows missing")
        for rows in audit_rows[arm]:
            validate_twin_rows(rows)
    return hidden, anchor, {
        "receipt": {
            "hidden_pair_count": int(hidden.pair_count),
            "hidden_row_count": int(hidden.pixels.shape[0]),
            "original_anchor_rows": int(anchor["pixels"].shape[0]),
            "original_anchor_pixels_sha256": _tensor_sha256(anchor["pixels"]),
            "original_anchor_actions_sha256": _tensor_sha256(anchor["actions"]),
            "same_original_anchor_for_all_arms": True,
            "audit_schedule_indices": list(AUDIT_SCHEDULE_INDICES),
            "audit_row_stream_sha256": {
                arm: _tensor_sha256(torch.stack(audit_rows[arm])) for arm in ARMS
            },
        },
        "rows": {
            arm: [
                {"hidden_row_indices": [int(value) for value in rows.tolist()]}
                for rows in audit_rows[arm]
            ]
            for arm in ARMS
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = assert_training_only_path(args.checkpoint)
    static = static_identity(
        checkpoint=checkpoint,
        catalog=args.catalog,
        schedule_dir=args.schedule_dir,
        original_h5=args.original_h5,
        original_lance=args.original_lance,
        schedule_files=args.schedule_files,
        expected_schedule_receipt_sha256=args.expected_schedule_receipt_sha256,
    )
    schedules = load_schedule_set(
        args.schedule_dir,
        explicit=args.schedule_files,
        expected_receipt_sha256=args.expected_schedule_receipt_sha256,
    )
    device = torch.device(args.device)
    _require(device.type == "cuda" and torch.cuda.is_available(), "CUDA required for formal zero-step audit")
    frozen_audit.native_gradient._prepare_optional_flash_attention()
    replay = frozen_audit.native_gradient.replay
    runtime_path, runtime = replay.completion.load_completion(replay.RUNTIME_COMPLETION)
    release_path, _ = replay.completion.load_source_release(runtime)
    _require(release_path.resolve() == replay.CURRENT_RELEASE.resolve(), "release changed")
    worktree = replay.completion._pinned_stable_worldmodel(runtime)
    trainer = replay.completion._configure_component_trainer("motion_damping", worktree)
    import contextworld.benchmarks.motion_damping_icl_score as motion_score
    import run_pusht_motion_damping_h3_train as motion

    _require(motion.trainer is trainer, "Motion trainer binding changed")
    mixed = trainer.mixed
    missing = object()
    previous_weight = mixed.VARIANT_WEIGHTS.get(ANALYSIS_VARIANT, missing)
    was_twin_variant = ANALYSIS_VARIANT in motion.TWIN_GROUP_VARIANTS
    was_diagnostic_variant = ANALYSIS_VARIANT in trainer.DIAGNOSTIC_VARIANTS["lewm"]
    outer_rng = replay.gradient_core._rng_snapshot()
    guard_counts: Mapping[str, int] = {}
    try:
        mixed.VARIANT_WEIGHTS[ANALYSIS_VARIANT] = ("native", replay.WEIGHT, "identifiable_future_only")
        motion.TWIN_GROUP_VARIANTS.add(ANALYSIS_VARIANT)
        trainer.DIAGNOSTIC_VARIANTS["lewm"].add(ANALYSIS_VARIANT)
        with replay.replay._install_fail_closed_guards(
            motion, motion_score, allow_training_table=True
        ) as guard_counts:
            hidden, anchor, inputs = _materialize_training_inputs(
                motion=motion,
                original_h5=args.original_h5,
                original_lance=args.original_lance,
                schedule_set=schedules,
            )
            mixed.pilot.set_reproducible_seed(EXPECTED_SCHEDULE_SEED)
            model, load_receipt = mixed.load_model_for_variant(
                checkpoint, variant=ANALYSIS_VARIANT, device=device
            )
            _require(
                load_receipt.get("sha256") == EXPECTED_CHECKPOINT_SHA256
                and load_receipt.get("model_state_sha256") == EXPECTED_MODEL_STATE_SHA256
                and load_receipt.get("strict_state_dict_load") is True,
                "checkpoint load identity changed",
            )
            query_latent, future_latent = frozen_audit._encode_all_latents(
                model=model,
                mixed=mixed,
                hidden=hidden,
                device=device,
                batch_size=int(args.latent_batch_size),
            )
            _, neighbours = frozen_audit.load_catalog(args.catalog)
            latent = weighted_latent_metrics(
                query_latent,
                future_latent,
                neighbours,
                {arm: schedules["arms"][arm]["weights"] for arm in ARMS},
            )
            rows = inputs["rows"]
            train_mode = _mode_gradient_audit(
                mode_name="train",
                model=model,
                mixed=mixed,
                hidden=hidden,
                anchor=anchor,
                rows_by_arm=rows,
                device=device,
            )
            eval_mode = _mode_gradient_audit(
                mode_name="eval",
                model=model,
                mixed=mixed,
                hidden=hidden,
                anchor=anchor,
                rows_by_arm=rows,
                device=device,
            )
        expected_guards = {
            "release_loader": 0,
            "release_auditor": 0,
            "optimizer_constructor": 0,
            "optimizer_step": 0,
            "development_scorer": 0,
            "public_scorer": 0,
            "training_table_reads": 1,
            "non_training_benchmark_reads": 0,
        }
        _require(
            all(guard_counts.get(name) == value for name, value in expected_guards.items()),
            f"forbidden zero-step action: {dict(guard_counts)}",
        )
    finally:
        replay.gradient_core._restore_rng(outer_rng)
        if previous_weight is missing:
            mixed.VARIANT_WEIGHTS.pop(ANALYSIS_VARIANT, None)
        else:
            mixed.VARIANT_WEIGHTS[ANALYSIS_VARIANT] = previous_weight
        if was_twin_variant:
            motion.TWIN_GROUP_VARIANTS.add(ANALYSIS_VARIANT)
        else:
            motion.TWIN_GROUP_VARIANTS.discard(ANALYSIS_VARIANT)
        if was_diagnostic_variant:
            trainer.DIAGNOSTIC_VARIANTS["lewm"].add(ANALYSIS_VARIANT)
        else:
            trainer.DIAGNOSTIC_VARIANTS["lewm"].discard(ANALYSIS_VARIANT)
    _require(replay.gradient_core._rng_equal(outer_rng, replay.gradient_core._rng_snapshot()), "outer RNG restoration failed")
    state_restoration = {
        "train_mode": train_mode["state_restoration"],
        "eval_mode": eval_mode["state_restoration"],
        "all_passed": all(train_mode["state_restoration"].values()) and all(eval_mode["state_restoration"].values()),
    }
    gradient = {
        "batch_count_per_arm": AUDIT_BATCH_COUNT,
        "twin_gradient_units_per_arm": AUDIT_BATCH_COUNT * TWINS_PER_BATCH,
        "train_mode": train_mode,
        "eval_mode": eval_mode,
        "comparisons": train_mode["comparisons"],
        "comparison": train_mode["comparisons"],
        "state_restoration": state_restoration,
    }
    decision = _decision(latent, gradient, state_restoration={"all_passed": state_restoration["all_passed"]})
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": decision["status"],
        "optimizer_updates": 0,
        "source": {"path": str(THIS_SOURCE), "sha256": _sha256(THIS_SOURCE)},
        "runtime": {
            "completion": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "release": {"path": str(release_path), "sha256": _sha256(release_path)},
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
        },
        "static_identity": static,
        "training_inputs": inputs["receipt"],
        "checkpoint_load": load_receipt,
        "latent_manipulation": latent,
        "gradient_visibility": gradient,
        "decision": decision,
        "guard_counts": dict(guard_counts),
        "evidence_boundary": {
            "training_only": True,
            "development_opened": False,
            "public_test_opened": False,
            "validation_opened": False,
            "cem_executed": False,
            "optimizer_steps": 0,
            "model_parameters_changed": False,
            "full_training_authorized": decision["status"] == "passed_go_for_motion_short_training",
            "initialization_mechanism_evidence_only": True,
        },
        "interpretation_boundary": {
            "train_mode": "Native train-mode pred_proj BatchNorm preserves real batch-composition effects.",
            "eval_mode": "Eval mode is a batch-coupling control; exposure arms have different row multisets, so no invariance claim is made across arms.",
            "comparisons": "Only REL50/HASH50, REL50/ABS50 and HASH50/D0 are pre-registered; no global repartition monotonicity gate is used.",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--original-h5", type=Path, default=DEFAULT_ORIGINAL_H5)
    parser.add_argument("--original-lance", type=Path, default=DEFAULT_ORIGINAL_LANCE)
    parser.add_argument("--schedule-dir", "--schedule-root", dest="schedule_dir", type=Path, default=DEFAULT_SCHEDULE_DIR)
    parser.add_argument("--expected-schedule-receipt-sha256", "--schedule-receipt-sha256", dest="expected_schedule_receipt_sha256")
    parser.add_argument("--d0-schedule", type=Path)
    parser.add_argument("--d0-multiplicity", type=Path)
    parser.add_argument("--rel50-schedule", type=Path)
    parser.add_argument("--rel50-multiplicity", type=Path)
    parser.add_argument("--abs50-schedule", type=Path)
    parser.add_argument("--abs50-multiplicity", type=Path)
    parser.add_argument("--hash50-schedule", type=Path)
    parser.add_argument("--hash50-multiplicity", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-batch-size", type=int, default=LATENT_ENCODE_BATCH_SIZE)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    _require(args.latent_batch_size > 0, "latent batch size must be positive")
    if args.expected_schedule_receipt_sha256 is not None:
        _require(
            bool(re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_schedule_receipt_sha256)),
            "expected schedule receipt SHA256 must be 64 hexadecimal characters",
        )
    args.schedule_files = {
        arm: {
            "schedule": getattr(args, f"{arm.lower()}_schedule"),
            "multiplicity": getattr(args, f"{arm.lower()}_multiplicity"),
        }
        for arm in ARMS
    }
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = assert_training_only_path(args.output)
    _require(not output.exists(), f"refusing to overwrite {output}")
    if args.check_only:
        payload = static_identity(
            checkpoint=args.checkpoint,
            catalog=args.catalog,
            schedule_dir=args.schedule_dir,
            original_h5=args.original_h5,
            original_lance=args.original_lance,
            schedule_files=args.schedule_files,
            expected_schedule_receipt_sha256=args.expected_schedule_receipt_sha256,
        )
        print(json.dumps({"status": "passed_static_identity", "identity": payload}, sort_keys=True))
        return 0
    payload = run(args)
    _write_exclusive(output, payload)
    receipt = output.parent / "receipt.json"
    schedule_receipt = payload["static_identity"]["schedule"]["receipt"]
    schedule_arms = payload["static_identity"]["schedule"]["arms"]
    _write_exclusive(
        receipt,
        {
            "schema_version": 1,
            "analysis_id": ANALYSIS_ID,
            "status": payload["status"],
            "input_sha256": {
                "checkpoint": EXPECTED_CHECKPOINT_SHA256,
                "d1_v1_catalog": EXPECTED_V1_CATALOG_SHA256,
                "d1_training_runner": EXPECTED_D1_RUNNER_SHA256,
                "schedule_receipt": schedule_receipt["sha256"],
                "schedule_outputs": {
                    arm: {
                        "schedule": schedule_arms[arm]["schedule_sha256"],
                        "multiplicity": schedule_arms[arm]["multiplicity_sha256"],
                    }
                    for arm in ARMS
                },
                "source": payload["source"]["sha256"],
            },
            "output_sha256": {"report.json": _sha256(output)},
            "decision_checks": payload["decision"]["checks"],
            **payload["evidence_boundary"],
        },
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "receipt": str(receipt),
                "receipt_sha256": _sha256(receipt),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed_go_for_motion_short_training" else 2


if __name__ == "__main__":
    raise SystemExit(main())
