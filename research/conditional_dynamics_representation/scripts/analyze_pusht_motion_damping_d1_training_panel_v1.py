#!/usr/bin/env python3
"""Audit paired conditional response on the frozen Motion Training pool.

The panel evaluates every frozen Training query pair exactly once under the
same D0 weighting (4,096 forward/reverse twins, 8,192 binary pairs).  It is an
inference-only audit of the D0 matched-native, D1 native, and final COJA
checkpoints.  No optimizer, Development, Public Test, or CEM path is opened.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD = ROOT.parent / "ContextWorld"
for source_root in (ROOT, CONTEXTWORLD, CONTEXTWORLD / "scripts"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    analyze_pusht_motion_damping_d1_latent_gradient_zero_step_v1 as zero_step,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    conditional_signal_metrics as signal_metrics,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_motion_damping_native_gradient_snr_diagnostic_v1 as native_gradient,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_d1_energy_stratified_native_v1 as d1_runner,
)


ANALYSIS_ID = "pusht_motion_damping_d1_training_panel_v1"
ANALYSIS_VARIANT = zero_step.ANALYSIS_VARIANT
EXPECTED_TWINS = 4096
EXPECTED_PAIRS = 8192
EXPECTED_HIDDEN_ROWS = 16384
EXPECTED_ROWS_PER_TWIN = 4
EXPECTED_BATCH_SIZE = 64
DEFAULT_BATCH_SIZE = 64
DEFAULT_BOOTSTRAP_SEED = 20260901
DEFAULT_BOOTSTRAP_RESAMPLES = 4096

DEFAULT_D0_CHECKPOINT = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_full_release_visible_joint_absolute_single_stage_"
    "native_control_step8192_v1/s14321_step8192_v1/"
    "mixed_frozen_image_identifiable_future_native_0p09_step8192.pt"
)
DEFAULT_D1_CHECKPOINT = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_energy_stratified_native_v1/"
    "s14321_step8192_v2_final/"
    "mixed_frozen_image_identifiable_future_native_0p09_step8192.pt"
)
DEFAULT_COJA_CHECKPOINT = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_full_release_visible_joint_absolute_single_stage_"
    "step8192_v1/s14321_step8192_v1/"
    "mixed_frozen_image_identifiable_future_native_0p09_step8192.pt"
)
DEFAULT_D1_REPORT = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_energy_stratified_native_v1/"
    "s14321_step8192_v2_final/training_report.json"
)
DEFAULT_CATALOG = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_metric_v1/d1_0_training_only_v1_final/"
    "per_twin_catalog.jsonl"
)
DEFAULT_SCHEDULE_DIR = d1_runner.DEFAULT_SCHEDULE_DIR
DEFAULT_OUTPUT = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_energy_stratified_native_v1/"
    "s14321_step8192_v2_final/training_d0_weighted_panel_v1.json"
)

EXPECTED_CHECKPOINT_SHA256 = {
    "D0": "6063e01524f49ac974f233bd876518e35ece6c8d406b19f7a25afd6fcf2ea3ff",
    "D1": "9bb4efd8831c851c706e206acb1153a79496422ffda1dc745a586c6827bdb32b",
    "COJA": "16466ce3410eb60a48812b3ff2bf31d12f5c426b8c2f1cbecab22ac6e03f6ff7",
}
EXPECTED_MODEL_STATE_SHA256 = {
    "D0": "a17749e3b98442efe17b330c21c0bbe3ec8375a24aee05370d37e327c7c060dc",
    "D1": "a2b13f6c3fbafb3a4d6236aa0bdf004838efe1aab96649477b11bfdfb68f29d6",
    "COJA": "80af7e9861c5f00f3ad257aedc5803809fd74ca742e217de22da8ef28a244c27",
}
EXPECTED_CATALOG_SHA256 = zero_step.EXPECTED_V1_CATALOG_SHA256
EXPECTED_D1_RUNNER_SHA256 = zero_step.EXPECTED_D1_RUNNER_SHA256


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


def _module_state_sha256(model: torch.nn.Module) -> str:
    """Hash parameters and buffers, including names, dtype, and shape."""

    digest = hashlib.sha256()
    values = [("parameter", name, value) for name, value in model.named_parameters()]
    values.extend(("buffer", name, value) for name, value in model.named_buffers())
    for kind, name, value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _encoder_identity_sha256(model: torch.nn.Module) -> str | None:
    """Hash the frozen visual target path, excluding action/prediction paths."""

    digest = hashlib.sha256()
    selected = []
    for name, value in model.named_parameters():
        if name.split(".", 1)[0] in {"encoder", "projector"}:
            selected.append(("parameter", name, value))
    for name, value in model.named_buffers():
        if name.split(".", 1)[0] in {"encoder", "projector"}:
            selected.append(("buffer", name, value))
    if not selected:
        return None
    for kind, name, value in selected:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def pair_ids(pair_count: int = EXPECTED_PAIRS) -> list[str]:
    _require(int(pair_count) > 0 and int(pair_count) % 2 == 0, "binary pair count must be positive and even")
    return [
        f"pmd-train-{index:06d}-{'forward' if index % 2 == 0 else 'reverse'}"
        for index in range(int(pair_count))
    ]


def validate_hidden_pool(
    pixels: torch.Tensor,
    actions: torch.Tensor,
    *,
    expected_twins: int = EXPECTED_TWINS,
) -> dict[str, Any]:
    """Validate all binary pairs and the complete four-row twin layout."""

    _require(torch.is_tensor(pixels) and torch.is_tensor(actions), "pool tensors required")
    expected_rows = int(expected_twins) * EXPECTED_ROWS_PER_TWIN
    _require(pixels.ndim >= 3 and pixels.shape[0] == expected_rows, "hidden pixel rows changed")
    _require(actions.ndim >= 1 and actions.shape[0] == expected_rows, "hidden action rows changed")
    _require(pixels.shape[1] >= 4, "hidden pixel frame count changed")
    _require(bool(torch.isfinite(pixels.float()).all()), "hidden pixels contain non-finite values")
    _require(bool(torch.isfinite(actions.float()).all()), "hidden actions contain non-finite values")

    grouped_pixels = pixels.reshape(int(expected_twins), 2, 2, *pixels.shape[1:])
    grouped_actions = actions.reshape(
        int(expected_twins), 2, 2, *actions.shape[1:]
    )
    query_equal = torch.eq(grouped_pixels[:, :, 0, 2], grouped_pixels[:, :, 1, 2])
    action_equal = torch.eq(grouped_actions[:, :, 0], grouped_actions[:, :, 1])
    history_different = torch.ne(
        grouped_pixels[:, :, 0, :2], grouped_pixels[:, :, 1, :2]
    ).reshape(int(expected_twins), 2, -1).any(dim=-1)
    future_different = torch.ne(
        grouped_pixels[:, :, 0, 3], grouped_pixels[:, :, 1, 3]
    ).reshape(int(expected_twins), 2, -1).any(dim=-1)
    checks = {
        "row_count_exact": int(pixels.shape[0]) == expected_rows,
        "complete_four_row_twins": True,
        "binary_pair_count_exact": int(pixels.shape[0] // 2)
        == int(expected_twins) * 2,
        "query_frame_exact_within_pair": bool(query_equal.flatten(1).all()),
        "action_exact_within_pair": bool(action_equal.flatten(1).all()),
        "history_differs_within_pair": bool(history_different.all()),
        "future_differs_within_pair": bool(future_different.all()),
    }
    _require(all(checks.values()), f"training pair identity failed: {checks}")
    return {
        "twins": int(expected_twins),
        "pairs": int(expected_twins) * 2,
        "rows": int(pixels.shape[0]),
        "rows_per_twin": EXPECTED_ROWS_PER_TWIN,
        "checks": checks,
    }


# Public aliases make the invariant easy to use from focused tests and callers.
validate_training_pool = validate_hidden_pool
validate_pair_identity = validate_hidden_pool


def _finite_distribution(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(array.size > 0 and bool(np.isfinite(array).all()), "empty or non-finite statistic")
    quantiles = np.quantile(array, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "minimum": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "maximum": float(quantiles[6]),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def _geometry_values(
    components: Mapping[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    target = components["target_delta_energy"].detach().double().cpu().numpy()
    prediction = components["prediction_delta_energy"].detach().double().cpu().numpy()
    cross = components["cross_energy"].detach().double().cpu().numpy()
    response = components["response_loss"].detach().double().cpu().numpy()
    return {
        "g_swap": components["g_swap"].detach().double().cpu().numpy(),
        "response_gain": np.asarray(
            [value if value is not None else np.nan for value in (
                _safe_ratio(c, t) for c, t in zip(cross, target, strict=True)
            )],
            dtype=np.float64,
        ),
        "alignment": np.asarray(
            [
                value if value is not None else np.nan
                for value in (
                    _safe_ratio(c, math.sqrt(max(p * t, 0.0)))
                    for c, p, t in zip(cross, prediction, target, strict=True)
                )
            ],
            dtype=np.float64,
        ),
        "nre": np.asarray(
            [value if value is not None else np.nan for value in (
                _safe_ratio(4.0 * e, t) for e, t in zip(response, target, strict=True)
            )],
            dtype=np.float64,
        ),
    }


def paired_records(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the common signal summary and one auditable row per pair."""

    prediction = predictions.detach().float()
    target = targets.detach().float()
    _require(prediction.shape == target.shape, "prediction/target shapes differ")
    _require(prediction.ndim >= 2 and prediction.shape[1] == 2, "paired tensors must have width two")
    _require(prediction.shape[0] == EXPECTED_PAIRS, "panel pair count changed")
    pair_id_values = list(ids) if ids is not None else pair_ids(int(prediction.shape[0]))
    _require(len(pair_id_values) == EXPECTED_PAIRS, "pair id count changed")

    summary = signal_metrics.paired_signal_summary(prediction, target)
    components = signal_metrics.paired_signal_components(prediction, target)
    values = _geometry_values(components)
    _require(all(np.isfinite(value).all() for value in values.values()), "pair statistic is non-finite")

    target_energy = values["nre"] * 0.0 + components["target_delta_energy"].detach().double().cpu().numpy()
    prediction_energy = components["prediction_delta_energy"].detach().double().cpu().numpy()
    cross = components["cross_energy"].detach().double().cpu().numpy()
    correct = components["correct_loss"].detach().double().cpu().numpy()
    swapped = components["swapped_loss"].detach().double().cpu().numpy()
    records = []
    for index, pair_id in enumerate(pair_id_values):
        twin = index // 2
        direction = "forward" if index % 2 == 0 else "reverse"
        records.append(
            {
                "pair_index": index,
                "pair_id": str(pair_id),
                "twin": twin,
                "twin_id": twin,
                "direction": direction,
                "correct_native_loss": float(correct[index]),
                "swapped_history_native_loss": float(swapped[index]),
                "g_swap": float(values["g_swap"][index]),
                "target_delta_energy": float(target_energy[index]),
                "prediction_delta_energy": float(prediction_energy[index]),
                "response_gain": float(values["response_gain"][index]),
                "alignment": float(values["alignment"][index]),
                "nre": float(values["nre"][index]),
            }
        )

    geometry = summary["response_geometry"]
    cross_total = float(cross.sum())
    target_total = float(target_energy.sum())
    prediction_total = float(prediction_energy.sum())
    geometry["alignment"] = _safe_ratio(
        cross_total, math.sqrt(max(prediction_total * target_total, 0.0))
    )
    geometry["per_pair"] = {
        name: _finite_distribution(values[name])
        for name in ("response_gain", "alignment", "nre")
    }
    summary["panel_weighting"] = {
        "scheme": "D0_weighted_equal_binary_query_pairs",
        "pair_count": EXPECTED_PAIRS,
        "twin_count": EXPECTED_TWINS,
        "each_pair_weight": 1.0 / EXPECTED_PAIRS,
        "each_twin_weight": 1.0 / EXPECTED_TWINS,
    }
    return summary, records


def _extract_metric_values(records: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    values = np.asarray([float(row[name]) for row in records], dtype=np.float64)
    _require(values.size == EXPECTED_PAIRS and bool(np.isfinite(values).all()), f"invalid {name} records")
    return values


def paired_delta_bootstrap(
    d0_values: Sequence[float] | np.ndarray,
    d1_values: Sequence[float] | np.ndarray,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap and sign-flip inference for paired D1 minus D0 values."""

    left = np.asarray(d0_values, dtype=np.float64).reshape(-1)
    right = np.asarray(d1_values, dtype=np.float64).reshape(-1)
    _require(left.shape == right.shape and left.size > 0, "paired delta width differs")
    _require(bool(np.isfinite(left).all() and np.isfinite(right).all()), "paired delta non-finite")
    _require(int(resamples) > 0, "bootstrap resamples must be positive")
    delta = right - left
    observed = float(delta.mean())
    generator = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    sign_flip = np.empty(int(resamples), dtype=np.float64)
    chunk = 256
    for start in range(0, int(resamples), chunk):
        stop = min(start + chunk, int(resamples))
        indices = generator.integers(0, delta.size, size=(stop - start, delta.size))
        bootstrap[start:stop] = delta[indices].mean(axis=1)
        signs = generator.integers(0, 2, size=(stop - start, delta.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        sign_flip[start:stop] = (delta[None, :] * signs).mean(axis=1)
    return {
        "unit": "frozen_training_binary_query_pair",
        "pair_count": int(delta.size),
        "observed_delta_mean": observed,
        "delta_distribution": _finite_distribution(delta),
        "bootstrap": {
            "seed": int(seed),
            "resamples": int(resamples),
            "mean": float(bootstrap.mean()),
            "ci_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
            "probability_delta_nonnegative": float(np.mean(bootstrap >= 0.0)),
        },
        "sign_flip": {
            "seed": int(seed),
            "resamples": int(resamples),
            "null_mean": float(sign_flip.mean()),
            "two_sided_monte_carlo_p": float(
                (np.count_nonzero(np.abs(sign_flip) >= abs(observed)) + 1)
                / (int(resamples) + 1)
            ),
            "interpretation": "paired randomization null for D1 minus D0",
        },
    }


def paired_delta_inference(
    d0_records: Sequence[Mapping[str, Any]],
    d1_records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    _require(len(d0_records) == len(d1_records) == EXPECTED_PAIRS, "paired records are incomplete")
    _require(
        [row["pair_id"] for row in d0_records] == [row["pair_id"] for row in d1_records],
        "D0/D1 pair identities differ",
    )
    return {
        metric: paired_delta_bootstrap(
            _extract_metric_values(d0_records, metric),
            _extract_metric_values(d1_records, metric),
            seed=int(seed),
            resamples=int(resamples),
        )
        for metric in ("g_swap", "response_gain", "alignment", "nre")
    }


def validate_d1_report_schedule_consumed(
    report_path: Path,
    *,
    expected_schedule_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind the formal D1 training report to one fully consumed schedule."""

    path = Path(report_path).expanduser().resolve()
    _require(path.is_file(), f"missing D1 training report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result", payload)
    contract = result.get("d1_ms50_schedule_contract")
    _require(isinstance(contract, dict), "D1 report has no schedule contract")
    checks = contract.get("checks", {})
    observed_sha = contract.get("schedule_sha256")
    expected_sha = dict(expected_schedule_sha256 or d1_runner.EXPECTED_SCHEDULE_SHA256)
    required = {
        "candidate_id": contract.get("candidate_id") == "D1-MS50",
        "optimizer_steps": int(contract.get("optimizer_steps", -1)) == d1_runner.EXPECTED_BATCHES,
        "full_schedule_consumed_once": checks.get("full_schedule_consumed_once") is True,
        "schedule_sha256_exact": checks.get("schedule_sha256_exact") is True,
        "public_test_opened": contract.get("public_test_opened") is False,
        "schedule_hash_mapping": observed_sha == expected_sha,
    }
    _require(all(required.values()), f"D1 schedule consumption gate failed: {required}")
    train_pairs = result.get("batch", {}).get("hidden_pairs")
    provenance_pairs = payload.get("provenance", {}).get("data", {}).get("train_pairs")
    _require(
        train_pairs in (None, EXPECTED_ROWS_PER_TWIN * 8, EXPECTED_PAIRS)
        and provenance_pairs in (None, EXPECTED_PAIRS),
        "D1 report training pair count changed",
    )
    final_checkpoint = result.get("final_checkpoint")
    if isinstance(final_checkpoint, dict):
        _require(
            final_checkpoint.get("sha256") in (None, EXPECTED_CHECKPOINT_SHA256["D1"]),
            "D1 report final checkpoint hash changed",
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "candidate_id": "D1-MS50",
        "optimizer_steps": d1_runner.EXPECTED_BATCHES,
        "schedule_sha256": observed_sha,
        "checks": required,
    }


def validate_scope_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless Development/Public and non-training reads are zero."""

    checks = {
        "development_scorer_zero": int(counts.get("development_scorer", 0)) == 0,
        "public_scorer_zero": int(counts.get("public_scorer", 0)) == 0,
        "release_loader_zero": int(counts.get("release_loader", 0)) == 0,
        "non_training_benchmark_reads_zero": int(counts.get("non_training_benchmark_reads", 0)) == 0,
    }
    _require(all(checks.values()), f"forbidden Development/Public action: {counts}")
    return checks


def static_identity(
    *,
    d0_checkpoint: Path = DEFAULT_D0_CHECKPOINT,
    d1_checkpoint: Path = DEFAULT_D1_CHECKPOINT,
    coja_checkpoint: Path = DEFAULT_COJA_CHECKPOINT,
    catalog: Path = DEFAULT_CATALOG,
    schedule_dir: Path = DEFAULT_SCHEDULE_DIR,
    d1_report: Path = DEFAULT_D1_REPORT,
) -> dict[str, Any]:
    checkpoints = {"D0": Path(d0_checkpoint), "D1": Path(d1_checkpoint), "COJA": Path(coja_checkpoint)}
    checkpoint_identity = {}
    for name, path in checkpoints.items():
        path = path.expanduser().resolve()
        _require(path.is_file(), f"missing {name} checkpoint: {path}")
        observed = _sha256(path)
        _require(observed == EXPECTED_CHECKPOINT_SHA256[name], f"{name} checkpoint changed")
        checkpoint_identity[name] = {"path": str(path), "sha256": observed}

    catalog_path = Path(catalog).expanduser().resolve()
    _require(_sha256(catalog_path) == EXPECTED_CATALOG_SHA256, "D1-0 catalog changed")
    catalog_rows = catalog_path.read_text(encoding="utf-8").splitlines()
    _require(len(catalog_rows) == EXPECTED_TWINS, "catalog twin count changed")
    schedule = d1_runner.verify_schedule_artifact(Path(schedule_dir))
    report = validate_d1_report_schedule_consumed(Path(d1_report))
    _require(
        _sha256(Path(d1_runner.__file__).resolve()) == EXPECTED_D1_RUNNER_SHA256,
        "D1 runner source changed",
    )
    return {
        "checkpoints": checkpoint_identity,
        "catalog": {
            "path": str(catalog_path),
            "sha256": EXPECTED_CATALOG_SHA256,
            "twin_count": EXPECTED_TWINS,
        },
        "schedule": {
            "dir": str(schedule["schedule_dir"]),
            "sha256": schedule["observed_sha256"],
            "runner_sha256": EXPECTED_D1_RUNNER_SHA256,
        },
        "d1_training_report": report,
        "authority": {
            "training_only": True,
            "optimizer_steps": 0,
            "development_opened": False,
            "public_test_opened": False,
            "cem_executed": False,
        },
        "source": {"path": str(THIS_SOURCE), "sha256": _sha256(THIS_SOURCE)},
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_model(
    *,
    model: torch.nn.Module,
    mixed: Any,
    hidden: Any,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Encode and predict all hidden rows in eval mode, retaining no gradients."""

    _require(int(batch_size) > 0, "batch size must be positive")
    pixels = hidden.pixels
    actions = hidden.action
    _require(pixels.shape[0] == actions.shape[0] == EXPECTED_HIDDEN_ROWS, "hidden row count changed")
    state_before = _module_state_sha256(model)
    encoder_before = _encoder_identity_sha256(model)
    model.eval()
    prediction_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, EXPECTED_HIDDEN_ROWS, int(batch_size)):
            stop = min(start + int(batch_size), EXPECTED_HIDDEN_ROWS)
            prepared_pixels = mixed.pilot.preprocess_pixels(pixels[start:stop], device)
            prepared_actions = actions[start:stop].to(device=device, non_blocking=True)
            with _autocast(device):
                encoded = model.encode({"pixels": prepared_pixels, "action": prepared_actions})
                embeddings = encoded["emb"]
                prediction = model.predict(embeddings[:, :3], encoded["act_emb"][:, :3])
            _require(embeddings.shape[1] >= 4 and prediction.shape[1] >= 3, "latent horizon changed")
            target_parts.append(embeddings[:, 3].detach().float().cpu())
            prediction_parts.append(prediction[:, -1].detach().float().cpu())
    predictions = torch.cat(prediction_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    state_after = _module_state_sha256(model)
    encoder_after = _encoder_identity_sha256(model)
    _require(state_before == state_after, "model state hash changed during evaluation")
    _require(encoder_before == encoder_after, "target encoder changed during evaluation")
    _require(predictions.shape == targets.shape and predictions.shape[0] == EXPECTED_HIDDEN_ROWS, "latent rows missing")
    return {
        "predictions": predictions.reshape(EXPECTED_PAIRS, 2, -1),
        "targets": targets.reshape(EXPECTED_PAIRS, 2, -1),
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "target_encoder_sha256": encoder_after,
        "target_latents_sha256": _tensor_sha256(targets),
        "eval_mode": all(not module.training for module in model.modules()),
        "batch_size": int(batch_size),
        "encoded_rows": int(predictions.shape[0]),
    }


def _prepare_training_context(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    """Reuse the zero-step loader/materialization path without opening eval splits."""

    replay = native_gradient.replay
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
    return replay, motion_score, motion, mixed, (missing, previous_weight, was_twin_variant, was_diagnostic_variant)


def run(args: argparse.Namespace) -> dict[str, Any]:
    static = static_identity(
        d0_checkpoint=args.d0_checkpoint,
        d1_checkpoint=args.d1_checkpoint,
        coja_checkpoint=args.coja_checkpoint,
        catalog=args.catalog,
        schedule_dir=args.schedule_dir,
        d1_report=args.d1_report,
    )
    device = torch.device(args.device)
    _require(device.type == "cuda" and torch.cuda.is_available(), "CUDA required for formal panel; use --check-only for CPU gates")
    native_gradient._prepare_optional_flash_attention()
    replay, motion_score, motion, mixed, restore = _prepare_training_context(args)
    missing, previous_weight, was_twin_variant, was_diagnostic_variant = restore
    # Initialize CUDA RNG state before taking the outer snapshot, matching the
    # frozen zero-step audit contract.
    torch.cuda.get_rng_state_all()
    outer_rng = replay.gradient_core._rng_snapshot()
    arms = {
        "D0": Path(args.d0_checkpoint).expanduser().resolve(),
        "D1": Path(args.d1_checkpoint).expanduser().resolve(),
        "COJA": Path(args.coja_checkpoint).expanduser().resolve(),
    }
    model_reports: dict[str, Any] = {}
    hidden = None
    pool_identity = None
    guard_counts: Mapping[str, Any] = {}
    try:
        mixed.VARIANT_WEIGHTS[ANALYSIS_VARIANT] = ("native", replay.WEIGHT, "identifiable_future_only")
        motion.TWIN_GROUP_VARIANTS.add(ANALYSIS_VARIANT)
        trainer = motion.trainer
        trainer.DIAGNOSTIC_VARIANTS["lewm"].add(ANALYSIS_VARIANT)
        with replay.replay._install_fail_closed_guards(
            motion, motion_score, allow_training_table=True
        ) as counts:
            hidden, anchor, inputs = zero_step._materialize_training_inputs(
                motion=motion,
                original_h5=args.original_h5.expanduser().resolve(),
                original_lance=args.original_lance.expanduser().resolve(),
            )
            pool_identity = validate_hidden_pool(hidden.pixels, hidden.action)
            for name, checkpoint in arms.items():
                mixed.pilot.set_reproducible_seed(d1_runner.EXPECTED_SEED)
                model, load_receipt = mixed.load_model_for_variant(
                    checkpoint, variant=ANALYSIS_VARIANT, device=device
                )
                expected_hash = EXPECTED_CHECKPOINT_SHA256[name]
                expected_state = EXPECTED_MODEL_STATE_SHA256[name]
                _require(load_receipt.get("sha256") == expected_hash, f"{name} checkpoint load hash changed")
                _require(load_receipt.get("model_state_sha256") == expected_state, f"{name} model state hash changed")
                _require(load_receipt.get("strict_state_dict_load") is True, f"{name} load was not strict")
                evaluated = evaluate_model(
                    model=model,
                    mixed=mixed,
                    hidden=hidden,
                    device=device,
                    batch_size=int(args.batch_size),
                )
                summary, records = paired_records(
                    evaluated["predictions"], evaluated["targets"], ids=pair_ids()
                )
                model_reports[name] = {
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": expected_hash,
                        "model_state_sha256": expected_state,
                    },
                    "load_receipt": load_receipt,
                    "evaluation": {
                        key: value
                        for key, value in evaluated.items()
                        if key not in {"predictions", "targets"}
                    },
                    "paired_signal": summary,
                    "records": records,
                }
                del model
            guard_counts = dict(counts)
        validate_scope_counts(guard_counts)
        _require(int(guard_counts.get("training_table_reads", 0)) == 1, "training split was not materialized exactly once")
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
            motion.trainer.DIAGNOSTIC_VARIANTS["lewm"].add(ANALYSIS_VARIANT)
        else:
            motion.trainer.DIAGNOSTIC_VARIANTS["lewm"].discard(ANALYSIS_VARIANT)
    _require(replay.gradient_core._rng_equal(outer_rng, replay.gradient_core._rng_snapshot()), "outer RNG restoration failed")

    encoder_hashes = {
        name: report["evaluation"]["target_encoder_sha256"]
        for name, report in model_reports.items()
    }
    comparable = len(set(value for value in encoder_hashes.values() if value is not None)) <= 1
    d0_records = model_reports["D0"]["records"]
    d1_records = model_reports["D1"]["records"]
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "completed_training_d0_weighted_inference_panel",
        "claim_scope": "frozen_Training_only_descriptive_mechanism_audit_not_generalization",
        "optimizer_updates": 0,
        "source": static["source"],
        "static_identity": static,
        "training_pool": {
            **pool_identity,
            "materialization": inputs["receipt"],
            "weighting": "D0 equal weighting over all 8,192 binary query pairs",
        },
        "target_encoder_identity": {
            "per_model_sha256": encoder_hashes,
            "per_model_target_latents_sha256": {
                name: report["evaluation"]["target_latents_sha256"]
                for name, report in model_reports.items()
            },
            "matched": comparable,
            "comparable_boundary": (
                "same frozen target encoder/latent basis; cross-model geometry is comparable"
                if comparable
                else "target encoder identity differs; report per-model metrics, do not compare absolute latent scales"
            ),
        },
        "models": model_reports,
        "D1_minus_D0": paired_delta_inference(
            d0_records,
            d1_records,
            seed=int(args.bootstrap_seed),
            resamples=int(args.bootstrap_resamples),
        ),
        "guard_counts": dict(guard_counts),
        "scope_read_counts": {
            "development": int(guard_counts.get("development_scorer", 0)),
            "public": int(guard_counts.get("public_scorer", 0)),
            "non_training_benchmark": int(
                guard_counts.get("non_training_benchmark_reads", 0)
            ),
        },
        "evidence_boundary": {
            "training_only": True,
            "development_opened": False,
            "public_test_opened": False,
            "cem_executed": False,
            "optimizer_steps": 0,
            "all_hidden_rows_encoded_and_predicted": True,
            "model_state_hash_unchanged_before_after_each_model": True,
        },
        "interpretation_boundary": {
            "G_swap": "Descriptive paired native-loss margin on the frozen Training population; not a held-out or generalization result.",
            "D0_weighting": "All 8,192 binary query pairs receive equal weight despite D1 training exposure multiplicity.",
            "gain_alignment_nre": "Comparable across models only when target encoder identity is matched; NRE is response-error over target-response energy.",
        },
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0-checkpoint", type=Path, default=DEFAULT_D0_CHECKPOINT)
    parser.add_argument("--d1-checkpoint", type=Path, default=DEFAULT_D1_CHECKPOINT)
    parser.add_argument("--coja-checkpoint", type=Path, default=DEFAULT_COJA_CHECKPOINT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--schedule-dir", type=Path, default=DEFAULT_SCHEDULE_DIR)
    parser.add_argument("--d1-report", type=Path, default=DEFAULT_D1_REPORT)
    parser.add_argument("--original-h5", type=Path, default=native_gradient.DEFAULT_ORIGINAL_H5)
    parser.add_argument("--original-lance", type=Path, default=native_gradient.DEFAULT_ORIGINAL_LANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    _require(args.batch_size > 0, "batch size must be positive")
    _require(args.bootstrap_resamples > 0, "bootstrap resamples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        identity = static_identity(
            d0_checkpoint=args.d0_checkpoint,
            d1_checkpoint=args.d1_checkpoint,
            coja_checkpoint=args.coja_checkpoint,
            catalog=args.catalog,
            schedule_dir=args.schedule_dir,
            d1_report=args.d1_report,
        )
        print(json.dumps({"status": "passed_static_identity", "identity": identity}, sort_keys=True))
        return 0
    output = args.output.expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    payload = run(args)
    _write_exclusive(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output), "sha256": _sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
