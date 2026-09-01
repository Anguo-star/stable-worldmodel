#!/usr/bin/env python3
"""Audit Motion Damping Development response after removing visual history.

This is an inference-only Development audit.  For each frozen paired query,
the terminal query RGB and action block are retained while the three history
frames are replaced by three copies of that query.  D0 native, D1 native, and
COJA final checkpoints use the same adapter and their own native target
encoder.  No training split, public split, optimizer, or planner is read.
"""

from __future__ import annotations

import argparse
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
for source_root in (ROOT, CONTEXTWORLD):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.benchmarks.adapters import (  # noqa: E402
    StableWorldModelLeWMMotionDampingAdapter,
    validate_adapter_protocol,
)
from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    MotionDampingICLDevelopmentDataset,
    load_motion_damping_icl_release,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    conditional_signal_metrics as signal_metrics,
)


ANALYSIS_ID = "pusht_motion_damping_d1_removed_history_v1"
HISTORY_TOKENS = 3
EXPECTED_DEVELOPMENT_PAIRS = 256
EXPECTED_OPTIMIZER_STEPS = 8192
DEFAULT_BATCH_SIZE = 64
DEFAULT_BOOTSTRAP_SEED = 20260901
DEFAULT_BOOTSTRAP_RESAMPLES = 4096

FORMAL_RUN = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_energy_stratified_native_v1/"
    "s14321_step8192_v2_final"
)
DEFAULT_OUTPUT = FORMAL_RUN / "development_removed_history_v1.json"
DEFAULT_RELEASE_CONFIG = CONTEXTWORLD / (
    "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml"
)
DEFAULT_D1_REPORT = FORMAL_RUN / "training_report.json"

_D0_RUN = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_full_release_visible_joint_absolute_single_stage_"
    "native_control_step8192_v1/s14321_step8192_v1"
)
_COJA_RUN = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_full_release_visible_joint_absolute_single_stage_"
    "step8192_v1/s14321_step8192_v1"
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "D0": {
        "checkpoint": _D0_RUN / "mixed_frozen_image_identifiable_future_native_0p09_step8192.pt",
        "config": _D0_RUN / "config.json",
        "report": _D0_RUN / "training_report.json",
        "checkpoint_sha256": "6063e01524f49ac974f233bd876518e35ece6c8d406b19f7a25afd6fcf2ea3ff",
        "config_sha256": "7c0853352043da6f17040af85dacaaf8b5de9dca4963aedb6908e1c4c6486b0b",
        "report_sha256": "14fb5dfdbeb32cca91b570572c472dd5ab5ed8256cbc5e1a3eaec7bcf175d68b",
        "model_state_sha256": "a17749e3b98442efe17b330c21c0bbe3ec8375a24aee05370d37e327c7c060dc",
    },
    "D1": {
        "checkpoint": FORMAL_RUN / "mixed_frozen_image_identifiable_future_native_0p09_step8192.pt",
        "config": FORMAL_RUN / "config.json",
        "report": DEFAULT_D1_REPORT,
        "checkpoint_sha256": "9bb4efd8831c851c706e206acb1153a79496422ffda1dc745a586c6827bdb32b",
        "config_sha256": "7c0853352043da6f17040af85dacaaf8b5de9dca4963aedb6908e1c4c6486b0b",
        "report_sha256": "54625ddb4f603fc9bd0120593ec34421904bd3e699c37587a72ef17aba71e4ed",
        "model_state_sha256": "a2b13f6c3fbafb3a4d6236aa0bdf004838efe1aab96649477b11bfdfb68f29d6",
    },
    "COJA": {
        "checkpoint": _COJA_RUN / "mixed_frozen_image_identifiable_future_native_0p09_step8192.pt",
        "config": _COJA_RUN / "config.json",
        "report": _COJA_RUN / "training_report.json",
        "checkpoint_sha256": "16466ce3410eb60a48812b3ff2bf31d12f5c426b8c2f1cbecab22ac6e03f6ff7",
        "config_sha256": "7c0853352043da6f17040af85dacaaf8b5de9dca4963aedb6908e1c4c6486b0b",
        "report_sha256": "79bba86281061d6208d1c7e587dac42cb44014073c0d38657260e529c9438e8d",
        "model_state_sha256": "80af7e9861c5f00f3ad257aedc5803809fd74ca742e217de22da8ef28a244c27",
    },
}
MODEL_ROLES = {
    "D0": "D0 matched native baseline",
    "D1": "D1 native final checkpoint",
    "COJA": "COJA positive control",
}
EXPECTED_RELEASE_CONFIG_SHA256 = "1795717d8bfa1d1cfcc01a69931b6241b0e7759dcf43553a6c4cca225ec9326b"
EXPECTED_MANIFEST_SHA256 = "48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b"
EXPECTED_DEVELOPMENT_TABLE_SHA256 = "64d43c931f106c2d53e3c3084e62381d2f2640c9d943e269475f3fb76aaa2de4"
EXPECTED_D1_SCHEDULE_SHA256 = {
    "config.json": "45c5ed29aa950877c2ccc091ee041762c64b7ba060a24af1ea9ca220fa169171",
    "multiplicity.jsonl": "4be57c44b5e9485902edabdfbfb1c629b4bf433ed375ab00c783ae0ed187abb8",
    "receipt.json": "6c043e3b0169e721b0c54289e9b449b3c8690e9cc3c8c88270829d8c6bf04ad6",
    "schedule.jsonl": "e058384b66f129ace7e30dec354373fc14c885581bd57a5e86fd446be6f45b96",
    "summary.json": "64bbbc9a39649c9e8d8283006226a738615b0b440167d6d66396114a670faa47",
}
EXPECTED_STABLEWM_REF = "875e607fc08aa72eacb94d5d178127804134cc06"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    _require(array.size > 0, f"{name} is empty")
    return array


def _finite_distribution(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(array.size > 0 and bool(np.isfinite(array).all()), "non-finite statistic")
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
    result = float(numerator) / float(denominator)
    return result if math.isfinite(result) else None


def pair_ids(pair_count: int = EXPECTED_DEVELOPMENT_PAIRS) -> list[str]:
    _require(int(pair_count) > 0, "pair count must be positive")
    return [f"motion-damping-development-{index:04d}" for index in range(int(pair_count))]


def validate_pair_identity(
    faster_pixels: Any,
    no_extra_pixels: Any,
    actions: Any,
    ids: Sequence[str] | None = None,
    *,
    expected_pairs: int = EXPECTED_DEVELOPMENT_PAIRS,
) -> dict[str, Any]:
    """Fail closed before construction when paired query/action identity is used."""

    faster = _as_numpy(faster_pixels, "faster_pixels")
    slower = _as_numpy(no_extra_pixels, "no_extra_pixels")
    action = _as_numpy(actions, "actions")
    expected = int(expected_pairs)
    _require(faster.shape == slower.shape, "paired pixel shapes differ")
    _require(faster.ndim == 5 and faster.shape[0] == expected, "Development pixel geometry changed")
    _require(faster.shape[1] >= HISTORY_TOKENS + 1 and faster.shape[-1] == 3, "history/future pixel geometry changed")
    _require(action.ndim >= 2 and action.shape[0] == expected, "Development action geometry changed")
    pair_values = list(ids) if ids is not None else pair_ids(expected)
    _require(len(pair_values) == expected and len(set(map(str, pair_values))) == expected, "pair identities changed")
    history_differs = np.any(
        faster[:, :HISTORY_TOKENS] != slower[:, :HISTORY_TOKENS],
        axis=tuple(range(1, faster[:, :HISTORY_TOKENS].ndim)),
    )
    future_differs = np.any(
        faster[:, 3] != slower[:, 3],
        axis=tuple(range(1, faster[:, 3].ndim)),
    )
    checks = {
        "pair_count_exact": faster.shape[0] == expected,
        "query_rgb_exact": bool(np.array_equal(faster[:, 2], slower[:, 2])),
        "action_block_finite": bool(np.isfinite(action).all()),
        "history_differs": bool(history_differs.all()),
        "future_differs": bool(future_differs.all()),
    }
    _require(all(checks.values()), f"Development pair identity failed: {checks}")
    return {
        "pair_count": expected,
        "condition_count": 2 * expected,
        "history_tokens": HISTORY_TOKENS,
        "pair_ids_sha256": hashlib.sha256("\n".join(map(str, pair_values)).encode("utf-8")).hexdigest(),
        "checks": checks,
    }


def validate_paired_modes(
    faster_pixels: Any,
    no_extra_pixels: Any,
    faster_actions: Any,
    no_extra_actions: Any,
    ids: Sequence[str] | None = None,
    *,
    expected_pairs: int = EXPECTED_DEVELOPMENT_PAIRS,
) -> dict[str, Any]:
    """Validate both mode arrays and their exact query/action overlap."""

    fp = _as_numpy(faster_pixels, "faster_pixels")
    npix = _as_numpy(no_extra_pixels, "no_extra_pixels")
    fa = _as_numpy(faster_actions, "faster_actions")
    na = _as_numpy(no_extra_actions, "no_extra_actions")
    _require(fa.shape == na.shape and bool(np.array_equal(fa, na)), "paired actions differ")
    identity = validate_pair_identity(fp, npix, fa, ids, expected_pairs=expected_pairs)
    identity["checks"]["paired_actions_exact"] = True
    return identity


validate_development_pair_identity = validate_paired_modes
validate_development_pairs = validate_paired_modes


def build_removed_history(pixels: Any) -> np.ndarray:
    """Replace a mode's history with [x_q, x_q, x_q] without changing x_q."""

    array = _as_numpy(pixels, "pixels")
    _require(array.ndim == 5 and array.shape[1] >= HISTORY_TOKENS + 1, "pixel history geometry changed")
    query = np.ascontiguousarray(array[:, 2])
    removed = np.repeat(query[:, None], HISTORY_TOKENS, axis=1)
    _require(bool(np.array_equal(removed[:, 0], query)), "removed query changed")
    _require(bool(np.array_equal(removed[:, 0], removed[:, 1]) and np.array_equal(removed[:, 1], removed[:, 2])), "removed history is not constant")
    return np.ascontiguousarray(removed)


make_removed_history = build_removed_history


def build_removed_histories(faster_pixels: Any, no_extra_pixels: Any) -> np.ndarray:
    fp = _as_numpy(faster_pixels, "faster_pixels")
    npix = _as_numpy(no_extra_pixels, "no_extra_pixels")
    _require(fp.shape == npix.shape, "paired pixel shapes differ")
    return np.concatenate([build_removed_history(fp), build_removed_history(npix)], axis=0)


def _response_arrays(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, np.ndarray]:
    components = signal_metrics.paired_signal_components(predictions, targets)
    target = components["target_delta_energy"].detach().double().cpu().numpy()
    prediction = components["prediction_delta_energy"].detach().double().cpu().numpy()
    cross = components["cross_energy"].detach().double().cpu().numpy()
    error = (4.0 * components["response_loss"]).detach().double().cpu().numpy()
    _require(bool(np.isfinite(target).all()) and bool((target > 0.0).all()), "target response energy is not positive")
    gain = cross / target
    alignment_denominator = np.sqrt(np.maximum(prediction * target, 0.0))
    alignment = np.divide(cross, alignment_denominator, out=np.zeros_like(cross), where=alignment_denominator > 0.0)
    nre = error / target
    return {
        "g_swap": components["g_swap"].detach().double().cpu().numpy(),
        "response_energy": prediction,
        "target_energy": target,
        "cross_energy": cross,
        "gain": gain,
        "alignment": alignment,
        "nre": nre,
    }


def _response_summary(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    response = np.asarray(values["response_energy"], dtype=np.float64)
    target = np.asarray(values["target_energy"], dtype=np.float64)
    cross = np.asarray(values["cross_energy"], dtype=np.float64)
    gain = np.asarray(values["gain"], dtype=np.float64)
    alignment = np.asarray(values["alignment"], dtype=np.float64)
    nre = np.asarray(values["nre"], dtype=np.float64)
    aggregate_gain = _safe_ratio(float(cross.sum()), float(target.sum()))
    aggregate_alignment = _safe_ratio(float(cross.sum()), math.sqrt(float(response.sum() * target.sum())))
    aggregate_nre = _safe_ratio(float((nre * target).sum()), float(target.sum()))
    return {
        "response_energy": _finite_distribution(response),
        "gain": {
            "distribution": _finite_distribution(gain),
            "positive_fraction": float(np.mean(gain > 0.0)),
            "aggregate": aggregate_gain,
        },
        "alignment": {
            "distribution": _finite_distribution(alignment),
            "aggregate": aggregate_alignment,
        },
        "nre": {
            "distribution": _finite_distribution(nre),
            "aggregate": aggregate_nre,
        },
        "g_swap": {
            "distribution": _finite_distribution(values["g_swap"]),
            "positive_fraction": float(np.mean(np.asarray(values["g_swap"]) > 0.0)),
        },
    }


def removed_history_metrics(
    correct_predictions: Any,
    removed_predictions: Any,
    targets: Any,
    ids: Sequence[str] | None = None,
    *,
    expected_pairs: int = EXPECTED_DEVELOPMENT_PAIRS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute target-MSE change and removed-history conditional response."""

    correct = torch.as_tensor(correct_predictions).detach().float().cpu()
    removed = torch.as_tensor(removed_predictions).detach().float().cpu()
    target = torch.as_tensor(targets).detach().float().cpu()
    _require(correct.shape == removed.shape == target.shape, "prediction/target shapes differ")
    _require(correct.ndim >= 3 and correct.shape[1] == 2, "paired prediction shape changed")
    expected = int(expected_pairs)
    _require(correct.shape[0] == expected, "Development pair count changed")
    _require(bool(torch.isfinite(correct).all() and torch.isfinite(removed).all() and torch.isfinite(target).all()), "prediction contains non-finite values")
    pair_values = list(ids) if ids is not None else pair_ids(expected)
    _require(len(pair_values) == expected and len(set(map(str, pair_values))) == expected, "pair identities changed")

    correct_mse = 0.5 * ((correct - target).square().mean(dim=tuple(range(2, correct.ndim))).sum(dim=1))
    removed_mse = 0.5 * ((removed - target).square().mean(dim=tuple(range(2, removed.ndim))).sum(dim=1))
    # Positive values mean that replacing history increased target error.
    removed_mse_increase = removed_mse - correct_mse
    correct_response = _response_arrays(correct, target)
    removed_response = _response_arrays(removed, target)
    correct_mse_np = correct_mse.double().numpy()
    removed_mse_np = removed_mse.double().numpy()
    increase_np = removed_mse_increase.double().numpy()
    records: list[dict[str, Any]] = []
    for index, pair_id in enumerate(pair_values):
        records.append(
            {
                "pair_index": int(index),
                "pair_id": str(pair_id),
                "mode_pair": "faster_decay_vs_no_extra_decay",
                "correct_target_mse": float(correct_mse_np[index]),
                "removed_target_mse": float(removed_mse_np[index]),
                "removed_target_mse_increase": float(increase_np[index]),
                "removed_target_mse_minus_correct": float(increase_np[index]),
                "correct_g_swap": float(correct_response["g_swap"][index]),
                "removed_g_swap": float(removed_response["g_swap"][index]),
                "correct_response_energy": float(correct_response["response_energy"][index]),
                "removed_response_energy": float(removed_response["response_energy"][index]),
                "target_response_energy": float(removed_response["target_energy"][index]),
                "removed_response_gain": float(removed_response["gain"][index]),
                "removed_response_alignment": float(removed_response["alignment"][index]),
                "removed_nre": float(removed_response["nre"][index]),
                "correct_response_gain": float(correct_response["gain"][index]),
                "correct_nre": float(correct_response["nre"][index]),
            }
        )
    summary = {
        "pair_count": expected,
        "target_mse": {
            "correct": _finite_distribution(correct_mse_np),
            "removed": _finite_distribution(removed_mse_np),
            "removed_minus_correct": _finite_distribution(increase_np),
            "mean_correct": float(correct_mse_np.mean()),
            "mean_removed": float(removed_mse_np.mean()),
            "mean_removed_minus_correct": float(increase_np.mean()),
        },
        "correct_response": _response_summary(correct_response),
        "removed_response": _response_summary(removed_response),
        # Flat aliases make the audit table convenient for downstream checks.
        "removed_gain": _response_summary(removed_response)["gain"],
        "removed_nre": _response_summary(removed_response)["nre"],
        "removed_alignment": _response_summary(removed_response)["alignment"],
        "weighting": {
            "unit": "Development_pair",
            "scheme": "equal_pair_weight",
            "pair_weight": 1.0 / expected,
        },
    }
    return summary, records


compute_removed_history_metrics = removed_history_metrics


def paired_delta_bootstrap(
    d0_values: Sequence[float] | np.ndarray,
    d1_values: Sequence[float] | np.ndarray,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap and paired sign-flip inference for D1 minus D0 values."""

    left = np.asarray(d0_values, dtype=np.float64).reshape(-1)
    right = np.asarray(d1_values, dtype=np.float64).reshape(-1)
    _require(left.shape == right.shape and left.size > 0, "paired delta shape differs")
    _require(bool(np.isfinite(left).all() and np.isfinite(right).all()), "paired delta is non-finite")
    _require(int(resamples) > 0, "resamples must be positive")
    delta = right - left
    generator = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    sign_flip = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), 256):
        stop = min(start + 256, int(resamples))
        indices = generator.integers(0, delta.size, size=(stop - start, delta.size))
        bootstrap[start:stop] = delta[indices].mean(axis=1)
        signs = generator.integers(0, 2, size=(stop - start, delta.size)).astype(np.float64)
        sign_flip[start:stop] = (delta[None, :] * (2.0 * signs - 1.0)).mean(axis=1)
    observed = float(delta.mean())
    return {
        "pair_count": int(delta.size),
        "difference": "D1_minus_D0",
        "observed_delta_mean": observed,
        "delta_distribution": _finite_distribution(delta),
        "bootstrap": {
            "seed": int(seed),
            "resamples": int(resamples),
            "ci_95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "probability_delta_nonnegative": float(np.mean(bootstrap >= 0.0)),
        },
        "sign_flip": {
            "seed": int(seed),
            "resamples": int(resamples),
            "null_mean": float(sign_flip.mean()),
            "two_sided_monte_carlo_p": float((np.count_nonzero(np.abs(sign_flip) >= abs(observed)) + 1) / (int(resamples) + 1)),
        },
    }


def paired_delta_inference(
    d0_records: Sequence[Mapping[str, Any]],
    d1_records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    _require(len(d0_records) == len(d1_records), "D0/D1 record count differs")
    _require([row["pair_id"] for row in d0_records] == [row["pair_id"] for row in d1_records], "D0/D1 pair identities differ")
    metrics = (
        "removed_target_mse_increase",
        "removed_response_gain",
        "removed_response_alignment",
        "removed_nre",
    )
    return {
        metric: paired_delta_bootstrap(
            [float(row[metric]) for row in d0_records],
            [float(row[metric]) for row in d1_records],
            seed=int(seed),
            resamples=int(resamples),
        )
        for metric in metrics
    }


def validate_source_guards(source: str | Path = THIS_SOURCE) -> dict[str, bool]:
    """Reject source snippets that could widen this audit's evidence boundary."""

    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else str(source)
    forbidden = {
        "public_dataset_reader": "MotionDamping" + "ICLEvalDataset",
        "public_table_reader": "validation" + ".lance",
        "optimizer_api": "torch." + "optim",
        "gradient_api": "autograd." + "grad",
        "backward_api": "." + "backward(",
        "step_api": "." + "step(",
        "planner_api": "C" + "EM",
    }
    checks = {name: token not in text for name, token in forbidden.items()}
    _require(all(checks.values()), f"source boundary failed: {checks}")
    return checks


def validate_scope_counts(counts: Mapping[str, Any] | None = None) -> dict[str, bool]:
    values = counts or {}
    checks = {
        "development_only": int(values.get("development_reads", 0)) >= 0,
        "public_reads_zero": int(values.get("public_reads", 0)) == 0,
        "cem_calls_zero": int(values.get("cem_calls", 0)) == 0,
        "optimizer_steps_zero": int(values.get("optimizer_steps", 0)) == 0,
    }
    _require(all(checks.values()), f"evidence boundary failed: {checks}")
    return checks


def _validate_report(label: str, path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label} training report: {path}")
    observed = _sha256(path)
    _require(observed == spec["report_sha256"], f"{label} report SHA changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict) and payload.get("status") == "completed", f"{label} report incomplete")
    result = payload.get("result", {})
    _require(int(payload.get("fixed_checkpoint_step", -1)) == EXPECTED_OPTIMIZER_STEPS, f"{label} fixed step changed")
    _require(int(result.get("optimizer_steps", -1)) == EXPECTED_OPTIMIZER_STEPS, f"{label} optimizer step changed")
    final = result.get("final_checkpoint", {})
    _require(final.get("sha256") == spec["checkpoint_sha256"], f"{label} report checkpoint SHA changed")
    _require(final.get("model_state_sha256") == spec["model_state_sha256"], f"{label} report model state SHA changed")
    provenance = payload.get("provenance", {})
    data = provenance.get("data", {})
    _require(data.get("train_pairs") == 8192, f"{label} report training population changed")
    _require(data.get("loader_validation_pairs") == EXPECTED_DEVELOPMENT_PAIRS, f"{label} Development count changed")
    _require(data.get("independent_validation_opened") is False, f"{label} independent validation opened")
    _require(payload.get("independent_validation_used_for_selection") is False, f"{label} independent selection changed")
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "train_pairs": 8192,
        "development_pairs": EXPECTED_DEVELOPMENT_PAIRS,
        "public_test_scoring_opened": False,
    }


def validate_d1_schedule_consumed(report_path: Path = DEFAULT_D1_REPORT) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("result", {}).get("d1_ms50_schedule_contract")
    _require(isinstance(contract, dict), "D1 schedule contract missing")
    checks = contract.get("checks", {})
    observed = contract.get("schedule_sha256")
    required = {
        "candidate_id": contract.get("candidate_id") == "D1-MS50",
        "optimizer_steps": int(contract.get("optimizer_steps", -1)) == EXPECTED_OPTIMIZER_STEPS,
        "full_schedule_consumed_once": checks.get("full_schedule_consumed_once") is True,
        "schedule_sha256_exact": checks.get("schedule_sha256_exact") is True,
        "public_test_opened": contract.get("public_test_opened") is False,
        "schedule_hash_mapping": observed == EXPECTED_D1_SCHEDULE_SHA256,
    }
    _require(all(required.values()), f"D1 schedule gate failed: {required}")
    return {"path": str(path), "sha256": _sha256(path), "schedule_sha256": dict(observed), "checks": required}


def validate_static_identity(
    *,
    release_config: Path = DEFAULT_RELEASE_CONFIG,
    d1_report: Path = DEFAULT_D1_REPORT,
    model_specs: Mapping[str, Mapping[str, Any]] = MODEL_SPECS,
) -> dict[str, Any]:
    validate_source_guards()
    config_path = Path(release_config).expanduser().resolve()
    _require(config_path.is_file(), f"missing release config: {config_path}")
    config_sha = _sha256(config_path)
    _require(config_sha == EXPECTED_RELEASE_CONFIG_SHA256, "release config SHA changed")
    release = load_motion_damping_icl_release(config_path)
    development = release["evaluation"]["development"]
    _require(development["split"] == "loader_validation", "Development split changed")
    _require(development["pair_count"] == EXPECTED_DEVELOPMENT_PAIRS, "Development pair count changed")
    _require(development["lance_table_sha256"] == EXPECTED_DEVELOPMENT_TABLE_SHA256, "Development table SHA changed")
    _require(development["data_manifest_sha256"] == EXPECTED_MANIFEST_SHA256, "release manifest SHA changed")
    _require(development["public_test"]["opened"] is False and development["public_test"]["read"] is False, "public boundary changed")
    bundles: dict[str, Any] = {}
    for label in ("D0", "D1", "COJA"):
        spec = model_specs[label]
        checkpoint = Path(spec["checkpoint"]).expanduser().resolve()
        config = Path(spec["config"]).expanduser().resolve()
        _require(checkpoint.is_file(), f"missing {label} checkpoint: {checkpoint}")
        _require(config.is_file(), f"missing {label} config: {config}")
        _require(_sha256(checkpoint) == spec["checkpoint_sha256"], f"{label} checkpoint SHA changed")
        _require(_sha256(config) == spec["config_sha256"], f"{label} config SHA changed")
        report = _validate_report(label, Path(spec["report"]).expanduser().resolve(), spec)
        bundles[label] = {
            "checkpoint": {"path": str(checkpoint), "sha256": spec["checkpoint_sha256"], "model_state_sha256": spec["model_state_sha256"]},
            "config": {"path": str(config), "sha256": spec["config_sha256"]},
            "report": report,
        }
    d1_schedule = validate_d1_schedule_consumed(Path(d1_report))
    return {
        "release_config": {"path": str(config_path), "sha256": config_sha},
        "development_contract": {
            "split": development["split"],
            "pair_count": int(development["pair_count"]),
            "lance_table_sha256": development["lance_table_sha256"],
            "data_manifest_sha256": development["data_manifest_sha256"],
            "public_test_opened": False,
        },
        "models": bundles,
        "d1_schedule": d1_schedule,
    }


def _target_encoder_state_hash(adapter: Any) -> str | None:
    model = getattr(adapter, "model", None)
    if model is None or not isinstance(model, torch.nn.Module):
        return None
    selected = []
    for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in values:
            if name.split(".", 1)[0] in {"encoder", "projector"}:
                selected.append((kind, name, value))
    if not selected:
        return None
    digest = hashlib.sha256()
    for kind, name, value in selected:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(kind.encode("ascii")); digest.update(b"\0")
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_runtime(adapter: Any, label: str) -> dict[str, Any]:
    protocol = validate_adapter_protocol(
        adapter,
        history_tokens=HISTORY_TOKENS,
        action_block_raw_steps=5,
        action_dim=2,
        minimum_future_action_blocks=1,
        task_name=f"{label} Motion Damping",
    )
    metadata = dict(adapter.metadata)
    _require(metadata.get("protocol", {}).get("native_target_encoder") is True, f"{label} target encoder is not native")
    _require(int(protocol.history_tokens) == HISTORY_TOKENS, f"{label} history length changed")
    _require(int(protocol.action_block_raw_steps) == 5 and int(protocol.action_dim) == 2, f"{label} action protocol changed")
    return {
        "adapter": metadata,
        "protocol": {
            "history_tokens": int(protocol.history_tokens),
            "action_block_raw_steps": int(protocol.action_block_raw_steps),
            "action_dim": int(protocol.action_dim),
            "future_action_blocks": int(protocol.future_action_blocks),
            "native_target_encoder": bool(protocol.native_target_encoder),
        },
        "target_encoder_sha256": _target_encoder_state_hash(adapter),
    }


def _rollout(adapter: Any, histories: np.ndarray, actions: np.ndarray, batch_size: int, label: str) -> np.ndarray:
    values = np.asarray(adapter.rollout_latents(histories, actions, batch_size=int(batch_size)))
    _require(values.ndim >= 2 and values.shape[0] == histories.shape[0], f"{label} rollout row count changed")
    if values.ndim == 3:
        _require(values.shape[1] >= 1, f"{label} rollout horizon empty")
        values = values[:, 0]
    _require(bool(np.isfinite(values).all()), f"{label} rollout non-finite")
    return np.ascontiguousarray(values, dtype=np.float32)


def _stack_mode_rows(values: np.ndarray, pair_count: int, label: str) -> np.ndarray:
    """Restore pair order after concatenating faster and no-extra rows."""

    array = np.asarray(values)
    _require(array.shape[0] == 2 * int(pair_count), f"{label} condition rows changed")
    return np.ascontiguousarray(
        np.stack([array[:pair_count], array[pair_count:]], axis=1)
    )


def evaluate_adapter(adapter: Any, arrays: Any, *, batch_size: int = DEFAULT_BATCH_SIZE, label: str = "model") -> dict[str, Any]:
    """Run correct and removed histories with one frozen native target path."""

    pair_count = int(arrays.pair_count)
    _require(pair_count == EXPECTED_DEVELOPMENT_PAIRS, "Development pair count changed")
    fp = np.asarray(arrays.faster_decay_pixels)
    npix = np.asarray(arrays.no_extra_decay_pixels)
    action = np.asarray(arrays.raw_action_blocks, dtype=np.float32)
    validate_paired_modes(
        fp, npix, action, action, arrays.pair_ids, expected_pairs=pair_count
    )
    correct_histories = np.concatenate([fp[:, :HISTORY_TOKENS], npix[:, :HISTORY_TOKENS]], axis=0)
    removed_histories = build_removed_histories(fp, npix)
    paired_actions = np.concatenate([action, action], axis=0)
    target_pixels = np.concatenate([fp[:, 3], npix[:, 3]], axis=0)
    before = adapter.frozen_state_hash()
    encoder_before = _target_encoder_state_hash(adapter)
    model = getattr(adapter, "model", None)
    if isinstance(model, torch.nn.Module):
        model.eval()
        _require(model.training is False, f"{label} model is not in eval mode")
    correct = _rollout(adapter, correct_histories, paired_actions, batch_size, label)
    removed = _rollout(adapter, removed_histories, paired_actions, batch_size, label)
    targets = np.asarray(adapter.encode_pixels(target_pixels, batch_size=int(batch_size)))
    if targets.ndim == 3:
        _require(targets.shape[1] >= 1, f"{label} target horizon empty")
        targets = targets[:, 0]
    _require(targets.shape[0] == target_pixels.shape[0], f"{label} target row count changed")
    _require(bool(np.isfinite(targets).all()), f"{label} target non-finite")
    after = adapter.frozen_state_hash()
    encoder_after = _target_encoder_state_hash(adapter)
    _require(before == after, f"{label} state hash changed during evaluation")
    _require(encoder_before == encoder_after, f"{label} target encoder changed during evaluation")
    correct_pairs = _stack_mode_rows(correct, pair_count, f"{label} correct")
    removed_pairs = _stack_mode_rows(removed, pair_count, f"{label} removed")
    target_pairs = _stack_mode_rows(targets, pair_count, f"{label} target")
    removed_gap = np.abs(removed_pairs[:, 0] - removed_pairs[:, 1])
    removed_max_gap = float(removed_gap.max()) if removed_gap.size else 0.0
    removed_scale = max(1.0, float(np.abs(removed_pairs).max()))
    removed_tolerance = 1.0e-5 * removed_scale
    _require(
        removed_max_gap <= removed_tolerance,
        f"{label} removed-history pair manipulation changed output",
    )
    return {
        "correct_predictions": correct_pairs,
        "removed_predictions": removed_pairs,
        "targets": target_pairs,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "target_encoder_sha256": encoder_after,
        "condition_count": int(correct.shape[0]),
        "correct_history_rows": int(correct.shape[0]),
        "removed_history_rows": int(removed.shape[0]),
        "target_rows": int(targets.shape[0]),
        "eval_mode": bool(model is None or model.training is False),
        "removed_history_manipulation_check": {
            "paired_query_action_identical": True,
            "prediction_exact_equal": bool(removed_max_gap == 0.0),
            "max_abs_prediction_gap": removed_max_gap,
            "absolute_tolerance": removed_tolerance,
            "passed": bool(removed_max_gap <= removed_tolerance),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    static = validate_static_identity(
        release_config=args.release_config,
        d1_report=args.d1_report,
    )
    release = load_motion_damping_icl_release(args.release_config.expanduser().resolve())
    dataset = MotionDampingICLDevelopmentDataset(release=release, repo_root=CONTEXTWORLD)
    identity = dataset.identity
    description = dataset.describe()
    _require(identity.get("passed") is True, "Development identity check failed")
    _require(identity.get("observed_data_manifest_sha256") == EXPECTED_MANIFEST_SHA256, "Development manifest changed")
    _require(identity.get("observed_lance_table_sha256") == EXPECTED_DEVELOPMENT_TABLE_SHA256, "Development table changed")
    _require(description.get("split") == "Development" and description.get("public_test_opened") is False, "Development boundary changed")
    arrays = dataset.arrays
    pool_identity = validate_paired_modes(
        arrays.faster_decay_pixels,
        arrays.no_extra_decay_pixels,
        arrays.raw_action_blocks,
        arrays.raw_action_blocks,
        arrays.pair_ids,
    )
    device = str(args.device)
    normalization = release["evaluation"]["action_normalization"]
    runtime = release["runtime"]["stable_worldmodel"]
    adapters: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    try:
        for label in ("D0", "D1", "COJA"):
            spec = MODEL_SPECS[label]
            adapter = StableWorldModelLeWMMotionDampingAdapter.from_checkpoint(
                Path(spec["checkpoint"]),
                action_mean=normalization["mean"],
                action_std=normalization["std_population"],
                repo_root=CONTEXTWORLD,
                stablewm_repo=str(ROOT),
                stablewm_ref=args.stablewm_ref,
                device=device,
            )
            runtime_identity = _validate_runtime(adapter, label)
            _require(runtime_identity["adapter"].get("checkpoint_sha256") == spec["checkpoint_sha256"], f"{label} loaded checkpoint SHA changed")
            evaluated = evaluate_adapter(adapter, arrays, batch_size=int(args.batch_size), label=label)
            summary, records = removed_history_metrics(
                evaluated["correct_predictions"],
                evaluated["removed_predictions"],
                evaluated["targets"],
                arrays.pair_ids,
            )
            reports[label] = {
                "role": MODEL_ROLES[label],
                "checkpoint": static["models"][label]["checkpoint"],
                "config": static["models"][label]["config"],
                "training_report": static["models"][label]["report"],
                "runtime": runtime_identity,
                "evaluation": {key: value for key, value in evaluated.items() if key not in {"correct_predictions", "removed_predictions", "targets"}},
                "removed_history": summary,
                "records": records,
            }
            adapters[label] = adapter
    finally:
        adapters.clear()
    target_hashes = {label: reports[label]["runtime"]["target_encoder_sha256"] for label in reports}
    _require(all(value is not None for value in target_hashes.values()), "target encoder identity unavailable")
    _require(len(set(target_hashes.values())) == 1, "target encoder identity differs across models")
    runtime_signatures = {
        (
            reports[label]["runtime"]["adapter"].get("stable_worldmodel_commit"),
            reports[label]["runtime"]["adapter"].get("model_class"),
            tuple(reports[label]["runtime"]["protocol"].items()),
        )
        for label in reports
    }
    _require(len(runtime_signatures) == 1, "model runtime/protocol differs across arms")
    scope = validate_scope_counts()
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "completed_development_removed_history_audit",
        "claim_scope": "Development_only_descriptive_not_Public_Test_or_generalization",
        "optimizer_steps": 0,
        "source": {"path": str(THIS_SOURCE), "sha256": _sha256(THIS_SOURCE)},
        "static_identity": static,
        "development_pool": {
            **pool_identity,
            "dataset": description,
            "identity": identity,
            "weighting": "equal weight over all frozen Development pairs",
        },
        "removed_history_protocol": {
            "query_frame": "terminal query RGB x_q at frame index 2",
            "history": "[x_q, x_q, x_q]",
            "actions": "original action block retained exactly",
            "paired_query_and_action_validated_before_construction": True,
            "input_support_boundary": (
                "Query-preserving temporal ablation; repeated-frame histories "
                "may be outside the native Training history distribution."
            ),
        },
        "target_encoder_identity": {
            "per_model_sha256": target_hashes,
            "matched": True,
            "runtime_stablewm_reference": runtime.get("expected_ref", EXPECTED_STABLEWM_REF),
            "runtime_signature_matched": True,
        },
        "models": reports,
        "D1_minus_D0": paired_delta_inference(
            reports["D0"]["records"],
            reports["D1"]["records"],
            seed=int(args.bootstrap_seed),
            resamples=int(args.bootstrap_resamples),
        ),
        "evidence_boundary": {
            "development_only": True,
            "public_test_opened": False,
            "cem_executed": False,
            "optimizer_steps": 0,
            "state_hash_unchanged_before_after_each_model": True,
            "scope_checks": scope,
        },
        "interpretation_boundary": {
            "removed_target_mse_increase": "removed target MSE minus correct target MSE; positive values mean history removal increased error.",
            "removed_response": "paired latent response produced after replacing all history frames by x_q.",
            "support": "Auxiliary off-support-sensitive ablation; it cannot replace the on-support correct-versus-swapped G_swap evidence.",
            "generalization": "Development-only mechanism evidence from one training seed; no Public Test or broad generalization claim.",
        },
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    _require(not target.exists(), f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


write_exclusive = _write_exclusive


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    parser.add_argument("--d1-report", type=Path, default=DEFAULT_D1_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--stablewm-ref", default="")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    _require(args.batch_size > 0 and args.bootstrap_resamples > 0, "batch size and resamples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        identity = validate_static_identity(release_config=args.release_config, d1_report=args.d1_report)
        print(json.dumps({"status": "passed_static_identity", "identity": identity}, sort_keys=True))
        return 0
    payload = run(args)
    _write_exclusive(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve()), "sha256": _sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
