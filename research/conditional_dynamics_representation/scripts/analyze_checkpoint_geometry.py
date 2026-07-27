#!/usr/bin/env python3
"""Diagnose conditional-dynamics representation contraction in LeWM/PLDM.

The script is intentionally read-only with respect to checkpoints and frozen
ContextWorld evaluation assets.  It reconstructs a training trajectory from
saved checkpoints and separates three questions:

1. Do two physically different next frames become hard to distinguish?
2. Is the change already present in the encoder, or introduced by the
   projector used by the prediction loss?
3. Does the whole representation degenerate, or only the paired futures that
   disagree because of the hidden rule?

The generated JSON is the source of truth.  CSV and Markdown are compact views
for humans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STABLE_COMMIT = "5864b74980f6ed328fd0045e777b3865962eff43"
HISTORY_CONDITIONS = (
    "observed_passable",
    "observed_blocked",
    "did_not_attempt_crossing",
)
MODULE_PREFIXES = (
    "encoder",
    "projector",
    "predictor",
    "pred_proj",
    "action_encoder",
)


@dataclass(frozen=True)
class CheckpointSpec:
    family: str
    training_mode: str
    training_method: str
    seed: int
    epoch: int
    label: str
    path: Path


@dataclass(frozen=True)
class FrozenBatch:
    query_ids: tuple[str, ...]
    histories: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            f"row_mse expects equal [N,D] arrays, got {left.shape}, "
            f"{right.shape}"
        )
    return np.mean(np.square(left - right), axis=1)


def effective_rank(values: np.ndarray) -> float:
    """Entropy effective rank of centered samples."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("effective_rank expects at least two [N,D] samples")
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(
        centered, full_matrices=False, compute_uv=False
    )
    energy = np.square(singular_values)
    total = float(energy.sum())
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    probabilities = energy[energy > 0.0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


def representation_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(
            "representation_summary expects at least two [N,D] samples"
        )
    centered = values - values.mean(axis=0, keepdims=True)
    feature_std = values.std(axis=0)
    return {
        "samples": int(values.shape[0]),
        "dimensions": int(values.shape[1]),
        "mean_feature_std": float(feature_std.mean()),
        "median_feature_std": float(np.median(feature_std)),
        "total_variance_per_dimension": float(np.square(centered).mean()),
        "effective_rank": effective_rank(values),
    }


def paired_and_unrelated_mse(
    passable: np.ndarray,
    blocked: np.ndarray,
) -> dict[str, float]:
    """Compare paired futures with targets from other physical queries."""

    passable = np.asarray(passable, dtype=np.float64)
    blocked = np.asarray(blocked, dtype=np.float64)
    if passable.shape != blocked.shape or passable.ndim != 2:
        raise ValueError("Target arrays must have equal [N,D] shapes")
    paired = row_mse(passable, blocked)
    stacked = np.concatenate([passable, blocked], axis=0)
    query_index = np.concatenate(
        [np.arange(len(passable)), np.arange(len(blocked))]
    )
    distances = np.mean(
        np.square(stacked[:, None, :] - stacked[None, :, :]), axis=-1
    )
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    unrelated = distances[
        upper & (query_index[:, None] != query_index[None, :])
    ]
    if not len(unrelated):
        raise ValueError("No unrelated target pairs are available")
    paired_mean = float(paired.mean())
    unrelated_mean = float(unrelated.mean())
    return {
        "paired_mean_mse": paired_mean,
        "paired_min_mse": float(paired.min()),
        "paired_max_mse": float(paired.max()),
        "unrelated_mean_mse": unrelated_mean,
        "paired_to_unrelated_ratio": (
            paired_mean / unrelated_mean
            if unrelated_mean > 0.0
            else float("nan")
        ),
    }


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("cosine_rows expects equal [N,D] arrays")
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    output = np.zeros_like(numerator)
    np.divide(
        numerator,
        denominator,
        out=output,
        where=denominator > np.finfo(np.float64).eps,
    )
    return output


def prediction_summary(
    *,
    predicted_passable_history: np.ndarray,
    predicted_blocked_history: np.ndarray,
    predicted_no_attempt_history: np.ndarray,
    target_passable: np.ndarray,
    target_blocked: np.ndarray,
) -> dict[str, Any]:
    pp = np.asarray(predicted_passable_history, dtype=np.float64)
    pb = np.asarray(predicted_blocked_history, dtype=np.float64)
    pn = np.asarray(predicted_no_attempt_history, dtype=np.float64)
    tp = np.asarray(target_passable, dtype=np.float64)
    tb = np.asarray(target_blocked, dtype=np.float64)
    if not (pp.shape == pb.shape == pn.shape == tp.shape == tb.shape):
        raise ValueError("Predictions and targets must have equal [N,D] shape")

    pp_to_pass = row_mse(pp, tp)
    pp_to_block = row_mse(pp, tb)
    pb_to_pass = row_mse(pb, tp)
    pb_to_block = row_mse(pb, tb)
    pass_choice = pp_to_pass < pp_to_block
    block_choice = pb_to_block < pb_to_pass
    selected_pass_history = np.where(pass_choice, "passable", "blocked")
    selected_block_history = np.where(
        block_choice, "blocked", "passable"
    )
    target_difference = tp - tb
    prediction_difference = pp - pb
    target_pair_mse = row_mse(tp, tb)
    prediction_pair_mse = row_mse(pp, pb)
    midpoint = 0.5 * (tp + tb)

    return {
        "queries": int(len(tp)),
        "passable_history_target_accuracy": float(pass_choice.mean()),
        "blocked_history_target_accuracy": float(block_choice.mean()),
        "correct_rule_switch_rate": float(
            np.logical_and(pass_choice, block_choice).mean()
        ),
        "history_changes_selected_target_rate": float(
            (selected_pass_history != selected_block_history).mean()
        ),
        "matching_target_mse": float(
            np.concatenate([pp_to_pass, pb_to_block]).mean()
        ),
        "other_target_mse": float(
            np.concatenate([pp_to_block, pb_to_pass]).mean()
        ),
        "no_attempt_to_passable_target_mse": float(row_mse(pn, tp).mean()),
        "no_attempt_to_blocked_target_mse": float(row_mse(pn, tb).mean()),
        "prediction_pair_mean_mse": float(prediction_pair_mse.mean()),
        "target_pair_mean_mse": float(target_pair_mse.mean()),
        "prediction_to_target_separation_ratio": float(
            prediction_pair_mse.mean() / target_pair_mse.mean()
        )
        if float(target_pair_mse.mean()) > 0.0
        else float("nan"),
        "difference_alignment_cosine_mean": float(
            cosine_rows(prediction_difference, target_difference).mean()
        ),
        "prediction_to_midpoint_mse": float(
            np.concatenate([row_mse(pp, midpoint), row_mse(pb, midpoint)]).mean()
        ),
    }


def _resolve_artifact_path(value: str, artifact_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "artifacts":
        return artifact_root.joinpath(*path.parts[1:]).resolve()
    return (artifact_root / path).resolve()


def load_frozen_batch(catalog_path: Path, artifact_root: Path) -> FrozenBatch:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    bundles = sorted(
        catalog["bundles"],
        key=lambda row: (int(row["eval_seed"]), int(row["evaluation_index"])),
    )
    histories: dict[str, list[np.ndarray]] = {
        condition: [] for condition in HISTORY_CONDITIONS
    }
    actions: dict[str, list[np.ndarray]] = {
        condition: [] for condition in HISTORY_CONDITIONS
    }
    targets: dict[str, list[np.ndarray]] = {
        "passable": [],
        "blocked": [],
    }
    query_ids: list[str] = []
    for bundle in bundles:
        payload_path = _resolve_artifact_path(
            str(bundle["payload"]), artifact_root
        )
        if not payload_path.is_file():
            raise FileNotFoundError(payload_path)
        if file_sha256(payload_path) != str(bundle["payload_sha256"]):
            raise ValueError(f"Frozen payload hash mismatch: {payload_path}")
        with np.load(payload_path, allow_pickle=False) as payload:
            for condition in HISTORY_CONDITIONS:
                histories[condition].append(
                    payload[f"{condition}_history_pixels"].copy()
                )
                actions[condition].append(
                    payload[f"{condition}_action_blocks"].copy()
                )
            targets["passable"].append(
                payload["target_passable_pixels"].copy()
            )
            targets["blocked"].append(
                payload["target_blocked_pixels"].copy()
            )
        query_ids.append(str(bundle["static_query_id"]))

    frozen = FrozenBatch(
        query_ids=tuple(query_ids),
        histories={
            key: np.stack(value, axis=0) for key, value in histories.items()
        },
        actions={
            key: np.stack(value, axis=0) for key, value in actions.items()
        },
        targets={
            key: np.stack(value, axis=0) for key, value in targets.items()
        },
    )
    if len(set(frozen.query_ids)) != len(frozen.query_ids):
        raise ValueError("Frozen diagnostic query IDs are not unique")
    if any(
        not np.array_equal(
            frozen.histories[HISTORY_CONDITIONS[0]][:, -1],
            frozen.histories[condition][:, -1],
        )
        for condition in HISTORY_CONDITIONS[1:]
    ):
        raise ValueError("History conditions do not end at the same query frame")
    return frozen


def _stable_adapter(
    *,
    checkpoint: Path,
    training_method: str,
    contextworld_repo: Path,
    stable_repo: Path,
    stable_ref: str,
    normalizer: Path,
    device: str,
) -> Any:
    if str(contextworld_repo) not in sys.path:
        sys.path.insert(0, str(contextworld_repo))
    from contextworld.benchmarks.adapters import (
        StableWorldModelLeWMAdapter,
        StableWorldModelPLDMAdapter,
    )

    adapter_classes = {
        "lewm": StableWorldModelLeWMAdapter,
        "pldm": StableWorldModelPLDMAdapter,
    }
    try:
        adapter_class = adapter_classes[training_method]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported training method: {training_method}"
        ) from exc
    return adapter_class.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=contextworld_repo,
        stablewm_repo=str(stable_repo),
        stablewm_ref=stable_ref,
        device=device,
    )


def encode_raw_and_projected(
    adapter: Any,
    pixels: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from contextworld.benchmarks.adapters import _preprocess_pixels

    values = np.asarray(pixels, dtype=np.uint8)
    raw_output: list[np.ndarray] = []
    projected_output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            transformed = _preprocess_pixels(
                values[start : start + batch_size],
                device=adapter.device,
            )
            raw = adapter.model.encoder(
                transformed,
                interpolate_pos_encoding=True,
            ).last_hidden_state[:, 0]
            projected = adapter.model.projector(raw)
            raw_output.append(raw.detach().float().cpu().numpy())
            projected_output.append(
                projected.detach().float().cpu().numpy()
            )
    return (
        np.concatenate(raw_output, axis=0),
        np.concatenate(projected_output, axis=0),
    )


def _image_pool(batch: FrozenBatch) -> np.ndarray:
    values = [
        batch.targets["passable"],
        batch.targets["blocked"],
    ]
    for condition in HISTORY_CONDITIONS:
        histories = batch.histories[condition]
        values.append(histories.reshape(-1, *histories.shape[2:]))
    return np.concatenate(values, axis=0)


def _target_encodings(
    adapter: Any,
    batch: FrozenBatch,
    *,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    images = np.concatenate(
        [batch.targets["passable"], batch.targets["blocked"]], axis=0
    )
    raw, projected = encode_raw_and_projected(
        adapter, images, batch_size=batch_size
    )
    count = len(batch.query_ids)
    return {
        "raw_encoder": {
            "passable": raw[:count],
            "blocked": raw[count:],
        },
        "prediction_space": {
            "passable": projected[:count],
            "blocked": projected[count:],
        },
    }


def _prediction_encodings(
    adapter: Any,
    batch: FrozenBatch,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for condition in HISTORY_CONDITIONS:
        rollout = adapter.rollout_latents(
            batch.histories[condition],
            batch.actions[condition],
            batch_size=batch_size,
        )
        if rollout.shape[1] != 1:
            raise ValueError(
                f"Expected one predicted future, got {rollout.shape}"
            )
        output[condition] = rollout[:, 0]
    return output


def _parameter_name_set(adapter: Any) -> set[str]:
    return {name for name, _ in adapter.model.named_parameters()}


def _load_state(path: Path) -> OrderedDict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, OrderedDict):
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint payload type: {type(state)}")
        state = OrderedDict(state)
    return state


def relative_parameter_drift(
    *,
    current_state: dict[str, Any],
    original_state: dict[str, Any],
    parameter_names: set[str],
) -> dict[str, float]:
    import torch

    output: dict[str, float] = {}
    for prefix in MODULE_PREFIXES:
        names = sorted(
            name
            for name in parameter_names
            if name == prefix or name.startswith(prefix + ".")
        )
        difference_energy = 0.0
        original_energy = 0.0
        for name in names:
            current = current_state[name].detach().float()
            original = original_state[name].detach().float()
            difference_energy += float(torch.square(current - original).sum())
            original_energy += float(torch.square(original).sum())
        output[prefix] = (
            math.sqrt(difference_energy / original_energy)
            if original_energy > 0.0
            else float("nan")
        )
    return output


def batch_norm_summary(
    current_state: dict[str, Any],
    original_state: dict[str, Any],
) -> dict[str, Any]:
    import torch

    output: dict[str, Any] = {}
    for prefix in ("projector", "pred_proj"):
        mean_key = f"{prefix}.net.1.running_mean"
        var_key = f"{prefix}.net.1.running_var"
        count_key = f"{prefix}.net.1.num_batches_tracked"
        current_mean = current_state[mean_key].detach().float()
        current_var = current_state[var_key].detach().float()
        original_mean = original_state[mean_key].detach().float()
        original_var = original_state[var_key].detach().float()
        output[prefix] = {
            "running_mean_l2": float(torch.linalg.vector_norm(current_mean)),
            "running_var_mean": float(current_var.mean()),
            "running_var_min": float(current_var.min()),
            "running_var_max": float(current_var.max()),
            "running_mean_change_l2": float(
                torch.linalg.vector_norm(current_mean - original_mean)
            ),
            "running_var_change_l2": float(
                torch.linalg.vector_norm(current_var - original_var)
            ),
            "batches_tracked": int(current_state[count_key]),
        }
    return output


def analyze_checkpoint(
    spec: CheckpointSpec,
    *,
    batch: FrozenBatch,
    contextworld_repo: Path,
    stable_repo: Path,
    stable_ref: str,
    normalizer: Path,
    device: str,
    batch_size: int,
    original_state: dict[str, Any],
    parameter_names: set[str],
) -> dict[str, Any]:
    adapter = _stable_adapter(
        checkpoint=spec.path,
        training_method=spec.training_method,
        contextworld_repo=contextworld_repo,
        stable_repo=stable_repo,
        stable_ref=stable_ref,
        normalizer=normalizer,
        device=device,
    )
    state_hash_before = adapter.frozen_state_hash()
    target_encodings = _target_encodings(
        adapter, batch, batch_size=batch_size
    )
    pool_raw, pool_projected = encode_raw_and_projected(
        adapter,
        _image_pool(batch),
        batch_size=batch_size,
    )
    predictions = _prediction_encodings(
        adapter, batch, batch_size=batch_size
    )
    state_hash_after = adapter.frozen_state_hash()
    if state_hash_before != state_hash_after:
        raise RuntimeError(
            f"Model state changed during read-only analysis: {spec.path}"
        )
    current_state = _load_state(spec.path)

    stage_metrics: dict[str, Any] = {}
    for stage, targets in target_encodings.items():
        stage_metrics[stage] = {
            "target_geometry": paired_and_unrelated_mse(
                targets["passable"], targets["blocked"]
            ),
        }
    stage_metrics["raw_encoder"]["global_geometry"] = representation_summary(
        pool_raw
    )
    stage_metrics["prediction_space"][
        "global_geometry"
    ] = representation_summary(pool_projected)

    prediction = prediction_summary(
        predicted_passable_history=predictions["observed_passable"],
        predicted_blocked_history=predictions["observed_blocked"],
        predicted_no_attempt_history=predictions[
            "did_not_attempt_crossing"
        ],
        target_passable=target_encodings["prediction_space"]["passable"],
        target_blocked=target_encodings["prediction_space"]["blocked"],
    )
    return {
        "family": spec.family,
        "training_mode": spec.training_mode,
        "training_method": spec.training_method,
        "seed": spec.seed,
        "epoch": spec.epoch,
        "label": spec.label,
        "checkpoint": str(spec.path),
        "checkpoint_sha256": file_sha256(spec.path),
        "model_state_hash_before": state_hash_before,
        "model_state_hash_after": state_hash_after,
        "representation": stage_metrics,
        "prediction": prediction,
        "relative_parameter_drift_from_original": relative_parameter_drift(
            current_state=current_state,
            original_state=original_state,
            parameter_names=parameter_names,
        ),
        "batch_norm": batch_norm_summary(current_state, original_state),
    }


def _parse_epochs(value: str) -> tuple[int, ...]:
    epochs = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not epochs or any(epoch <= 0 for epoch in epochs):
        raise argparse.ArgumentTypeError(
            "Epochs must be a non-empty comma-separated list of positive ints"
        )
    return epochs


def checkpoint_specs(
    artifact_root: Path,
    *,
    scope: str,
    tiny_epochs: Iterable[int],
    formal_epochs: Iterable[int],
    include_single_rule: bool,
    candidate_run_pattern: str | None = None,
    candidate_seeds: Iterable[int] = (3072,),
    candidate_training_method: str = "lewm",
    candidate_label: str = "候选联合训练",
) -> list[CheckpointSpec]:
    checkpoint_root = artifact_root / "training/runs/checkpoints"
    original = (
        checkpoint_root
        / "h3_origheldout_s3072"
        / "weights_final_step_6420.pt"
    )
    specs = [
        CheckpointSpec(
            family="original",
            training_mode="original_h3",
            training_method="lewm",
            seed=3072,
            epoch=0,
            label="原始 H3",
            path=original,
        )
    ]
    if scope in {"tiny", "all"}:
        tiny_runs = (
            (
                "tiny_joint",
                "encoder_predictor_joint_update",
                "联合训练小样本",
                "h3_passage_tiny_paired_overfit_s3072",
            ),
            (
                "tiny_fixed",
                "fixed_encoder_and_projector",
                "固定图像表示小样本",
                "h3_passage_tiny_frozen_representation_s3072",
            ),
        )
        for family, mode, label, run in tiny_runs:
            for epoch in tiny_epochs:
                specs.append(
                    CheckpointSpec(
                        family=family,
                        training_mode=mode,
                        training_method="lewm",
                        seed=3072,
                        epoch=epoch,
                        label=f"{label} epoch {epoch}",
                        path=checkpoint_root
                        / run
                        / f"weights_epoch_{epoch}.pt",
                    )
                )
    if scope in {"formal", "all"}:
        for seed in (3072, 4096, 5120):
            formal_runs: tuple[tuple[str, str, str, str, str], ...] = (
                (
                    "formal_joint",
                    "encoder_predictor_joint_update",
                    "lewm",
                    "联合训练正式数据",
                    f"h3_passage_mixed_rules_passage_formal_s{seed}",
                ),
                (
                    "formal_fixed",
                    "fixed_encoder_and_projector",
                    "lewm",
                    "固定图像表示正式数据",
                    "h3_passage_mixed_rules_fixed_representation_v2_"
                    f"passage_formal_s{seed}",
                ),
            )
            if include_single_rule:
                formal_runs += (
                    (
                        "formal_passable_only",
                        "encoder_predictor_joint_update_single_rule",
                        "lewm",
                        "只用门可通过数据",
                        f"h3_passage_passable_only_passage_formal_s{seed}",
                    ),
                    (
                        "formal_blocked_only",
                        "encoder_predictor_joint_update_single_rule",
                        "lewm",
                        "只用门不可通过数据",
                        f"h3_passage_blocked_only_passage_formal_s{seed}",
                    ),
                )
            for family, mode, method, label, run in formal_runs:
                for epoch in formal_epochs:
                    specs.append(
                        CheckpointSpec(
                            family=family,
                            training_mode=mode,
                            training_method=method,
                            seed=seed,
                            epoch=epoch,
                            label=f"{label} seed {seed} epoch {epoch}",
                            path=checkpoint_root
                            / run
                            / f"weights_epoch_{epoch}.pt",
                        )
                    )
    if scope in {"pldm", "all"}:
        for seed in (3072, 4096, 5120):
            for family, mode, label, run in (
                (
                    "pldm_joint",
                    "encoder_predictor_joint_update",
                    "PLDM 联合训练正式数据",
                    f"h3_passage_mixed_rules_pldm_objective_"
                    f"passage_formal_s{seed}",
                ),
                (
                    "pldm_fixed",
                    "fixed_encoder_and_projector",
                    "PLDM 固定图像表示正式数据",
                    "h3_passage_mixed_rules_pldm_objective_"
                    "fixed_representation_passage_formal_"
                    f"s{seed}",
                ),
            ):
                for epoch in formal_epochs:
                    specs.append(
                        CheckpointSpec(
                            family=family,
                            training_mode=mode,
                            training_method="pldm",
                            seed=seed,
                            epoch=epoch,
                            label=f"{label} seed {seed} epoch {epoch}",
                            path=checkpoint_root
                            / run
                            / f"weights_epoch_{epoch}.pt",
                        )
                    )
    if scope == "candidate":
        if not candidate_run_pattern:
            raise ValueError(
                "--candidate-run-pattern is required for candidate scope"
            )
        seeds = tuple(candidate_seeds)
        if len(seeds) > 1 and "{seed}" not in candidate_run_pattern:
            raise ValueError(
                "--candidate-run-pattern must contain {seed} when more "
                "than one candidate seed is requested"
            )
        for seed in seeds:
            try:
                run = candidate_run_pattern.format(seed=seed)
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError(
                    "Invalid --candidate-run-pattern; only the {seed} "
                    "placeholder is supported"
                ) from exc
            for epoch in formal_epochs:
                specs.append(
                    CheckpointSpec(
                        family="candidate_joint",
                        training_mode="encoder_predictor_joint_update",
                        training_method=candidate_training_method,
                        seed=seed,
                        epoch=epoch,
                        label=(
                            f"{candidate_label} seed {seed} epoch {epoch}"
                        ),
                        path=checkpoint_root
                        / run
                        / f"weights_epoch_{epoch}.pt",
                    )
                )
    missing = [spec.path for spec in specs if not spec.path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s):\n" + "\n".join(map(str, missing))
        )
    return specs


def _flat_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["representation"]["raw_encoder"]
    projected = row["representation"]["prediction_space"]
    prediction = row["prediction"]
    drift = row["relative_parameter_drift_from_original"]
    return {
        "family": row["family"],
        "training_mode": row["training_mode"],
        "training_method": row["training_method"],
        "seed": row["seed"],
        "epoch": row["epoch"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "raw_paired_target_mse": raw["target_geometry"]["paired_mean_mse"],
        "raw_paired_to_unrelated_ratio": raw["target_geometry"][
            "paired_to_unrelated_ratio"
        ],
        "raw_global_variance": raw["global_geometry"][
            "total_variance_per_dimension"
        ],
        "raw_effective_rank": raw["global_geometry"]["effective_rank"],
        "projected_paired_target_mse": projected["target_geometry"][
            "paired_mean_mse"
        ],
        "projected_paired_to_unrelated_ratio": projected[
            "target_geometry"
        ]["paired_to_unrelated_ratio"],
        "projected_global_variance": projected["global_geometry"][
            "total_variance_per_dimension"
        ],
        "projected_effective_rank": projected["global_geometry"][
            "effective_rank"
        ],
        "passable_history_accuracy": prediction[
            "passable_history_target_accuracy"
        ],
        "blocked_history_accuracy": prediction[
            "blocked_history_target_accuracy"
        ],
        "correct_rule_switch_rate": prediction["correct_rule_switch_rate"],
        "prediction_pair_mse": prediction["prediction_pair_mean_mse"],
        "difference_alignment_cosine": prediction[
            "difference_alignment_cosine_mean"
        ],
        "encoder_parameter_drift": drift["encoder"],
        "projector_parameter_drift": drift["projector"],
        "predictor_parameter_drift": drift["predictor"],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    flat = [_flat_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contextworld-repo",
        type=Path,
        required=True,
        help="Path to the ContextWorld checkout",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Defaults to <ag_data>/data/world_model/context_world",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help=(
            "Frozen hidden-passage catalog. Defaults to the eight-query "
            "training-trajectory diagnostic catalog."
        ),
    )
    parser.add_argument(
        "--stable-repo",
        type=Path,
        default=None,
        help="Defaults to the stable-worldmodel sibling of ContextWorld",
    )
    parser.add_argument("--stable-ref", default=STABLE_COMMIT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--scope",
        choices=("tiny", "formal", "pldm", "candidate", "all"),
        default="all",
    )
    parser.add_argument(
        "--tiny-epochs",
        type=_parse_epochs,
        default=(1, 2, 4, 8, 16, 32, 64),
    )
    parser.add_argument(
        "--formal-epochs",
        type=_parse_epochs,
        default=(1, 2, 3, 4),
    )
    parser.add_argument(
        "--include-single-rule",
        action="store_true",
        help=(
            "Include passable-only and blocked-only formal controls. "
            "Useful for endpoint diagnostics."
        ),
    )
    parser.add_argument(
        "--candidate-run-pattern",
        default=None,
        help=(
            "Checkpoint run directory pattern for --scope candidate. "
            "Use {seed} for a multi-seed trajectory."
        ),
    )
    parser.add_argument(
        "--candidate-seeds",
        type=_parse_epochs,
        default=(3072,),
        help="Comma-separated training seeds for --scope candidate.",
    )
    parser.add_argument(
        "--candidate-training-method",
        choices=("lewm", "pldm"),
        default="lewm",
    )
    parser.add_argument(
        "--candidate-label",
        default="候选联合训练",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contextworld_repo = args.contextworld_repo.expanduser().resolve()
    if args.artifact_root is None:
        artifact_root = (
            contextworld_repo.parents[1]
            / "data/world_model/context_world"
        ).resolve()
    else:
        artifact_root = args.artifact_root.expanduser().resolve()
    stable_repo = (
        args.stable_repo.expanduser().resolve()
        if args.stable_repo is not None
        else (contextworld_repo.parent / "stable-worldmodel").resolve()
    )
    catalog = (
        args.catalog.expanduser().resolve()
        if args.catalog is not None
        else artifact_root
        / "evaluation/history3/"
        "hidden_passage_frozen_representation_diagnostic_v1/catalog.json"
    )
    normalizer = (
        artifact_root
        / "splits/tworoom_original_train_s3072_normalizer.json"
    )
    for path in (contextworld_repo, stable_repo, catalog, normalizer):
        if not path.exists():
            raise FileNotFoundError(path)

    batch = load_frozen_batch(catalog, artifact_root)
    specs = checkpoint_specs(
        artifact_root,
        scope=args.scope,
        tiny_epochs=args.tiny_epochs,
        formal_epochs=args.formal_epochs,
        include_single_rule=args.include_single_rule,
        candidate_run_pattern=args.candidate_run_pattern,
        candidate_seeds=args.candidate_seeds,
        candidate_training_method=args.candidate_training_method,
        candidate_label=args.candidate_label,
    )

    original_adapter = _stable_adapter(
        checkpoint=specs[0].path,
        training_method=specs[0].training_method,
        contextworld_repo=contextworld_repo,
        stable_repo=stable_repo,
        stable_ref=args.stable_ref,
        normalizer=normalizer,
        device=args.device,
    )
    parameter_names = _parameter_name_set(original_adapter)
    original_state = _load_state(specs[0].path)
    del original_adapter

    results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] {spec.label}: {spec.path}",
            flush=True,
        )
        results.append(
            analyze_checkpoint(
                spec,
                batch=batch,
                contextworld_repo=contextworld_repo,
                stable_repo=stable_repo,
                stable_ref=args.stable_ref,
                normalizer=normalizer,
                device=args.device,
                batch_size=args.batch_size,
                original_state=original_state,
                parameter_names=parameter_names,
            )
        )

    payload = {
        "schema_version": 1,
        "status": "retrospective_read_only_checkpoint_diagnostic",
        "scope": args.scope,
        "question": (
            "联合训练失败是否伴随可复现的条件动力学表示收缩，且该收缩"
            "是否只是局部现象而非整个表示空间退化？"
        ),
        "provenance": {
            "contextworld_repo": str(contextworld_repo),
            "stable_worldmodel_repo": str(stable_repo),
            "stable_worldmodel_commit": args.stable_ref,
            "artifact_root": str(artifact_root),
            "catalog": str(catalog),
            "catalog_sha256": file_sha256(catalog),
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
            "device": args.device,
            "environment_calls": 0,
            "checkpoints_modified": False,
        },
        "frozen_batch": {
            "queries": len(batch.query_ids),
            "history_conditions": list(HISTORY_CONDITIONS),
            "true_next_frames_per_query": 2,
        },
        "checkpoints": results,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "checkpoint_geometry.json"
    csv_path = output_dir / "checkpoint_geometry.csv"
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(results, csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
