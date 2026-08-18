#!/usr/bin/env python3
"""Measure which LeWM loss gradients contract the hidden-rule separation.

This is a read-only, single-batch autograd diagnostic.  The measured loss can
come from either the balanced frozen queries or the exact first batch of the
tiny LeWM training protocol.  In both cases it asks what an infinitesimal
gradient-descent step would do to a separate, frozen paired distance between
the passable and blocked futures.

The prediction MSE is viewed through two gradient-only partitions:

* ``pred_context_branch`` detaches the target embedding, so gradients flow
  through context Encoder/Projector, ActionEncoder, Predictor and PredProj.
* ``pred_target_branch`` detaches the prediction, so gradients flow only
  through the online target Encoder/Projector.

These two entries have the same scalar value and are not additive scalar loss
terms.  Their parameter gradients add to the native undetached prediction
loss gradient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import analyze_checkpoint_geometry as geometry


MODULE_GROUPS = (
    "encoder",
    "projector",
    "predictor",
    "pred_proj",
    "action_encoder",
)
LEWM_SIGREG_WEIGHT = 0.09
TAIL_SIGREG_FRACTION = 0.10
TUNED_SIGREG_REFERENCE_WEIGHT = 0.90
LEWM_VISREG_REFERENCE_WEIGHT = 0.09
VISREG_KWARGS = {
    "num_projections": 1024,
    "lambda_scale": 1.0,
    "lambda_shape": 1.0,
    "lambda_center": 1.0,
}
PLDM_WEIGHTS = {
    "std_loss": 18.0,
    "std_t_loss": 0.7,
    "cov_loss": 12.0,
    "temp_align_loss": 0.2,
}


def parameter_group(name: str) -> str:
    for group in MODULE_GROUPS:
        if name == group or name.startswith(group + "."):
            return group
    return "other"


def tensor_collection_sha256(values: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_balanced_rule_batch(
    adapter: Any,
    frozen: geometry.FrozenBatch,
) -> dict[str, Any]:
    """Create [passable histories, blocked histories] four-frame clips."""

    from contextworld.benchmarks.adapters import _preprocess_pixels

    passable_pixels = np.concatenate(
        [
            frozen.histories["observed_passable"],
            frozen.targets["passable"][:, None],
        ],
        axis=1,
    )
    blocked_pixels = np.concatenate(
        [
            frozen.histories["observed_blocked"],
            frozen.targets["blocked"][:, None],
        ],
        axis=1,
    )
    pixels = np.concatenate([passable_pixels, blocked_pixels], axis=0)
    passable_actions = adapter._normalize_actions(
        frozen.actions["observed_passable"]
    )
    blocked_actions = adapter._normalize_actions(
        frozen.actions["observed_blocked"]
    )
    actions = np.concatenate([passable_actions, blocked_actions], axis=0)
    batch, frames = pixels.shape[:2]
    transformed = _preprocess_pixels(
        pixels.reshape(-1, *pixels.shape[2:]),
        device=adapter.device,
    ).reshape(batch, frames, 3, pixels.shape[2], pixels.shape[3])

    import torch

    action_tensor = torch.from_numpy(actions).to(
        device=adapter.device,
        dtype=next(adapter.model.parameters()).dtype,
    )
    return {
        "pixels": transformed,
        "actions": action_tensor,
        "query_count": len(frozen.query_ids),
        "pixels_sha256": tensor_collection_sha256([pixels]),
        "actions_sha256": tensor_collection_sha256([actions]),
    }


def build_exact_tiny_training_batch(
    *,
    stable_worldmodel: Any,
    contextworld_repo: Path,
    stable_commit: str,
    device: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the tiny protocol's first shuffled training batch."""

    import torch

    from contextworld.training.tworoom_data import (
        build_tworoom_grouped_data,
    )

    benchmark_config = (
        contextworld_repo
        / "configs/benchmark/"
        "tworoom_hidden_passage_h3_tiny_overfit_training_v1.yaml"
    )
    grouped = build_tworoom_grouped_data(
        stable_worldmodel,
        repo_root=contextworld_repo,
        benchmark_config=benchmark_config,
        model_id="H3_Passage_TinyPairedOverfit",
        epoch_size=256,
        validation_epoch_size=160,
        frameskip=5,
        num_steps=4,
        img_size=224,
        seed=seed,
        expected_stablewm_commit=stable_commit,
    )

    class IndexedDataset:
        def __init__(self, dataset: Any) -> None:
            self.dataset = dataset

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = dict(self.dataset[index])
            sample["__contextworld_index__"] = int(index)
            return sample

    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        IndexedDataset(grouped.train),
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    raw = next(iter(loader))
    indices = [int(value) for value in raw.pop("__contextworld_index__")]
    pixels = raw["pixels"].to(device=device)
    actions = torch.nan_to_num(raw["action"], 0.0).to(device=device)
    loss_batch = {
        "pixels": pixels,
        "actions": actions,
    }
    metadata = {
        "source": "exact_tiny_training_protocol_first_shuffled_batch",
        "benchmark_config": str(benchmark_config),
        "benchmark_config_sha256": geometry.file_sha256(benchmark_config),
        "model_id": "H3_Passage_TinyPairedOverfit",
        "seed": seed,
        "epoch_size": 256,
        "batch_size": 16,
        "indices": indices,
        "pixels_sha256": tensor_collection_sha256(
            [pixels.detach().cpu().numpy()]
        ),
        "actions_sha256": tensor_collection_sha256(
            [actions.detach().cpu().numpy()]
        ),
        "model_visible_keys": ["pixels", "actions"],
        "data_audit_passed": bool(
            grouped.metadata["training_exclusion_audit"]["passed"]
            and grouped.metadata["paired_collection_audit"]["passed"]
        ),
    }
    return loss_batch, metadata


def gradient_energy_by_group(
    names: list[str],
    gradients: tuple[Any | None, ...],
) -> dict[str, float]:
    import torch

    energy = {group: 0.0 for group in (*MODULE_GROUPS, "other")}
    for name, gradient in zip(names, gradients):
        if gradient is None:
            continue
        value = gradient.detach().float()
        energy[parameter_group(name)] += float(torch.square(value).sum())
    return {
        group: math.sqrt(value)
        for group, value in energy.items()
    }


def descent_direction_summary(
    names: list[str],
    distance_gradients: tuple[Any | None, ...],
    loss_gradients: tuple[Any | None, ...],
) -> dict[str, dict[str, float | str | None]]:
    """Summarize d(distance)/d(lr) under theta <- theta - lr*grad(loss)."""

    import torch

    accumulators = {
        group: {"dot": 0.0, "distance_energy": 0.0, "loss_energy": 0.0}
        for group in (*MODULE_GROUPS, "other", "all")
    }
    for name, distance_gradient, loss_gradient in zip(
        names, distance_gradients, loss_gradients
    ):
        if distance_gradient is None or loss_gradient is None:
            continue
        distance_value = distance_gradient.detach().float()
        loss_value = loss_gradient.detach().float()
        dot = float(torch.sum(distance_value * loss_value))
        distance_energy = float(torch.square(distance_value).sum())
        loss_energy = float(torch.square(loss_value).sum())
        group = parameter_group(name)
        for key in (group, "all"):
            accumulators[key]["dot"] += dot
            accumulators[key]["distance_energy"] += distance_energy
            accumulators[key]["loss_energy"] += loss_energy

    output: dict[str, dict[str, float | str | None]] = {}
    for group, values in accumulators.items():
        dot = values["dot"]
        denominator = math.sqrt(
            values["distance_energy"] * values["loss_energy"]
        )
        cosine = dot / denominator if denominator > 0.0 else None
        predicted_change = -dot
        if denominator == 0.0:
            effect = "no_shared_gradient"
        elif predicted_change < 0.0:
            effect = "contracts_pair_distance"
        elif predicted_change > 0.0:
            effect = "expands_pair_distance"
        else:
            effect = "neutral"
        output[group] = {
            "distance_gradient_norm": math.sqrt(
                values["distance_energy"]
            ),
            "loss_gradient_norm_on_shared_parameters": math.sqrt(
                values["loss_energy"]
            ),
            "gradient_dot": dot,
            "descent_cosine": -cosine if cosine is not None else None,
            "predicted_distance_change_per_unit_lr": predicted_change,
            "effect": effect,
        }
    return output


def paired_distance(values: Any, query_count: int) -> Any:
    import torch

    passable = values[:query_count]
    blocked = values[query_count:]
    return torch.mean(torch.square(passable - blocked))


def representation_distances(
    model: Any,
    distance_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure paired target distances on a graph separate from the loss."""

    from einops import rearrange

    pixels = distance_batch["pixels"].to(
        dtype=next(model.encoder.parameters()).dtype
    )
    actions = distance_batch["actions"].to(
        dtype=next(model.action_encoder.parameters()).dtype
    )
    batch_size, frames = pixels.shape[:2]
    flat_pixels = rearrange(pixels, "b t ... -> (b t) ...")
    raw = model.encoder(
        flat_pixels, interpolate_pos_encoding=True
    ).last_hidden_state[:, 0]
    projected = model.projector(raw)
    raw = rearrange(raw, "(b t) d -> b t d", b=batch_size)
    projected = rearrange(
        projected, "(b t) d -> b t d", b=batch_size
    )
    action_embeddings = model.action_encoder(actions)
    prediction = model.predict(
        projected[:, : frames - 1],
        action_embeddings[:, : frames - 1],
    )
    query_count = int(distance_batch["query_count"])
    return {
        "raw_encoder_pair_mse": paired_distance(
            raw[:, -1], query_count
        ),
        "prediction_space_pair_mse": paired_distance(
            projected[:, -1], query_count
        ),
        "causal_prediction_pair_mse": paired_distance(
            prediction[:, -1], query_count
        ),
    }


def native_losses(
    model: Any,
    loss_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the native LeWM loss plus gradient-only branch views."""

    import torch
    import torch.nn.functional as functional
    from einops import rearrange

    from stable_worldmodel.wm.loss import (
        JointTemporalCovarianceSIGReg,
        PLDMLoss,
        SIGReg,
        VISRegLoss,
    )

    pixels = loss_batch["pixels"].to(
        dtype=next(model.encoder.parameters()).dtype
    )
    actions = loss_batch["actions"].to(
        dtype=next(model.action_encoder.parameters()).dtype
    )
    batch_size, frames = pixels.shape[:2]
    flat_pixels = rearrange(pixels, "b t ... -> (b t) ...")
    raw = model.encoder(
        flat_pixels, interpolate_pos_encoding=True
    ).last_hidden_state[:, 0]
    projected = model.projector(raw)
    embeddings = rearrange(
        projected, "(b t) d -> b t d", b=batch_size
    )
    action_embeddings = model.action_encoder(actions)

    context = embeddings[:, :3]
    context_actions = action_embeddings[:, :3]
    targets = embeddings[:, 1:]
    predictions = model.predict(context, context_actions)
    if predictions.shape != targets.shape:
        raise RuntimeError(
            f"Unexpected prediction/target shapes: "
            f"{predictions.shape} vs {targets.shape}"
        )

    prediction_loss = functional.mse_loss(predictions, targets)
    prediction_context_branch = functional.mse_loss(
        predictions, targets.detach()
    )
    prediction_target_branch = functional.mse_loss(
        predictions.detach(), targets
    )
    sigreg = SIGReg(knots=17, num_proj=1024).to(
        device=embeddings.device
    )
    tail_sigreg = SIGReg(
        knots=17,
        num_proj=1024,
        tail_fraction=TAIL_SIGREG_FRACTION,
    ).to(device=embeddings.device)
    projections = torch.randn(
        embeddings.size(-1),
        sigreg.num_proj,
        device=embeddings.device,
    )
    projections = projections.div_(projections.norm(p=2, dim=0))
    projection_statistics = sigreg._projected_statistic(
        embeddings.transpose(0, 1),
        projections,
    )
    temporally_centered_embeddings = (
        embeddings - embeddings.mean(dim=1, keepdim=True)
    )
    temporally_centered_statistics = sigreg._projected_statistic(
        temporally_centered_embeddings.transpose(0, 1),
        projections,
    )
    projected_values = embeddings.transpose(0, 1) @ projections
    # Mean and tail variants deliberately share the exact same projections.
    sigreg_loss = sigreg._aggregate_projected_statistic(
        projection_statistics
    )
    temporally_centered_sigreg_loss = (
        sigreg._aggregate_projected_statistic(
            temporally_centered_statistics
        )
    )
    jt_cov = JointTemporalCovarianceSIGReg(
        knots=17,
        num_proj=1024,
        rho=None,
    ).to(device=embeddings.device)
    predictive_trajectory = torch.cat(
        [embeddings[:, :1], predictions],
        dim=1,
    )
    target_modes = jt_cov.normalized_temporal_modes(
        embeddings.transpose(0, 1)
    )
    predictive_modes = jt_cov.normalized_temporal_modes(
        predictive_trajectory.transpose(0, 1)
    )
    target_joint = target_modes.permute(1, 0, 2).reshape(
        1, batch_size, -1
    )
    predictive_joint = predictive_modes.permute(1, 0, 2).reshape(
        1, batch_size, -1
    )
    joint_projections = torch.randn(
        target_joint.size(-1),
        jt_cov.num_proj,
        device=embeddings.device,
    )
    joint_projections = joint_projections.div_(
        joint_projections.norm(p=2, dim=0)
    )
    target_jt_cov_loss = jt_cov._aggregate_projected_statistic(
        jt_cov._projected_statistic(target_joint, joint_projections)
    )
    predictive_jt_cov_loss = jt_cov._aggregate_projected_statistic(
        jt_cov._projected_statistic(
            predictive_joint,
            joint_projections,
        )
    )
    tail_sigreg_loss = tail_sigreg._aggregate_projected_statistic(
        projection_statistics
    )
    tail_count = math.ceil(
        TAIL_SIGREG_FRACTION * projection_statistics.size(-1)
    )
    projected_std = projected_values.float().std(dim=-2)
    low_variance_indices = torch.topk(
        projected_std,
        k=tail_count,
        dim=-1,
        largest=False,
        sorted=False,
    ).indices
    high_variance_indices = torch.topk(
        projected_std,
        k=tail_count,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    low_variance_tail_sigreg_loss = torch.gather(
        projection_statistics,
        dim=-1,
        index=low_variance_indices,
    ).mean()
    high_variance_tail_sigreg_loss = torch.gather(
        projection_statistics,
        dim=-1,
        index=high_variance_indices,
    ).mean()
    characteristic_arguments = (
        projected_values.unsqueeze(-1) * sigreg.t
    )
    cosine_residual = (
        characteristic_arguments.cos().mean(dim=-3) - sigreg.phi
    )
    sine_mean = characteristic_arguments.sin().mean(dim=-3)
    lower_cf_statistic = (
        torch.relu(cosine_residual).square() @ sigreg.weights
    ) * embeddings.size(0)
    lower_cf_with_sine_statistic = (
        (
            torch.relu(cosine_residual).square()
            + sine_mean.square()
        )
        @ sigreg.weights
    ) * embeddings.size(0)
    lower_cf_sigreg_loss = lower_cf_statistic.mean()
    lower_cf_with_sine_sigreg_loss = (
        lower_cf_with_sine_statistic.mean()
    )
    asymmetric_cf_sigreg_0p25_loss = (
        lower_cf_with_sine_sigreg_loss
        + 0.25 * (sigreg_loss - lower_cf_with_sine_sigreg_loss)
    )
    asymmetric_cf_sigreg_0p50_loss = (
        lower_cf_with_sine_sigreg_loss
        + 0.50 * (sigreg_loss - lower_cf_with_sine_sigreg_loss)
    )
    visreg_loss = VISRegLoss(**VISREG_KWARGS).to(
        device=embeddings.device
    )(embeddings.transpose(0, 1))
    pldm = PLDMLoss().to(device=embeddings.device)
    pldm_terms = pldm(embeddings)

    losses: dict[str, Any] = {
        "pred_loss": prediction_loss,
        "pred_context_branch": prediction_context_branch,
        "pred_target_branch": prediction_target_branch,
        "sigreg_loss": sigreg_loss,
        "weighted_sigreg_loss": LEWM_SIGREG_WEIGHT * sigreg_loss,
        "lewm_total_loss": (
            prediction_loss + LEWM_SIGREG_WEIGHT * sigreg_loss
        ),
        "temporally_centered_sigreg_loss": (
            temporally_centered_sigreg_loss
        ),
        "weighted_temporally_centered_sigreg_0p09_loss": (
            LEWM_SIGREG_WEIGHT * temporally_centered_sigreg_loss
        ),
        "lewm_temporally_centered_sigreg_0p09_total_loss": (
            prediction_loss
            + LEWM_SIGREG_WEIGHT * temporally_centered_sigreg_loss
        ),
        "target_jt_cov_sigreg_loss": target_jt_cov_loss,
        "weighted_target_jt_cov_sigreg_0p09_loss": (
            LEWM_SIGREG_WEIGHT * target_jt_cov_loss
        ),
        "lewm_target_jt_cov_sigreg_0p09_total_loss": (
            prediction_loss + LEWM_SIGREG_WEIGHT * target_jt_cov_loss
        ),
        "causal_predictive_jt_cov_sigreg_loss": predictive_jt_cov_loss,
        "weighted_causal_predictive_jt_cov_sigreg_0p09_loss": (
            LEWM_SIGREG_WEIGHT * predictive_jt_cov_loss
        ),
        "lewm_causal_predictive_jt_cov_sigreg_0p09_total_loss": (
            prediction_loss
            + LEWM_SIGREG_WEIGHT * predictive_jt_cov_loss
        ),
        "tail_sigreg_loss": tail_sigreg_loss,
        "weighted_tail_sigreg_0p09_loss": (
            LEWM_SIGREG_WEIGHT * tail_sigreg_loss
        ),
        "lewm_tail_sigreg_0p09_total_loss": (
            prediction_loss + LEWM_SIGREG_WEIGHT * tail_sigreg_loss
        ),
        "low_variance_tail_sigreg_loss": (
            low_variance_tail_sigreg_loss
        ),
        "high_variance_tail_sigreg_loss": (
            high_variance_tail_sigreg_loss
        ),
        "lower_cf_sigreg_loss": lower_cf_sigreg_loss,
        "lower_cf_with_sine_sigreg_loss": (
            lower_cf_with_sine_sigreg_loss
        ),
        "asymmetric_cf_sigreg_0p25_loss": (
            asymmetric_cf_sigreg_0p25_loss
        ),
        "asymmetric_cf_sigreg_0p50_loss": (
            asymmetric_cf_sigreg_0p50_loss
        ),
        "visreg_loss": visreg_loss,
        "weighted_visreg_loss": (
            LEWM_VISREG_REFERENCE_WEIGHT * visreg_loss
        ),
        "lewm_visreg_total_loss": (
            prediction_loss
            + LEWM_VISREG_REFERENCE_WEIGHT * visreg_loss
        ),
    }
    weighted_pldm_terms = []
    for name, weight in PLDM_WEIGHTS.items():
        weighted = weight * pldm_terms[name]
        losses[f"weighted_pldm_{name}"] = weighted
        weighted_pldm_terms.append(weighted)
    pldm_regularizer = torch.stack(weighted_pldm_terms).sum()
    losses["pldm_regularizer_total"] = pldm_regularizer
    losses["pldm_total_without_idm"] = prediction_loss + pldm_regularizer

    return losses


def autograd_diagnostic(
    model: Any,
    distance_batch: Mapping[str, Any],
    loss_batch: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    distances = representation_distances(model, distance_batch)
    distance_gradients: dict[str, tuple[Any | None, ...]] = {}
    distance_items = list(distances.items())
    for index, (name, value) in enumerate(distance_items):
        distance_gradients[name] = torch.autograd.grad(
            value,
            parameters,
            retain_graph=index < len(distance_items) - 1,
            allow_unused=True,
        )

    losses = native_losses(model, loss_batch)
    loss_rows: dict[str, Any] = {}
    loss_items = list(losses.items())
    for index, (name, value) in enumerate(loss_items):
        gradients = torch.autograd.grad(
            value,
            parameters,
            retain_graph=index < len(loss_items) - 1,
            allow_unused=True,
        )
        loss_rows[name] = {
            "value": float(value.detach()),
            "gradient_norms": gradient_energy_by_group(names, gradients),
            "distance_descent_direction": {
                distance_name: descent_direction_summary(
                    names, distance_gradient, gradients
                )
                for distance_name, distance_gradient in (
                    distance_gradients.items()
                )
            },
        }

    native_sigreg_row = loss_rows["sigreg_loss"]
    tail_sigreg_row = loss_rows["tail_sigreg_loss"]
    native_encoder_norm = native_sigreg_row["gradient_norms"]["encoder"]
    tail_encoder_norm = tail_sigreg_row["gradient_norms"]["encoder"]
    if tail_encoder_norm <= 0:
        raise RuntimeError("Tail-SIGReg has zero Encoder gradient norm")
    matched_tail_weight = (
        TUNED_SIGREG_REFERENCE_WEIGHT
        * native_encoder_norm
        / tail_encoder_norm
    )
    budget_comparison = {
        "matching_rule": (
            "match the Tail-SIGReg Encoder gradient norm to the "
            "0.90-weight native SIGReg Encoder gradient norm"
        ),
        "native_reference_weight": TUNED_SIGREG_REFERENCE_WEIGHT,
        "tail_fraction": TAIL_SIGREG_FRACTION,
        "matched_tail_weight": matched_tail_weight,
        "unweighted_encoder_gradient_norms": {
            "native_sigreg": native_encoder_norm,
            "tail_sigreg": tail_encoder_norm,
        },
        "distances": {},
    }
    for distance_name in distances:
        native_effect = native_sigreg_row["distance_descent_direction"][
            distance_name
        ]["all"]["predicted_distance_change_per_unit_lr"]
        tail_effect = tail_sigreg_row["distance_descent_direction"][
            distance_name
        ]["all"]["predicted_distance_change_per_unit_lr"]
        prediction_effect = loss_rows["pred_loss"][
            "distance_descent_direction"
        ][distance_name]["all"]["predicted_distance_change_per_unit_lr"]
        budget_comparison["distances"][distance_name] = {
            "native_sigreg_0p90_regularizer_effect": (
                TUNED_SIGREG_REFERENCE_WEIGHT * native_effect
            ),
            "matched_tail_sigreg_regularizer_effect": (
                matched_tail_weight * tail_effect
            ),
            "native_sigreg_effect_per_encoder_gradient_norm": (
                native_effect / native_encoder_norm
            ),
            "tail_sigreg_effect_per_encoder_gradient_norm": (
                tail_effect / tail_encoder_norm
            ),
            "prediction_plus_native_sigreg_0p90_effect": (
                prediction_effect
                + TUNED_SIGREG_REFERENCE_WEIGHT * native_effect
            ),
            "prediction_plus_matched_tail_sigreg_effect": (
                prediction_effect + matched_tail_weight * tail_effect
            ),
            "prediction_plus_tail_sigreg_0p09_effect": (
                prediction_effect + LEWM_SIGREG_WEIGHT * tail_effect
            ),
        }

    candidate_rows = {
        "target_jt_cov_sigreg": loss_rows[
            "target_jt_cov_sigreg_loss"
        ],
        "causal_predictive_jt_cov_sigreg": loss_rows[
            "causal_predictive_jt_cov_sigreg_loss"
        ],
        "temporally_centered_sigreg": loss_rows[
            "temporally_centered_sigreg_loss"
        ],
        "tail_by_statistic": tail_sigreg_row,
        "tail_by_low_variance": loss_rows[
            "low_variance_tail_sigreg_loss"
        ],
        "tail_by_high_variance": loss_rows[
            "high_variance_tail_sigreg_loss"
        ],
        "one_sided_lower_cf": loss_rows["lower_cf_sigreg_loss"],
        "one_sided_lower_cf_with_sine": loss_rows[
            "lower_cf_with_sine_sigreg_loss"
        ],
        "asymmetric_cf_overdispersion_0p25": loss_rows[
            "asymmetric_cf_sigreg_0p25_loss"
        ],
        "asymmetric_cf_overdispersion_0p50": loss_rows[
            "asymmetric_cf_sigreg_0p50_loss"
        ],
    }
    reduction_comparison = {
        "matching_rule": (
            "for each candidate, match its Encoder gradient norm to the "
            "0.90-weight native mean-SIGReg Encoder gradient norm"
        ),
        "tail_fraction": TAIL_SIGREG_FRACTION,
        "native_reference_weight": TUNED_SIGREG_REFERENCE_WEIGHT,
        "candidates": {},
    }
    for candidate_name, candidate_row in candidate_rows.items():
        candidate_norm = candidate_row["gradient_norms"]["encoder"]
        if candidate_norm <= 0:
            raise RuntimeError(
                f"{candidate_name} has zero Encoder gradient norm"
            )
        candidate_weight = (
            TUNED_SIGREG_REFERENCE_WEIGHT
            * native_encoder_norm
            / candidate_norm
        )
        candidate_summary = {
            "unweighted_loss": candidate_row["value"],
            "unweighted_encoder_gradient_norm": candidate_norm,
            "matched_weight": candidate_weight,
            "distances": {},
        }
        for distance_name in distances:
            candidate_effect = candidate_row[
                "distance_descent_direction"
            ][distance_name]["all"][
                "predicted_distance_change_per_unit_lr"
            ]
            prediction_effect = loss_rows["pred_loss"][
                "distance_descent_direction"
            ][distance_name]["all"][
                "predicted_distance_change_per_unit_lr"
            ]
            candidate_summary["distances"][distance_name] = {
                "regularizer_effect_per_encoder_gradient_norm": (
                    candidate_effect / candidate_norm
                ),
                "matched_regularizer_effect": (
                    candidate_weight * candidate_effect
                ),
                "prediction_plus_matched_regularizer_effect": (
                    prediction_effect
                    + candidate_weight * candidate_effect
                ),
            }
        reduction_comparison["candidates"][candidate_name] = (
            candidate_summary
        )

    predictive_jt_cov_row = loss_rows[
        "causal_predictive_jt_cov_sigreg_loss"
    ]
    predictive_contract_checks = {
        "predictor_gradient_nonzero": (
            predictive_jt_cov_row["gradient_norms"]["predictor"] > 0.0
        ),
        "prediction_projector_gradient_nonzero": (
            predictive_jt_cov_row["gradient_norms"]["pred_proj"] > 0.0
        ),
        "context_encoder_gradient_nonzero": (
            predictive_jt_cov_row["gradient_norms"]["encoder"] > 0.0
        ),
        "action_encoder_gradient_nonzero": (
            predictive_jt_cov_row["gradient_norms"]["action_encoder"] > 0.0
        ),
    }
    return {
        "distance_values": {
            name: float(value.detach())
            for name, value in distances.items()
        },
        "losses": loss_rows,
        "tail_sigreg_gradient_budget_comparison": budget_comparison,
        "sigreg_reduction_gradient_budget_comparison": (
            reduction_comparison
        ),
        "causal_predictive_jt_cov_contract": {
            "checks": predictive_contract_checks,
            "passed": all(predictive_contract_checks.values()),
            "target_future_direct_regularizer_edge": False,
            "target_future_no_direct_edge_verified_by_unit_test": True,
        },
        "gradient_partition_contract": {
            "pred_context_branch": (
                "target embedding detached; gradients follow the prediction "
                "and context path"
            ),
            "pred_target_branch": (
                "prediction detached; gradients follow only the online "
                "target representation path"
            ),
            "note": (
                "The branch entries are gradient views of the same scalar "
                "MSE, not two scalar terms to add."
            ),
        },
    }


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "--checkpoint must be LABEL=/path/to/weights.pt"
        )
    return label.strip(), Path(raw_path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextworld-repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--stable-repo", type=Path, default=None)
    parser.add_argument("--stable-ref", default=geometry.STABLE_COMMIT)
    parser.add_argument(
        "--data-stable-ref",
        default=geometry.STABLE_COMMIT,
        help=(
            "StableWorldModel commit recorded by the immutable synthesized "
            "training data. This may differ from the runtime checkout used "
            "to evaluate a new loss implementation."
        ),
    )
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--normalizer", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_labeled_path,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--autograd-seed",
        type=int,
        default=None,
        help=(
            "RNG seed for model train-mode stochasticity and random "
            "regularizer projections. Defaults to --seed; keeping --seed "
            "fixed while varying this value reuses the exact data batch."
        ),
    )
    parser.add_argument(
        "--loss-batch-source",
        choices=("frozen_diagnostic", "tiny_training_first_batch"),
        default="frozen_diagnostic",
        help=(
            "Use the frozen paired clips for both loss and distance, or "
            "reconstruct the exact first batch of the tiny training run "
            "for the loss while retaining the frozen paired distance."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    autograd_seed = (
        args.seed if args.autograd_seed is None else args.autograd_seed
    )

    contextworld_repo = args.contextworld_repo.expanduser().resolve()
    artifact_root = (
        args.artifact_root.expanduser().resolve()
        if args.artifact_root is not None
        else (
            contextworld_repo.parents[1]
            / "data/world_model/context_world"
        ).resolve()
    )
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
        args.normalizer.expanduser().resolve()
        if args.normalizer is not None
        else artifact_root
        / "splits/tworoom_original_train_s3072_normalizer.json"
    )
    checkpoints = [(label, path) for label, path in args.checkpoint]
    required = [
        contextworld_repo,
        stable_repo,
        catalog,
        normalizer,
        *(path for _, path in checkpoints),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )

    import torch

    frozen = geometry.load_frozen_batch(catalog, artifact_root)
    adapter = geometry._stable_adapter(
        checkpoint=checkpoints[0][1],
        training_method="lewm",
        contextworld_repo=contextworld_repo,
        stable_repo=stable_repo,
        stable_ref=args.stable_ref,
        normalizer=normalizer,
        device=args.device,
    )
    balanced_batch = build_balanced_rule_batch(adapter, frozen)
    if args.loss_batch_source == "frozen_diagnostic":
        loss_batch = balanced_batch
        loss_batch_metadata = {
            "source": "frozen_diagnostic_balanced_rule_batch",
            "clips": 2 * len(frozen.query_ids),
            "pixels_sha256": balanced_batch["pixels_sha256"],
            "actions_sha256": balanced_batch["actions_sha256"],
        }
    else:
        import stable_worldmodel as stable_worldmodel_module

        loss_batch, loss_batch_metadata = build_exact_tiny_training_batch(
            stable_worldmodel=stable_worldmodel_module,
            contextworld_repo=contextworld_repo,
            stable_commit=args.data_stable_ref,
            device=args.device,
            seed=args.seed,
        )
    model = adapter.model
    model.requires_grad_(True)
    results: list[dict[str, Any]] = []
    for index, (label, path) in enumerate(checkpoints, start=1):
        print(
            f"[{index}/{len(checkpoints)}] gradient mechanism for "
            f"{label}: {path}",
            flush=True,
        )
        state = geometry._load_state(path)
        model.load_state_dict(state, strict=True)
        model.train()
        torch.manual_seed(autograd_seed)
        if str(args.device).startswith("cuda"):
            torch.cuda.manual_seed_all(autograd_seed)
        row = autograd_diagnostic(
            model,
            distance_batch=balanced_batch,
            loss_batch=loss_batch,
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        row.update(
            {
                "label": label,
                "checkpoint": str(path),
                "checkpoint_sha256": geometry.file_sha256(path),
                "model_state_hash_after_restore": adapter.frozen_state_hash(),
            }
        )
        results.append(row)

    payload = {
        "schema_version": 1,
        "status": "retrospective_read_only_single_batch_autograd_diagnostic",
        "question": (
            "LeWM 的 prediction target 分支、prediction context 分支和 "
            "mean/Tail-SIGReg/VISReg 分别推动门规则配对未来收缩还是"
            "分离；Tail-SIGReg 在相同 Encoder 梯度预算下是否更聚焦？"
        ),
        "provenance": {
            "contextworld_repo": str(contextworld_repo),
            "stable_worldmodel_repo": str(stable_repo),
            "runtime_stable_worldmodel_commit": args.stable_ref,
            "synthesis_data_stable_worldmodel_commit": args.data_stable_ref,
            "artifact_root": str(artifact_root),
            "catalog": str(catalog),
            "catalog_sha256": geometry.file_sha256(catalog),
            "normalizer": str(normalizer),
            "normalizer_sha256": geometry.file_sha256(normalizer),
            "device": args.device,
            "training_batch_seed": args.seed,
            "autograd_seed": autograd_seed,
            "checkpoints_modified": False,
            "environment_calls": 0,
        },
        "paired_distance_batch": {
            "queries_per_rule": len(frozen.query_ids),
            "clips": 2 * len(frozen.query_ids),
            "frames_per_clip": 4,
            "pixels_sha256": balanced_batch["pixels_sha256"],
            "actions_sha256": balanced_batch["actions_sha256"],
            "source": (
                "frozen diagnostic observed-passable/passable-target and "
                "observed-blocked/blocked-target pairs"
            ),
        },
        "loss_batch": loss_batch_metadata,
        "weights": {
            "lewm_sigreg": LEWM_SIGREG_WEIGHT,
            "tail_sigreg_fraction": TAIL_SIGREG_FRACTION,
            "tuned_sigreg_reference": TUNED_SIGREG_REFERENCE_WEIGHT,
            "lewm_visreg_reference": LEWM_VISREG_REFERENCE_WEIGHT,
            "visreg_kwargs": VISREG_KWARGS,
            "pldm_active_regularizers": PLDM_WEIGHTS,
        },
        "sign_convention": (
            "predicted_distance_change_per_unit_lr < 0 means an "
            "infinitesimal gradient-descent step contracts the paired "
            "door-rule distance; > 0 means expansion."
        ),
        "checkpoints": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            geometry._json_safe(payload),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
