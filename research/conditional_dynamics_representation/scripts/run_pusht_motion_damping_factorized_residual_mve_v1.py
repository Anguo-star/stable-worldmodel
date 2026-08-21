#!/usr/bin/env python3
"""Development-only factorized residual diagnostics for PushT Motion Damping.

The frozen seed-3073 PushT LeWM supplies the History-3 latents, action
embeddings, base prediction, and target.  This runner reads only the formal
Training and loader-validation (Development) Motion Damping tables.  It does
not instantiate a Public Test dataset or score Public Test data.

The residual head is the deliberately small factorization

    delta = W_o(phi(W_q q) * phi(W_c c)).

Here ``q = [z2, z2-z1, a2]``.  Learned ``c`` is
``[z1-z0, z2-z1, a0, a1]``; the oracle diagnostic replaces it with a two-class
one-hot damping label and is reported only as an upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


# The ContextWorld checkout is a sibling of this repository.  This is the
# same import arrangement used by the existing ContextWorld runners.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
CONTEXTWORLD_SCRIPTS = CONTEXTWORLD_ROOT / "scripts"
for source_root in (REPO_ROOT, CONTEXTWORLD_ROOT, CONTEXTWORLD_SCRIPTS):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import run_pusht_frozen_history_residual_pilot as frozen_pilot  # noqa: E402
from contextworld.training.pusht_history_residual import (  # noqa: E402
    complete_twin_group_batch_stream,
    paired_center_response_loss,
    paired_prediction_metrics,
)


FrozenRows = frozen_pilot.FrozenRows

ORACLE_CLASS_NAMES = (
    "faster_decay_0p2",
    "no_extra_decay_1p0",
)
FAST_DAMPING_LABEL = 0
NO_EXTRA_DAMPING_LABEL = 1
FIXED_SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_history_inputs(
    latents: torch.Tensor, action_embeddings: torch.Tensor
) -> tuple[int, int, int]:
    if latents.ndim != 3 or action_embeddings.ndim != 3:
        raise ValueError("latents and action_embeddings must have shape (B,3,D)")
    if latents.shape[1] != 3 or action_embeddings.shape[1] != 3:
        raise ValueError("History-3 inputs are required")
    if latents.shape[0] != action_embeddings.shape[0]:
        raise ValueError("latent and action batch sizes must match")
    if latents.shape[-1] <= 0 or action_embeddings.shape[-1] <= 0:
        raise ValueError("latent and action dimensions must be positive")
    return int(latents.shape[0]), int(latents.shape[-1]), int(action_embeddings.shape[-1])


def _coerce_oracle_indices(labels: torch.Tensor | Sequence[int] | Sequence[float]) -> torch.Tensor:
    values = torch.as_tensor(labels)
    if values.ndim == 2 and values.shape[-1] == 2:
        if not bool(torch.isfinite(values.float()).all()):
            raise ValueError("one-hot damping context contains non-finite values")
        if not bool(torch.allclose(values.sum(dim=-1), torch.ones(values.shape[0], device=values.device, dtype=values.dtype), atol=1e-6, rtol=0.0)):
            raise ValueError("two-column oracle context must be one-hot")
        if not bool(((values == 0) | (values == 1)).all()):
            raise ValueError("two-column oracle context must be one-hot")
        return values.argmax(dim=-1).to(torch.long)
    if values.ndim == 2 and values.shape[-1] == 1:
        values = values.reshape(-1)
    if values.ndim != 1:
        raise ValueError("damping labels must have shape (B,)")
    if values.dtype.is_floating_point:
        rounded = values.round()
        if torch.allclose(values, rounded, atol=1e-6, rtol=0.0) and bool(
            ((rounded == 0) | (rounded == 1)).all()
        ):
            values = rounded.to(torch.long)
        else:
            fast = torch.isclose(values, torch.tensor(0.2, dtype=values.dtype, device=values.device), atol=1e-5, rtol=0.0)
            no_extra = torch.isclose(values, torch.tensor(1.0, dtype=values.dtype, device=values.device), atol=1e-5, rtol=0.0)
            if not bool((fast | no_extra).all()):
                raise ValueError("floating damping labels must be 0/1 or 0.2/1.0")
            values = torch.where(fast, torch.zeros_like(values), torch.ones_like(values)).to(torch.long)
    else:
        values = values.to(torch.long)
    if values.numel() and not bool(((values == 0) | (values == 1)).all()):
        raise ValueError("damping labels must use class indices 0 or 1")
    return values


def oracle_context(
    labels: torch.Tensor | Sequence[int] | Sequence[float],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return [faster-decay, no-extra-decay] one-hot context in that order."""

    indices = _coerce_oracle_indices(labels)
    if device is not None:
        indices = indices.to(device)
    return F.one_hot(indices, num_classes=2).to(dtype=dtype)


oracle_label_context = oracle_context


def motion_damping_labels(pair_count: int) -> torch.Tensor:
    """Return the canonical row labels: [0.2, 1.0] for every paired row."""

    if pair_count <= 0 or pair_count % 2:
        raise ValueError("Motion Damping pair_count must be a positive even number")
    return torch.tensor([FAST_DAMPING_LABEL, NO_EXTRA_DAMPING_LABEL], dtype=torch.long).repeat(pair_count)


def motion_damping_twin_group_ids(pair_count: int) -> torch.Tensor:
    """Return four contiguous row ids per forward/reverse twin group."""

    if pair_count <= 0 or pair_count % 2:
        raise ValueError("Motion Damping pair_count must be a positive even number")
    return torch.arange(pair_count // 2, dtype=torch.long).repeat_interleave(4)


def _query_features(
    latents: torch.Tensor, action_embeddings: torch.Tensor
) -> torch.Tensor:
    _validate_history_inputs(latents, action_embeddings)
    z0, z1, z2 = latents.unbind(dim=1)
    _a0, _a1, a2 = action_embeddings.unbind(dim=1)
    return torch.cat((z2, z2 - z1, a2), dim=-1)


def _learned_context_features(
    latents: torch.Tensor, action_embeddings: torch.Tensor
) -> torch.Tensor:
    _validate_history_inputs(latents, action_embeddings)
    z0, z1, z2 = latents.unbind(dim=1)
    a0, a1, _a2 = action_embeddings.unbind(dim=1)
    return torch.cat((z1 - z0, z2 - z1, a0, a1), dim=-1)


class FactorizedResidualHead(nn.Module):
    """Low-rank query/context residual head with an exactly zero start."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        rank: int,
        variant: str = "learned",
    ) -> None:
        super().__init__()
        if variant not in {"learned", "oracle"}:
            raise ValueError("variant must be 'learned' or 'oracle'")
        if latent_dim <= 0 or action_dim <= 0 or rank <= 0:
            raise ValueError("latent_dim, action_dim, and rank must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.rank = int(rank)
        self.variant = variant
        self.query_dim = 2 * self.latent_dim + self.action_dim
        self.learned_context_dim = 2 * self.latent_dim + 2 * self.action_dim
        self.context_dim = 2 if variant == "oracle" else self.learned_context_dim

        # Bias-free maps make the implementation match the stated
        # factorization exactly.  The output map is zero initialized so a
        # newly attached head is an exact identity adapter for the base model.
        self.W_q = nn.Linear(self.query_dim, self.rank, bias=False)
        self.W_c = nn.Linear(self.context_dim, self.rank, bias=False)
        self.W_o = nn.Linear(self.rank, self.latent_dim, bias=False)
        self.phi = nn.SiLU()
        nn.init.zeros_(self.W_o.weight)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def query(self, latents: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        return _query_features(latents, action_embeddings)

    q_features = query

    def learned_context(
        self, latents: torch.Tensor, action_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return _learned_context_features(latents, action_embeddings)

    c_features = learned_context

    def context(
        self,
        latents: torch.Tensor,
        action_embeddings: torch.Tensor,
        labels: torch.Tensor | Sequence[int] | Sequence[float] | None = None,
    ) -> torch.Tensor:
        if self.variant == "learned":
            if labels is not None:
                raise ValueError("learned variant cannot consume oracle labels")
            return self.learned_context(latents, action_embeddings)
        if labels is None:
            raise ValueError("oracle variant requires damping labels")
        return oracle_context(labels, device=latents.device, dtype=latents.dtype)

    def oracle_context(
        self, labels: torch.Tensor | Sequence[int] | Sequence[float], *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if self.variant != "oracle":
            raise ValueError("oracle_context is available only for the oracle variant")
        return oracle_context(labels, dtype=dtype)

    def forward(
        self,
        latents: torch.Tensor,
        action_embeddings: torch.Tensor,
        labels: torch.Tensor | Sequence[int] | Sequence[float] | None = None,
        *,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, latent_dim, action_dim = _validate_history_inputs(
            latents, action_embeddings
        )
        if latent_dim != self.latent_dim or action_dim != self.action_dim:
            raise ValueError("FactorizedResidualHead input dimensions do not match")
        q = self.query(latents, action_embeddings)
        c = self.context(latents, action_embeddings, labels) if context is None else context
        if c.ndim != 2 or c.shape != (batch_size, self.context_dim):
            raise ValueError("context must have shape (B, context_dim)")
        return self.W_o(self.phi(self.W_q(q)) * self.phi(self.W_c(c)))


LowRankFactorizedResidualHead = FactorizedResidualHead


class OracleMLPResidualHead(nn.Module):
    """Nonlinear oracle-conditioned response upper bound with a zero start."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim, action_dim, and hidden_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.variant = "oracle_mlp"
        self.head_family = "oracle_mlp"
        self.query_dim = 2 * self.latent_dim + self.action_dim
        self.oracle_context_dim = 2
        self.input_dim = self.query_dim + self.oracle_context_dim

        self.norm = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.phi = nn.SiLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def query(self, latents: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        return _query_features(latents, action_embeddings)

    q_features = query

    def oracle_context(
        self,
        labels: torch.Tensor | Sequence[int] | Sequence[float],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return oracle_context(labels, device=device, dtype=dtype)

    def context(
        self,
        labels: torch.Tensor | Sequence[int] | Sequence[float] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError("oracle_mlp variant requires damping labels")
        return self.oracle_context(labels, device=device, dtype=dtype)

    def forward(
        self,
        latents: torch.Tensor,
        action_embeddings: torch.Tensor,
        labels: torch.Tensor | Sequence[int] | Sequence[float] | None = None,
    ) -> torch.Tensor:
        batch_size, latent_dim, action_dim = _validate_history_inputs(
            latents, action_embeddings
        )
        if latent_dim != self.latent_dim or action_dim != self.action_dim:
            raise ValueError("OracleMLPResidualHead input dimensions do not match")
        q = self.query(latents, action_embeddings)
        c = self.context(labels, device=q.device, dtype=q.dtype)
        if c.shape != (batch_size, self.oracle_context_dim):
            raise ValueError("oracle labels must have one context row per input row")
        features = torch.cat((q, c), dim=-1)
        return self.fc2(self.phi(self.fc1(self.norm(features))))


OracleMLPResponseHead = OracleMLPResidualHead


def make_probe_features(
    latents: torch.Tensor, action_embeddings: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Build strict query-only and history-bearing probe populations."""

    _validate_history_inputs(latents, action_embeddings)
    _z0, _z1, z2 = latents.unbind(dim=1)
    _a0, _a1, a2 = action_embeddings.unbind(dim=1)
    # The response head may use z2-z1 for dynamic state estimation, but the
    # query-only negative control must contain no history-bearing difference.
    query = torch.cat((z2, a2), dim=-1)
    history = _learned_context_features(latents, action_embeddings)
    return {
        "query_only": query,
        "history_only": history,
        "full_history": torch.cat((query, history), dim=-1),
    }


def target_paired_response_vectors(
    target: torch.Tensor, pair_indices: torch.Tensor | None = None
) -> torch.Tensor:
    """Return target right-minus-left response vectors for explicit pairs."""

    if target.ndim != 2:
        raise ValueError("target must have shape (rows, latent_dim)")
    if pair_indices is None:
        if target.shape[0] % 2:
            raise ValueError("target rows must be even")
        pair_indices = torch.arange(target.shape[0], device=target.device).reshape(-1, 2)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must have shape (pairs,2)")
    pair_indices = pair_indices.to(device=target.device, dtype=torch.long)
    return target[pair_indices[:, 1]] - target[pair_indices[:, 0]]


def target_response_svd_diagnostics(
    response_vectors: torch.Tensor,
) -> dict[str, Any]:
    """Compute singular spectrum, entropy effective rank, and energy ranks."""

    if response_vectors.ndim == 3 and response_vectors.shape[1] == 2:
        response_vectors = response_vectors[:, 1] - response_vectors[:, 0]
    if response_vectors.ndim != 2:
        raise ValueError("response_vectors must have shape (pairs, latent_dim)")
    values = response_vectors.detach().float().cpu()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("response_vectors contain non-finite values")
    singular_values = torch.linalg.svdvals(values)
    energy = singular_values.square()
    total_energy = energy.sum()
    if float(total_energy) <= 1e-12:
        entropy_effective_rank = 0.0
        energy_entropy_effective_rank = 0.0
        energy_fraction = torch.zeros_like(energy)
        ranks = {"r90": 0, "r95": 0, "r99": 0}
    else:
        energy_fraction = energy / total_energy
        # Effective rank conventionally uses normalized singular values;
        # explained-energy thresholds use squared singular values.
        singular_probability = singular_values / singular_values.sum().clamp_min(1e-12)
        nonzero_singular = singular_probability > 0
        singular_entropy = -(
            singular_probability[nonzero_singular]
            * singular_probability[nonzero_singular].log()
        ).sum()
        entropy_effective_rank = float(singular_entropy.exp())
        nonzero_energy = energy_fraction > 0
        energy_entropy = -(
            energy_fraction[nonzero_energy] * energy_fraction[nonzero_energy].log()
        ).sum()
        energy_entropy_effective_rank = float(energy_entropy.exp())
        cumulative = torch.cumsum(energy_fraction, dim=0)
        ranks = {
            f"r{int(threshold * 100)}": int(
                torch.nonzero(cumulative >= threshold, as_tuple=False)[0].item() + 1
            )
            for threshold in (0.90, 0.95, 0.99)
        }
    return {
        "response_count": int(values.shape[0]),
        "response_dimension": int(values.shape[1]),
        "singular_values": [float(value) for value in singular_values],
        "energy_fractions": [float(value) for value in energy_fraction],
        "entropy_effective_rank": entropy_effective_rank,
        "effective_rank_entropy": entropy_effective_rank,
        "energy_entropy_effective_rank": energy_entropy_effective_rank,
        **ranks,
    }


# Short aliases make the diagnostic functions convenient to use from focused
# tests and from later read-only analysis notebooks.
svd_diagnostics = target_response_svd_diagnostics
compute_svd_diagnostics = target_response_svd_diagnostics


def leave_forward_reverse_twin_group_out_folds(
    row_count: int,
    group_ids: torch.Tensor | None = None,
    *,
    fold_count: int = 8,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return deterministic group-isolated folds over forward/reverse twins."""

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if group_ids is None:
        if row_count % 4:
            raise ValueError("four rows are required per forward/reverse twin group")
        group_ids = torch.arange(row_count, dtype=torch.long) // 4
    group_ids = torch.as_tensor(group_ids, dtype=torch.long).reshape(-1).cpu()
    if group_ids.numel() != row_count:
        raise ValueError("group_ids must have one entry per row")
    groups = torch.unique(group_ids, sorted=True)
    if groups.numel() < 2:
        raise ValueError("group-isolated probing needs at least two twin groups")
    actual_fold_count = min(int(fold_count), int(groups.numel()))
    folds: list[tuple[torch.Tensor, torch.Tensor]] = []
    all_rows = torch.arange(row_count, dtype=torch.long)
    for group in groups:
        if int((group_ids == group).sum()) != 4:
            raise ValueError("each twin group must contain exactly four rows")
    for fold_index in range(actual_fold_count):
        held_out_groups = groups[fold_index::actual_fold_count]
        held_out_mask = torch.isin(group_ids, held_out_groups)
        held_out = all_rows[held_out_mask]
        train = all_rows[~held_out_mask]
        if train.numel() == 0:
            raise ValueError("group-isolated probing needs a non-empty training fold")
        if torch.isin(train, held_out).any():
            raise RuntimeError("twin-group fold leaked held-out rows into training")
        folds.append((train, held_out))
    return folds


def _fit_ridge_binary(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")
    train_features = train_features.float()
    test_features = test_features.float()
    train_labels = train_labels.float()
    squeeze = train_labels.ndim == 1
    if squeeze:
        train_labels = train_labels[:, None]
    if train_labels.ndim != 2 or train_labels.shape[0] != train_features.shape[0]:
        raise ValueError(
            "ridge labels must have shape (train_rows,) or (train_rows,K)"
        )
    mean = train_features.mean(dim=0, keepdim=True)
    scale = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_scaled = (train_features - mean) / scale
    test_scaled = (test_features - mean) / scale
    label_mean = train_labels.mean(dim=0, keepdim=True)
    centered_labels = train_labels - label_mean
    rows, dimensions = train_scaled.shape
    try:
        if dimensions <= rows:
            gram = train_scaled.T @ train_scaled + alpha * torch.eye(dimensions)
            coefficients = torch.linalg.solve(
                gram, train_scaled.T @ centered_labels
            )
        else:
            gram = train_scaled @ train_scaled.T + alpha * torch.eye(rows)
            dual = torch.linalg.solve(gram, centered_labels)
            coefficients = train_scaled.T @ dual
    except RuntimeError:
        if dimensions <= rows:
            coefficients = torch.linalg.pinv(gram) @ (
                train_scaled.T @ centered_labels
            )
        else:
            coefficients = train_scaled.T @ (
                torch.linalg.pinv(gram) @ centered_labels
            )
    prediction = label_mean + test_scaled @ coefficients
    return prediction[:, 0] if squeeze else prediction


def _binary_probe_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    target = target.to(torch.long).reshape(-1)
    predicted_labels = (predicted.reshape(-1) >= 0.5).to(torch.long)
    accuracy = (predicted_labels == target).float().mean()
    class_rates = []
    for label in (0, 1):
        selected = target == label
        if bool(selected.any()):
            class_rates.append((predicted_labels[selected] == target[selected]).float().mean())
    balanced = torch.stack(class_rates).mean() if class_rates else torch.tensor(float("nan"))
    return {"accuracy": float(accuracy), "balanced_accuracy": float(balanced)}


def leave_forward_reverse_twin_group_out_ridge_probes(
    latents: torch.Tensor,
    action_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    group_ids: torch.Tensor | None = None,
    alpha: float = 1.0,
    seed: int = 3073,
    fold_count: int = 8,
) -> dict[str, Any]:
    """Run query/history/full ridge probes with within-fold shuffle controls."""

    if latents.ndim != 3 or action_embeddings.ndim != 3:
        raise ValueError("probe inputs must be History-3 tensors")
    row_count = int(latents.shape[0])
    _require(action_embeddings.shape[0] == row_count, "probe row counts must match")
    labels = _coerce_oracle_indices(labels).cpu()
    if labels.numel() != row_count:
        raise ValueError("probe labels must have one entry per row")
    features = {name: value.detach().float().cpu() for name, value in make_probe_features(latents.cpu(), action_embeddings.cpu()).items()}
    folds = leave_forward_reverse_twin_group_out_folds(
        row_count, group_ids, fold_count=fold_count
    )
    group_ids_cpu = (
        torch.arange(row_count, dtype=torch.long) // 4
        if group_ids is None
        else torch.as_tensor(group_ids, dtype=torch.long).reshape(-1).cpu()
    )
    fold_receipts = []
    for train_indices, held_out_indices in folds:
        train_groups = torch.unique(group_ids_cpu[train_indices], sorted=True)
        held_out_groups = torch.unique(group_ids_cpu[held_out_indices], sorted=True)
        if torch.isin(train_groups, held_out_groups).any():
            raise RuntimeError("fold group isolation failed")
        fold_receipts.append(
            {
                "held_out_groups": [int(value) for value in held_out_groups],
                "held_out_indices": [int(value) for value in held_out_indices],
                "train_groups": [int(value) for value in train_groups],
                "train_indices": [int(value) for value in train_indices],
                "group_isolation_passed": True,
            }
        )

    results: dict[str, Any] = {}
    for feature_offset, (feature_name, feature_matrix) in enumerate(features.items()):
        fold_metrics = []
        shuffled_fold_metrics = []
        for fold_number, (train_indices, held_out_indices) in enumerate(folds):
            x_train = feature_matrix[train_indices]
            y_train = labels[train_indices]
            x_test = feature_matrix[held_out_indices]
            y_test = labels[held_out_indices]
            generator = torch.Generator(device="cpu").manual_seed(
                int(seed) + 1_000_003 * fold_number + 10_007 * feature_offset
            )
            shuffled_labels = y_train[torch.randperm(y_train.numel(), generator=generator)]
            joint_prediction = _fit_ridge_binary(
                x_train,
                torch.stack((y_train, shuffled_labels), dim=1),
                x_test,
                alpha=alpha,
            )
            prediction = joint_prediction[:, 0]
            shuffled_prediction = joint_prediction[:, 1]
            fold_metrics.append(_binary_probe_metrics(prediction, y_test))
            shuffled_fold_metrics.append(_binary_probe_metrics(shuffled_prediction, y_test))

        accuracy = sum(row["accuracy"] for row in fold_metrics) / len(fold_metrics)
        balanced = sum(row["balanced_accuracy"] for row in fold_metrics) / len(fold_metrics)
        shuffled_accuracy = sum(row["accuracy"] for row in shuffled_fold_metrics) / len(shuffled_fold_metrics)
        shuffled_balanced = sum(row["balanced_accuracy"] for row in shuffled_fold_metrics) / len(shuffled_fold_metrics)
        results[feature_name] = {
            "feature_dimension": int(feature_matrix.shape[1]),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "fold_metrics": fold_metrics,
            "within_fold_label_shuffle": {
                "accuracy": shuffled_accuracy,
                "balanced_accuracy": shuffled_balanced,
                "fold_metrics": shuffled_fold_metrics,
                "labels_shuffled_only_in_training_fold": True,
            },
            # Flat aliases keep the JSON easy to grep.
            "label_shuffle_accuracy": shuffled_accuracy,
            "label_shuffle_balanced_accuracy": shuffled_balanced,
        }
    controls = {
        f"{name}_label_shuffle": result["within_fold_label_shuffle"]
        for name, result in results.items()
    }
    return {
        "split": "Development",
        "fold_unit": "forward_reverse_twin_group_isolated_kfold",
        "fold_count": len(folds),
        "alpha": float(alpha),
        "seed": int(seed),
        "fold_isolation_passed": True,
        "folds": fold_receipts,
        "feature_sets": results,
        "query_only": results["query_only"],
        "history_only": results["history_only"],
        "full_history": results["full_history"],
        "controls": controls,
    }


# Concise alias used by downstream diagnostics.
ridge_probe_diagnostics = leave_forward_reverse_twin_group_out_ridge_probes
leave_twin_group_out_ridge_probes = leave_forward_reverse_twin_group_out_ridge_probes


@torch.no_grad()
def corrected_prediction(
    head: nn.Module,
    rows: FrozenRows,
    *,
    labels: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    head.eval()
    predictions = []
    for start in range(0, rows.count, batch_size):
        indices = torch.arange(start, min(start + batch_size, rows.count), dtype=torch.long)
        batch = frozen_pilot.take(rows, indices, device)
        batch_labels = None if labels is None else labels[indices].to(device)
        residual = head(batch.latents, batch.action_embeddings, batch_labels)
        predictions.append((batch.base_prediction + residual).float().cpu())
    return torch.cat(predictions, dim=0)


def _response_fields(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted_delta = prediction[1::2] - prediction[0::2]
    target_delta = target[1::2] - target[0::2]
    predicted_double = predicted_delta.double()
    target_double = target_delta.double()
    prediction_energy = predicted_double.square().sum()
    target_energy = target_double.square().sum().clamp_min(1e-12)
    cross_energy = (predicted_double * target_double).sum()
    error_energy = (predicted_double - target_double).square().sum()
    cosine = F.cosine_similarity(predicted_delta, target_delta, dim=-1, eps=1e-8)
    return {
        "response_cosine_mean": float(cosine.mean()),
        "response_gain": float(cross_energy / target_energy),
        "response_alignment": float(
            cross_energy
            / (prediction_energy.sqrt() * target_energy.sqrt()).clamp_min(1e-12)
        ),
        "normalized_response_error": float(error_energy / target_energy),
        "prediction_to_target_energy_ratio": float(
            prediction_energy / target_energy
        ),
        "response_mse": float((predicted_delta - target_delta).square().mean()),
        "prediction_response_energy": float(predicted_delta.square().mean()),
        "target_response_energy": float(target_delta.square().mean()),
    }


@torch.no_grad()
def paired_metrics(
    *,
    base_prediction: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Report native paired metrics plus response cosine/error gains."""

    if prediction.shape != target.shape or base_prediction.shape != target.shape:
        raise ValueError("base_prediction, prediction, and target shapes must match")
    if prediction.ndim != 2 or prediction.shape[0] % 2:
        raise ValueError("paired metrics require an even (rows, latent_dim) tensor")
    metrics = dict(paired_prediction_metrics(prediction=prediction, target=target))
    base_fields = _response_fields(base_prediction, target)
    corrected_fields = _response_fields(prediction, target)
    metrics.update(
        {
            "base_prediction_mse": float((base_prediction - target).square().mean()),
            "corrected_prediction_mse": float((prediction - target).square().mean()),
            "base_response_cosine_mean": base_fields["response_cosine_mean"],
            "corrected_response_cosine_mean": corrected_fields[
                "response_cosine_mean"
            ],
            "response_cosine_mean_improvement": (
                corrected_fields["response_cosine_mean"]
                - base_fields["response_cosine_mean"]
            ),
            "base_response_gain": base_fields["response_gain"],
            "response_gain": corrected_fields["response_gain"],
            "base_response_alignment": base_fields["response_alignment"],
            "response_alignment": corrected_fields["response_alignment"],
            "prediction_to_target_energy_ratio": corrected_fields[
                "prediction_to_target_energy_ratio"
            ],
            "base_normalized_response_error": base_fields["normalized_response_error"],
            "corrected_normalized_response_error": corrected_fields["normalized_response_error"],
            "normalized_response_error": corrected_fields["normalized_response_error"],
            "normalized_error": corrected_fields["normalized_response_error"],
            "normalized_error_gain": base_fields["normalized_response_error"] - corrected_fields["normalized_response_error"],
            "base_response_mse": base_fields["response_mse"],
            "corrected_response_mse": corrected_fields["response_mse"],
        }
    )
    return metrics


def _parameter_counts(base_model: nn.Module, head: nn.Module) -> dict[str, int]:
    base_total = sum(parameter.numel() for parameter in base_model.parameters())
    base_trainable = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    head_total = sum(parameter.numel() for parameter in head.parameters())
    head_trainable = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    return {
        "base_total": int(base_total),
        "base_trainable": int(base_trainable),
        "base_frozen": int(base_total - base_trainable),
        "head_total": int(head_total),
        "head_trainable": int(head_trainable),
        "optimizer_trainable": int(head_trainable),
    }


def _gradient_bootstrap_norms(
    head: nn.Module,
    rows: FrozenRows,
    labels: torch.Tensor | None,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    head.train()
    batch = frozen_pilot.take(rows, indices, device)
    batch_labels = None if labels is None else labels[indices].to(device)
    prediction = batch.base_prediction + head(batch.latents, batch.action_embeddings, batch_labels)
    pair_indices = torch.arange(indices.numel(), device=device, dtype=torch.long).reshape(-1, 2)
    loss, components = paired_center_response_loss(
        prediction=prediction,
        target=batch.target,
        pair_indices=pair_indices,
    )
    head.zero_grad(set_to_none=True)
    loss.backward()
    norms = {}
    squared_total = 0.0
    nonzero = 0
    for name, parameter in head.named_parameters():
        norm = 0.0 if parameter.grad is None else float(parameter.grad.detach().norm())
        norms[name] = norm
        squared_total += norm * norm
        nonzero += int(norm > 0.0)
    head.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach()),
        "components": {name: float(value.detach()) for name, value in components.items()},
        "parameter_l2_norms": norms,
        "total_l2_norm": math.sqrt(squared_total),
        "nonzero_parameter_gradient_count": nonzero,
        "bootstrap_indices": [int(value) for value in indices],
        "complete_twin_groups": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("oracle", "oracle_mlp", "learned"), default="learned"
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3073)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    # These optional paths preserve the pinned helper defaults while making a
    # CPU smoke run or an explicitly bound Development release reproducible.
    parser.add_argument("--release-config", type=Path, default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--action-normalizer-source", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--probe-fold-count", type=int, default=8)
    return parser.parse_args(argv)


def _output_paths(output: Path) -> tuple[Path, Path, bool]:
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    json_mode = output.suffix.lower() == ".json"
    report_path = output if json_mode else output / "report.json"
    head_path = (
        output.with_name(output.stem + "_head.pt")
        if json_mode
        else output / "factorized_residual_head.pt"
    )
    if report_path.exists() or head_path.exists():
        raise FileExistsError(f"Refusing to overwrite output artifacts: {report_path}")
    return report_path, head_path, json_mode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_twin_batch_size(pair_count: int, requested: int) -> int:
    if requested <= 0 or requested % 4:
        raise ValueError("batch-size must be a positive multiple of four")
    group_count = pair_count // 2
    requested_groups = requested // 4
    if group_count % requested_groups == 0:
        return requested
    divisors = [value for value in range(1, min(group_count, requested_groups) + 1) if group_count % value == 0]
    if not divisors:
        raise ValueError("could not choose a complete-twin-group batch size")
    return 4 * max(divisors)


def _load_motion_damping_train_development(
    *,
    release_config: Path | None,
    source_checkpoint: Path | None,
    action_normalizer_source: Path | None,
    device: torch.device,
    encoder_batch_size: int,
) -> tuple[nn.Module, FrozenRows, FrozenRows, dict[str, Any], dict[str, Any]]:
    """Load only the pinned train and Development tables through pilot helpers."""

    if encoder_batch_size <= 0 or encoder_batch_size % 2:
        raise ValueError("encoder-batch-size must be a positive even number")
    release_path = release_config or frozen_pilot.DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
    release = frozen_pilot.load_motion_damping_icl_release(release_path)
    data_root = frozen_pilot.resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=CONTEXTWORLD_ROOT
    )
    tables = release["data"]["lance_tables"]
    pair_counts = release["data"]["pair_counts"]
    # Deliberately enumerate only train and loader_validation.  No Public Test
    # table path is resolved, read, decoded, or scored here.
    train_arrays = frozen_pilot.read_damping_pairs(
        data_root / tables["train"],
        expected_pairs=int(pair_counts["train"]),
        expected_split="train",
    )
    development_arrays = frozen_pilot.read_damping_pairs(
        data_root / tables["loader_validation"],
        expected_pairs=int(pair_counts["loader_validation"]),
        expected_split="loader_validation",
    )
    twin_audit = frozen_pilot.motion_twin_audit(train_arrays)
    checkpoint = source_checkpoint or frozen_pilot.pilot.DEFAULT_CHECKPOINT
    action_source = action_normalizer_source or frozen_pilot.pilot.DEFAULT_ORIGINAL_DATASET
    action_stats = frozen_pilot.pilot.original_action_stats(Path(action_source).expanduser().resolve())
    base_model, source_receipt = frozen_pilot.pilot.load_model(
        Path(checkpoint).expanduser().resolve(), device=device
    )
    frozen_pilot.freeze_model(base_model)
    if any(parameter.requires_grad for parameter in base_model.parameters()):
        raise RuntimeError("the entire base model was not frozen")
    batch_pairs = max(1, encoder_batch_size // 2)
    train_rows = frozen_pilot.encode_paired_arrays(
        base_model,
        train_arrays,
        action_stats=action_stats,
        device=device,
        batch_pairs=batch_pairs,
    )
    development_rows = frozen_pilot.encode_paired_arrays(
        base_model,
        development_arrays,
        action_stats=action_stats,
        device=device,
        batch_pairs=batch_pairs,
    )
    if train_rows.count != 2 * train_arrays.pair_count or development_rows.count != 2 * development_arrays.pair_count:
        raise RuntimeError("encoded Motion Damping rows do not preserve pair counts")
    release_receipt = {
        "config": str(Path(release_path).expanduser().resolve()),
        "config_sha256": _sha256(Path(release_path).expanduser().resolve()),
        "training_pair_count": int(train_arrays.pair_count),
        "development_pair_count": int(development_arrays.pair_count),
        "development_split": "loader_validation",
        "training_twin_audit": twin_audit,
        "public_test_opened": False,
    }
    return base_model, train_rows, development_rows, source_receipt, release_receipt


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.rank <= 0 or args.steps < 0:
        raise ValueError("rank must be positive and steps must be non-negative")
    if args.mlp_hidden_dim <= 0:
        raise ValueError("mlp-hidden-dim must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.ridge_alpha <= 0:
        raise ValueError("optimizer and ridge controls are invalid")
    if args.probe_fold_count < 2:
        raise ValueError("probe-fold-count must be at least two")
    report_path, head_path, json_mode = _output_paths(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    frozen_pilot.pilot.set_reproducible_seed(args.seed)

    base_model, train_rows, development_rows, source_receipt, release_receipt = _load_motion_damping_train_development(
        release_config=args.release_config,
        source_checkpoint=args.source_checkpoint,
        action_normalizer_source=args.action_normalizer_source,
        device=device,
        encoder_batch_size=args.encoder_batch_size,
    )
    train_pair_count = train_rows.count // 2
    development_pair_count = development_rows.count // 2
    train_labels = motion_damping_labels(train_pair_count)
    development_labels = motion_damping_labels(development_pair_count)
    development_group_ids = motion_damping_twin_group_ids(development_pair_count)

    train_response = target_paired_response_vectors(train_rows.target)
    development_response = target_paired_response_vectors(development_rows.target)
    svd = {
        "Training": target_response_svd_diagnostics(train_response),
        "Development": target_response_svd_diagnostics(development_response),
    }
    ridge = leave_forward_reverse_twin_group_out_ridge_probes(
        development_rows.latents,
        development_rows.action_embeddings,
        development_labels,
        group_ids=development_group_ids,
        alpha=args.ridge_alpha,
        seed=args.seed,
        fold_count=args.probe_fold_count,
    )

    latent_dim = int(train_rows.latents.shape[-1])
    action_dim = int(train_rows.action_embeddings.shape[-1])
    if args.variant == "oracle_mlp":
        head = OracleMLPResidualHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=args.mlp_hidden_dim,
        ).to(device)
    else:
        head = FactorizedResidualHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            rank=args.rank,
            variant=args.variant,
        ).to(device)
    if any(parameter.requires_grad is False for parameter in head.parameters()):
        raise RuntimeError("all factorized head parameters must be trainable")
    if any(parameter.requires_grad for parameter in base_model.parameters()):
        raise RuntimeError("base model became trainable")

    effective_batch_size = _valid_twin_batch_size(train_pair_count, args.batch_size)
    bootstrap_stream = iter(
        complete_twin_group_batch_stream(
            pair_count=train_pair_count,
            rows_per_batch=effective_batch_size,
            seed=args.seed,
        )
    )
    bootstrap_indices = next(bootstrap_stream)
    uses_oracle_labels = args.variant in {"oracle", "oracle_mlp"}
    oracle_train_labels = train_labels if uses_oracle_labels else None
    gradient_bootstrap = _gradient_bootstrap_norms(
        head,
        train_rows,
        oracle_train_labels,
        bootstrap_indices,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_stream = iter(
        complete_twin_group_batch_stream(
            pair_count=train_pair_count,
            rows_per_batch=effective_batch_size,
            seed=args.seed,
        )
    )
    local_pair_indices = torch.arange(
        effective_batch_size, device=device, dtype=torch.long
    ).reshape(-1, 2)
    snapshot_steps = sorted({step for step in FIXED_SNAPSHOT_STEPS if step <= args.steps} | {args.steps})
    snapshots: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    counts = _parameter_counts(base_model, head)

    def snapshot(step: int) -> None:
        train_prediction = corrected_prediction(
            head,
            train_rows,
            labels=oracle_train_labels,
            device=device,
            batch_size=args.encoder_batch_size,
        )
        development_prediction = corrected_prediction(
            head,
            development_rows,
            labels=development_labels if uses_oracle_labels else None,
            device=device,
            batch_size=args.encoder_batch_size,
        )
        row = {
            "optimizer_step": int(step),
            "parameter_counts": counts,
            "training": paired_metrics(
                base_prediction=train_rows.base_prediction,
                prediction=train_prediction,
                target=train_rows.target,
            ),
            "development": paired_metrics(
                base_prediction=development_rows.base_prediction,
                prediction=development_prediction,
                target=development_rows.target,
            ),
        }
        snapshots.append(row)

    snapshot(0)
    for step in range(1, args.steps + 1):
        head.train()
        indices = next(train_stream)
        batch = frozen_pilot.take(train_rows, indices, device)
        batch_labels = None if oracle_train_labels is None else oracle_train_labels[indices].to(device)
        prediction = batch.base_prediction + head(batch.latents, batch.action_embeddings, batch_labels)
        loss, components = paired_center_response_loss(
            prediction=prediction,
            target=batch.target,
            pair_indices=local_pair_indices,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("factorized residual loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step in snapshot_steps:
            trace.append(
                {
                    "optimizer_step": step,
                    "loss": float(loss.detach()),
                    "components": {name: float(value.detach()) for name, value in components.items()},
                    "gradient_norm_before_clip": gradient_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "batch_indices": [int(value) for value in indices],
                    "complete_twin_groups": True,
                }
            )
        if step in snapshot_steps:
            snapshot(step)

    # Create the output only after all read-only diagnostics and head updates
    # have completed.  Existing paths were rejected before model loading.
    if json_mode:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        report_path.parent.mkdir(parents=True, exist_ok=False)
    head_state = {name: value.detach().cpu() for name, value in head.state_dict().items()}
    torch.save(head_state, head_path)
    head_family = "oracle_mlp" if args.variant == "oracle_mlp" else "factorized_residual"
    head_report = {
        "head_family": head_family,
        "type": "OracleMLPResidualHead" if args.variant == "oracle_mlp" else "FactorizedResidualHead",
        "variant": args.variant,
        "rank": int(args.rank),
        "latent_dim": latent_dim,
        "action_embedding_dim": action_dim,
        "query_formula": "[z2, z2-z1, a2]",
        "learned_context_formula": "[z1-z0, z2-z1, a0, a1]",
        "oracle_context_classes": list(ORACLE_CLASS_NAMES),
        "oracle_upper_bound_only": uses_oracle_labels,
        "initial_output_weight_zero": True,
        "checkpoint": str(head_path.absolute()),
        "checkpoint_sha256": _sha256(head_path),
        "trainable_parameter_count": int(head.trainable_parameter_count),
    }
    if args.variant == "oracle_mlp":
        head_report["hidden_dim"] = int(args.mlp_hidden_dim)
    report = {
        "schema_version": 1,
        "status": "completed_training_and_development_only",
        "development_only": True,
        "public_test_opened": False,
        "public_test_read": False,
        "public_test_scored": False,
        "head_family": head_family,
        "variant": args.variant,
        "rank": int(args.rank),
        "seed": int(args.seed),
        "device": str(device),
        "steps": int(args.steps),
        "release": release_receipt,
        "base_model": {
            **_json_safe(source_receipt),
            "checkpoint_role": "frozen_seed3073_LeWM",
            "all_parameters_frozen": True,
            "trainable_parameter_count": 0,
        },
        "head": head_report,
        "parameter_counts": counts,
        "training": {
            "batch_size_requested": int(args.batch_size),
            "batch_size_effective": int(effective_batch_size),
            "complete_four_row_twin_groups": True,
            "loss": "existing_paired_center_response_loss",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
        },
        "gradient_bootstrap_norms": gradient_bootstrap,
        "target_paired_response_svd": svd,
        "development_ridge_probes": ridge,
        "fixed_snapshot_steps": [int(step) for step in snapshot_steps],
        "step0": snapshots[0],
        "snapshots": snapshots,
        "trace": trace,
    }
    if report_path.exists():
        # The head was just created; a pre-existing report would violate the
        # create-once contract and must never be replaced.
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path), "development_only": True}, indent=2))


if __name__ == "__main__":
    main()
