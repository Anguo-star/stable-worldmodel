#!/usr/bin/env python3
"""Run a bounded mechanism pilot for narrowly targeted LeWM objectives.

This is not a benchmark confirmation run.  It reuses the frozen tiny
History-3 training protocol and native optimizer schedule for a bounded
repeated-data run (256 CPU steps by default), solely to test causal
predictions made by the checkpoint and autograd diagnostics:

* ``native``: prediction MSE + 0.09 * SIGReg.
* ``target_detach``: detach the prediction target, retaining 0.09 * SIGReg.
* ``lewm_plus_std``: native LeWM plus PLDM's std loss at its configured
  weight 18.
* ``lewm_plus_std_cov``: native LeWM plus PLDM's std/covariance pair.
* ``pldm_active``: the exact active PLDM model-loss terms (IDM has weight 0).
* ``sigreg_0p3/0p9/2p05``: prediction MSE with a controlled SIGReg sweep.
* ``underdispersion_sigreg_0p45/0p75``: replace SIGReg's two-sided real
  characteristic-function residual by its positive part, retaining the same
  Gaussian target, projection sketch, and one-regularizer objective.
* ``asymmetric_sigreg_g0p25_w0p80``: retain 25% of the overdispersion
  pressure while prioritizing underdispersion, at a gradient-calibrated
  regularizer weight of 0.80.
* ``temporally_centered_sigreg_0p09/0p60/0p79``: apply the unchanged SIGReg
  statistic to each clip's temporally centered residuals instead of its
  latent marginals, with native shuffled batches and no pair metadata.  The
  second weight matches the Encoder gradient budget of 0.90 native SIGReg
  on the exact first batch.
* ``visreg_0p09``: replace SIGReg with the pinned official VISReg loss.
* ``paired_native``: native LeWM on a visible-condition paired batch order.
* ``conditional_sigreg_0p09``: replace each rule-varying marginal sketch by
  a matched Haar high-pass sketch, retaining one 0.09-weight regularizer.
* ``conditional_full_haar_0p09``: negative ablation that retains both the
  matched Haar low-pass and high-pass rows in one marginal sketch.

All variants start from the same checkpoint and see the same shuffled sample
indices.  Frozen door-rule geometry and rule switching are measured at a
predeclared set of optimizer steps without online environment calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

import analyze_checkpoint_geometry as geometry
import analyze_lewm_module_swaps as module_swaps


VARIANTS = (
    "native",
    "target_detach",
    "lewm_plus_std",
    "lewm_plus_std_cov",
    "pldm_active",
    "sigreg_0p3",
    "sigreg_0p9",
    "sigreg_2p05",
    "underdispersion_sigreg_0p45",
    "underdispersion_sigreg_0p75",
    "asymmetric_sigreg_g0p25_w0p80",
    "temporally_centered_sigreg_0p09",
    "temporally_centered_sigreg_0p60",
    "temporally_centered_sigreg_0p79",
    "visreg_0p09",
    "paired_native",
    "conditional_sigreg_0p09",
    "conditional_full_haar_0p09",
)
SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 1.0e-3
GRADIENT_CLIP_NORM = 1.0
SIGREG_WEIGHT = 0.09
SIGREG_VARIANT_WEIGHTS = {
    "sigreg_0p3": 0.3,
    "sigreg_0p9": 0.9,
    "sigreg_2p05": 2.05,
}
UNDERDISPERSION_SIGREG_WEIGHTS = {
    "underdispersion_sigreg_0p45": 0.45,
    "underdispersion_sigreg_0p75": 0.75,
}
ASYMMETRIC_SIGREG_CONFIGS = {
    "asymmetric_sigreg_g0p25_w0p80": {
        "overdispersion_weight": 0.25,
        "loss_weight": 0.80,
    },
}
TEMPORALLY_CENTERED_SIGREG_WEIGHTS = {
    "temporally_centered_sigreg_0p09": 0.09,
    "temporally_centered_sigreg_0p60": 0.60,
    "temporally_centered_sigreg_0p79": 0.79,
}
VISREG_WEIGHT = 0.09
STD_WEIGHT = 18.0
STD_T_WEIGHT = 0.7
COV_WEIGHT = 12.0
TEMP_ALIGN_WEIGHT = 0.2
WARMUP_STEPS = 10
SCHEDULER_MAX_STEPS = 1024


class IndexedDataset:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        sample["__contextworld_index__"] = int(index)
        return sample


class CompleteHaarSIGReg(torch.nn.Module):
    """Negative ablation: apply a complete Haar transform before SIGReg."""

    def __init__(self, *, knots: int, num_proj: int) -> None:
        super().__init__()
        from stable_worldmodel.wm.loss import ConditionalSIGReg, SIGReg

        self.native = SIGReg(knots=knots, num_proj=num_proj)
        self._validate_pairs = ConditionalSIGReg._validate_pairs

    def forward(self, proj, *, pairs=None, active=None):
        if pairs is None:
            return self.native(proj)
        pairs = pairs.to(device=proj.device)
        active = active.to(device=proj.device)
        self._validate_pairs(proj, pairs, active)
        if pairs.numel() == 0 or not bool(active.any()):
            return self.native(proj)

        projections = torch.randn(
            proj.size(-1),
            self.native.num_proj,
            device=proj.device,
        )
        projections = projections.div_(projections.norm(p=2, dim=0))
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        rows = []
        for time_index in range(proj.size(0)):
            selected = active[time_index]
            if not bool(selected.any()):
                population = proj[time_index]
            else:
                selected_pairs = pairs[selected]
                pair_sums = (
                    proj[time_index, selected_pairs[:, 0]]
                    + proj[time_index, selected_pairs[:, 1]]
                ) * inverse_sqrt_two
                pair_differences = (
                    proj[time_index, selected_pairs[:, 0]]
                    - proj[time_index, selected_pairs[:, 1]]
                ) * inverse_sqrt_two
                signs = torch.empty(
                    pair_differences.size(0),
                    1,
                    device=pair_differences.device,
                    dtype=pair_differences.dtype,
                )
                signs.bernoulli_(0.5).mul_(2.0).sub_(1.0)
                population = proj[time_index].clone()
                population[selected_pairs[:, 0]] = pair_sums
                population[selected_pairs[:, 1]] = (
                    pair_differences * signs
                )
            rows.append(
                self.native._projected_statistic(
                    population.unsqueeze(0),
                    projections,
                ).mean()
            )
        return torch.stack(rows).mean()


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def visible_condition_pairs(
    dataset: list[dict[str, Any]],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Pair samples using only the first visible frame and action sequence."""

    condition_groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset):
        condition = _tensor_digest(sample["pixels"][0], sample["action"])
        condition_groups.setdefault(condition, []).append(index)

    pairs: list[tuple[int, int]] = []
    multiplicity: dict[int, int] = {}
    active_counts = torch.zeros(4, dtype=torch.long)
    for condition, indices in sorted(condition_groups.items()):
        future_groups: dict[str, list[int]] = {}
        for index in indices:
            future = _tensor_digest(dataset[index]["pixels"][-1])
            future_groups.setdefault(future, []).append(index)
        if len(future_groups) != 2:
            raise RuntimeError(
                "A visible-condition group must contain exactly two future "
                f"outcomes: condition={condition}, outcomes="
                f"{len(future_groups)}"
            )
        outcomes = [
            sorted(values)
            for _, values in sorted(future_groups.items())
        ]
        if len(outcomes[0]) != len(outcomes[1]):
            raise RuntimeError(
                "Paired future outcomes have unequal multiplicity: "
                f"condition={condition}, sizes="
                f"{[len(values) for values in outcomes]}"
            )
        multiplicity[len(indices)] = multiplicity.get(len(indices), 0) + 1
        for left, right in zip(*outcomes):
            left_sample = dataset[left]
            right_sample = dataset[right]
            if not torch.equal(
                left_sample["pixels"][0], right_sample["pixels"][0]
            ):
                raise RuntimeError("Visible pair has unequal condition pixels")
            if not torch.equal(
                left_sample["action"], right_sample["action"]
            ):
                raise RuntimeError("Visible pair has unequal action sequence")
            differences = (
                left_sample["pixels"] != right_sample["pixels"]
            ).flatten(1).any(dim=1)
            if not bool(differences[-1]):
                raise RuntimeError("Visible pair has no distinct future")
            active_counts += differences.to(dtype=torch.long)
            pairs.append((left, right))

    flattened = [index for pair in pairs for index in pair]
    if sorted(flattened) != list(range(len(dataset))):
        raise RuntimeError(
            "Visible-condition pairing must cover every sample exactly once"
        )
    return pairs, {
        "source": "model_visible_first_frame_pixels_plus_full_action_sequence",
        "uses_rule_labels": False,
        "uses_pair_id": False,
        "condition_groups": len(condition_groups),
        "sample_pairs": len(pairs),
        "condition_group_size_histogram": {
            str(size): count for size, count in sorted(multiplicity.items())
        },
        "active_pair_counts_by_time": [
            int(value) for value in active_counts
        ],
        "covers_every_sample_once": True,
    }


class VisibleConditionPairBatchSampler:
    """Shuffle visible-condition pairs while keeping each pair adjacent."""

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("paired batch_size must be a positive even value")
        self.pairs, self.audit = visible_condition_pairs(dataset)
        self.pairs_per_batch = batch_size // 2
        if len(self.pairs) % self.pairs_per_batch:
            raise ValueError(
                "visible-condition pair count must divide evenly into batches"
            )
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        order = torch.randperm(
            len(self.pairs), generator=self.generator
        ).tolist()
        for start in range(0, len(order), self.pairs_per_batch):
            batch = []
            for pair_index in order[start : start + self.pairs_per_batch]:
                batch.extend(self.pairs[pair_index])
            yield batch

    def __len__(self) -> int:
        return len(self.pairs) // self.pairs_per_batch


def materialize_training_dataset(
    dataset: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode and transform the bounded tiny dataset exactly once."""

    samples = [dict(dataset[index]) for index in range(len(dataset))]
    tensor_bytes = sum(
        value.numel() * value.element_size()
        for sample in samples
        for value in sample.values()
        if torch.is_tensor(value)
    )
    return samples, {
        "enabled": True,
        "samples": len(samples),
        "tensor_bytes": tensor_bytes,
        "purpose": (
            "avoid repeatedly decoding the same PNG-backed tiny samples "
            "across logical epochs"
        ),
    }


def build_tiny_training_dataset(
    *,
    stable_worldmodel: Any,
    contextworld_repo: Path,
    stable_commit: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
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
    metadata = {
        "benchmark_config": str(benchmark_config),
        "benchmark_config_sha256": geometry.file_sha256(benchmark_config),
        "model_id": "H3_Passage_TinyPairedOverfit",
        "epoch_size": len(grouped.train),
        "validation_epoch_size": len(grouped.val),
        "raw_train_clips": grouped.metadata["groups"][
            "passage_tiny_overfit"
        ]["train_clips_raw"],
        "paired_collection_audit_passed": grouped.metadata[
            "paired_collection_audit"
        ]["passed"],
        "training_exclusion_audit_passed": grouped.metadata[
            "training_exclusion_audit"
        ]["passed"],
    }
    return grouped.train, metadata


def make_loader(
    dataset: Any,
    *,
    seed: int,
    paired: bool = False,
) -> torch.utils.data.DataLoader:
    if paired:
        batch_sampler = VisibleConditionPairBatchSampler(
            dataset,
            batch_size=16,
            seed=seed,
        )
        return torch.utils.data.DataLoader(
            IndexedDataset(dataset),
            batch_sampler=batch_sampler,
            num_workers=0,
        )
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        IndexedDataset(dataset),
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )


def native_components(
    model: Any,
    batch: dict[str, Any],
    *,
    marginal_regularizer: Any,
    marginal_regularizer_name: str,
    pldm: Any,
) -> dict[str, Any]:
    pixels = batch["pixels"].to(
        device=next(model.parameters()).device,
        dtype=next(model.encoder.parameters()).dtype,
    )
    actions = torch.nan_to_num(batch["action"], 0.0).to(
        device=next(model.parameters()).device,
        dtype=next(model.action_encoder.parameters()).dtype,
    )
    output = model.encode({"pixels": pixels, "action": actions})
    embeddings = output["emb"]
    action_embeddings = output["act_emb"]
    targets = embeddings[:, 1:]
    predictions = model.predict(
        embeddings[:, :3],
        action_embeddings[:, :3],
    )
    if predictions.shape != targets.shape:
        raise RuntimeError(
            f"Unexpected prediction/target shape: "
            f"{predictions.shape} vs {targets.shape}"
        )
    prediction_loss = torch.mean(torch.square(predictions - targets))
    target_detached_prediction_loss = torch.mean(
        torch.square(predictions - targets.detach())
    )
    pldm_terms = pldm(embeddings)
    components = {
        "pred_loss": prediction_loss,
        "target_detached_pred_loss": target_detached_prediction_loss,
        "std_loss": pldm_terms["std_loss"],
        "std_t_loss": pldm_terms["std_t_loss"],
        "cov_loss": pldm_terms["cov_loss"],
        "temp_align_loss": pldm_terms["temp_align_loss"],
    }
    if marginal_regularizer_name == "conditional_sigreg":
        if embeddings.size(0) % 2:
            raise RuntimeError(
                "Conditional SIGReg requires an even paired batch"
            )
        pairs = torch.arange(
            embeddings.size(0), device=embeddings.device
        ).view(-1, 2)
        pair_pixels = pixels.view(
            embeddings.size(0) // 2,
            2,
            *pixels.shape[1:],
        )
        pair_actions = actions.view(
            embeddings.size(0) // 2,
            2,
            *actions.shape[1:],
        )
        if not bool(torch.eq(pair_pixels[:, 0, 0], pair_pixels[:, 1, 0]).all()):
            raise RuntimeError(
                "Conditional SIGReg batch has unequal visible condition frames"
            )
        if not bool(torch.eq(pair_actions[:, 0], pair_actions[:, 1]).all()):
            raise RuntimeError(
                "Conditional SIGReg batch has unequal action sequences"
            )
        active = torch.ne(
            pair_pixels[:, 0], pair_pixels[:, 1]
        ).flatten(2).any(dim=-1).transpose(0, 1)
        components["conditional_pair_count"] = torch.as_tensor(
            pairs.size(0), device=embeddings.device
        )
        components["conditional_active_fraction"] = active.float().mean()
        components["conditional_sigreg_loss"] = marginal_regularizer(
            embeddings.transpose(0, 1),
            pairs=pairs,
            active=active,
        )
    else:
        components[f"{marginal_regularizer_name}_loss"] = (
            marginal_regularizer(embeddings.transpose(0, 1))
        )
    return components


def objective(
    components: dict[str, Any],
    *,
    variant: str,
) -> Any:
    if variant == "native":
        return (
            components["pred_loss"]
            + SIGREG_WEIGHT * components["sigreg_loss"]
        )
    if variant == "paired_native":
        return (
            components["pred_loss"]
            + SIGREG_WEIGHT * components["sigreg_loss"]
        )
    if variant == "conditional_sigreg_0p09":
        return (
            components["pred_loss"]
            + SIGREG_WEIGHT * components["conditional_sigreg_loss"]
        )
    if variant == "conditional_full_haar_0p09":
        return (
            components["pred_loss"]
            + SIGREG_WEIGHT * components["conditional_sigreg_loss"]
        )
    if variant in TEMPORALLY_CENTERED_SIGREG_WEIGHTS:
        return (
            components["pred_loss"]
            + TEMPORALLY_CENTERED_SIGREG_WEIGHTS[variant]
            * components["temporally_centered_sigreg_loss"]
        )
    if variant == "target_detach":
        return (
            components["target_detached_pred_loss"]
            + SIGREG_WEIGHT * components["sigreg_loss"]
        )
    if variant == "lewm_plus_std":
        return (
            components["pred_loss"]
            + SIGREG_WEIGHT * components["sigreg_loss"]
            + STD_WEIGHT * components["std_loss"]
        )
    if variant == "lewm_plus_std_cov":
        return (
            components["pred_loss"]
            + SIGREG_WEIGHT * components["sigreg_loss"]
            + STD_WEIGHT * components["std_loss"]
            + COV_WEIGHT * components["cov_loss"]
        )
    if variant == "pldm_active":
        return (
            components["pred_loss"]
            + STD_WEIGHT * components["std_loss"]
            + STD_T_WEIGHT * components["std_t_loss"]
            + COV_WEIGHT * components["cov_loss"]
            + TEMP_ALIGN_WEIGHT * components["temp_align_loss"]
        )
    if variant in SIGREG_VARIANT_WEIGHTS:
        return (
            components["pred_loss"]
            + SIGREG_VARIANT_WEIGHTS[variant] * components["sigreg_loss"]
        )
    if variant in UNDERDISPERSION_SIGREG_WEIGHTS:
        return (
            components["pred_loss"]
            + UNDERDISPERSION_SIGREG_WEIGHTS[variant]
            * components["underdispersion_sigreg_loss"]
        )
    if variant in ASYMMETRIC_SIGREG_CONFIGS:
        return (
            components["pred_loss"]
            + ASYMMETRIC_SIGREG_CONFIGS[variant]["loss_weight"]
            * components["asymmetric_sigreg_loss"]
        )
    if variant == "visreg_0p09":
        return (
            components["pred_loss"]
            + VISREG_WEIGHT * components["visreg_loss"]
        )
    raise ValueError(f"Unsupported variant: {variant}")


def diagnostic_snapshot(
    adapter: Any,
    frozen: geometry.FrozenBatch,
    actions: dict[str, Any],
    *,
    step: int,
    batch_size: int,
    original_state: dict[str, Any],
    parameter_names: set[str],
) -> dict[str, Any]:
    model = adapter.model
    was_training = model.training
    model.eval()
    cache = module_swaps.representation_cache(
        adapter, frozen, batch_size=batch_size
    )
    predictions = module_swaps.predict_from_cached_histories(
        adapter,
        cache["projected_histories"],
        actions,
        batch_size=batch_size,
    )
    targets = cache["projected_targets"]
    current_state = model.state_dict()
    drift = geometry.relative_parameter_drift(
        current_state=current_state,
        original_state=original_state,
        parameter_names=parameter_names,
    )
    row = {
        "optimizer_step": step,
        "target_geometry": module_swaps.target_geometry(cache),
        "prediction": geometry.prediction_summary(
            predicted_passable_history=predictions["observed_passable"],
            predicted_blocked_history=predictions["observed_blocked"],
            predicted_no_attempt_history=predictions[
                "did_not_attempt_crossing"
            ],
            target_passable=targets["passable"],
            target_blocked=targets["blocked"],
        ),
        "relative_parameter_drift_from_initialization": drift,
        "batch_norm": geometry.batch_norm_summary(
            current_state, original_state
        ),
    }
    model.train(was_training)
    return row


def run_variant(
    adapter: Any,
    *,
    variant: str,
    dataset: Any,
    frozen: geometry.FrozenBatch,
    frozen_actions: dict[str, Any],
    initial_state: dict[str, Any],
    parameter_names: set[str],
    seed: int,
    max_steps: int,
    batch_size: int,
) -> dict[str, Any]:
    from stable_pretraining.optim.lr_scheduler import (
        LinearWarmupCosineAnnealingLR,
    )
    from stable_worldmodel.wm.loss import (
        ConditionalSIGReg,
        PLDMLoss,
        SIGReg,
        TemporallyCenteredSIGReg,
        VISRegLoss,
    )

    model = adapter.model
    model.load_state_dict(initial_state, strict=True)
    model.requires_grad_(True)
    model.train()
    torch.manual_seed(seed)
    if variant == "visreg_0p09":
        marginal_regularizer_name = "visreg"
        marginal_regularizer = VISRegLoss(
            num_projections=1024,
            lambda_scale=1.0,
            lambda_shape=1.0,
            lambda_center=1.0,
        ).to(adapter.device)
    elif variant == "conditional_sigreg_0p09":
        marginal_regularizer_name = "conditional_sigreg"
        marginal_regularizer = ConditionalSIGReg(
            knots=17,
            num_proj=1024,
        ).to(adapter.device)
    elif variant == "conditional_full_haar_0p09":
        marginal_regularizer_name = "conditional_sigreg"
        marginal_regularizer = CompleteHaarSIGReg(
            knots=17,
            num_proj=1024,
        ).to(adapter.device)
    elif variant in UNDERDISPERSION_SIGREG_WEIGHTS:
        marginal_regularizer_name = "underdispersion_sigreg"
        marginal_regularizer = SIGReg(
            knots=17,
            num_proj=1024,
            overdispersion_weight=0.0,
        ).to(adapter.device)
    elif variant in ASYMMETRIC_SIGREG_CONFIGS:
        marginal_regularizer_name = "asymmetric_sigreg"
        marginal_regularizer = SIGReg(
            knots=17,
            num_proj=1024,
            overdispersion_weight=(
                ASYMMETRIC_SIGREG_CONFIGS[variant][
                    "overdispersion_weight"
                ]
            ),
        ).to(adapter.device)
    elif variant in TEMPORALLY_CENTERED_SIGREG_WEIGHTS:
        marginal_regularizer_name = "temporally_centered_sigreg"
        marginal_regularizer = TemporallyCenteredSIGReg(
            knots=17,
            num_proj=1024,
        ).to(adapter.device)
    else:
        marginal_regularizer_name = "sigreg"
        marginal_regularizer = SIGReg(knots=17, num_proj=1024).to(
            adapter.device
        )
    pldm = PLDMLoss().to(adapter.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        max_steps=SCHEDULER_MAX_STEPS,
        warmup_start_lr=0.0,
        eta_min=0.0,
    )
    paired_loader = variant in {
        "paired_native",
        "conditional_sigreg_0p09",
        "conditional_full_haar_0p09",
    }
    loader = make_loader(dataset, seed=seed, paired=paired_loader)
    snapshot_steps = {
        step for step in SNAPSHOT_STEPS if step <= max_steps
    } | {max_steps}
    snapshots = [
        diagnostic_snapshot(
            adapter,
            frozen,
            frozen_actions,
            step=0,
            batch_size=batch_size,
            original_state=initial_state,
            parameter_names=parameter_names,
        )
    ]
    trace: list[dict[str, Any]] = []
    first_batch_indices: list[int] | None = None
    step = 0
    epoch = 0
    while step < max_steps:
        epoch += 1
        for batch in loader:
            indices = [
                int(value)
                for value in batch.pop("__contextworld_index__")
            ]
            if first_batch_indices is None:
                first_batch_indices = indices
            components = native_components(
                model,
                batch,
                marginal_regularizer=marginal_regularizer,
                marginal_regularizer_name=marginal_regularizer_name,
                pldm=pldm,
            )
            loss = objective(components, variant=variant)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            step += 1

            if step == 1 or step in snapshot_steps:
                trace.append(
                    {
                        "optimizer_step": step,
                        "logical_epoch": epoch,
                        "learning_rate_used": learning_rate,
                        "gradient_norm_before_clip": float(gradient_norm),
                        "loss": float(loss.detach()),
                        "components": {
                            name: float(value.detach())
                            for name, value in components.items()
                        },
                    }
                )
            if step in snapshot_steps:
                snapshots.append(
                    diagnostic_snapshot(
                        adapter,
                        frozen,
                        frozen_actions,
                        step=step,
                        batch_size=batch_size,
                        original_state=initial_state,
                        parameter_names=parameter_names,
                    )
                )
                print(
                    f"[{variant}] step={step} "
                    f"raw_ratio="
                    f"{snapshots[-1]['target_geometry']['raw_encoder']['paired_to_unrelated_ratio']:.6f} "
                    f"projected_ratio="
                    f"{snapshots[-1]['target_geometry']['prediction_space']['paired_to_unrelated_ratio']:.6f} "
                    f"switch="
                    f"{snapshots[-1]['prediction']['correct_rule_switch_rate']:.3f}",
                    flush=True,
                )
            if step >= max_steps:
                break

    return {
        "variant": variant,
        "loader_mode": (
            "visible_condition_paired"
            if paired_loader
            else "native_sample_shuffle"
        ),
        "optimizer_steps": step,
        "logical_epochs": epoch,
        "first_batch_indices": first_batch_indices,
        "loss_trace": trace,
        "snapshots": snapshots,
        "final_model_state_hash": adapter.frozen_state_hash(),
    }


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
            "data. This is separate from --stable-ref so a new objective "
            "implementation can be audited against unchanged data."
        ),
    )
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--normalizer", type=Path, default=None)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"Comma-separated subset of: {', '.join(VARIANTS)}",
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--torch-num-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    variants = tuple(
        value.strip() for value in args.variants.split(",") if value.strip()
    )
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("--variants must be a non-empty unique list")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.torch_num_threads <= 0:
        raise ValueError("--torch-num-threads must be positive")
    torch.set_num_threads(args.torch_num_threads)
    torch.set_num_interop_threads(1)

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
    initial_checkpoint = (
        args.initial_checkpoint.expanduser().resolve()
        if args.initial_checkpoint is not None
        else artifact_root
        / "training/runs/checkpoints/h3_origheldout_s3072/"
        "weights_final_step_6420.pt"
    )
    required = [
        contextworld_repo,
        stable_repo,
        catalog,
        normalizer,
        initial_checkpoint,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    frozen = geometry.load_frozen_batch(catalog, artifact_root)
    adapter = geometry._stable_adapter(
        checkpoint=initial_checkpoint,
        training_method="lewm",
        contextworld_repo=contextworld_repo,
        stable_repo=stable_repo,
        stable_ref=args.stable_ref,
        normalizer=normalizer,
        device=args.device,
    )
    import stable_worldmodel as stable_worldmodel_module

    dataset, dataset_metadata = build_tiny_training_dataset(
        stable_worldmodel=stable_worldmodel_module,
        contextworld_repo=contextworld_repo,
        stable_commit=args.data_stable_ref,
        seed=args.seed,
    )
    print(
        f"Materializing {len(dataset)} transformed training samples once",
        flush=True,
    )
    dataset, materialization = materialize_training_dataset(dataset)
    dataset_metadata["materialization"] = materialization
    initial_state = {
        name: value.to(adapter.device)
        for name, value in geometry._load_state(initial_checkpoint).items()
    }
    parameter_names = geometry._parameter_name_set(adapter)
    frozen_actions = module_swaps.normalized_actions(adapter, frozen)

    results = []
    for index, variant in enumerate(variants, start=1):
        print(
            f"[{index}/{len(variants)}] mechanism pilot: {variant}",
            flush=True,
        )
        results.append(
            run_variant(
                adapter,
                variant=variant,
                dataset=dataset,
                frozen=frozen,
                frozen_actions=frozen_actions,
                initial_state=initial_state,
                parameter_names=parameter_names,
                seed=args.seed,
                max_steps=args.max_steps,
                batch_size=args.eval_batch_size,
            )
        )

    first_batches_by_loader: dict[str, set[tuple[int, ...]]] = {}
    for row in results:
        first_batches_by_loader.setdefault(
            row["loader_mode"], set()
        ).add(tuple(row["first_batch_indices"] or []))
    unequal_loader_modes = {
        mode: batches
        for mode, batches in first_batches_by_loader.items()
        if len(batches) != 1
    }
    if unequal_loader_modes:
        raise RuntimeError(
            "Objective variants using the same loader mode did not receive "
            f"the same first batch: {unequal_loader_modes}"
        )
    _, visible_pairing_audit = visible_condition_pairs(dataset)
    payload = {
        "schema_version": 1,
        "status": "bounded_mechanism_pilot_not_benchmark_confirmation",
        "question": (
            "target detach 是否足以阻止 LeWM 局部收缩，以及 PLDM 的 std "
            "防坍缩项是否足以改变该训练路径？"
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
            "initial_checkpoint": str(initial_checkpoint),
            "initial_checkpoint_sha256": geometry.file_sha256(
                initial_checkpoint
            ),
            "device": args.device,
            "precision": "float32",
            "checkpoints_written": False,
            "environment_calls": 0,
        },
        "training_data": dataset_metadata,
        "training_contract": {
            "seed": args.seed,
            "batch_size": 16,
            "max_steps": args.max_steps,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "warmup_steps": WARMUP_STEPS,
            "scheduler_max_steps": SCHEDULER_MAX_STEPS,
            "snapshot_steps": sorted(
                {
                    step
                    for step in SNAPSHOT_STEPS
                    if step <= args.max_steps
                }
                | {args.max_steps}
            ),
            "first_batch_indices_equal_within_loader_mode": True,
            "loader_modes": sorted(first_batches_by_loader),
            "visible_condition_pairing": visible_pairing_audit,
            "torch_num_threads": args.torch_num_threads,
            "torch_num_interop_threads": 1,
        },
        "objective_contract": {
            "native": "pred_loss + 0.09 * sigreg_loss",
            "target_detach": (
                "pred_loss(prediction, target.detach()) + "
                "0.09 * sigreg_loss"
            ),
            "lewm_plus_std": (
                "pred_loss + 0.09 * sigreg_loss + 18 * std_loss"
            ),
            "lewm_plus_std_cov": (
                "pred_loss + 0.09 * sigreg_loss + 18 * std_loss + "
                "12 * cov_loss"
            ),
            "pldm_active": (
                "pred_loss + 18 * std_loss + 0.7 * std_t_loss + "
                "12 * cov_loss + 0.2 * temp_align_loss"
            ),
            "sigreg_0p3": "pred_loss + 0.3 * sigreg_loss",
            "sigreg_0p9": "pred_loss + 0.9 * sigreg_loss",
            "sigreg_2p05": "pred_loss + 2.05 * sigreg_loss",
            "underdispersion_sigreg_0p45": (
                "pred_loss + 0.45 * underdispersion_sigreg_loss"
            ),
            "underdispersion_sigreg_0p75": (
                "pred_loss + 0.75 * underdispersion_sigreg_loss"
            ),
            "asymmetric_sigreg_g0p25_w0p80": (
                "pred_loss + 0.80 * asymmetric_sigreg_loss; "
                "overdispersion_weight=0.25"
            ),
            "temporally_centered_sigreg_0p09": (
                "pred_loss + 0.09 * temporally_centered_sigreg_loss; "
                "residual = embedding - per-clip temporal mean"
            ),
            "temporally_centered_sigreg_0p60": (
                "pred_loss + 0.60 * temporally_centered_sigreg_loss; "
                "residual = embedding - per-clip temporal mean; "
                "weight exceeds the exact-first-batch zero-crossing for "
                "all three audited projection streams"
            ),
            "temporally_centered_sigreg_0p79": (
                "pred_loss + 0.79 * temporally_centered_sigreg_loss; "
                "residual = embedding - per-clip temporal mean; "
                "weight matches the exact-first-batch Encoder gradient "
                "budget of 0.90 native SIGReg"
            ),
            "visreg_0p09": "pred_loss + 0.09 * visreg_loss",
            "paired_native": (
                "pred_loss + 0.09 * sigreg_loss; visible-condition "
                "paired batch order only"
            ),
            "conditional_sigreg_0p09": (
                "pred_loss + 0.09 * conditional_sigreg_loss; selected "
                "time marginals are replaced by visible-condition Haar "
                "high-pass contrasts"
            ),
            "conditional_full_haar_0p09": (
                "negative ablation: pred_loss + 0.09 * one SIGReg "
                "statistic after a complete visible-condition Haar "
                "low-pass/high-pass transform"
            ),
        },
        "limitations": [
            (
                "single-device float32 rather than the formal 8-GPU "
                "bf16-mixed topology"
            ),
            "tiny 160-clip repeated-training diagnostic rather than formal data",
            "single seed and a bounded repeated-data optimizer budget",
            "mechanism evidence only; not a registered benchmark pass",
        ],
        "variants": results,
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
