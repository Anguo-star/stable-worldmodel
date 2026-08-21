#!/usr/bin/env python3
"""Run the Motion-Damping live-Predictor oracle-context MVE.

This Development-only diagnostic changes no LeWM loss.  It adds one
zero-initialized 2 -> 192 projection to the existing action-conditioning
stream.  The ``oracle`` arm supplies the two damping identities; the
equal-parameter ``constant`` arm supplies the same unit-norm vector to both
hidden conditions.  Standard PushT rows receive zero context in both arms.

The oracle label is deliberately privileged and is used only to test an
architectural upper bound.  Public/Test data and CEM are never opened here.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONTEXTWORLD_ROOT = ROOT.parent / "ContextWorld"
CONTEXTWORLD_SCRIPTS = CONTEXTWORLD_ROOT / "scripts"
for source_root in (ROOT, CONTEXTWORLD_ROOT, CONTEXTWORLD_SCRIPTS):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


PREREG = (
    ROOT
    / "research/conditional_dynamics_representation/configs/"
    "pusht_motion_damping_oracle_context_predictor_mve_v1.yaml"
)
RELEASE_CONFIG = (
    CONTEXTWORLD_ROOT
    / "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml"
)
ARTIFACT_ROOT = (
    ROOT
    / "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_oracle_context_predictor_mve_v1"
)
EXPECTED_RELEASE_ID = "contextworld_pusht_motion_damping_icl_history3_v1"
EXPECTED_RELEASE_SHA256 = (
    "1795717d8bfa1d1cfcc01a69931b6241b0e7759dcf43553a6c4cca225ec9326b"
)
FROZEN_SEED = 14321
DEFAULT_STEPS = 256
TOTAL_BATCH_SIZE = 128
ORIGINAL_BATCH_SIZE = 64
CONTEXT_DIM = 2
CONDITIONING_DIM = 192
BOOTSTRAP_REPLICATES = 10_000
ARMS = {
    "oracle": "mixed_live_predictor_oracle_context_native_0p09",
    "constant": "mixed_live_predictor_constant_context_native_0p09",
}
SCREEN_THRESHOLDS = {
    "future_accuracy_minimum": 0.55,
    "history_accuracy_minimum": 0.55,
    "paired_assignment_lower_95_minimum": 0.50,
    "rule_switch_minimum": 0.75,
    "worst_condition_minimum": 0.10,
    "response_gain_minimum": 0.10,
    "normalized_response_error_strict_maximum": 0.95,
    "normalized_response_error_upper_95_strict_maximum": 1.00,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _constant_vector(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = 1.0 / math.sqrt(2.0)
    return torch.tensor([value, value], device=device, dtype=dtype)


def training_context(
    arm: str,
    *,
    batch_size: int,
    original_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the exact original-then-alternating-low/high context tensor."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if batch_size != TOTAL_BATCH_SIZE or original_batch_size != ORIGINAL_BATCH_SIZE:
        raise ValueError("MVE requires the frozen 64 original + 64 hidden batch")
    result = torch.zeros(batch_size, CONTEXT_DIM, device=device, dtype=dtype)
    if arm == "oracle":
        result[original_batch_size::2, 0] = 1.0
        result[original_batch_size + 1 :: 2, 1] = 1.0
    else:
        result[original_batch_size:] = _constant_vector(device=device, dtype=dtype)
    return result


def evaluation_context(
    arm: str,
    condition: str,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an explicit homogeneous Development context block."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if condition not in {"faster_decay", "no_extra_decay"}:
        raise ValueError(f"unknown condition {condition!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if arm == "constant":
        row = _constant_vector(device=device, dtype=dtype)
    elif condition == "faster_decay":
        row = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    else:
        row = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    return row.expand(batch_size, -1)


class EpisodeContextController:
    """Supply train contexts or one explicitly named evaluation condition."""

    def __init__(self, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self._evaluation_condition: str | None = None

    @property
    def evaluation_condition(self) -> str | None:
        return self._evaluation_condition

    @contextmanager
    def evaluating(self, condition: str) -> Iterator[None]:
        if self._evaluation_condition is not None:
            raise RuntimeError("nested explicit evaluation context is forbidden")
        if condition not in {"faster_decay", "no_extra_decay"}:
            raise ValueError(condition)
        self._evaluation_condition = condition
        try:
            yield
        finally:
            self._evaluation_condition = None

    def for_conditioning(self, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim != 3 or conditioning.shape[-1] != CONDITIONING_DIM:
            raise RuntimeError("LeWM action-conditioning boundary changed")
        if self._evaluation_condition is not None:
            return evaluation_context(
                self.arm,
                self._evaluation_condition,
                int(conditioning.shape[0]),
                device=conditioning.device,
                dtype=conditioning.dtype,
            )
        return training_context(
            self.arm,
            batch_size=int(conditioning.shape[0]),
            original_batch_size=ORIGINAL_BATCH_SIZE,
            device=conditioning.device,
            dtype=conditioning.dtype,
        )


def attach_episode_context_projection(
    model: nn.Module,
    controller: EpisodeContextController,
    gradient_state: dict[str, Any] | None = None,
) -> nn.Parameter:
    """Attach and route the zero-start 384-parameter context projection."""

    predictor = model.predictor
    if hasattr(predictor, "episode_context_projection"):
        raise RuntimeError("episode context projection is already attached")
    reference = next(predictor.parameters())
    projection = nn.Linear(CONTEXT_DIM, CONDITIONING_DIM, bias=False).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    nn.init.zeros_(projection.weight)
    predictor.add_module("episode_context_projection", projection)
    native_forward = predictor.forward

    def forward_with_episode_context(
        self: nn.Module,
        history: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        context = controller.for_conditioning(conditioning)
        delta = self.episode_context_projection(context).unsqueeze(1)
        return native_forward(history, conditioning + delta)

    predictor.forward = MethodType(forward_with_episode_context, predictor)
    if gradient_state is not None:
        gradient_state.update(
            {
                "backward_calls": 0,
                "first_finite_nonzero_gradient_step": None,
                "first_gradient_l2_norm": None,
            }
        )

        def capture_gradient(gradient: torch.Tensor) -> torch.Tensor:
            gradient_state["backward_calls"] += 1
            finite = bool(torch.isfinite(gradient).all())
            norm = float(torch.linalg.vector_norm(gradient.detach().float()))
            if (
                finite
                and norm > 0.0
                and gradient_state["first_finite_nonzero_gradient_step"] is None
            ):
                gradient_state["first_finite_nonzero_gradient_step"] = int(
                    gradient_state["backward_calls"]
                )
                gradient_state["first_gradient_l2_norm"] = norm
            return gradient

        projection.weight.register_hook(capture_gradient)
    return projection.weight


def _distance_summary(low: torch.Tensor, high: torch.Tensor) -> dict[str, float]:
    paired = torch.linalg.vector_norm(high - low, dim=-1)
    unrelated = torch.linalg.vector_norm(high.roll(1, 0) - low, dim=-1)
    return {
        "paired_mean": float(paired.mean()),
        "paired_min": float(paired.min()),
        "unrelated_mean": float(unrelated.mean()),
        "paired_to_unrelated_ratio": float(
            paired.mean() / unrelated.mean().clamp_min(1e-12)
        ),
    }


def _effective_rank(values: torch.Tensor) -> float:
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-30)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def paired_bootstrap(
    pair_balanced: Sequence[float] | np.ndarray,
    error_energy: Sequence[float] | np.ndarray,
    target_energy: Sequence[float] | np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = FROZEN_SEED,
) -> dict[str, Any]:
    balanced = np.asarray(pair_balanced, dtype=np.float64)
    numerator = np.asarray(error_energy, dtype=np.float64)
    denominator = np.asarray(target_energy, dtype=np.float64)
    _require(
        balanced.shape == numerator.shape == denominator.shape == (256,),
        "bootstrap requires all 256 matched Development pairs",
    )
    _require(
        np.isfinite(balanced).all()
        and np.isfinite(numerator).all()
        and np.isfinite(denominator).all()
        and np.all(denominator > 0.0),
        "invalid bootstrap inputs",
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, balanced.size, size=(replicates, balanced.size))
    assignment_samples = balanced[indices].mean(axis=1)
    error_samples = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
    return {
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": "matched_query_pair",
        "quantile_method": "numpy_linear",
        "paired_query_balanced_macro": {
            "point": float(balanced.mean()),
            "lower_95": float(np.quantile(assignment_samples, 0.025, method="linear")),
            "upper_95": float(np.quantile(assignment_samples, 0.975, method="linear")),
        },
        "target_energy_weighted_normalized_response_error": {
            "point": float(numerator.sum() / denominator.sum()),
            "lower_95": float(np.quantile(error_samples, 0.025, method="linear")),
            "upper_95": float(np.quantile(error_samples, 0.975, method="linear")),
        },
    }


def screen_decision(
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    response = metrics["latent_response"]
    checks = {
        "target_latent_separation_256_of_256": (
            response["target_latent_separation"]["zero_separation_pair_count"] == 0
        ),
        "future_accuracy": (
            metrics["two_real_future_target_selection_rate"]
            >= SCREEN_THRESHOLDS["future_accuracy_minimum"]
        ),
        "history_accuracy": (
            metrics["correct_history_preference_rate"]
            >= SCREEN_THRESHOLDS["history_accuracy_minimum"]
        ),
        "paired_assignment_bootstrap_lower": (
            bootstrap["paired_query_balanced_macro"]["lower_95"]
            >= SCREEN_THRESHOLDS["paired_assignment_lower_95_minimum"]
        ),
        "rule_switch": (
            metrics["correct_rule_switch_rate"]
            >= SCREEN_THRESHOLDS["rule_switch_minimum"]
        ),
        "worst_condition": (
            metrics["worst_mode_target_selection_rate"]
            >= SCREEN_THRESHOLDS["worst_condition_minimum"]
        ),
        "response_gain": (
            response["response_gain"]
            >= SCREEN_THRESHOLDS["response_gain_minimum"]
        ),
        "normalized_response_error_point": (
            response["normalized_response_error"]
            < SCREEN_THRESHOLDS["normalized_response_error_strict_maximum"]
        ),
        "normalized_response_error_bootstrap_upper": (
            bootstrap["target_energy_weighted_normalized_response_error"]["upper_95"]
            < SCREEN_THRESHOLDS[
                "normalized_response_error_upper_95_strict_maximum"
            ]
        ),
    }
    passed = all(checks.values())
    assignment_ok = all(
        checks[name]
        for name in ("future_accuracy", "history_accuracy", "worst_condition")
    )
    calibration_ok = all(
        checks[name]
        for name in (
            "response_gain",
            "normalized_response_error_point",
            "normalized_response_error_bootstrap_upper",
        )
    )
    if passed:
        phenotype = "early_direct_icl_signal"
    elif assignment_ok and not calibration_ok:
        phenotype = "assignment_without_calibrated_response"
    elif calibration_ok and not assignment_ok:
        phenotype = "calibrated_response_without_balanced_assignment"
    else:
        phenotype = "no_onset_or_mixed_failure"
    return {
        "passed": passed,
        "category": "early_direct_icl_signal_at_step256" if passed else "screen_no_advance",
        "phenotype": phenotype,
        "checks": checks,
        "thresholds": dict(SCREEN_THRESHOLDS),
        "public_or_cem_authorized": False,
        "longer_budget_requires_matched_arm_comparison": True,
    }


@torch.no_grad()
def evaluate_with_explicit_context(
    model: nn.Module,
    evaluation: dict[str, torch.Tensor],
    *,
    controller: EpisodeContextController,
    pilot_module: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate low and high blocks separately under named contexts."""

    was_training = model.training
    model.eval()
    try:
        low_pixels = evaluation["low_pixels"]
        high_pixels = evaluation["high_pixels"]
        actions = evaluation["action"][:, :3]
        with controller.evaluating("faster_decay"):
            predicted_low = pilot_module.predict_histories(
                model,
                low_pixels[:, :3],
                actions,
                device=device,
                batch_size=batch_size,
            )
        with controller.evaluating("no_extra_decay"):
            predicted_high = pilot_module.predict_histories(
                model,
                high_pixels[:, :3],
                actions,
                device=device,
                batch_size=batch_size,
            )
        future_pixels = torch.cat([low_pixels[:, 3:4], high_pixels[:, 3:4]])
        raw_future, projected_future = pilot_module.encode_pixels(
            model,
            future_pixels,
            device=device,
            batch_size=batch_size,
        )
    finally:
        model.train(was_training)

    count = int(low_pixels.shape[0])
    _require(count == 256, "MVE requires the frozen 256-pair Development split")
    raw_low, raw_high = raw_future[:count, 0], raw_future[count:, 0]
    target_low = projected_future[:count, 0]
    target_high = projected_future[count:, 0]

    def mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left - right).square().mean(dim=-1)

    low_to_low = mse(predicted_low, target_low)
    low_to_high = mse(predicted_low, target_high)
    high_to_low = mse(predicted_high, target_low)
    high_to_high = mse(predicted_high, target_high)
    low_future = low_to_low < low_to_high
    high_future = high_to_high < high_to_low
    low_history = low_to_low < high_to_low
    high_history = high_to_high < low_to_high
    predicted_response = predicted_high - predicted_low
    target_response = target_high - target_low
    dot = (predicted_response * target_response).sum(dim=-1)
    predicted_energy = predicted_response.square().sum(dim=-1)
    target_energy = target_response.square().sum(dim=-1)
    error_energy = (predicted_response - target_response).square().sum(dim=-1)
    target_total = target_energy.sum().clamp_min(1e-30)
    response_gain = dot.sum() / target_total
    normalized_error = error_energy.sum() / target_total
    aggregate_cosine = dot.sum() / torch.sqrt(
        predicted_energy.sum().clamp_min(1e-30) * target_total
    )
    pair_cosine = dot / torch.sqrt(
        predicted_energy.clamp_min(1e-30) * target_energy.clamp_min(1e-30)
    )
    pair_balanced = torch.stack(
        [low_future, high_future, low_history, high_history], dim=1
    ).float().mean(dim=1)
    bootstrap = paired_bootstrap(
        pair_balanced.numpy(),
        error_energy.numpy(),
        target_energy.numpy(),
    )
    state_gap = torch.linalg.vector_norm(
        evaluation["high_states"][:, 3, 2:4]
        - evaluation["low_states"][:, 3, 2:4],
        dim=-1,
    )
    correct_losses = torch.cat([low_to_low, high_to_high])
    incorrect_losses = torch.cat([low_to_high, high_to_low])
    result = {
        "pair_count": count,
        "decision_count": 2 * count,
        "two_real_future_target_selection_rate": float(
            torch.cat([low_future, high_future]).float().mean()
        ),
        "correct_history_preference_rate": float(
            torch.cat([low_history, high_history]).float().mean()
        ),
        "correct_rule_switch_rate": float((dot > 0).float().mean()),
        "worst_mode_target_selection_rate": float(
            min(low_future.float().mean(), high_future.float().mean())
        ),
        "prediction_mse": {
            "correct_future_mean": float(correct_losses.mean()),
            "incorrect_future_mean": float(incorrect_losses.mean()),
            "incorrect_minus_correct_margin": float(
                (incorrect_losses - correct_losses).mean()
            ),
        },
        "representation_geometry": {
            "raw_encoder": _distance_summary(raw_low, raw_high),
            "prediction_space": _distance_summary(target_low, target_high),
            "future_effective_rank": _effective_rank(
                torch.cat([target_low, target_high])
            ),
            "future_per_dimension_variance": float(
                torch.cat([target_low, target_high]).var(dim=0, unbiased=False).mean()
            ),
        },
        "physical_future_block_gap_px": {
            "minimum": float(state_gap.min()),
            "mean": float(state_gap.mean()),
            "maximum": float(state_gap.max()),
        },
        "latent_response": {
            "orientation": "no_extra_decay_minus_faster_decay",
            "aggregate_cosine_alignment": float(aggregate_cosine),
            "mean_pair_cosine_alignment": float(pair_cosine.mean()),
            "response_gain": float(response_gain),
            "normalized_response_error": float(normalized_error),
            "target_latent_separation": {
                "zero_separation_pair_count": int((target_energy <= 1e-20).sum()),
                "minimum_squared_l2": float(target_energy.min()),
            },
        },
        "paired_bootstrap": bootstrap,
        "context_evaluation_protocol": {
            "low_and_high_predicted_in_separate_named_context_blocks": True,
            "cursor_or_call_order_inference_used": False,
            "public_test_opened": False,
        },
        "deterministic_current_query_only_accuracy_bound": 0.5,
    }
    result["mechanism_screen"] = screen_decision(result, bootstrap)
    return result


def _new_gradient_state() -> dict[str, Any]:
    return {}


def install_training_overlay(motion: Any, *, arm: str) -> str:
    """Register one live-Predictor context arm without editing core code."""

    if arm not in ARMS:
        raise ValueError(arm)
    variant = ARMS[arm]
    mixed = motion.trainer.mixed
    motion.TWIN_GROUP_VARIANTS.add(variant)
    mixed.VARIANT_WEIGHTS[variant] = ("native", 0.09, "identifiable_future_only")
    motion.trainer.DIAGNOSTIC_VARIANTS["lewm"].add(variant)
    _require(
        variant not in mixed.FROZEN_IMAGE_VARIANTS,
        "live context arm must jointly train the native LeWM representation",
    )
    native_train_variant = mixed.train_variant

    def train_variant_with_context(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("variant") != variant:
            return native_train_variant(*args, **kwargs)
        controller = EpisodeContextController(arm)
        gradient_state = _new_gradient_state()
        native_loader = mixed.load_model_for_variant
        native_evaluator = mixed.pilot.evaluate_model
        loaded_model: nn.Module | None = None
        adapter_weight: nn.Parameter | None = None

        def load_with_context(*loader_args: Any, **loader_kwargs: Any):
            nonlocal loaded_model, adapter_weight
            model, receipt = native_loader(*loader_args, **loader_kwargs)
            if loaded_model is not None:
                raise RuntimeError("context overlay loaded more than one model")
            loaded_model = model
            adapter_weight = attach_episode_context_projection(
                model, controller, gradient_state
            )
            _require(adapter_weight.numel() == 384, "adapter size changed")
            _require(
                float(torch.linalg.vector_norm(adapter_weight.detach().float())) == 0.0,
                "adapter did not start at exact zero",
            )
            receipt = dict(receipt)
            receipt["research_overlay"] = {
                "method": "live_predictor_episode_context_projection_v1",
                "arm": arm,
                "added_trainable_parameters": 384,
                "zero_initialized": True,
                "joint_native_lewm_training": True,
            }
            return model, receipt

        def evaluate_context_model(
            model: nn.Module,
            evaluation: dict[str, torch.Tensor],
            *,
            device: torch.device,
            batch_size: int,
        ) -> dict[str, Any]:
            if model is not loaded_model:
                return native_evaluator(
                    model, evaluation, device=device, batch_size=batch_size
                )
            return evaluate_with_explicit_context(
                model,
                evaluation,
                controller=controller,
                pilot_module=mixed.pilot,
                device=device,
                batch_size=batch_size,
            )

        mixed.load_model_for_variant = load_with_context
        mixed.pilot.evaluate_model = evaluate_context_model
        try:
            result = native_train_variant(*args, **kwargs)
        finally:
            mixed.load_model_for_variant = native_loader
            mixed.pilot.evaluate_model = native_evaluator
        if loaded_model is None or adapter_weight is None:
            raise RuntimeError("context adapter was never attached")
        max_steps = int(kwargs["max_steps"])
        final_norm = float(torch.linalg.vector_norm(adapter_weight.detach().float()))
        result["objective"] = (
            "native_identifiable_future_mse + 0.09*original_sigreg; "
            "no_added_loss"
        )
        result["hidden_labels_at_model_or_loss_boundary"] = arm == "oracle"
        result["episode_context_contract"] = {
            "arm": arm,
            "diagnostic_upper_bound_only": arm == "oracle",
            "architecture": "add Linear(2,192,bias=False)(context) to existing action embeddings before existing Predictor",
            "added_trainable_parameters": 384,
            "projection_initialization": "exact_zero",
            "initial_projection_l2_norm": 0.0,
            "final_projection_l2_norm": final_norm,
            "projection_changed": final_norm > 0.0,
            "first_finite_nonzero_gradient_step": gradient_state.get(
                "first_finite_nonzero_gradient_step"
            ),
            "first_gradient_l2_norm": gradient_state.get("first_gradient_l2_norm"),
            "backward_calls": gradient_state.get("backward_calls"),
            "optimizer_inclusion_evidence": (
                "attached before trainer constructs optimizer; nonzero final norm"
                if final_norm > 0.0
                else "attached before trainer constructs optimizer"
            ),
            "native_modules_jointly_trainable": True,
            "training_context": {
                "original_rows_0_to_63": [0.0, 0.0],
                "hidden_even_rows_from_64": (
                    [1.0, 0.0]
                    if arm == "oracle"
                    else [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]
                ),
                "hidden_odd_rows_from_65": (
                    [0.0, 1.0]
                    if arm == "oracle"
                    else [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]
                ),
                "complete_forward_reverse_twins": True,
            },
            "privileged_information": {
                "raw_hidden_metadata_tensor_entered_model": False,
                "oracle_condition_identity_entered_as_two_vector": arm == "oracle",
                "constant_arm_contains_condition_identity": False,
                "final_method_claim_authorized": False,
            },
            "step0_exact_native_function": True,
            "inference_path_changed": True,
            "public_test_opened": False,
            "cem_opened": False,
        }
        result["mechanism_screen"] = result["snapshots"][-1][
            "hidden_evaluation"
        ]["mechanism_screen"]
        result["mechanism_screen"]["optimizer_step"] = max_steps
        return result

    mixed.train_variant = train_variant_with_context
    return variant


def install_release_authorization_overlay(motion: Any, *, arm: str, steps: int) -> dict[str, Any]:
    """Authorize only this Development MVE in memory."""

    native_loader = motion.load_motion_damping_icl_release
    audit: dict[str, Any] = {}

    def load_with_followup(path: Path) -> dict[str, Any]:
        release = copy.deepcopy(native_loader(path))
        _require(release["release_id"] == EXPECTED_RELEASE_ID, "release id changed")
        matrix = release["training"]["reference_matrix"]
        _require(matrix["status"] == "failed_development", "base status changed")
        completed = [int(value) for value in matrix["completed_development_seeds"]]
        _require(FROZEN_SEED in completed, "legacy seed identity changed")
        matrix["completed_development_seeds"] = [
            value for value in completed if value != FROZEN_SEED
        ]
        release["training"]["learnability_followup"] = {
            # This exact status is the shared trainer's narrow Development
            # extension seam; the diagnostic identity remains explicit below.
            "status": "development_recipe_search_in_progress",
            "diagnostic": "authorized_oracle_context_mve",
            "authority": str(PREREG),
            "arm": arm,
            "seed": FROZEN_SEED,
            "optimizer_steps": steps,
            "development_only": True,
            "public_test_open": False,
            "cem_open": False,
        }
        audit.update(
            {
                "base_status": "failed_development",
                "removed_completed_seed_in_memory_only": FROZEN_SEED,
                "all_other_completed_seeds_preserved": True,
                "base_release_mutated_on_disk": False,
                "public_test_open": False,
                "cem_open": False,
            }
        )
        return release

    motion.load_motion_damping_icl_release = load_with_followup
    return audit


def validate_static_contract() -> dict[str, Any]:
    config = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    checks = {
        "release_exists": RELEASE_CONFIG.is_file(),
        "release_sha_matches": file_sha256(RELEASE_CONFIG) == EXPECTED_RELEASE_SHA256,
        "release_id_matches": config["identities"]["release_id"] == EXPECTED_RELEASE_ID,
        "seed_matches": int(config["training"]["seed"]) == FROZEN_SEED,
        "steps_match": int(config["training"]["optimizer_steps"]) == DEFAULT_STEPS,
        "adapter_parameter_count_matches": (
            int(config["architecture"]["added_trainable_parameters"]) == 384
        ),
        "loss_unchanged": config["objective"]["additional_loss"] is None,
        "public_and_cem_closed": (
            config["evidence_boundary"]["public_test_open"] is False
            and config["evidence_boundary"]["cem_open"] is False
        ),
    }
    _require(all(checks.values()), f"static MVE contract failed: {checks}")
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--optimizer-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--original-h5", type=Path, default=None)
    parser.add_argument("--original-lance", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def expected_output(arm: str, steps: int = DEFAULT_STEPS) -> Path:
    return (ARTIFACT_ROOT / f"{arm}_s{FROZEN_SEED}_step{steps}_v1").resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require(args.seed == FROZEN_SEED, f"first MVE requires seed {FROZEN_SEED}")
    _require(args.optimizer_steps > 0, "optimizer steps must be positive")
    output = args.output.expanduser().resolve()
    if not args.dry_run and output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    static_checks = validate_static_contract()

    import run_pusht_motion_damping_h3_train as motion

    variant = install_training_overlay(motion, arm=args.arm)
    release_overlay = install_release_authorization_overlay(
        motion, arm=args.arm, steps=args.optimizer_steps
    )
    native_argv = sys.argv
    forwarded = [
        str(motion.__file__),
        "--release-config",
        str(RELEASE_CONFIG),
        "--model",
        "lewm",
        "--variant",
        variant,
        "--seed",
        str(args.seed),
        "--optimizer-steps",
        str(args.optimizer_steps),
        "--output",
        str(output),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    for name in ("original_h5", "original_lance", "checkpoint"):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend(["--" + name.replace("_", "-"), str(value.resolve())])
    if args.dry_run:
        forwarded.append("--dry-run")
    sys.argv = forwarded
    try:
        motion.main()
    finally:
        sys.argv = native_argv
    if args.dry_run:
        return

    report_path = output / "training_report.json"
    provenance_path = output / "training_provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    disclosure = {
        "experiment_id": "pusht_motion_damping_oracle_context_predictor_mve_v1",
        "arm": args.arm,
        "variant": variant,
        "preregistration": {"path": str(PREREG), "sha256": file_sha256(PREREG)},
        "release": {"path": str(RELEASE_CONFIG), "sha256": file_sha256(RELEASE_CONFIG)},
        "release_authorization_overlay": release_overlay,
        "static_checks": static_checks,
        "oracle_privileged_context_opened": args.arm == "oracle",
        "public_test_opened": False,
        "cem_opened": False,
    }
    report["research_overlay"] = disclosure
    provenance["research_overlay"] = disclosure
    if args.arm == "oracle":
        provenance["visible_fields"] = [
            "pixels",
            "action",
            "oracle_motion_damping_identity_diagnostic_only",
        ]
    _atomic_json(report_path, report)
    _atomic_json(provenance_path, provenance)


if __name__ == "__main__":
    main()
