#!/usr/bin/env python3
"""Audit native conditional-gradient strength and SNR on frozen train pairs.

This runner takes zero optimizer steps and opens neither Development nor Public
Test.  It replays the frozen MotionDamping training batch and decomposes paired
terminal MSE into its exact center and response terms.  Parameter-space
statistics are restricted to the trainable Predictor and prediction projection.
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


ROOT = Path(__file__).resolve().parents[3]
CONTEXTWORLD = ROOT.parent / "ContextWorld"
for source_root in (ROOT, CONTEXTWORLD, CONTEXTWORLD / "scripts"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    analyze_pusht_motion_damping_exact_penalty_boundary_comparator_v1 as replay,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    conditional_signal_metrics as signal_metrics,
)
from contextworld.evaluation.protocol import (  # noqa: E402
    _prepare_optional_flash_attention,
)


ANALYSIS_VARIANT = "analysis_motion_native_gradient_snr_v1"
ALLOWED_PARAMETER_GROUPS = ("predictor", "pred_proj")
ORIGINAL_ROWS = replay.ORIGINAL_ROWS
SEED = replay.SEED
DEFAULT_ORIGINAL_H5 = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "lewm-pusht/pusht_expert_train.h5"
)
DEFAULT_ORIGINAL_LANCE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/lance-format/"
    "LeWorldModel/data/lewm_pusht.lance"
)
BATCH_SIZES = (1, 8, 16, 32, 64, 128)
EXPECTED_TRAINABLE_PARAMETER_COUNT = 11_584_128


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(name: str) -> str:
    return name.split(".", 1)[0]


def _gradient_tuple(
    scalar: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        torch.autograd.grad(
            scalar,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
    )


def _cpu_gradient(
    values: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        None if value is None else value.detach().to(device="cpu", dtype=torch.float64)
        for value in values
    )


def _linear_combination(
    terms: Sequence[tuple[float, Sequence[torch.Tensor | None]]],
) -> tuple[torch.Tensor | None, ...]:
    _require(bool(terms), "gradient combination is empty")
    width = len(terms[0][1])
    _require(all(len(values) == width for _, values in terms), "gradient widths differ")
    output: list[torch.Tensor | None] = []
    for index in range(width):
        combined = None
        for weight, values in terms:
            value = values[index]
            if value is None:
                continue
            weighted = float(weight) * value
            combined = weighted.clone() if combined is None else combined + weighted
        output.append(combined)
    return tuple(output)


def _aggregate_norms(
    gradient: Sequence[torch.Tensor | None], parameter_groups: Sequence[str]
) -> dict[str, float]:
    scopes = signal_metrics.gradient_population_summary(
        [gradient], parameter_groups=parameter_groups, batch_sizes=(1,)
    )["scopes"]
    return {
        name: float(row["mean_gradient_norm"])
        for name, row in scopes.items()
    }


def _gradient_parity(
    observed: Sequence[torch.Tensor | None],
    expected: Sequence[torch.Tensor | None],
) -> dict[str, float | bool]:
    difference_squared = 0.0
    reference_squared = 0.0
    maximum_absolute_error = 0.0
    for one, two in zip(observed, expected, strict=True):
        if one is None and two is None:
            continue
        if one is None:
            one = torch.zeros_like(two)
        if two is None:
            two = torch.zeros_like(one)
        delta = one.detach().double().cpu() - two.detach().double().cpu()
        difference_squared += float(delta.square().sum())
        reference_squared += float(two.detach().double().cpu().square().sum())
        maximum_absolute_error = max(
            maximum_absolute_error,
            float(delta.abs().max()) if delta.numel() else 0.0,
        )
    relative = math.sqrt(difference_squared) / max(
        math.sqrt(reference_squared), 1.0e-30
    )
    return {
        "relative_l2_error": relative,
        "maximum_absolute_error": maximum_absolute_error,
        "passed_bf16_tolerance": relative <= 2.0e-2,
    }


def _audit(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    expected_model_state_sha256: str,
    batch: Mapping[str, Any],
    mixed: Any,
    device: torch.device,
) -> dict[str, Any]:
    mixed.pilot.set_reproducible_seed(SEED)
    model, load_receipt = mixed.load_model_for_variant(
        checkpoint, variant=ANALYSIS_VARIANT, device=device
    )
    _require(
        load_receipt.get("sha256") == expected_checkpoint_sha256,
        "checkpoint load SHA differs",
    )
    _require(
        load_receipt.get("model_state_sha256") == expected_model_state_sha256,
        "loaded model-state SHA differs",
    )
    _require(load_receipt.get("strict_state_dict_load") is True, "load was not strict")

    modes_before = replay.exact_audit._module_modes(model)
    buffers_before = replay.live_ccrm._buffer_snapshot(model)
    parameter_hash = replay.live_ccrm.parameter_value_sha256(model)
    model.train()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and _group(name) in ALLOWED_PARAMETER_GROUPS
    ]
    _require(bool(named_parameters), "no trainable predictor parameters selected")
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    parameter_groups = [_group(name) for name in names]
    _require(
        set(parameter_groups) == set(ALLOWED_PARAMETER_GROUPS),
        "trainable parameter route differs",
    )
    selected_parameter_count = sum(parameter.numel() for parameter in parameters)
    _require(
        selected_parameter_count == EXPECTED_TRAINABLE_PARAMETER_COUNT,
        "selected optimizer parameter count differs from the frozen training report",
    )

    pixels = mixed.pilot.preprocess_pixels(batch["pixels"], device)
    actions = batch["actions"].to(device=device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        encoded = model.encode({"pixels": pixels, "action": actions})
        embeddings = encoded["emb"]
        prediction = model.predict(embeddings[:, :3], encoded["act_emb"][:, :3])
        target = embeddings[:, 1:].detach()
        error = (prediction - target).square().mean(dim=-1)
        original_loss_by_row = error[:ORIGINAL_ROWS].mean(dim=1)

    groups = replay.live_ccrm.binary_hidden_groups(
        original_batch_size=ORIGINAL_ROWS,
        batch_size=int(embeddings.shape[0]),
        device=device,
    )
    selected_prediction = prediction[groups, -1].float()
    selected_target = target[groups, -1].float()
    parts = signal_metrics.paired_signal_components(
        selected_prediction, selected_target
    )
    hidden_direct = error[ORIGINAL_ROWS:, -1].float().mean()
    center_mean = parts["center_loss"].mean()
    response_mean = parts["response_loss"].mean()
    hidden_decomposed = center_mean + response_mean
    original_mean = original_loss_by_row.float().mean()
    recipe_mean = 0.5 * (original_mean + hidden_direct)
    _require(
        torch.allclose(hidden_direct, hidden_decomposed, atol=2.0e-5, rtol=0.0),
        "paired hidden loss decomposition differs",
    )

    direct_gradients = {
        "original": _cpu_gradient(_gradient_tuple(original_mean, parameters)),
        "center": _cpu_gradient(_gradient_tuple(center_mean, parameters)),
        "response": _cpu_gradient(_gradient_tuple(response_mean, parameters)),
        "hidden_correct": _cpu_gradient(_gradient_tuple(hidden_direct, parameters)),
        "native_recipe": _cpu_gradient(_gradient_tuple(recipe_mean, parameters)),
    }

    accumulators = {
        name: signal_metrics.GradientPopulationAccumulator(
            parameter_groups=parameter_groups, batch_sizes=BATCH_SIZES
        )
        for name in ("center", "response", "hidden_correct")
    }
    for pair_index in range(int(groups.shape[0])):
        center_gradient = _gradient_tuple(
            parts["center_loss"][pair_index], parameters
        )
        response_gradient = _gradient_tuple(
            parts["response_loss"][pair_index], parameters
        )
        hidden_gradient = _linear_combination(
            ((1.0, center_gradient), (1.0, response_gradient))
        )
        accumulators["center"].add(center_gradient)
        accumulators["response"].add(response_gradient)
        accumulators["hidden_correct"].add(hidden_gradient)
        del center_gradient, response_gradient, hidden_gradient

    population_means = {
        name: accumulator.mean_gradients()
        for name, accumulator in accumulators.items()
    }
    parity = {
        name: _gradient_parity(population_means[name], direct_gradients[name])
        for name in accumulators
    }
    _require(
        all(row["passed_bf16_tolerance"] for row in parity.values()),
        f"per-pair aggregate gradient parity failed: {parity}",
    )

    nonconditional = _linear_combination(
        ((0.5, direct_gradients["original"]), (0.5, direct_gradients["center"]))
    )
    weighted_response = _linear_combination(((0.5, direct_gradients["response"]),))
    reconstructed_recipe = _linear_combination(
        ((1.0, nonconditional), (1.0, weighted_response))
    )
    recipe_parity = _gradient_parity(
        reconstructed_recipe, direct_gradients["native_recipe"]
    )
    _require(recipe_parity["passed_bf16_tolerance"], "native recipe gradient differs")

    aggregate_gradients = {
        **direct_gradients,
        "weighted_nonconditional": nonconditional,
        "weighted_response": weighted_response,
        "reconstructed_native_recipe": reconstructed_recipe,
    }
    aggregate_norms = {
        name: _aggregate_norms(values, parameter_groups)
        for name, values in aggregate_gradients.items()
    }
    gradient_strength = {}
    for scope in ("all", *ALLOWED_PARAMETER_GROUPS):
        response_norm = aggregate_norms["weighted_response"][scope]
        nonconditional_norm = aggregate_norms["weighted_nonconditional"][scope]
        recipe_norm = aggregate_norms["native_recipe"][scope]
        gradient_strength[scope] = {
            "weighted_response_gradient_norm": response_norm,
            "weighted_nonconditional_gradient_norm": nonconditional_norm,
            "native_recipe_gradient_norm": recipe_norm,
            "response_to_nonconditional_norm_ratio": (
                response_norm / nonconditional_norm
                if nonconditional_norm > 0.0
                else None
            ),
            "response_to_recipe_norm_ratio": (
                response_norm / recipe_norm if recipe_norm > 0.0 else None
            ),
            "response_two_term_energy_share": (
                response_norm**2 / (response_norm**2 + nonconditional_norm**2)
                if response_norm > 0.0 or nonconditional_norm > 0.0
                else None
            ),
        }

    report = {
        "checkpoint": load_receipt,
        "precision": "cuda_bf16_forward_float32_loss_decomposition",
        "parameter_names": names,
        "parameter_groups": {
            group: sum(value == group for value in parameter_groups)
            for group in ALLOWED_PARAMETER_GROUPS
        },
        "selected_optimizer_parameter_count": selected_parameter_count,
        "selected_optimizer_route_source": (
            "adjacent training_report.json representation_freeze contract"
        ),
        "losses": {
            "original_prediction_mse": float(original_mean.detach().cpu()),
            "hidden_center_mse": float(center_mean.detach().cpu()),
            "hidden_response_mse": float(response_mean.detach().cpu()),
            "hidden_correct_mse": float(hidden_direct.detach().cpu()),
            "native_recipe_prediction_mse": float(recipe_mean.detach().cpu()),
            "hidden_decomposition_absolute_error": float(
                (hidden_direct - hidden_decomposed).detach().abs().cpu()
            ),
        },
        "paired_output_signal": signal_metrics.paired_signal_summary(
            selected_prediction.detach().cpu(), selected_target.detach().cpu(),
            batch_sizes=BATCH_SIZES,
        ),
        "gradient_populations": {
            name: accumulator.summary()
            for name, accumulator in accumulators.items()
        },
        "aggregate_gradient_norms": aggregate_norms,
        "aggregate_gradient_strength": gradient_strength,
        "aggregate_gradient_relations": {
            "weighted_nonconditional_vs_weighted_response": (
                signal_metrics.gradient_relation_summary(
                    nonconditional,
                    weighted_response,
                    parameter_groups=parameter_groups,
                )
            ),
            "native_recipe_vs_weighted_response": (
                signal_metrics.gradient_relation_summary(
                    direct_gradients["native_recipe"],
                    weighted_response,
                    parameter_groups=parameter_groups,
                )
            ),
        },
        "gradient_parity": {
            "per_pair_mean_vs_direct": parity,
            "native_recipe_reconstruction": recipe_parity,
        },
    }

    buffers_restored = replay.exact_audit._restore_buffers(model, buffers_before)
    modes_restored = replay.exact_audit._restore_module_modes(model, modes_before)
    _require(
        replay.live_ccrm.parameter_value_sha256(model) == parameter_hash,
        "diagnostic changed model parameters",
    )
    _require(
        all(parameter.grad is None for parameter in model.parameters()),
        "autograd.grad populated parameter .grad",
    )
    report["state_restoration"] = {
        "buffers_restored": buffers_restored,
        "module_modes_restored": modes_restored,
        "parameters_unchanged": True,
        "parameter_grad_slots_remain_none": True,
    }
    return report


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-model-state-sha256", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original-h5", type=Path, default=DEFAULT_ORIGINAL_H5)
    parser.add_argument(
        "--original-lance", type=Path, default=DEFAULT_ORIGINAL_LANCE
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source = Path(__file__).resolve()
    _require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
    _require(
        _sha256(checkpoint) == args.expected_checkpoint_sha256,
        "checkpoint file SHA differs before load",
    )
    _require(not output.exists(), f"refusing to overwrite {output}")
    if args.check_only:
        print(json.dumps({"status": "passed_static_identity", "checkpoint": str(checkpoint)}))
        return

    device = torch.device(args.device)
    _require(device.type == "cuda" and torch.cuda.is_available(), "CUDA required")
    _prepare_optional_flash_attention()
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
    torch.cuda.get_rng_state_all()
    outer_rng = replay.gradient_core._rng_snapshot()
    try:
        mixed.VARIANT_WEIGHTS[ANALYSIS_VARIANT] = (
            "native", replay.WEIGHT, "identifiable_future_only"
        )
        motion.TWIN_GROUP_VARIANTS.add(ANALYSIS_VARIANT)
        trainer.DIAGNOSTIC_VARIANTS["lewm"].add(ANALYSIS_VARIANT)
        with replay.replay._install_fail_closed_guards(
            motion, motion_score, allow_training_table=True
        ) as guard_counts:
            batch, batch_receipt = replay.replay._build_real_training_batch(
                motion=motion,
                runner=replay.live_ccrm,
                original_h5=args.original_h5.expanduser().resolve(),
                original_lance=args.original_lance.expanduser().resolve(),
            )
            for name, expected in replay.EXPECTED_BATCH_HASHES.items():
                _require(batch_receipt[name] == expected, f"{name} changed")
            audit = _audit(
                checkpoint=checkpoint,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                expected_model_state_sha256=args.expected_model_state_sha256,
                batch=batch,
                mixed=mixed,
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
            all(guard_counts[name] == value for name, value in expected_guards.items()),
            f"forbidden diagnostic action: {guard_counts}",
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
    _require(
        replay.gradient_core._rng_equal(outer_rng, replay.gradient_core._rng_snapshot()),
        "RNG restoration failed",
    )

    report = {
        "schema_version": 1,
        "analysis_id": "motion_damping_native_gradient_snr_diagnostic_v1",
        "label": args.label,
        "status": "completed_zero_optimizer_steps_training_only_gradient_diagnostic",
        "optimizer_updates": 0,
        "claim_scope": {
            "training_batch_only": True,
            "development_opened": False,
            "public_test_opened": False,
            "cem_executed": False,
            "single_frozen_batch_is_not_training_outcome": True,
        },
        "source": {"path": str(source), "sha256": _sha256(source)},
        "runtime": {
            "completion": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "release": {"path": str(release_path), "sha256": _sha256(release_path)},
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
        },
        "training_batch": batch_receipt,
        "guard_counts": dict(guard_counts),
        "objective_contract": {
            "native_prediction_recipe": "0.5 * original_all_transition_mse + 0.5 * hidden_terminal_mse",
            "paired_identity": "hidden_terminal_mse = center_mse + response_mse",
            "weighted_nonconditional": "0.5 * original_mse + 0.5 * center_mse",
            "weighted_response": "0.5 * response_mse",
            "sigreg_route": "zero for selected predictor/pred_proj parameters",
            "targets_detached": True,
            "gradient_units": "one complete hidden condition pair",
        },
        "audit": audit,
        "interpretation_boundary": {
            "data_vs_optimization": (
                "rho_cond and paired output geometry describe this frozen batch; "
                "parameter Bcrit additionally includes the checkpoint Jacobian."
            ),
            "snr": (
                "sqrt(B/Bcrit) is a trace-energy population estimate over the 32 "
                "paired units, not a confidence interval over training runs."
            ),
            "data_decision": (
                "A high response Bcrit with a small response-to-nonconditional norm "
                "supports coverage or sampling changes; it does not by itself prove "
                "that data-only changes are sufficient across tasks."
            ),
        },
    }
    _require(_sha256(checkpoint) == args.expected_checkpoint_sha256, "checkpoint changed")
    _write_exclusive(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": _sha256(output)}))


if __name__ == "__main__":
    main()
