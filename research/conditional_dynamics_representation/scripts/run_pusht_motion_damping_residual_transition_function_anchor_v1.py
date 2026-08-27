#!/usr/bin/env python3
"""Bootstrap Motion ICL while anchoring the source planning function.

This bounded discovery arm starts from the fixed residual-transition checkpoint
that retained 17/20 standard PushT CEM successes.  It keeps the existing action
encoder fixed, trains the existing Predictor/pred_proj with the already-defined
pair-normalized exact-future objective, and adds one training-only consistency
constraint against a frozen copy of the source Predictor/pred_proj on ordinary
PushT rows.  No learned parameter, model module, head, or inference branch is
added to the saved checkpoint.

The anchor is normalized by the frozen teacher's displacement energy::

    L_anchor = MSE(p_student, sg(p_source))
               / max(MSE(sg(p_source) - history), 1e-8)

The single existing auxiliary weight is reused for both conditional bootstrap
and function preservation::

    L = L_native + 0.09 * (L_pair + L_anchor)

This is a weight-only 1,024-step restart from a 2,048-step source checkpoint;
it is not presented as a budget-matched 1,024-step control.
"""

from __future__ import annotations

import argparse
import copy
from einops import rearrange
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Callable, Sequence

import torch
from torch import nn


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_worldmodel.wm.lewm.lewm import causal_transition_basis  # noqa: E402
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    pair_normalized_exact_future_residual_v1 as exact_future,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_causal_transition_basis_v1 as causal,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_native_twin_sampler_v1 as native_twin,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_residual_output_basis_v1 as residual,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_residual_transition_ccrm_v1 as paired,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_temporal_homotopy_exact_future_v1 as hashing,
)


CANDIDATE = "pusht_motion_damping_residual_transition_function_anchor_v1"
SOURCE_CHECKPOINT_SHA256 = (
    "520bfa3738265cbe059873019f20c05e3e1c060b061d43158b057ab0ccf6cd75"
)
OPTIMIZER_STEPS = 1024
AUXILIARY_WEIGHT = 0.09
ANCHOR_EPSILON = 1e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _buffer_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_buffers()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_source_checkpoint(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args, _ = parser.parse_known_args(list(argv))
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed = _sha256(checkpoint)
    if observed != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Function-anchor source checkpoint changed: "
            f"expected {SOURCE_CHECKPOINT_SHA256}, observed {observed}"
        )
    return checkpoint


def frozen_reference_prediction(
    *,
    predictor: nn.Module,
    pred_proj: nn.Module,
    history: torch.Tensor,
    action_embedding: torch.Tensor,
) -> torch.Tensor:
    """Apply the frozen causal-transition/residual source function."""

    transformed = causal_transition_basis(history)
    displacement = predictor(transformed, action_embedding)
    displacement = pred_proj(
        rearrange(displacement, "b t d -> (b t) d")
    )
    displacement = rearrange(
        displacement,
        "(b t) d -> b t d",
        b=history.size(0),
    )
    return history + displacement


def normalized_function_anchor(
    *,
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    history: torch.Tensor,
    epsilon: float = ANCHOR_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dimensionless source-function drift and its frozen scale."""

    if student_prediction.shape != teacher_prediction.shape:
        raise ValueError("Student and teacher prediction shapes differ")
    if student_prediction.shape != history.shape:
        raise ValueError("Function anchor history shape differs")
    scale = torch.square(teacher_prediction.detach() - history.detach()).mean()
    scale = scale.clamp_min(float(epsilon))
    loss = torch.square(
        student_prediction - teacher_prediction.detach()
    ).mean() / scale
    return loss, scale


def function_anchored_prediction_loss(
    *,
    native_loss: Callable[..., torch.Tensor],
    mixed_module: Any,
    state: dict[str, Any],
    prediction: torch.Tensor,
    embeddings: torch.Tensor,
    original_batch_size: int,
    conditional_population: str,
) -> torch.Tensor:
    native_value = native_loss(
        prediction=prediction,
        embeddings=embeddings,
        original_batch_size=original_batch_size,
        conditional_population=conditional_population,
    )
    model_state = state.get("model")
    if not model_state:
        raise RuntimeError("Function-anchor model was not loaded")
    capture = model_state["capture"]
    if capture.get("prediction") is not prediction:
        raise RuntimeError("Function anchor did not receive captured forward")
    history = capture["history"]
    action_embedding = capture["action_embedding"]
    with torch.random.fork_rng(
        devices=[prediction.device.index or 0]
        if prediction.device.type == "cuda"
        else []
    ):
        with mixed_module.temporary_eval_modules(
            model_state["model"].predictor,
            model_state["model"].pred_proj,
        ):
            deterministic_prediction = model_state["native_predict"](
                history.detach(), action_embedding.detach()
            )
        with torch.no_grad():
            teacher_prediction = frozen_reference_prediction(
                predictor=model_state["teacher_predictor"],
                pred_proj=model_state["teacher_pred_proj"],
                history=history.detach(),
                action_embedding=action_embedding.detach(),
            )
    groups = paired.binary_hidden_groups(
        original_batch_size=original_batch_size,
        batch_size=prediction.size(0),
        device=prediction.device,
    )
    response = exact_future.pair_normalized_exact_future_residual(
        deterministic_prediction,
        embeddings[:, 1:],
        groups,
    )
    pair_loss = response["loss"]
    anchor_loss, anchor_scale = normalized_function_anchor(
        student_prediction=deterministic_prediction[:original_batch_size],
        teacher_prediction=teacher_prediction[:original_batch_size],
        history=history[:original_batch_size],
    )
    if state["loss_calls"] == 0 and float(anchor_loss.detach().float()) != 0.0:
        raise RuntimeError(
            "Source-function anchor is nonzero before the first optimizer step"
        )
    if not all(
        torch.isfinite(value).all()
        for value in (native_value, pair_loss, anchor_loss, anchor_scale)
    ):
        raise RuntimeError("Function-anchor loss produced a non-finite value")
    total = native_value + float(state["weight"]) * (
        pair_loss + anchor_loss
    )
    components = {
        "native_prediction_loss": float(native_value.detach().float()),
        "pair_normalized_exact_future_loss": float(
            pair_loss.detach().float()
        ),
        "normalized_source_function_anchor_loss": float(
            anchor_loss.detach().float()
        ),
        "source_displacement_energy": float(anchor_scale.detach().float()),
        "total_prediction_loss": float(total.detach().float()),
    }
    state["loss_calls"] += 1
    if state["first_components"] is None:
        state["first_components"] = copy.deepcopy(components)
    state["last_components"] = components
    capture.clear()
    return total


def _install_function_anchor(trainer: Any) -> dict[str, Any]:
    mixed = trainer.trainer.mixed
    native_loader = mixed.load_model_for_variant
    native_model_config = mixed.model_config
    native_loss = mixed.mixed_prediction_loss
    state: dict[str, Any] = {
        "model": None,
        "loss_calls": 0,
        "first_components": None,
        "last_components": None,
        "weight": AUXILIARY_WEIGHT,
    }

    def load_model(checkpoint: Path, *args: Any, **kwargs: Any):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if _sha256(checkpoint) != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError("Function-anchor source checkpoint changed")
        model, receipt = native_loader(checkpoint, *args, **kwargs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        source_state = hashing._state_sha256(model)
        model.temporal_input_basis = "causal_transition"
        model.temporal_output_basis = "residual"
        model.action_encoder.requires_grad_(False)
        action_state = hashing._state_sha256(model.action_encoder)
        teacher_predictor = copy.deepcopy(model.predictor).eval().requires_grad_(False)
        teacher_pred_proj = copy.deepcopy(model.pred_proj).eval().requires_grad_(False)
        teacher_state = (
            hashing._state_sha256(teacher_predictor),
            hashing._state_sha256(teacher_pred_proj),
        )
        pred_proj_buffers = _buffer_state_sha256(model.pred_proj)
        native_predict = model.predict
        native_train = model.train
        capture: dict[str, torch.Tensor] = {}

        def train_with_inference_aligned_head(
            self: Any,
            mode: bool = True,
        ) -> Any:
            result = native_train(mode)
            if mode:
                # pred_proj owns BatchNorm buffers.  Updating those buffers in
                # the native train-mode forward would move the saved planning
                # function before the first optimizer step and make the
                # source-function equality constraint ill posed.
                self.pred_proj.eval()
            return result

        def predict_with_capture(
            self: Any,
            history: torch.Tensor,
            action_embedding: torch.Tensor,
        ) -> torch.Tensor:
            result = native_predict(history, action_embedding)
            if self.training and torch.is_grad_enabled():
                capture.clear()
                capture.update(
                    {
                        "history": history,
                        "action_embedding": action_embedding,
                        "prediction": result,
                    }
                )
            return result

        model.predict = types.MethodType(predict_with_capture, model)
        model.train = types.MethodType(train_with_inference_aligned_head, model)
        if sum(parameter.numel() for parameter in model.parameters()) != parameter_count:
            raise RuntimeError("Function anchor changed saved model parameter count")
        state["model"] = {
            "model": model,
            "native_predict": native_predict,
            "capture": capture,
            "teacher_predictor": teacher_predictor,
            "teacher_pred_proj": teacher_pred_proj,
            "parameter_count": parameter_count,
            "source_state_sha256": source_state,
            "action_encoder_state_sha256_before": action_state,
            "teacher_state_sha256_before": teacher_state,
            "pred_proj_buffer_state_sha256_before": pred_proj_buffers,
        }
        updated = dict(receipt)
        updated["residual_transition_function_anchor"] = {
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "temporal_input_basis": "causal_transition",
            "temporal_output_basis": "residual",
            "action_encoder_frozen": True,
            "teacher_scope": ["predictor", "pred_proj"],
            "teacher_training_only": True,
            "pred_proj_buffers_frozen_in_inference_mode": True,
            "parameter_reset": False,
            "learned_parameters_added_to_saved_model": 0,
        }
        return model, updated

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(native_model_config(*args, **kwargs))
        config["temporal_input_basis"] = "causal_transition"
        config["temporal_output_basis"] = "residual"
        return config

    def augmented_loss(**kwargs: Any) -> torch.Tensor:
        return function_anchored_prediction_loss(
            native_loss=native_loss,
            mixed_module=mixed,
            state=state,
            **kwargs,
        )

    mixed.load_model_for_variant = load_model
    mixed.model_config = model_config
    mixed.mixed_prediction_loss = augmented_loss
    return state


def _rewrite_training_report(output: Path, *, state: dict[str, Any]) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    model_state = state.get("model") or {}
    model = model_state.get("model")
    action_after = hashing._state_sha256(model.action_encoder)
    teacher_after = (
        hashing._state_sha256(model_state["teacher_predictor"]),
        hashing._state_sha256(model_state["teacher_pred_proj"]),
    )
    pred_proj_buffers_after = _buffer_state_sha256(model.pred_proj)
    grouping = result["batch"].get("motion_damping_twin_grouping")
    trainable_roots = tuple(result["representation_freeze"]["trainable_top_level_modules"])
    checks = {
        "source_checkpoint_exact": (
            result["source_checkpoint"]["sha256"] == SOURCE_CHECKPOINT_SHA256
        ),
        "optimizer_steps_exact": int(result["optimizer_steps"]) == OPTIMIZER_STEPS,
        "native_mse_sigreg_0p09_retained": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "complete_twin_grouping": bool(
            grouping
            and grouping.get("enabled") is True
            and grouping.get("condition_rows_per_group") == 4
        ),
        "one_joint_and_anchor_call_per_step": (
            int(state["loss_calls"]) == OPTIMIZER_STEPS
        ),
        "step1_anchor_exact_zero": (
            float(state["first_components"][
                "normalized_source_function_anchor_loss"
            ]) == 0.0
        ),
        "anchor_scale_positive": (
            float(state["first_components"]["source_displacement_energy"]) > 0.0
            and float(state["last_components"]["source_displacement_energy"]) > 0.0
        ),
        "action_encoder_frozen_and_unchanged": (
            model_state["action_encoder_state_sha256_before"] == action_after
            and all(not parameter.requires_grad for parameter in model.action_encoder.parameters())
        ),
        "teacher_unchanged": model_state["teacher_state_sha256_before"] == teacher_after,
        "pred_proj_buffers_unchanged": (
            model_state["pred_proj_buffer_state_sha256_before"]
            == pred_proj_buffers_after
            and model.pred_proj.training is False
        ),
        "saved_model_parameter_count_unchanged": (
            model_state["parameter_count"]
            == sum(parameter.numel() for parameter in model.parameters())
        ),
        "only_existing_predictor_and_head_trainable": (
            trainable_roots == ("pred_proj", "predictor")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Function-anchor contract failed: {checks}")
    result["residual_transition_function_anchor_contract"] = {
        "checks": checks,
        "continuation_source": {
            "checkpoint": result["source_checkpoint"]["path"],
            "sha256": SOURCE_CHECKPOINT_SHA256,
            "source_optimizer_steps": 2048,
            "source_motion_six_gate_passed": False,
            "source_cem_screen_successes": "17/20",
        },
        "objective": (
            "native_mse + 0.09*SIGReg + "
            "0.09*(pair_normalized_exact_future + "
            "normalized_source_function_anchor)"
        ),
        "first_loss_components": state["first_components"],
        "last_loss_components": state["last_components"],
        "optimizer_semantics": {
            "kind": "weight_only_restart",
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "fresh_optimizer_steps": OPTIMIZER_STEPS,
            "total_parameter_update_exposure": 2048 + OPTIMIZER_STEPS,
            "matched_native_1024_comparison": False,
        },
        "temporal_input_basis": "causal_transition",
        "temporal_output_basis": "residual",
        "action_encoder_frozen": True,
        "teacher_training_only": True,
        "teacher_scope": ["predictor", "pred_proj"],
        "pred_proj_buffers_frozen_in_inference_mode": True,
        "teacher_or_anchor_present_at_inference": False,
        "learned_parameters_added_to_saved_model": 0,
        "model_modules_added_to_saved_model": 0,
        "inference_compute_added": 0,
        "auxiliary_terms_added": 2,
    }
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    _validate_source_checkpoint(effective)
    args = residual._discovery_args(effective)
    if (
        args.input_basis != "causal_transition"
        or args.optimizer_steps != OPTIMIZER_STEPS
    ):
        raise RuntimeError("Function-anchor continuation is fixed to 1024 steps")
    trainer = causal._load_trainer()
    native_twin._install_runtime(trainer)
    state = _install_function_anchor(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *residual._trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(output, state=state)
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("temporal_input_basis") != "causal_transition"
            or config.get("temporal_output_basis") != "residual"
        ):
            raise RuntimeError("Function-anchor checkpoint saved wrong semantics")
        sidecar = output / "residual_transition_function_anchor_method_v1.json"
        if sidecar.exists():
            raise FileExistsError(sidecar)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": CANDIDATE,
                    "source": str(THIS_SOURCE),
                    "source_sha256": _sha256(THIS_SOURCE),
                    "training_report": str(report),
                    "training_report_sha256": _sha256(report),
                    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                    "objective": (
                        "native_mse + 0.09*SIGReg + "
                        "0.09*(pair_exact_future + source_function_anchor)"
                    ),
                    "optimizer_steps_added": OPTIMIZER_STEPS,
                    "total_parameter_update_exposure": 2048 + OPTIMIZER_STEPS,
                    "action_encoder_frozen": True,
                    "training_only_frozen_teacher": True,
                    "learned_parameters_added_to_saved_model": 0,
                    "model_modules_added_to_saved_model": 0,
                    "inference_compute_added": 0,
                    "claim_boundary": {
                        "development_mechanism_screen_only": True,
                        "single_seed_discovery": True,
                        "public_test_opened": False,
                        "additional_seeds_opened": False,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
