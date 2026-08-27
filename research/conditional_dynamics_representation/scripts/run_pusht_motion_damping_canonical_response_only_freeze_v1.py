#!/usr/bin/env python3
"""Freeze-only source continuation for canonical conditional response.

Starting from the fixed planning-capable residual-transition source, this arm
adds only the canonical response-only paired auxiliary for 1,024 fresh AdamW
steps.  It freezes the existing action encoder and pred-proj BatchNorm buffers,
but uses no teacher, function anchor, new module, new parameter, gradient
route, or inference computation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Callable, Sequence

import torch


sys.modules.setdefault("flash_attn", None)

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    canonical_response_only_v1 as objective,
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
    run_pusht_motion_damping_residual_transition_function_anchor_v1 as anchor,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_temporal_homotopy_exact_future_v1 as hashing,
)


CANDIDATE = "pusht_motion_damping_canonical_response_only_freeze_v1"
SOURCE_CHECKPOINT_SHA256 = anchor.SOURCE_CHECKPOINT_SHA256
OPTIMIZER_STEPS = 1024
AUXILIARY_WEIGHT = 0.09


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--input-basis")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args(list(argv))
    checkpoint = args.checkpoint.expanduser().resolve()
    checks = {
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_sha_exact": (
            checkpoint.is_file()
            and _sha256(checkpoint) == SOURCE_CHECKPOINT_SHA256
        ),
        "model_exact": args.model == "lewm",
        "variant_exact": args.variant in {None, causal.NATIVE_VARIANT},
        "seed_exact": args.seed == 14321,
        "optimizer_steps_exact": args.optimizer_steps == OPTIMIZER_STEPS,
        "input_basis_exact": args.input_basis == "causal_transition",
        "output_present": args.output is not None,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Motion freeze-only continuation contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return args


def _freeze_only_prediction_loss(
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
        raise RuntimeError("Freeze-only continuation model was not loaded")
    capture = model_state["capture"]
    if capture.get("prediction") is not prediction:
        raise RuntimeError("Freeze-only loss did not receive captured forward")
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
    groups = paired.binary_hidden_groups(
        original_batch_size=original_batch_size,
        batch_size=prediction.size(0),
        device=prediction.device,
    )
    response = objective.canonical_response_only(
        deterministic_prediction,
        embeddings[:, 1:],
        groups,
    )
    auxiliary = response["loss"]
    total = native_value + AUXILIARY_WEIGHT * auxiliary
    components = {
        "native_prediction_loss": float(native_value.detach().float()),
        "canonical_response_only_loss": float(auxiliary.detach().float()),
        "centered_response_loss": float(
            response["response_loss"].detach().float()
        ),
        "canonical_margin_loss": float(
            response["canonical_margin_loss"].detach().float()
        ),
        "excluded_direct_common_center_mse": float(
            response["normalized_common_center_mse_by_group"]
            .mean()
            .detach()
            .float()
        ),
        "total_prediction_loss": float(total.detach().float()),
    }
    state["loss_calls"] += 1
    if state["first_components"] is None:
        state["first_components"] = copy.deepcopy(components)
    state["last_components"] = components
    capture.clear()
    return total


def _install_freeze_only(trainer: Any) -> dict[str, Any]:
    mixed = trainer.trainer.mixed
    native_loader = mixed.load_model_for_variant
    native_model_config = mixed.model_config
    native_loss = mixed.mixed_prediction_loss
    state: dict[str, Any] = {
        "model": None,
        "loss_calls": 0,
        "first_components": None,
        "last_components": None,
    }

    def load_model(checkpoint: Path, *args: Any, **kwargs: Any):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if _sha256(checkpoint) != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError("Freeze-only source checkpoint changed")
        model, receipt = native_loader(checkpoint, *args, **kwargs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model.temporal_input_basis = "causal_transition"
        model.temporal_output_basis = "residual"
        model.action_encoder.requires_grad_(False)
        action_state = hashing._state_sha256(model.action_encoder)
        pred_proj_buffers = anchor._buffer_state_sha256(model.pred_proj)
        native_predict = model.predict
        native_train = model.train
        capture: dict[str, torch.Tensor] = {}

        def train_with_frozen_existing_state(
            self: Any,
            mode: bool = True,
        ) -> Any:
            result = native_train(mode)
            if mode:
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

        model.train = types.MethodType(train_with_frozen_existing_state, model)
        model.predict = types.MethodType(predict_with_capture, model)
        state["model"] = {
            "model": model,
            "native_predict": native_predict,
            "capture": capture,
            "parameter_count": parameter_count,
            "action_encoder_state_sha256_before": action_state,
            "pred_proj_buffer_state_sha256_before": pred_proj_buffers,
        }
        updated = dict(receipt)
        updated["canonical_response_only_freeze"] = {
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "action_encoder_frozen": True,
            "pred_proj_buffers_frozen_in_inference_mode": True,
            "training_only_teacher": False,
            "learned_parameters_added": 0,
            "model_modules_added": 0,
        }
        return model, updated

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(native_model_config(*args, **kwargs))
        config["temporal_input_basis"] = "causal_transition"
        config["temporal_output_basis"] = "residual"
        return config

    def augmented_loss(**kwargs: Any) -> torch.Tensor:
        return _freeze_only_prediction_loss(
            native_loss=native_loss,
            mixed_module=mixed,
            state=state,
            **kwargs,
        )

    mixed.load_model_for_variant = load_model
    mixed.model_config = model_config
    mixed.mixed_prediction_loss = augmented_loss
    return state


def _rewrite_report(output: Path, *, state: dict[str, Any]) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    model_state = state["model"]
    model = model_state["model"]
    trainable_roots = tuple(
        result["representation_freeze"]["trainable_top_level_modules"]
    )
    checks = {
        "source_checkpoint_exact": (
            result["source_checkpoint"]["sha256"]
            == SOURCE_CHECKPOINT_SHA256
        ),
        "optimizer_steps_exact": (
            int(result["optimizer_steps"]) == OPTIMIZER_STEPS
        ),
        "seed_exact": int(result["seed"]) == 14321,
        "variant_exact": result["variant"] == causal.NATIVE_VARIANT,
        "native_mse_sigreg_0p09_retained": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "one_auxiliary_call_per_step": (
            int(state["loss_calls"]) == OPTIMIZER_STEPS
        ),
        "action_encoder_frozen_and_unchanged": (
            model_state["action_encoder_state_sha256_before"]
            == hashing._state_sha256(model.action_encoder)
            and all(
                not parameter.requires_grad
                for parameter in model.action_encoder.parameters()
            )
        ),
        "pred_proj_buffers_frozen": (
            model_state["pred_proj_buffer_state_sha256_before"]
            == anchor._buffer_state_sha256(model.pred_proj)
            and model.pred_proj.training is False
        ),
        "only_existing_predictor_and_head_trainable": (
            trainable_roots == ("pred_proj", "predictor")
        ),
        "saved_model_parameter_count_unchanged": (
            model_state["parameter_count"]
            == sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Freeze-only terminal contract failed: {checks}")
    result["motion_canonical_response_only_freeze_contract"] = {
        "checks": checks,
        "objective": (
            "native_mse + 0.09*SIGReg + "
            "0.09*(centered_response + canonical_assignment_0p5)"
        ),
        "continuation_source": {
            "checkpoint": result["source_checkpoint"]["path"],
            "sha256": SOURCE_CHECKPOINT_SHA256,
            "source_optimizer_steps": 2048,
        },
        "first_loss_components": state["first_components"],
        "last_loss_components": state["last_components"],
        "fresh_optimizer_steps": OPTIMIZER_STEPS,
        "action_encoder_frozen": True,
        "pred_proj_buffers_frozen_in_inference_mode": True,
        "training_only_frozen_teacher": False,
        "direct_normalized_common_center_mse_included": False,
        "learned_parameters_added_to_saved_model": 0,
        "model_modules_added_to_saved_model": 0,
        "inference_compute_added": 0,
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
    args = _validate_args(effective)
    trainer = causal._load_trainer()
    native_twin._install_runtime(trainer)
    state = _install_freeze_only(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *residual._trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous
    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_report(output, state=state)
        sidecar = output / "motion_canonical_response_only_freeze_method_v1.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": CANDIDATE,
                    "source": str(THIS_SOURCE),
                    "source_sha256": _sha256(THIS_SOURCE),
                    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                    "fresh_optimizer_steps": OPTIMIZER_STEPS,
                    "training_report": str(report),
                    "training_report_sha256": _sha256(report),
                    "training_only_frozen_teacher": False,
                    "learned_parameters_added_to_saved_model": 0,
                    "model_modules_added_to_saved_model": 0,
                    "inference_compute_added": 0,
                    "single_seed_discovery": True,
                    "public_test_opened": False,
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
