#!/usr/bin/env python3
"""Add the existing single CCRM term to residual-transition LeWM.

Residual-transition LeWM fixes the absolute-future parameterization and gets
Motion response gain close to the mechanism gate, while the already-defined
CCRM objective directly calibrates the remaining paired response.  This arm
adds no parameter or module and uses exactly one existing auxiliary term at
its frozen 0.09 weight.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Callable, Sequence

import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    centered_conditional_response_matching_v1 as ccrm,
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


CANDIDATE = "pusht_motion_damping_residual_transition_ccrm_v1"
NATIVE_VARIANT = causal.NATIVE_VARIANT
CCRM_WEIGHT = 0.09


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_hidden_groups(
    *,
    original_batch_size: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    hidden_count = int(batch_size) - int(original_batch_size)
    if original_batch_size <= 0 or hidden_count <= 0 or hidden_count % 2:
        raise ValueError("CCRM requires complete adjacent hidden pairs")
    return torch.arange(
        original_batch_size,
        batch_size,
        device=device,
        dtype=torch.long,
    ).reshape(-1, 2)


def residual_transition_ccrm_prediction_loss(
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
        raise RuntimeError("Residual-transition CCRM model was not loaded")
    capture = model_state["capture"]
    if capture.get("prediction") is not prediction:
        raise RuntimeError("CCRM loss did not receive the captured native forward")
    history = capture["history"]
    action = capture["action"]
    with torch.random.fork_rng(
        devices=[prediction.device.index or 0]
        if prediction.device.type == "cuda"
        else []
    ):
        with mixed_module.temporary_eval_modules(
            model_state["model"].predictor,
            model_state["model"].pred_proj,
        ):
            auxiliary_prediction = model_state["native_predict"](
                history.detach(), action.detach()
            )
    groups = binary_hidden_groups(
        original_batch_size=original_batch_size,
        batch_size=prediction.size(0),
        device=prediction.device,
    )
    response = state["objective"](
        auxiliary_prediction,
        embeddings[:, 1:],
        groups,
    )
    auxiliary = response["loss"]
    metric = response[state["response_metric_key"]]
    total = native_value + float(state["weight"]) * auxiliary
    components = {
        "native_prediction_loss": float(native_value.detach().float()),
        f"auxiliary_{state['objective_name']}_loss": float(
            auxiliary.detach().float()
        ),
        "total_prediction_loss": float(total.detach().float()),
        "response_error_minimum": float(
            metric.detach().min()
        ),
        "response_error_maximum": float(
            metric.detach().max()
        ),
    }
    state["loss_calls"] += 1
    if state["first_components"] is None:
        state["first_components"] = copy.deepcopy(components)
    state["last_components"] = components
    capture.clear()
    return total


def _install_paired_auxiliary(
    trainer: Any,
    *,
    objective: Callable[..., dict[str, Any]],
    objective_name: str,
    response_metric_key: str,
    weight: float = CCRM_WEIGHT,
) -> dict[str, Any]:
    if not torch.isfinite(torch.tensor(float(weight))) or float(weight) <= 0.0:
        raise ValueError("paired auxiliary weight must be finite and positive")
    mixed = trainer.trainer.mixed
    native_loader = mixed.load_model_for_variant
    native_loss = mixed.mixed_prediction_loss
    state: dict[str, Any] = {
        "model": None,
        "loss_calls": 0,
        "first_components": None,
        "last_components": None,
        "objective": objective,
        "objective_name": objective_name,
        "response_metric_key": response_metric_key,
        "weight": float(weight),
    }

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = native_loader(*args, **kwargs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        native_predict = model.predict
        capture: dict[str, torch.Tensor] = {}

        def predict_with_capture(
            self: Any,
            history: torch.Tensor,
            action: torch.Tensor,
        ) -> torch.Tensor:
            prediction = native_predict(history, action)
            if self.training and torch.is_grad_enabled():
                capture.clear()
                capture.update(
                    {
                        "history": history,
                        "action": action,
                        "prediction": prediction,
                    }
                )
            return prediction

        model.predict = types.MethodType(predict_with_capture, model)
        if sum(parameter.numel() for parameter in model.parameters()) != parameter_count:
            raise RuntimeError("CCRM capture changed parameter count")
        state["model"] = {
            "model": model,
            "native_predict": native_predict,
            "capture": capture,
            "parameter_count": parameter_count,
        }
        updated = dict(receipt)
        updated["residual_transition_paired_auxiliary"] = {
            "objective": objective_name,
            "weight": float(weight),
            "learned_parameters_added": 0,
            "model_modules_added": 0,
        }
        return model, updated

    def augmented_loss(**kwargs: Any) -> torch.Tensor:
        return residual_transition_ccrm_prediction_loss(
            native_loss=native_loss,
            mixed_module=mixed,
            state=state,
            **kwargs,
        )

    mixed.load_model_for_variant = load_model
    mixed.mixed_prediction_loss = augmented_loss
    return state


def _install_ccrm(trainer: Any) -> dict[str, Any]:
    return _install_paired_auxiliary(
        trainer,
        objective=ccrm.centered_conditional_response_matching,
        objective_name="ccrm",
        response_metric_key="normalized_error_by_group",
    )


def _rewrite_training_report(
    output: Path,
    *,
    release_state: dict[str, Any],
    residual_state: dict[str, Any],
    ccrm_state: dict[str, Any],
) -> Path:
    report = residual._rewrite_training_report(
        output,
        input_basis="causal_transition",
        release_state=release_state,
        method_state=residual_state,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    checks = {
        "one_ccrm_call_per_optimizer_step": (
            int(ccrm_state["loss_calls"]) == int(result["optimizer_steps"])
        ),
        "residual_transition_contract_passed": all(
            result["residual_output_basis_contract"]["checks"].values()
        ),
        "parameter_count_unchanged": (
            ccrm_state["model"]["parameter_count"]
            == residual_state["model"]["parameter_count_after"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Residual-transition CCRM contract failed: {checks}")
    result["residual_transition_ccrm_contract"] = {
        "checks": checks,
        "auxiliary": "centered_conditional_response_matching_v1",
        "auxiliary_weight": CCRM_WEIGHT,
        "first_loss_components": ccrm_state["first_components"],
        "last_loss_components": ccrm_state["last_components"],
        "predictor_inputs_detached": True,
        "targets_detached_inside_objective": True,
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "inference_path_changed_beyond_residual_basis": False,
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
    args = residual._discovery_args(effective)
    if args.input_basis != "causal_transition" or args.optimizer_steps != 2048:
        raise RuntimeError("Residual-transition CCRM is fixed to 2048 steps")
    trainer = causal._load_trainer()
    release_state = native_twin._install_runtime(trainer)
    residual_state = residual._install_residual_output(
        trainer,
        input_basis="causal_transition",
    )
    ccrm_state = _install_ccrm(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *residual._trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(
            output,
            release_state=release_state,
            residual_state=residual_state,
            ccrm_state=ccrm_state,
        )
        sidecar = output / "residual_transition_ccrm_method_v1.json"
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
                    "objective": "native_mse + 0.09*SIGReg + 0.09*CCRM",
                    "temporal_input_basis": "causal_transition",
                    "temporal_output_basis": "residual",
                    "learned_parameters_added": 0,
                    "model_modules_added": 0,
                    "auxiliary_terms_added": 1,
                    "claim_boundary": {
                        "development_only": True,
                        "single_seed_discovery": True,
                        "public_test_opened": False,
                        "cem_opened": False,
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
