#!/usr/bin/env python3
"""Warm-start the successful residual-transition joint objective continuously.

The prior residual-transition arm reset the existing final prediction projection
to zero.  That arm passed the Motion mechanism screen but destroyed standard
PushT CEM.  This runner changes only the initialization path: one fixed scalar
``alpha`` continuously moves both temporal coordinates and output semantics
from the loaded native LeWM function to the final residual-transition function::

    T_alpha(H) = [z0, z1 - alpha*z0, ...]
    prediction = F_theta(T_alpha(H), A) + alpha*H

At alpha=0 this is exactly the loaded absolute LeWM predictor.  At alpha=1 it
is exactly the ordinary causal-transition/residual LeWM saved in config.json.
Alpha ramps linearly over the first 1,024 optimizer steps and is then held at
one for the remaining 1,024 steps.  The only auxiliary is the already-defined
pair-normalized exact-future term at its existing 0.09 weight.  No parameter,
module, head, hidden label, or inference-only wrapper is added.
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


CANDIDATE = "pusht_motion_damping_temporal_homotopy_exact_future_v1"
OBJECTIVE = "pair_normalized_exact_future"
TOTAL_STEPS = 2048
RAMP_STEPS = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def alpha_for_optimizer_step(optimizer_step: int) -> float:
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be nonnegative")
    return min(1.0, float(optimizer_step) / float(RAMP_STEPS))


def temporal_homotopy_input(
    embedding: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if embedding.ndim != 3 or embedding.size(1) < 1:
        raise ValueError("embedding must have shape (B,T,D) with T >= 1")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    return torch.cat(
        [
            embedding[:, :1],
            embedding[:, 1:] - float(alpha) * embedding[:, :-1],
        ],
        dim=1,
    )


def temporal_homotopy_prediction(
    *,
    model: torch.nn.Module,
    native_predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    embedding: torch.Tensor,
    action_embedding: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply the continuous training transform using the unchanged model."""

    transformed = temporal_homotopy_input(embedding, alpha=alpha)
    previous_input = getattr(model, "temporal_input_basis", "absolute")
    previous_output = getattr(model, "temporal_output_basis", "absolute")
    model.temporal_input_basis = "absolute"
    model.temporal_output_basis = "absolute"
    try:
        base_prediction = native_predict(transformed, action_embedding)
    finally:
        model.temporal_input_basis = previous_input
        model.temporal_output_basis = previous_output
    return base_prediction + float(alpha) * embedding


def _install_temporal_homotopy(trainer: Any) -> dict[str, Any]:
    mixed = trainer.trainer.mixed
    native_loader = mixed.load_model_for_variant
    native_model_config = mixed.model_config
    native_evaluate = mixed.pilot.evaluate_model
    native_adamw = torch.optim.AdamW
    state: dict[str, Any] = {
        "optimizer_step": 0,
        "alpha": 0.0,
        "model": None,
        "grad_predict_calls": 0,
        "grad_calls_by_optimizer_step": {},
    }

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = native_loader(*args, **kwargs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        initial_state_sha256 = _state_sha256(model)
        native_predict = model.predict
        # The wrapper owns both transforms during training.  Keeping the core
        # attributes absolute makes alpha=0 bit-identical to the loaded model.
        model.temporal_input_basis = "absolute"
        model.temporal_output_basis = "absolute"

        def predict_with_temporal_homotopy(
            self: Any,
            embedding: torch.Tensor,
            action_embedding: torch.Tensor,
        ) -> torch.Tensor:
            if self.training and torch.is_grad_enabled():
                completed = int(state["optimizer_step"])
                state["grad_predict_calls"] = int(state["grad_predict_calls"]) + 1
                state["grad_calls_by_optimizer_step"].setdefault(
                    completed, []
                ).append(float(state["alpha"]))
            return temporal_homotopy_prediction(
                model=self,
                native_predict=native_predict,
                embedding=embedding,
                action_embedding=action_embedding,
                alpha=float(state["alpha"]),
            )

        model.predict = types.MethodType(predict_with_temporal_homotopy, model)
        observed_count = sum(parameter.numel() for parameter in model.parameters())
        observed_state_sha256 = _state_sha256(model)
        if observed_count != parameter_count:
            raise RuntimeError("Temporal homotopy changed LeWM parameter count")
        if observed_state_sha256 != initial_state_sha256:
            raise RuntimeError("Temporal homotopy mutated loaded model state")
        state["model"] = {
            "model": model,
            "parameter_count_before": parameter_count,
            "parameter_count_after": observed_count,
            "loaded_state_sha256": initial_state_sha256,
            "installed_state_sha256": observed_state_sha256,
            "hard_parameter_reset": False,
        }
        updated = dict(receipt)
        updated["temporal_homotopy"] = {
            "input_formula": "[z0,z1-alpha*z0,...]",
            "output_formula": "F_theta(T_alpha(H),A)+alpha*H",
            "alpha_start": 0.0,
            "alpha_end": 1.0,
            "ramp_steps": RAMP_STEPS,
            "hold_steps": TOTAL_STEPS - RAMP_STEPS,
            "loaded_state_unchanged": True,
            "hard_parameter_reset": False,
            "learned_parameters_added": 0,
        }
        return model, updated

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(native_model_config(*args, **kwargs))
        # A saved checkpoint has alpha=1 and therefore needs no wrapper.
        config["temporal_input_basis"] = "causal_transition"
        config["temporal_output_basis"] = "residual"
        return config

    def evaluate(
        model: torch.nn.Module,
        evaluation: dict[str, torch.Tensor],
        *,
        device: torch.device,
        batch_size: int,
    ) -> dict[str, Any]:
        metrics = native_evaluate(
            model,
            evaluation,
            device=device,
            batch_size=batch_size,
        )
        metrics["temporal_homotopy"] = {
            "optimizer_step": int(state["optimizer_step"]),
            "alpha": float(state["alpha"]),
        }
        return metrics

    class TemporalHomotopyAdamW(native_adamw):
        def step(self, closure=None):
            result = super().step(closure=closure)
            state["optimizer_step"] = int(state["optimizer_step"]) + 1
            state["alpha"] = alpha_for_optimizer_step(state["optimizer_step"])
            return result

    mixed.load_model_for_variant = load_model
    mixed.model_config = model_config
    mixed.pilot.evaluate_model = evaluate
    torch.optim.AdamW = TemporalHomotopyAdamW
    state["restore"] = {"adamw": native_adamw, "evaluate": native_evaluate}
    return state


def _rewrite_training_report(
    output: Path,
    *,
    homotopy_state: dict[str, Any],
    auxiliary_state: dict[str, Any],
) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    model_state = homotopy_state.get("model") or {}
    grouping = result["batch"].get("motion_damping_twin_grouping")
    snapshots = result["snapshots"]
    observed_schedule = [
        {
            "optimizer_step": int(row["optimizer_step"]),
            "alpha": float(row["hidden_evaluation"]["temporal_homotopy"]["alpha"]),
        }
        for row in snapshots
    ]
    expected_schedule = [
        {"optimizer_step": 0, "alpha": 0.0},
        {"optimizer_step": 1024, "alpha": 1.0},
        {"optimizer_step": 2048, "alpha": 1.0},
    ]
    grad_calls = homotopy_state["grad_calls_by_optimizer_step"]
    paired_call_clock_exact = (
        tuple(sorted(grad_calls)) == tuple(range(TOTAL_STEPS))
        and all(
            len(grad_calls[step]) == 2
            and grad_calls[step][0] == grad_calls[step][1]
            and grad_calls[step][0] == alpha_for_optimizer_step(step)
            for step in range(TOTAL_STEPS)
        )
    )
    checks = {
        "native_mse_sigreg_0p09": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "complete_twin_grouping": bool(
            grouping
            and grouping.get("enabled") is True
            and grouping.get("condition_rows_per_group") == 4
        ),
        "one_auxiliary_call_per_optimizer_step": (
            int(auxiliary_state["loss_calls"]) == TOTAL_STEPS
        ),
        "two_predict_calls_share_one_preupdate_clock": (
            int(homotopy_state["grad_predict_calls"]) == 2 * TOTAL_STEPS
            and paired_call_clock_exact
        ),
        "optimizer_step_exact": int(homotopy_state["optimizer_step"]) == TOTAL_STEPS,
        "terminal_alpha_exact": float(homotopy_state["alpha"]) == 1.0,
        "snapshot_schedule_exact": observed_schedule == expected_schedule,
        "parameter_count_unchanged": (
            model_state.get("parameter_count_before")
            == model_state.get("parameter_count_after")
        ),
        "loaded_state_unmutated_at_install": (
            model_state.get("loaded_state_sha256")
            == model_state.get("installed_state_sha256")
        ),
        "no_hard_parameter_reset": model_state.get("hard_parameter_reset") is False,
        "objective_exact": auxiliary_state["objective_name"] == OBJECTIVE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Temporal-homotopy exact-future contract failed: {checks}")
    result["temporal_homotopy_exact_future_contract"] = {
        "checks": checks,
        "input_formula": "T_alpha(H)=[z0,z1-alpha*z0,...]",
        "output_formula": "prediction=F_theta(T_alpha(H),A)+alpha*H",
        "step0_function": "loaded_native_absolute_LeWM_exact",
        "terminal_function": "causal_transition_input_plus_residual_output",
        "alpha_schedule": {
            "linear_ramp_optimizer_steps": RAMP_STEPS,
            "final_hold_optimizer_steps": TOTAL_STEPS - RAMP_STEPS,
            "snapshots": observed_schedule,
            "grad_predict_calls": int(homotopy_state["grad_predict_calls"]),
            "same_alpha_for_native_and_auxiliary_each_step": (
                paired_call_clock_exact
            ),
            "representative_preupdate_calls": {
                str(step): grad_calls[step]
                for step in (0, RAMP_STEPS - 1, RAMP_STEPS, TOTAL_STEPS - 1)
            },
        },
        "auxiliary": "pair_normalized_exact_future_residual_v1",
        "auxiliary_weight": paired.CCRM_WEIGHT,
        "first_loss_components": auxiliary_state["first_components"],
        "last_loss_components": auxiliary_state["last_components"],
        "predictor_inputs_detached_for_auxiliary": True,
        "targets_detached_inside_auxiliary": True,
        "hard_parameter_reset": False,
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "component_balance_hyperparameters_added": 0,
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
    if args.input_basis != "causal_transition" or args.optimizer_steps != TOTAL_STEPS:
        raise RuntimeError("Temporal homotopy exact-future is fixed to 2048 steps")
    trainer = causal._load_trainer()
    native_twin._install_runtime(trainer)
    homotopy_state = _install_temporal_homotopy(trainer)
    auxiliary_state = paired._install_paired_auxiliary(
        trainer,
        objective=exact_future.pair_normalized_exact_future_residual,
        objective_name=OBJECTIVE,
        response_metric_key="normalized_exact_future_residual_by_group",
    )
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *residual._trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous
        torch.optim.AdamW = homotopy_state["restore"]["adamw"]

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(
            output,
            homotopy_state=homotopy_state,
            auxiliary_state=auxiliary_state,
        )
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("temporal_input_basis") != "causal_transition"
            or config.get("temporal_output_basis") != "residual"
        ):
            raise RuntimeError("Saved terminal model config does not match alpha=1")
        sidecar = output / "temporal_homotopy_exact_future_method_v1.json"
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
                    "objective": (
                        "native_mse + 0.09*SIGReg + "
                        "0.09*pair_normalized_exact_future"
                    ),
                    "input_formula": "[z0,z1-alpha*z0,...]",
                    "output_formula": "F_theta(T_alpha(H),A)+alpha*H",
                    "terminal_temporal_input_basis": "causal_transition",
                    "terminal_temporal_output_basis": "residual",
                    "hard_parameter_reset": False,
                    "learned_parameters_added": 0,
                    "model_modules_added": 0,
                    "auxiliary_terms_added": 1,
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
