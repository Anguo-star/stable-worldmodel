#!/usr/bin/env python3
"""Motion MVE for joint history- and action-intervention anchoring.

The matched hidden rows train the canonical history-conditioned response.  On
ordinary rows, the existing point-function anchor is retained and one fixed
counterfactual is added: support actions stay unchanged while final query
action embeddings are cyclically permuted across the ordinary batch.  The
student action response is matched to the frozen source action response.

The source checkpoint, optimizer budget, weights, cycle, and gate order are
fixed.  The saved LeWM gains no parameter, module, or inference computation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch


sys.modules.setdefault("flash_attn", None)

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    action_intervention_function_anchor_v1 as action_anchor,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_canonical_response_function_anchor_v1 as base,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_residual_transition_function_anchor_v1 as anchor,
)


CANDIDATE = "pusht_motion_damping_action_intervention_anchor_v1"
OPTIMIZER_STEPS = 1024
CYCLIC_SHIFT = 1


def cyclic_final_query_action(
    action_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep support actions and cycle only the final query action."""

    if not torch.is_tensor(action_embedding) or action_embedding.ndim != 3:
        raise ValueError("action_embedding must have shape (B,T,D)")
    if action_embedding.size(0) < 2 or action_embedding.size(1) < 1:
        raise ValueError("action intervention requires B>=2 and T>=1")
    indices = torch.arange(
        action_embedding.size(0),
        device=action_embedding.device,
    )
    source_indices = torch.roll(indices, shifts=CYCLIC_SHIFT, dims=0)
    if bool((indices == source_indices).any()):
        raise RuntimeError("fixed cyclic action intervention is not a derangement")
    counterfactual = action_embedding.detach().clone()
    counterfactual[:, -1] = action_embedding.detach()[source_indices, -1]
    if action_embedding.size(1) > 1 and not torch.equal(
        counterfactual[:, :-1], action_embedding.detach()[:, :-1]
    ):
        raise RuntimeError("action intervention changed support actions")
    return counterfactual, source_indices


def _install_action_intervention(
    *,
    mixed_module: Any,
    state_ref: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    original = anchor.normalized_function_anchor
    trace: dict[str, Any] = {
        "calls": 0,
        "first": None,
        "last": None,
    }

    def point_and_action_anchor(
        *,
        student_prediction: torch.Tensor,
        teacher_prediction: torch.Tensor,
        history: torch.Tensor,
        epsilon: float = anchor.ANCHOR_EPSILON,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_loss, point_scale = original(
            student_prediction=student_prediction,
            teacher_prediction=teacher_prediction,
            history=history,
            epsilon=epsilon,
        )
        state = state_ref.get("state")
        if not state or not state.get("model"):
            raise RuntimeError("action-intervention model state is unavailable")
        model_state = state["model"]
        model = model_state["model"]
        capture = model_state["capture"]
        action_embedding = capture.get("action_embedding")
        captured_history = capture.get("history")
        if action_embedding is None or captured_history is None:
            raise RuntimeError("action-intervention forward capture is unavailable")
        count = int(student_prediction.size(0))
        ordinary_history = captured_history[:count].detach()
        ordinary_actions = action_embedding[:count].detach()
        if ordinary_history.shape != history.shape or not torch.equal(
            ordinary_history, history.detach()
        ):
            raise RuntimeError("ordinary history slice changed before action anchor")
        counterfactual_actions, source_indices = cyclic_final_query_action(
            ordinary_actions
        )
        query_action_change = (
            counterfactual_actions[:, -1] - ordinary_actions[:, -1]
        ).square().mean()
        if not bool(query_action_change > 0.0):
            raise RuntimeError("cyclic query-action intervention has zero energy")

        with mixed_module.temporary_eval_modules(
            model.predictor,
            model.pred_proj,
        ):
            student_counterfactual = model_state["native_predict"](
                ordinary_history,
                counterfactual_actions,
            )
        with torch.no_grad():
            teacher_counterfactual = anchor.frozen_reference_prediction(
                predictor=model_state["teacher_predictor"],
                pred_proj=model_state["teacher_pred_proj"],
                history=ordinary_history,
                action_embedding=counterfactual_actions,
            )
        action_result = action_anchor.normalized_action_response_anchor(
            student_base=student_prediction[:, -1],
            student_counterfactual=student_counterfactual[:, -1],
            teacher_base=teacher_prediction[:, -1],
            teacher_counterfactual=teacher_counterfactual[:, -1],
            epsilon=epsilon,
        )
        source_energy = action_result["source_action_response_energy"]
        if not bool(source_energy > 0.0):
            raise RuntimeError("source action response has zero energy")
        combined = point_loss + action_result["loss"]
        row = {
            "point_function_anchor_loss": float(point_loss.detach().float()),
            "action_response_anchor_loss": float(
                action_result["loss"].detach().float()
            ),
            "combined_function_anchor_loss": float(
                combined.detach().float()
            ),
            "source_displacement_energy": float(
                point_scale.detach().float()
            ),
            "source_action_response_energy": float(
                source_energy.detach().float()
            ),
            "query_action_embedding_change_energy": float(
                query_action_change.detach().float()
            ),
            "ordinary_batch_size": count,
            "cycle_shift": CYCLIC_SHIFT,
            "cycle_is_derangement": bool(
                (source_indices != torch.arange(
                    count, device=source_indices.device
                )).all()
            ),
            "support_actions_unchanged": True,
            "teacher_requires_grad": bool(
                teacher_prediction.requires_grad
                or teacher_counterfactual.requires_grad
            ),
        }
        trace["calls"] += 1
        if trace["first"] is None:
            trace["first"] = copy.deepcopy(row)
        trace["last"] = row
        return combined, point_scale

    anchor.normalized_function_anchor = point_and_action_anchor
    return trace, original


def _rewrite_report(
    output: Path,
    *,
    state: dict[str, Any],
    response_trace: dict[str, Any],
    action_trace: dict[str, Any],
) -> Path:
    report = base._rewrite_report(
        output,
        state=state,
        trace=response_trace,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    inherited = result.pop(
        "motion_canonical_response_function_anchor_contract"
    )
    first = action_trace["first"] or {}
    last = action_trace["last"] or {}
    inherited["objective"] = (
        "native_mse + 0.09*SIGReg + "
        "0.09*(canonical_centered_response + canonical_assignment_0p5 + "
        "ordinary_point_function_anchor + ordinary_action_response_anchor)"
    )
    inherited["auxiliary_terms_added"] = 3
    for component_key in ("first_loss_components", "last_loss_components"):
        components = inherited.get(component_key)
        if isinstance(components, dict):
            combined = components.pop(
                "normalized_source_function_anchor_loss", None
            )
            if combined is not None:
                components["combined_point_and_action_function_anchor_loss"] = (
                    combined
                )
    inherited["action_intervention_function_anchor"] = {
        "formula": (
            "normalized MSE of student-vs-source terminal action response"
        ),
        "intervention": (
            "fixed cyclic derangement of final ordinary query-action "
            "embedding; support actions unchanged"
        ),
        "calls": int(action_trace["calls"]),
        "first": first,
        "last": last,
    }
    inherited["checks"].update(
        {
            "action_anchor_calls_exact": (
                int(action_trace["calls"]) == OPTIMIZER_STEPS
            ),
            "step1_point_anchor_exact_zero": (
                float(first.get("point_function_anchor_loss", -1.0)) == 0.0
            ),
            "step1_action_anchor_exact_zero": (
                float(first.get("action_response_anchor_loss", -1.0)) == 0.0
            ),
            "source_action_response_energy_positive": (
                float(first.get("source_action_response_energy", 0.0)) > 0.0
                and float(last.get("source_action_response_energy", 0.0)) > 0.0
            ),
            "query_action_intervention_nonzero": (
                float(first.get(
                    "query_action_embedding_change_energy", 0.0
                )) > 0.0
            ),
            "cycle_fixed_derangement": (
                first.get("cycle_is_derangement") is True
                and int(first.get("cycle_shift", 0)) == CYCLIC_SHIFT
            ),
            "support_actions_unchanged": (
                first.get("support_actions_unchanged") is True
            ),
            "teacher_has_no_gradient_path": (
                first.get("teacher_requires_grad") is False
                and last.get("teacher_requires_grad") is False
            ),
        }
    )
    if not all(inherited["checks"].values()):
        raise RuntimeError(
            "Action-intervention terminal contract failed: "
            f"{inherited['checks']}"
        )
    result["motion_action_intervention_anchor_contract"] = inherited
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": anchor._sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = base._args(effective)
    anchor._validate_source_checkpoint(effective)
    trainer = base.causal._load_trainer()
    base.native_twin._install_runtime(trainer)
    response_trace, original_response = base._install_response_objective()
    state_ref: dict[str, Any] = {}
    action_trace, original_point_anchor = _install_action_intervention(
        mixed_module=trainer.trainer.mixed,
        state_ref=state_ref,
    )
    state = anchor._install_function_anchor(trainer)
    state_ref["state"] = state
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *base.residual._trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous
        anchor.exact_future.pair_normalized_exact_future_residual = (
            original_response
        )
        anchor.normalized_function_anchor = original_point_anchor
    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_report(
            output,
            state=state,
            response_trace=response_trace,
            action_trace=action_trace,
        )
        sidecar = output / "motion_action_intervention_anchor_v1.json"
        if sidecar.exists():
            raise FileExistsError(sidecar)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": CANDIDATE,
                    "source": str(THIS_SOURCE),
                    "source_sha256": anchor._sha256(THIS_SOURCE),
                    "source_checkpoint_sha256": (
                        anchor.SOURCE_CHECKPOINT_SHA256
                    ),
                    "fresh_optimizer_steps": OPTIMIZER_STEPS,
                    "fixed_cycle_shift": CYCLIC_SHIFT,
                    "training_report": str(report),
                    "training_report_sha256": anchor._sha256(report),
                    "training_only_frozen_teacher": True,
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
