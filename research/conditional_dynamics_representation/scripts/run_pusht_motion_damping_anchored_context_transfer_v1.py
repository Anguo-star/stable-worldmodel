#!/usr/bin/env python3
"""Run one zero-parameter cross-query context-transfer Motion MVE.

Complete forward/reverse twins expose two different queries under the same
two damping endpoints.  For every destination row, this diagnostic takes the
support transitions from the opposite-query row with the same damping mode,
translates that support history so that it ends at the destination current
latent, and retains the destination query action and true future::

    H_s = [z0_s, z1_s, z2_s]
    H_{s->d} = [z0_s + delta, z1_s + delta, z2_d]
    delta = z2_d - z2_s
    A_{s->d} = [a0_s, a1_s, a2_d]

The native LeWM Predictor, parameter count, absolute temporal basis, SIGReg,
optimizer, and inference path are unchanged.  The existing 0.5 hidden-data
weight is split equally between real hidden rows and transferred hidden rows;
there is no tunable auxiliary coefficient.  This is a paired-catalog
diagnostic, not a claim that ordinary unlabelled episodes already provide the
twin correspondence.
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


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_causal_transition_basis_v1 as causal,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_native_twin_sampler_v1 as native_twin,
)


CANDIDATE = "pusht_motion_damping_anchored_context_transfer_v1"
NATIVE_VARIANT = causal.NATIVE_VARIANT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_condition_opposite_query_indices(
    row_count: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Map [F-low,F-high,R-low,R-high] to [R-low,R-high,F-low,F-high]."""

    if row_count <= 0 or row_count % 4:
        raise ValueError("Twin transfer requires complete four-row groups")
    rows = torch.arange(row_count, device=device).reshape(-1, 4)
    return rows[:, [2, 3, 0, 1]].reshape(-1)


def anchored_context_transfer(
    history: torch.Tensor,
    action_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return anchored opposite-query support, actions, and source indices."""

    if history.ndim != 3 or action_embedding.ndim != 3:
        raise ValueError("Transfer inputs must be rank-3 tensors")
    if history.shape[:2] != action_embedding.shape[:2]:
        raise ValueError("History and action token axes must match")
    if history.size(1) < 2:
        raise ValueError("Transfer requires support and query tokens")

    source = same_condition_opposite_query_indices(
        history.size(0), device=history.device
    )
    source_history = history.index_select(0, source)
    destination_current = history[:, -1:]
    offset = destination_current - source_history[:, -1:]
    transferred_history = torch.cat(
        [source_history[:, :-1] + offset, destination_current], dim=1
    )

    source_actions = action_embedding.index_select(
        0, source.to(device=action_embedding.device)
    )
    transferred_actions = torch.cat(
        [source_actions[:, :-1], action_embedding[:, -1:]], dim=1
    )
    return transferred_history, transferred_actions, source


def _install_transfer_overlay(trainer: Any) -> dict[str, Any]:
    """Capture the native training forward and add one transferred forward."""

    mixed = trainer.trainer.mixed
    native_loader = mixed.load_model_for_variant
    native_loss: Callable[..., torch.Tensor] = mixed.mixed_prediction_loss
    state: dict[str, Any] = {
        "forward_calls": 0,
        "loss_calls": 0,
        "first_components": None,
        "last_components": None,
        "model": None,
    }

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = native_loader(*args, **kwargs)
        parameter_count = sum(p.numel() for p in model.parameters())
        native_predict = model.predict
        capture: dict[str, Any] = {}

        def predict_with_capture(
            self: Any,
            history: torch.Tensor,
            action_embedding: torch.Tensor,
        ) -> torch.Tensor:
            prediction = native_predict(history, action_embedding)
            if self.training and torch.is_grad_enabled():
                capture.clear()
                capture.update(
                    {
                        "history": history,
                        "action_embedding": action_embedding,
                        "prediction": prediction,
                    }
                )
                state["forward_calls"] += 1
            return prediction

        model.predict = types.MethodType(predict_with_capture, model)
        observed_count = sum(p.numel() for p in model.parameters())
        if observed_count != parameter_count:
            raise RuntimeError("Context transfer changed parameter count")
        state["model"] = {
            "model": model,
            "native_predict": native_predict,
            "capture": capture,
            "parameter_count_before": int(parameter_count),
            "parameter_count_after": int(observed_count),
        }
        receipt = dict(receipt)
        receipt["anchored_context_transfer"] = {
            "learned_parameters_added": 0,
            "inference_path_changed": False,
            "temporal_input_basis": "absolute",
        }
        return model, receipt

    def transferred_prediction_loss(
        *,
        prediction: torch.Tensor,
        embeddings: torch.Tensor,
        original_batch_size: int,
        conditional_population: str,
    ) -> torch.Tensor:
        if conditional_population != "identifiable_future_only":
            raise RuntimeError("Context transfer requires identifiable-future rows")
        model_state = state.get("model")
        if not model_state:
            raise RuntimeError("Context-transfer model was not loaded")
        capture = model_state["capture"]
        if capture.get("prediction") is not prediction:
            raise RuntimeError("Loss did not receive the captured native forward")
        if prediction.ndim != 3 or embeddings.ndim != 3:
            raise ValueError("Prediction and embeddings must be rank-3")
        if prediction.shape[:2] != (
            embeddings.size(0), embeddings.size(1) - 1
        ):
            raise ValueError("Prediction/target transition shapes changed")
        if not 0 < original_batch_size < prediction.size(0):
            raise ValueError("Transfer requires both original and hidden rows")

        hidden_history = capture["history"][original_batch_size:]
        hidden_actions = capture["action_embedding"][original_batch_size:]
        transferred_history, transferred_actions, _ = (
            anchored_context_transfer(hidden_history, hidden_actions)
        )
        transferred_prediction = model_state["native_predict"](
            transferred_history, transferred_actions
        )

        original = torch.square(
            prediction[:original_batch_size]
            - embeddings[:original_batch_size, 1:]
        ).mean()
        hidden_native = torch.square(
            prediction[original_batch_size:, -1]
            - embeddings[original_batch_size:, -1]
        ).mean()
        hidden_transfer = torch.square(
            transferred_prediction[:, -1]
            - embeddings[original_batch_size:, -1]
        ).mean()
        total = 0.5 * original + 0.25 * hidden_native + 0.25 * hidden_transfer
        components = {
            "original_full_horizon_mse": float(original.detach().float()),
            "hidden_native_terminal_mse": float(
                hidden_native.detach().float()
            ),
            "hidden_transfer_terminal_mse": float(
                hidden_transfer.detach().float()
            ),
            "total_prediction_loss": float(total.detach().float()),
        }
        state["loss_calls"] += 1
        if state["first_components"] is None:
            state["first_components"] = copy.deepcopy(components)
        state["last_components"] = components
        capture.clear()
        return total

    mixed.load_model_for_variant = load_model
    mixed.mixed_prediction_loss = transferred_prediction_loss
    state["native_prediction_loss"] = native_loss
    return state


def _rewrite_training_report(
    output: Path,
    transfer_state: dict[str, Any],
    release_state: dict[str, Any],
) -> Path:
    report = native_twin._rewrite_training_report(output, release_state)
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    model_state = transfer_state.get("model") or {}
    checks = {
        "complete_twin_grouping": result[
            "native_twin_sampler_contract"
        ]["checks"]["complete_twin_grouping"],
        "absolute_temporal_basis": result[
            "native_twin_sampler_contract"
        ]["checks"]["absolute_temporal_basis"],
        "native_sigreg_0p09": (
            result["regularizer"] == "native"
            and float(result["regularizer_weight"]) == 0.09
        ),
        "one_loss_call_per_optimizer_step": (
            int(transfer_state["loss_calls"])
            == int(result["optimizer_steps"])
        ),
        "parameter_count_unchanged": (
            model_state.get("parameter_count_before")
            == model_state.get("parameter_count_after")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Anchored-transfer contract failed: {checks}")
    result["anchored_context_transfer_contract"] = {
        "checks": checks,
        "formula": (
            "H_sd=[z0_s+(z2_d-z2_s),z1_s+(z2_d-z2_s),z2_d]; "
            "A_sd=[a0_s,a1_s,a2_d]"
        ),
        "source_mapping_within_twin": [2, 3, 0, 1],
        "same_damping_mode_preserved_by_row_position": True,
        "query_direction_changed": True,
        "population_weights": {
            "original_full_horizon": 0.5,
            "hidden_native_terminal": 0.25,
            "hidden_transfer_terminal": 0.25,
        },
        "first_loss_components": transfer_state["first_components"],
        "last_loss_components": transfer_state["last_components"],
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "inference_path_changed": False,
        "loss_family": "native_mse_on_real_and_transferred_examples",
        "tunable_auxiliary_weight_added": False,
        "hidden_value_entered_model": False,
        "paired_catalog_row_order_used_by_trainer": True,
        "ordinary_episode_sampler": False,
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
    args: argparse.Namespace = causal._discovery_args(effective)
    trainer = causal._load_trainer()
    release_state = native_twin._install_runtime(trainer)
    transfer_state = _install_transfer_overlay(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *effective]
        trainer.main()
    finally:
        sys.argv = previous

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(
            output, transfer_state, release_state
        )
        sidecar = output / "anchored_context_transfer_method_v1.json"
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
                    "learned_parameters_added": 0,
                    "model_modules_added": 0,
                    "inference_path_changed": False,
                    "tunable_auxiliary_weight_added": False,
                    "paired_catalog_row_order_used_by_trainer": True,
                    "claim_boundary": {
                        "development_only": True,
                        "single_seed_discovery": True,
                        "paired_catalog_diagnostic": True,
                        "ordinary_episode_sampler": False,
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
