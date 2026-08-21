#!/usr/bin/env python3
"""Run one zero-parameter terminal-aligned Motion Damping MVE.

The support portion of a history is represented as observed transitions and
paired with the actions that caused them.  The final token remains the current
absolute state paired with the query action::

    x = [z1-z0, ..., z(T-1)-z(T-2), z(T-1)]
    c = [a0,    ..., a(T-2),         a(T-1)]

Only the final predictor output is supervised.  Earlier outputs would leak
their own targets through the transition tokens and are deliberately excluded.
The transform adds no learned parameter and consumes no hidden label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Sequence

import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
for root in (REPO_ROOT,):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_causal_transition_basis_v1 as causal,
)


CANDIDATE = "pusht_motion_damping_terminal_aligned_transition_v1"
NATIVE_VARIANT = causal.NATIVE_VARIANT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_aligned_transition_basis(
    embedding: torch.Tensor,
) -> torch.Tensor:
    """Expose action-aligned support transitions and retain the query state."""

    if embedding.ndim != 3:
        raise ValueError(
            "Terminal-aligned input must have shape (B,T,D), "
            f"got {tuple(embedding.shape)}"
        )
    if embedding.size(1) < 1:
        raise ValueError("Terminal-aligned input needs at least one token")
    if embedding.size(1) == 1:
        return embedding
    transitions = embedding[:, 1:] - embedding[:, :-1]
    return torch.cat([transitions, embedding[:, -1:]], dim=1)


def terminal_only_population_balanced_prediction_loss(
    *,
    prediction: torch.Tensor,
    embeddings: torch.Tensor,
    original_batch_size: int,
    conditional_population: str,
) -> torch.Tensor:
    """Apply the same terminal MSE to original and hidden populations."""

    del conditional_population
    if prediction.ndim != 3 or embeddings.ndim != 3:
        raise ValueError("prediction and embeddings must be rank-3 tensors")
    if prediction.shape[:2] != (
        embeddings.shape[0],
        embeddings.shape[1] - 1,
    ):
        raise ValueError("prediction/embedding transition shapes do not match")
    if not 0 < original_batch_size <= prediction.shape[0]:
        raise ValueError("original_batch_size is outside the batch")

    target = embeddings[:, -1]
    terminal_prediction = prediction[:, -1]
    original_loss = torch.square(
        terminal_prediction[:original_batch_size]
        - target[:original_batch_size]
    ).mean()
    if original_batch_size == prediction.shape[0]:
        return original_loss
    hidden_loss = torch.square(
        terminal_prediction[original_batch_size:]
        - target[original_batch_size:]
    ).mean()
    return 0.5 * (original_loss + hidden_loss)


def _install_model_predict(model: Any) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    native_predict = model.predict
    model.temporal_input_basis = "absolute"

    def predict_terminal_aligned(
        self: Any,
        embedding: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.shape[:2] != action_embedding.shape[:2]:
            raise ValueError(
                "Terminal-aligned embedding/action lengths must match: "
                f"embedding={tuple(embedding.shape)}, "
                f"action={tuple(action_embedding.shape)}"
            )
        transformed = terminal_aligned_transition_basis(embedding)
        previous = self.temporal_input_basis
        self.temporal_input_basis = "absolute"
        try:
            return native_predict(transformed, action_embedding)
        finally:
            self.temporal_input_basis = previous

    model.predict = types.MethodType(predict_terminal_aligned, model)
    observed_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if observed_count != parameter_count:
        raise RuntimeError("Terminal alignment changed model parameter count")
    return {
        "formula": "[z1-z0,...,z(T-1)-z(T-2),z(T-1)]",
        "action_formula": "[a0,...,a(T-2),a(T-1)]",
        "learned_parameters_added": 0,
        "parameter_count_before": int(parameter_count),
        "parameter_count_after": int(observed_count),
    }


def _install_overlay(trainer: Any) -> dict[str, Any]:
    causal_state = causal._install_overlay(trainer)
    mixed = trainer.trainer.mixed
    causal_load_model = mixed.load_model_for_variant
    causal_model_config = mixed.model_config
    causal_release_loader = trainer.load_motion_damping_icl_release
    state: dict[str, Any] = {
        "causal_overlay": causal_state,
        "model": None,
        "loss": None,
    }

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = causal_load_model(*args, **kwargs)
        method = _install_model_predict(model)
        state["model"] = method
        receipt = dict(receipt)
        receipt["temporal_input_basis"] = "terminal_aligned_transition"
        receipt["terminal_aligned_transition"] = method
        return model, receipt

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(causal_model_config(*args, **kwargs))
        # The experiment wrapper owns the transform.  Persisting a second
        # core basis in the config would apply two transforms on reload.
        config["temporal_input_basis"] = "absolute"
        return config

    def load_release(path: Path) -> dict[str, Any]:
        release = causal_release_loader(path)
        followup = release["training"]["learnability_followup"]
        followup.update(
            {
                "candidate": "terminal_aligned_transition_v1",
                "fixed_checkpoint_step": 1024,
                "prediction_supervision": "terminal_only_all_populations",
            }
        )
        return release

    def prediction_loss(**kwargs: Any) -> torch.Tensor:
        loss = terminal_only_population_balanced_prediction_loss(**kwargs)
        state["loss"] = {
            "scope": "terminal_only_all_populations",
            "population_balanced": True,
            "new_loss_term_added": False,
        }
        return loss

    mixed.load_model_for_variant = load_model
    mixed.model_config = model_config
    mixed.mixed_prediction_loss = prediction_loss
    trainer.load_motion_damping_icl_release = load_release
    return state


def _rewrite_training_report(output: Path, state: dict[str, Any]) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    result["prediction_supervision"] = {
        "standard_rows_transition_indices": [2],
        "hidden_rows_transition_indices": [2],
        "hidden_rows_excluded_transition_indices": [0, 1],
        "standard_rows_excluded_transition_indices": [0, 1],
        "reason": (
            "only the final output is leakage-free under the "
            "terminal-aligned transition representation"
        ),
        "public_test_used": False,
    }
    result["terminal_aligned_transition_contract"] = {
        **(state["model"] or {}),
        **(state["loss"] or {}),
        "hidden_labels_consumed": False,
        "pair_metadata_consumed": False,
        "inference_uses_same_terminal_representation": True,
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
    state = _install_overlay(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *effective]
        trainer.main()
    finally:
        sys.argv = previous

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(output, state)
        sidecar = output / "terminal_aligned_transition_method_v1.json"
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
                    "representation": state["model"],
                    "prediction_supervision": state["loss"],
                    "learned_parameters_added": 0,
                    "loss_terms_added": 0,
                    "hidden_labels_consumed": False,
                    "pair_metadata_consumed": False,
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
