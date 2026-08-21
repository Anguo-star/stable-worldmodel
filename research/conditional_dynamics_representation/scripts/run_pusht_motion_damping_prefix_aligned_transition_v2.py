#!/usr/bin/env python3
"""Train Motion Damping with leakage-free aligned transition prefixes.

For every transition target ``z[k]`` the predictor receives only the prefix
ending at ``z[k-1]``.  Historical differences are paired with their causing
actions and the final token remains the current state paired with the query
action.  This preserves the native supervision set while adding no parameter,
loss term, label, or pair metadata.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_causal_transition_basis_v1 as causal,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_terminal_aligned_transition_v1 as terminal,
)


CANDIDATE = "pusht_motion_damping_prefix_aligned_transition_v2"
NATIVE_VARIANT = causal.NATIVE_VARIANT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_model_predict(model: Any) -> dict[str, Any]:
    """Replace one causal pass by one leakage-free pass per prefix."""

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    native_predict = model.predict
    model.temporal_input_basis = "absolute"

    def predict_prefix_aligned(
        self: Any,
        embedding: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim != 3 or action_embedding.ndim != 3:
            raise ValueError("Prefix-aligned inputs must be rank-3 tensors")
        if embedding.shape[:2] != action_embedding.shape[:2]:
            raise ValueError(
                "Prefix-aligned embedding/action lengths must match: "
                f"embedding={tuple(embedding.shape)}, "
                f"action={tuple(action_embedding.shape)}"
            )
        outputs = []
        for length in range(1, embedding.size(1) + 1):
            history = embedding[:, :length]
            aligned = terminal.terminal_aligned_transition_basis(history)
            previous = self.temporal_input_basis
            self.temporal_input_basis = "absolute"
            try:
                prediction = native_predict(
                    aligned,
                    action_embedding[:, :length],
                )
            finally:
                self.temporal_input_basis = previous
            outputs.append(prediction[:, -1:])
        return torch.cat(outputs, dim=1)

    model.predict = types.MethodType(predict_prefix_aligned, model)
    observed_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if observed_count != parameter_count:
        raise RuntimeError("Prefix alignment changed model parameter count")
    return {
        "per_prefix_formula": (
            "predict z[k] from [z1-z0,...,z[k-1]-z[k-2],z[k-1]] "
            "and [a0,...,a[k-1]]"
        ),
        "learned_parameters_added": 0,
        "predictor_calls_per_history3_forward": 3,
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
    }

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = causal_load_model(*args, **kwargs)
        method = _install_model_predict(model)
        state["model"] = method
        receipt = dict(receipt)
        receipt["temporal_input_basis"] = "prefix_aligned_transition"
        receipt["prefix_aligned_transition"] = method
        return model, receipt

    def model_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        config = copy.deepcopy(causal_model_config(*args, **kwargs))
        config["temporal_input_basis"] = "absolute"
        return config

    def load_release(path: Path) -> dict[str, Any]:
        release = causal_release_loader(path)
        followup = release["training"]["learnability_followup"]
        followup.update(
            {
                "candidate": "prefix_aligned_transition_v2",
                "fixed_checkpoint_step": 1024,
                "prediction_supervision": "native_identifiable_future",
            }
        )
        return release

    mixed.load_model_for_variant = load_model
    mixed.model_config = model_config
    trainer.load_motion_damping_icl_release = load_release
    return state


def _rewrite_training_report(output: Path, state: dict[str, Any]) -> Path:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["result"]["prefix_aligned_transition_contract"] = {
        **(state["model"] or {}),
        "native_prediction_supervision_retained": True,
        "standard_rows_transition_indices": [0, 1, 2],
        "hidden_rows_transition_indices": [2],
        "loss_terms_added": 0,
        "hidden_labels_consumed": False,
        "pair_metadata_consumed": False,
        "inference_uses_same_final_prefix_representation": True,
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
        sidecar = output / "prefix_aligned_transition_method_v2.json"
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
                    "native_prediction_supervision_retained": True,
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
