#!/usr/bin/env python3
"""Run one zero-parameter causal-transition-basis ActionDelay candidate.

This is a thin discovery wrapper around ContextWorld's existing H7 trainer.
It changes only the coordinates presented to the existing LeWM Predictor:

    [z0, z1, ..., zT] -> [z0, z1-z0, ..., zT-z(T-1)].

The transform is prefix-invertible, adds no parameters, keeps the native
absolute-future MSE + SIGReg objective, and is used by both training and
autoregressive rollout through ``LeWM.predict``.  The historical synthesized
release remains bound to its original StableWM commit; this wrapper installs
the candidate method on that runtime without changing its state dict.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
TRAINER_SOURCE = CONTEXTWORLD_ROOT / "scripts/train_tworoom_step1.py"

BASIS_NAME = "causal_transition"
BASE_RUNTIME_COMMIT = "ad2bc44579f2b5b65c004fd2c9d8edc8ebaa43ce"
CANDIDATE_ID = "action_delay_h7_causal_transition_basis_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _causal_transition_basis(embedding: Any) -> Any:
    if embedding.ndim != 3:
        raise ValueError(
            "LeWM temporal input must have shape (B,T,D), "
            f"got {tuple(embedding.shape)}"
        )
    if embedding.size(1) < 1:
        raise ValueError("LeWM temporal input must contain at least one token")
    import torch

    return torch.cat(
        [embedding[:, :1], embedding[:, 1:] - embedding[:, :-1]],
        dim=1,
    )


def _install_runtime_basis(lewm_class: type[Any]) -> dict[str, Any]:
    """Install the same config-carried inference transform on an old runtime."""

    if "temporal_input_basis" in inspect.signature(lewm_class.__init__).parameters:
        return {"native_support": True, "patched": False}
    if getattr(lewm_class, "_causal_transition_basis_v1_installed", False):
        return {"native_support": False, "patched": False}

    native_init = lewm_class.__init__
    native_predict = lewm_class.predict

    def init(
        self: Any,
        *args: Any,
        temporal_input_basis: str = "absolute",
        **kwargs: Any,
    ) -> None:
        native_init(self, *args, **kwargs)
        value = str(temporal_input_basis).strip().lower()
        if value not in {"absolute", BASIS_NAME}:
            raise ValueError(f"Unsupported temporal input basis: {value!r}")
        self.temporal_input_basis = value

    def predict(self: Any, embedding: Any, action_embedding: Any) -> Any:
        value = getattr(self, "temporal_input_basis", "absolute")
        if value == BASIS_NAME:
            embedding = _causal_transition_basis(embedding)
        elif value != "absolute":
            raise ValueError(
                "Loaded LeWM checkpoint has unsupported temporal input basis "
                f"{value!r}"
            )
        return native_predict(self, embedding, action_embedding)

    lewm_class.__init__ = init
    lewm_class.predict = predict
    lewm_class._causal_transition_basis_v1_installed = True
    return {"native_support": False, "patched": True}


def _load_trainer() -> Any:
    name = "_contextworld_action_delay_causal_transition_basis_v1"
    specification = importlib.util.spec_from_file_location(name, TRAINER_SOURCE)
    if specification is None or specification.loader is None:
        raise ImportError(TRAINER_SOURCE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _install_trainer_overlay(trainer: Any) -> None:
    native_load = trainer.load_stable_worldmodel
    native_compose = trainer._compose_model_config
    native_objective = trainer._training_objective_spec

    def load_stable_worldmodel(*args: Any, **kwargs: Any):
        swm, stable_repo, stable_commit = native_load(*args, **kwargs)
        if stable_commit != BASE_RUNTIME_COMMIT:
            raise RuntimeError(
                "Transition-basis discovery must use the release-compatible "
                f"runtime {BASE_RUNTIME_COMMIT}, got {stable_commit}"
            )
        from stable_worldmodel.wm.lewm.lewm import LeWM

        _install_runtime_basis(LeWM)
        return swm, stable_repo, stable_commit

    def compose_model_config(*args: Any, **kwargs: Any):
        from omegaconf import open_dict

        cfg = native_compose(*args, **kwargs)
        with open_dict(cfg):
            cfg.model.temporal_input_basis = BASIS_NAME
        return cfg

    def training_objective_spec(*args: Any, **kwargs: Any):
        result = dict(native_objective(*args, **kwargs))
        result["temporal_input_basis"] = BASIS_NAME
        result["learned_parameters_added"] = 0
        result["future_target"] = "absolute_latent"
        return result

    trainer.load_stable_worldmodel = load_stable_worldmodel
    trainer._compose_model_config = compose_model_config
    trainer._training_objective_spec = training_objective_spec


def _validate_discovery_args(args: Any) -> None:
    checks = {
        "model": args.model_id == "H7_ActionDelay_Paired_LeWM",
        "profile": args.profile == "icl_formal",
        "steps": int(args.expected_optimizer_steps) == 1024,
        "seed": int(args.seed) == 3072,
        "split_seed": int(args.data_split_seed) == 3072,
        "regularizer": args.lewm_regularizer == "sigreg",
        "sigreg_weight": float(args.lewm_sigreg_weight) == 0.09,
        "no_vcreg": (
            float(args.lewm_std_weight) == 0.0
            and float(args.lewm_cov_weight) == 0.0
        ),
        "fresh": args.resume_policy == "never",
        "runtime": args.stablewm_ref == BASE_RUNTIME_COMMIT,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Causal-transition discovery contract failed: "
            + json.dumps(checks, sort_keys=True)
        )


def main(argv: Sequence[str] | None = None) -> int:
    trainer = _load_trainer()
    _install_trainer_overlay(trainer)
    original_argv = sys.argv
    if argv is not None:
        sys.argv = [str(THIS_SOURCE), *argv]
    try:
        args = trainer.parse_args()
        _validate_discovery_args(args)
        result = trainer.run(args)
    finally:
        sys.argv = original_argv

    if trainer._process_is_global_zero():
        sidecar = Path(args.report).resolve().with_suffix(".method.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": CANDIDATE_ID,
                    "source": str(THIS_SOURCE),
                    "source_sha256": _sha256(THIS_SOURCE),
                    "temporal_input_basis": BASIS_NAME,
                    "formula": "[z0,z1-z0,...,zT-z(T-1)]",
                    "prefix_invertible": True,
                    "learned_parameters_added": 0,
                    "loss_added": 0,
                    "absolute_future_target_retained": True,
                    "training_report": str(Path(args.report).resolve()),
                    "training_passed": bool(result.get("passed")),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "passed": bool(result.get("passed")),
                    "candidate": CANDIDATE_ID,
                    "global_step": result.get("training", {}).get(
                        "global_step"
                    ),
                    "report": str(Path(args.report).resolve()),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
