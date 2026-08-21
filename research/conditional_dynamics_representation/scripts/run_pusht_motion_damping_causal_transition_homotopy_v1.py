#!/usr/bin/env python3
"""Warm-start Motion Damping through a zero-parameter temporal homotopy."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Iterator, Sequence

import torch


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
for root in (REPO_ROOT, CONTEXTWORLD_ROOT, CONTEXTWORLD_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_causal_transition_basis_v1 as fixed,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_oracle_context_predictor_mve_v1 as response,
)


TOTAL_STEPS = 2048
RAMP_STEPS = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _NoContextIntervention:
    @contextmanager
    def evaluating(self, _name: str) -> Iterator[None]:
        yield


def _args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args(argv)
    checks = {
        "model": args.model == "lewm",
        "variant": args.variant == fixed.NATIVE_VARIANT,
        "seed": args.seed == 14321,
        "optimizer_steps": args.optimizer_steps == TOTAL_STEPS,
        "output": args.output is not None,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Motion transition-homotopy contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return args


def _install_homotopy(trainer: Any) -> dict[str, Any]:
    fixed_state = fixed._install_overlay(trainer)
    mixed = trainer.trainer.mixed
    fixed_release_loader = trainer.load_motion_damping_icl_release
    fixed_load_model = mixed.load_model_for_variant
    native_evaluate = mixed.pilot.evaluate_model
    native_adamw = torch.optim.AdamW
    state: dict[str, Any] = {
        "optimizer_step": 0,
        "alpha": 0.0,
        "model": None,
        "fixed_overlay": fixed_state,
    }

    def load_release(path: Path) -> dict[str, Any]:
        release = fixed_release_loader(path)
        followup = release["training"]["learnability_followup"]
        followup.update(
            {
                "candidate": "causal_transition_homotopy_v1",
                "fixed_checkpoint_step": TOTAL_STEPS,
                "ramp_steps": RAMP_STEPS,
            }
        )
        return release

    def load_model(*args: Any, **kwargs: Any):
        model, receipt = fixed_load_model(*args, **kwargs)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        native_predict = model.predict
        model.temporal_input_basis = "absolute"

        def predict_with_homotopy(
            self: Any,
            embedding: torch.Tensor,
            action_embedding: torch.Tensor,
        ) -> torch.Tensor:
            alpha = float(state["alpha"])
            transformed = torch.cat(
                [
                    embedding[:, :1],
                    embedding[:, 1:] - alpha * embedding[:, :-1],
                ],
                dim=1,
            )
            previous = self.temporal_input_basis
            self.temporal_input_basis = "absolute"
            try:
                return native_predict(transformed, action_embedding)
            finally:
                self.temporal_input_basis = previous

        model.predict = types.MethodType(predict_with_homotopy, model)
        if sum(parameter.numel() for parameter in model.parameters()) != parameter_count:
            raise RuntimeError("Homotopy unexpectedly changed parameter count")
        state["model"] = model
        receipt = dict(receipt)
        receipt["temporal_input_homotopy"] = {
            "formula": "[z0,z1-alpha*z0,...]",
            "alpha_start": 0.0,
            "alpha_end": 1.0,
            "ramp_steps": RAMP_STEPS,
            "hold_steps": TOTAL_STEPS - RAMP_STEPS,
            "learned_parameters_added": 0,
        }
        return model, receipt

    def evaluate(
        model: torch.nn.Module,
        evaluation: dict[str, torch.Tensor],
        *,
        device: torch.device,
        batch_size: int,
    ) -> dict[str, Any]:
        metrics = response.evaluate_with_explicit_context(
            model,
            evaluation,
            controller=_NoContextIntervention(),
            pilot_module=mixed.pilot,
            device=device,
            batch_size=batch_size,
        )
        metrics["temporal_homotopy"] = {
            "optimizer_step": int(state["optimizer_step"]),
            "alpha": float(state["alpha"]),
        }
        return metrics

    class HomotopyAdamW(native_adamw):
        def step(self, closure=None):
            result = super().step(closure=closure)
            state["optimizer_step"] = int(state["optimizer_step"]) + 1
            state["alpha"] = min(
                1.0,
                float(state["optimizer_step"]) / float(RAMP_STEPS),
            )
            return result

    trainer.load_motion_damping_icl_release = load_release
    mixed.load_model_for_variant = load_model
    mixed.pilot.evaluate_model = evaluate
    torch.optim.AdamW = HomotopyAdamW
    state["restore"] = {
        "evaluate": native_evaluate,
        "adamw": native_adamw,
    }
    return state


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = _args(effective)
    trainer = fixed._load_trainer()
    state = _install_homotopy(trainer)
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *effective]
        trainer.main()
    finally:
        sys.argv = previous
        torch.optim.AdamW = state["restore"]["adamw"]

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = output / "training_report.json"
        payload = json.loads(report.read_text(encoding="utf-8"))
        snapshots = payload["result"]["snapshots"]
        observed = [
            {
                "optimizer_step": int(row["optimizer_step"]),
                "alpha": float(
                    row["hidden_evaluation"]["temporal_homotopy"]["alpha"]
                ),
            }
            for row in snapshots
        ]
        if observed != [
            {"optimizer_step": 0, "alpha": 0.0},
            {"optimizer_step": 1024, "alpha": 1.0},
            {"optimizer_step": 2048, "alpha": 1.0},
        ]:
            raise RuntimeError(f"Unexpected homotopy snapshots: {observed}")
        sidecar = output / "causal_transition_homotopy_method_v1.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": "pusht_motion_damping_causal_transition_homotopy_v1",
                    "source": str(THIS_SOURCE),
                    "source_sha256": _sha256(THIS_SOURCE),
                    "training_report": str(report),
                    "training_report_sha256": _sha256(report),
                    "formula": "[z0,z1-alpha*z0,...,zT-alpha*z(T-1)]",
                    "prefix_invertible_for_every_alpha": True,
                    "alpha_schedule": {
                        "start": 0.0,
                        "end": 1.0,
                        "linear_ramp_optimizer_steps": RAMP_STEPS,
                        "final_basis_hold_optimizer_steps": TOTAL_STEPS - RAMP_STEPS,
                    },
                    "learned_parameters_added": 0,
                    "loss_added": 0,
                    "snapshots": observed,
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
