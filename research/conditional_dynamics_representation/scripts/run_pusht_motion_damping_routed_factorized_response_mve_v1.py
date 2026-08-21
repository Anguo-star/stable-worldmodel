#!/usr/bin/env python3
"""Run the source-routed rank-64 Motion-Damping factorization MVE.

Standard PushT prediction MSE and the original SIGReg train native LeWM.
Hidden Motion-Damping terminal MSE trains only a zero-start low-rank
query-by-context response branch.  No auxiliary loss is added.  This isolates
the directly observed failure: the branch gradient was weakly corrective at
step 256 while the jointly trained native Predictor trunk moved strongly in
the wrong conditional-response direction.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Callable, Sequence

import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONTEXTWORLD_ROOT = ROOT.parent / "ContextWorld"
CONTEXTWORLD_SCRIPTS = CONTEXTWORLD_ROOT / "scripts"
for source_root in (ROOT, CONTEXTWORLD_ROOT, CONTEXTWORLD_SCRIPTS):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_factorized_context_response_mve_v1 as factorized,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_oracle_context_predictor_mve_v1 as additive,
)


PREREG = (
    ROOT
    / "research/conditional_dynamics_representation/configs/"
    "pusht_motion_damping_routed_factorized_response_mve_v1.yaml"
)
RELEASE_CONFIG = additive.RELEASE_CONFIG
ARTIFACT_ROOT = (
    ROOT
    / "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_routed_factorized_response_mve_v1"
)
EXPECTED_RELEASE_ID = additive.EXPECTED_RELEASE_ID
EXPECTED_RELEASE_SHA256 = additive.EXPECTED_RELEASE_SHA256
FROZEN_SEED = additive.FROZEN_SEED
DEFAULT_STEPS = additive.DEFAULT_STEPS
RANK = 64
ADDED_PARAMETER_COUNT = (
    factorized.QUERY_DIM * RANK
    + factorized.CONTEXT_DIM * RANK
    + RANK * factorized.LATENT_DIM
)
ARMS = {
    "oracle": "mixed_routed_factorized_oracle_context_native_0p09",
    "constant": "mixed_routed_factorized_constant_context_native_0p09",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def routed_prediction_loss(
    *,
    prediction: torch.Tensor,
    embeddings: torch.Tensor,
    original_batch_size: int,
    conditional_population: str,
) -> torch.Tensor:
    """Keep the native scalar loss while detaching the hidden target path."""

    _require(
        conditional_population == "identifiable_future_only",
        "routed MVE requires identifiable-future supervision",
    )
    _require(
        prediction.ndim == 3
        and embeddings.ndim == 3
        and prediction.shape[:2]
        == (embeddings.shape[0], embeddings.shape[1] - 1),
        "prediction/embedding boundary changed",
    )
    _require(
        original_batch_size == additive.ORIGINAL_BATCH_SIZE,
        "original batch partition changed",
    )
    original_loss = torch.square(
        prediction[:original_batch_size]
        - embeddings[:original_batch_size, 1:]
    ).mean()
    hidden_loss = torch.square(
        prediction[original_batch_size:, -1]
        - embeddings[original_batch_size:, -1].detach()
    ).mean()
    return 0.5 * (original_loss + hidden_loss)


def attach_routed_factorized_response(
    model: nn.Module,
    controller: additive.EpisodeContextController,
    gradient_state: dict[str, Any] | None = None,
) -> factorized.FactorizedContextResponse:
    """Attach rank-64 response and stop hidden gradients at native boundaries."""

    predictor = model.predictor
    if hasattr(predictor, "episode_context_response"):
        raise RuntimeError("routed response is already attached")
    reference = next(predictor.parameters())
    response = factorized.FactorizedContextResponse(rank=RANK).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    predictor.add_module("episode_context_response", response)
    native_predict = model.predict

    def predict_with_routed_response(
        self: nn.Module,
        history: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        native = native_predict(history, action_embedding)
        context = controller.for_conditioning(action_embedding)
        training_route = controller.evaluation_condition is None
        branch_history = history.detach() if training_route else history
        branch_action = action_embedding.detach() if training_route else action_embedding
        terminal_delta = self.predictor.episode_context_response(
            branch_history, branch_action, context
        )
        residual = torch.cat(
            [torch.zeros_like(native[:, :-1]), terminal_delta.unsqueeze(1)],
            dim=1,
        )
        if not training_route:
            return native + residual
        split = additive.ORIGINAL_BATCH_SIZE
        _require(native.shape[0] == additive.TOTAL_BATCH_SIZE, "batch size changed")
        # Original rows retain the ordinary native graph. Hidden rows expose
        # values from native LeWM but route their terminal loss only to B(q)c.
        routed_native = torch.cat([native[:split], native[split:].detach()], dim=0)
        return routed_native + residual

    model.predict = MethodType(predict_with_routed_response, model)
    if gradient_state is not None:
        gradient_state.update(
            {
                "backward_calls": 0,
                "first_finite_nonzero_output_gradient_step": None,
                "first_output_gradient_l2_norm": None,
            }
        )

        def capture_output_gradient(gradient: torch.Tensor) -> torch.Tensor:
            gradient_state["backward_calls"] += 1
            norm = float(torch.linalg.vector_norm(gradient.detach().float()))
            if (
                bool(torch.isfinite(gradient).all())
                and norm > 0.0
                and gradient_state["first_finite_nonzero_output_gradient_step"]
                is None
            ):
                gradient_state["first_finite_nonzero_output_gradient_step"] = int(
                    gradient_state["backward_calls"]
                )
                gradient_state["first_output_gradient_l2_norm"] = norm
            return gradient

        response.W_o.weight.register_hook(capture_output_gradient)
    return response


def install_training_overlay(motion: Any, *, arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(arm)
    variant = ARMS[arm]
    mixed = motion.trainer.mixed
    motion.TWIN_GROUP_VARIANTS.add(variant)
    mixed.VARIANT_WEIGHTS[variant] = ("native", 0.09, "identifiable_future_only")
    motion.trainer.DIAGNOSTIC_VARIANTS["lewm"].add(variant)
    _require(
        variant not in mixed.FROZEN_IMAGE_VARIANTS,
        "routed MVE keeps native original/SIGReg training active",
    )
    native_train_variant = mixed.train_variant

    def train_variant_with_routing(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("variant") != variant:
            return native_train_variant(*args, **kwargs)
        controller = additive.EpisodeContextController(arm)
        gradient_state: dict[str, Any] = {}
        native_loader = mixed.load_model_for_variant
        native_evaluator = mixed.pilot.evaluate_model
        native_loss: Callable[..., torch.Tensor] = mixed.mixed_prediction_loss
        loaded_model: nn.Module | None = None
        response: factorized.FactorizedContextResponse | None = None

        def load_with_response(*loader_args: Any, **loader_kwargs: Any):
            nonlocal loaded_model, response
            model, receipt = native_loader(*loader_args, **loader_kwargs)
            if loaded_model is not None:
                raise RuntimeError("routed overlay loaded more than one model")
            loaded_model = model
            response = attach_routed_factorized_response(
                model, controller, gradient_state
            )
            _require(
                response.trainable_parameter_count == ADDED_PARAMETER_COUNT,
                "routed factorized parameter count changed",
            )
            receipt = dict(receipt)
            receipt["research_overlay"] = {
                "method": "source_routed_factorized_context_response_v1",
                "arm": arm,
                "rank": RANK,
                "added_trainable_parameters": ADDED_PARAMETER_COUNT,
                "hidden_terminal_gradient_destination": "response_branch_only",
            }
            return model, receipt

        def evaluate_context_model(
            model: nn.Module,
            evaluation: dict[str, torch.Tensor],
            *,
            device: torch.device,
            batch_size: int,
        ) -> dict[str, Any]:
            if model is not loaded_model:
                return native_evaluator(
                    model, evaluation, device=device, batch_size=batch_size
                )
            return additive.evaluate_with_explicit_context(
                model,
                evaluation,
                controller=controller,
                pilot_module=mixed.pilot,
                device=device,
                batch_size=batch_size,
            )

        mixed.load_model_for_variant = load_with_response
        mixed.pilot.evaluate_model = evaluate_context_model
        mixed.mixed_prediction_loss = routed_prediction_loss
        try:
            result = native_train_variant(*args, **kwargs)
        finally:
            mixed.load_model_for_variant = native_loader
            mixed.pilot.evaluate_model = native_evaluator
            mixed.mixed_prediction_loss = native_loss
        if loaded_model is None or response is None:
            raise RuntimeError("routed response was never attached")
        max_steps = int(kwargs["max_steps"])
        component_norms = {
            name: float(torch.linalg.vector_norm(parameter.detach().float()))
            for name, parameter in response.named_parameters()
        }
        result["objective"] = (
            "same native identifiable-future MSE scalar + 0.09*original_sigreg; "
            "hidden terminal MSE graph routed to B(q)c only; no added loss"
        )
        result["hidden_labels_at_model_or_loss_boundary"] = arm == "oracle"
        result["routed_factorized_response_contract"] = {
            "arm": arm,
            "diagnostic_upper_bound_only": arm == "oracle",
            "formula": "W_o(SiLU(W_q[sg(z_query),sg(a_query)])*W_c(c_episode))",
            "rank": RANK,
            "added_trainable_parameters": ADDED_PARAMETER_COUNT,
            "terminal_prediction_only": True,
            "function_at_step0_exactly_native": True,
            "gradient_at_step0_intentionally_routed": True,
            "original_full_horizon_mse_destination": "native_lewm",
            "hidden_terminal_mse_prediction_destination": "response_branch_only",
            "hidden_terminal_target_detached": True,
            "branch_query_inputs_detached_during_training": True,
            "original_sigreg_unchanged": True,
            "native_modules_receive_original_mse_and_original_sigreg": True,
            "additional_loss": None,
            "final_component_l2_norms": component_norms,
            "first_finite_nonzero_output_gradient_step": gradient_state.get(
                "first_finite_nonzero_output_gradient_step"
            ),
            "first_output_gradient_l2_norm": gradient_state.get(
                "first_output_gradient_l2_norm"
            ),
            "backward_calls": gradient_state.get("backward_calls"),
            "oracle_condition_identity_entered_as_two_vector": arm == "oracle",
            "constant_arm_contains_condition_identity": False,
            "public_test_opened": False,
            "cem_opened": False,
        }
        result["mechanism_screen"] = result["snapshots"][-1][
            "hidden_evaluation"
        ]["mechanism_screen"]
        result["mechanism_screen"]["optimizer_step"] = max_steps
        return result

    mixed.train_variant = train_variant_with_routing
    return variant


def install_release_authorization_overlay(
    motion: Any, *, arm: str, steps: int
) -> dict[str, Any]:
    native_loader = motion.load_motion_damping_icl_release
    audit: dict[str, Any] = {}

    def load_with_followup(path: Path) -> dict[str, Any]:
        release = copy.deepcopy(native_loader(path))
        _require(release["release_id"] == EXPECTED_RELEASE_ID, "release id changed")
        matrix = release["training"]["reference_matrix"]
        _require(matrix["status"] == "failed_development", "base status changed")
        completed = [int(value) for value in matrix["completed_development_seeds"]]
        _require(FROZEN_SEED in completed, "legacy seed identity changed")
        matrix["completed_development_seeds"] = [
            value for value in completed if value != FROZEN_SEED
        ]
        release["training"]["learnability_followup"] = {
            "status": "development_recipe_search_in_progress",
            "diagnostic": "authorized_routed_factorized_response_mve",
            "authority": str(PREREG),
            "arm": arm,
            "seed": FROZEN_SEED,
            "optimizer_steps": steps,
            "development_only": True,
            "public_test_open": False,
            "cem_open": False,
        }
        audit.update(
            {
                "base_status": "failed_development",
                "removed_completed_seed_in_memory_only": FROZEN_SEED,
                "all_other_completed_seeds_preserved": True,
                "base_release_mutated_on_disk": False,
                "public_test_open": False,
                "cem_open": False,
            }
        )
        return release

    motion.load_motion_damping_icl_release = load_with_followup
    return audit


def validate_static_contract() -> dict[str, Any]:
    config = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    checks = {
        "release_sha_matches": (
            additive.file_sha256(RELEASE_CONFIG) == EXPECTED_RELEASE_SHA256
        ),
        "seed_matches": int(config["training"]["seed"]) == FROZEN_SEED,
        "steps_match": int(config["training"]["optimizer_steps"]) == DEFAULT_STEPS,
        "rank_matches": int(config["architecture"]["rank"]) == RANK,
        "parameter_count_matches": (
            int(config["architecture"]["added_trainable_parameters"])
            == ADDED_PARAMETER_COUNT
        ),
        "no_added_loss": config["objective"]["additional_loss"] is None,
        "hidden_route_matches": (
            config["gradient_routing"]["hidden_terminal_prediction"]
            == "factorized_response_branch_only"
        ),
        "public_and_cem_closed": (
            config["evidence_boundary"]["public_test_open"] is False
            and config["evidence_boundary"]["cem_open"] is False
        ),
    }
    _require(all(checks.values()), f"static routed contract failed: {checks}")
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--optimizer-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--original-h5", type=Path, default=None)
    parser.add_argument("--original-lance", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require(args.seed == FROZEN_SEED, f"first MVE requires seed {FROZEN_SEED}")
    _require(args.optimizer_steps > 0, "optimizer steps must be positive")
    output = args.output.expanduser().resolve()
    if not args.dry_run and output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    static_checks = validate_static_contract()

    import run_pusht_motion_damping_h3_train as motion

    variant = install_training_overlay(motion, arm=args.arm)
    release_overlay = install_release_authorization_overlay(
        motion, arm=args.arm, steps=args.optimizer_steps
    )
    native_argv = sys.argv
    forwarded = [
        str(motion.__file__),
        "--release-config",
        str(RELEASE_CONFIG),
        "--model",
        "lewm",
        "--variant",
        variant,
        "--seed",
        str(args.seed),
        "--optimizer-steps",
        str(args.optimizer_steps),
        "--output",
        str(output),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    for name in ("original_h5", "original_lance", "checkpoint"):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend(["--" + name.replace("_", "-"), str(value.resolve())])
    if args.dry_run:
        forwarded.append("--dry-run")
    sys.argv = forwarded
    try:
        motion.main()
    finally:
        sys.argv = native_argv
    if args.dry_run:
        return

    report_path = output / "training_report.json"
    provenance_path = output / "training_provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    disclosure = {
        "experiment_id": "pusht_motion_damping_routed_factorized_response_mve_v1",
        "arm": args.arm,
        "variant": variant,
        "preregistration": {
            "path": str(PREREG),
            "sha256": additive.file_sha256(PREREG),
        },
        "release": {
            "path": str(RELEASE_CONFIG),
            "sha256": additive.file_sha256(RELEASE_CONFIG),
        },
        "release_authorization_overlay": release_overlay,
        "static_checks": static_checks,
        "oracle_privileged_context_opened": args.arm == "oracle",
        "public_test_opened": False,
        "cem_opened": False,
    }
    report["research_overlay"] = disclosure
    provenance["research_overlay"] = disclosure
    if args.arm == "oracle":
        provenance["visible_fields"] = [
            "pixels",
            "action",
            "oracle_motion_damping_identity_diagnostic_only",
        ]
    _atomic_json(report_path, report)
    _atomic_json(provenance_path, provenance)


if __name__ == "__main__":
    main()
