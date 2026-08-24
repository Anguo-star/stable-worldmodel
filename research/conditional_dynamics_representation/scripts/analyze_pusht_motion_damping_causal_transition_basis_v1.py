#!/usr/bin/env python3
"""Extract exact Development response metrics for the transition-basis MVE."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterator

import torch


sys.modules.setdefault("flash_attn", None)

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
for root in (REPO_ROOT, CONTEXTWORLD_ROOT, CONTEXTWORLD_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_oracle_context_predictor_mve_v1 as response,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_terminal_aligned_transition_v1 as terminal_aligned,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_prefix_aligned_transition_v2 as prefix_aligned,
)
import run_pusht_motion_damping_h3_train as motion  # noqa: E402


ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
DEFAULT_RUN = ROOT / (
    "artifacts/pusht_motion_damping_causal_transition_basis_v1/"
    "s14321_step1024_v1"
)
DEFAULT_BASELINE = CONTEXTWORLD_ROOT / (
    "artifacts/evaluation/history3/pusht_motion_damping_strict_causal_reference/"
    "reference_training/lewm_seed14321/training_report.json"
)
DEFAULT_H5 = REPO_ROOT.parents[1] / (
    "data/world_model/quentinll/lewm-pusht/pusht_expert_train.h5"
)
VARIANT = "mixed_frozen_image_identifiable_future_native_0p09"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--original-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer-step", type=int, default=1024)
    parser.add_argument(
        "--basis",
        choices=(
            "causal_transition",
            "absolute_native_twin",
            "absolute_anchored_transfer",
            "transition_anchored_transfer",
            "transition_transfer_only",
            "transition_transfer_consistency",
            "response_calibrated_transfer",
            "residual_absolute",
            "residual_transition",
            "residual_transition_ccrm",
            "residual_transition_exact_future",
            "residual_transition_canonical_response_only",
            "residual_transition_canonical_response_only_freeze",
            "residual_transition_cartesian_action_pair",
            "residual_transition_cartesian_action_pair_legacy_scale",
            "residual_transition_contact_cartesian_action_pair",
            "residual_transition_replay_cartesian_action_pair",
            "residual_transition_replay_cartesian_action_pair_single_stage",
            "residual_transition_replay_cartesian_action_pair_single_stage_matched_budget",
            "absolute_replay_cartesian_action_pair_single_stage",
            "absolute_replay_cartesian_action_pair_single_stage_step2048",
            "residual_transition_canonical_response_function_anchor",
            "residual_transition_action_intervention_anchor",
            "temporal_homotopy_exact_future",
            "residual_transition_exact_future_weight003",
            "residual_transition_native_consolidation",
            "residual_transition_function_anchor",
            "residual_transition_original_only_consolidation",
            "terminal_aligned_transition",
            "prefix_aligned_transition",
        ),
        default="causal_transition",
    )
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    checkpoint = run / f"{VARIANT}_step{int(args.optimizer_step)}.pt"
    training_report = run / "training_report.json"
    output = run / "development_response_analysis_v1.json"
    if output.exists():
        raise FileExistsError(output)
    for path in (checkpoint, training_report, args.baseline, args.original_h5):
        if not Path(path).expanduser().resolve().exists():
            raise FileNotFoundError(path)
    training_payload = json.loads(training_report.read_text(encoding="utf-8"))
    if int(training_payload["result"]["optimizer_steps"]) != int(
        args.optimizer_step
    ):
        raise RuntimeError("Checkpoint and requested optimizer step mismatch")

    release = motion.load_motion_damping_icl_release(
        motion.DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
    )
    data_root = CONTEXTWORLD_ROOT / release["data"]["artifact_tree"]["root"]
    action_stats = motion.trainer.mixed.pilot.original_action_stats(
        args.original_h5.expanduser().resolve()
    )
    # Reproduce the two capability bindings installed by motion.main()
    # without opening a training loop.
    motion.trainer._read_lance_pairs = motion._read_lance_pairs
    motion.trainer.HIDDEN_FIELD = "hidden_motion_damping"
    evaluation = motion.trainer._loader_validation(
        data_root / "loader_validation.lance",
        expected_pairs=int(release["data"]["pair_counts"]["loader_validation"]),
        action_stats=action_stats,
    )
    device = torch.device(args.device)
    motion.trainer.mixed.VARIANT_WEIGHTS[VARIANT] = (
        "native",
        0.09,
        "identifiable_future_only",
    )
    model, load_receipt = motion.trainer.mixed.load_model_for_variant(
        checkpoint,
        variant=VARIANT,
        device=device,
    )
    if not hasattr(model, "temporal_input_basis"):
        raise RuntimeError("LeWM runtime lacks temporal-basis support")
    if args.basis == "causal_transition":
        model.temporal_input_basis = "causal_transition"
        candidate = "pusht_motion_damping_causal_transition_basis_v1"
        comparison_label = "causal_transition_step1024"
    elif args.basis == "absolute_native_twin":
        model.temporal_input_basis = "absolute"
        candidate = "pusht_motion_damping_native_twin_sampler_v1"
        comparison_label = "native_twin_sampler_step1024"
    elif args.basis == "absolute_anchored_transfer":
        model.temporal_input_basis = "absolute"
        candidate = "pusht_motion_damping_anchored_context_transfer_v1"
        comparison_label = "anchored_context_transfer_step1024"
    elif args.basis == "transition_anchored_transfer":
        prefix_aligned._install_model_predict(model)
        candidate = "pusht_motion_damping_transition_context_transfer_v1"
        comparison_label = "transition_context_transfer_step1024"
    elif args.basis == "transition_transfer_only":
        prefix_aligned._install_model_predict(model)
        candidate = "pusht_motion_damping_transition_transfer_only_v1"
        comparison_label = "transition_transfer_only_step1024"
    elif args.basis == "transition_transfer_consistency":
        prefix_aligned._install_model_predict(model)
        candidate = "pusht_motion_damping_transition_transfer_consistency_v1"
        comparison_label = "transition_transfer_consistency_step1024"
    elif args.basis == "response_calibrated_transfer":
        prefix_aligned._install_model_predict(model)
        candidate = "pusht_motion_damping_response_calibrated_transfer_v1"
        comparison_label = "response_calibrated_transfer_step1024"
    elif args.basis == "residual_absolute":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None) != "absolute"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Residual-absolute checkpoint basis mismatch")
        candidate = "pusht_motion_damping_residual_output_basis_v1"
        comparison_label = f"residual_absolute_step{int(args.optimizer_step)}"
    elif args.basis == "residual_transition":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Residual-transition checkpoint basis mismatch")
        candidate = "pusht_motion_damping_residual_output_basis_v1"
        comparison_label = (
            f"residual_transition_step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_ccrm":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Residual-transition CCRM basis mismatch")
        candidate = "pusht_motion_damping_residual_transition_ccrm_v1"
        comparison_label = (
            f"residual_transition_ccrm_step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_exact_future":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Residual-transition exact-future basis mismatch")
        candidate = "pusht_motion_damping_residual_transition_exact_future_v1"
        comparison_label = (
            f"residual_transition_exact_future_step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_canonical_response_only":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError(
                "Canonical-response-only checkpoint basis mismatch"
            )
        candidate = "pusht_motion_damping_canonical_response_only_v1"
        comparison_label = (
            "residual_transition_canonical_response_only_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_canonical_response_only_freeze":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError(
                "Canonical-response freeze checkpoint basis mismatch"
            )
        candidate = (
            "pusht_motion_damping_canonical_response_only_freeze_v1"
        )
        comparison_label = (
            "residual_transition_canonical_response_only_freeze_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_cartesian_action_pair":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Cartesian action-pair basis mismatch")
        candidate = "pusht_motion_damping_cartesian_action_pair_v1"
        comparison_label = (
            "residual_transition_cartesian_action_pair_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_cartesian_action_pair_legacy_scale":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Legacy-scale Cartesian action-pair basis mismatch")
        candidate = (
            "pusht_motion_damping_cartesian_action_pair_legacy_scale_v2"
        )
        comparison_label = (
            "residual_transition_cartesian_action_pair_legacy_scale_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_contact_cartesian_action_pair":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError(
                "Contact Cartesian action-pair basis mismatch"
            )
        candidate = (
            "pusht_motion_damping_contact_cartesian_action_pair_v1"
        )
        comparison_label = (
            "residual_transition_contact_cartesian_action_pair_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_replay_cartesian_action_pair":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError(
                "Replay Cartesian action-pair basis mismatch"
            )
        candidate = (
            "pusht_motion_damping_replay_cartesian_action_pair_v1"
        )
        comparison_label = (
            "residual_transition_replay_cartesian_action_pair_"
            f"step{int(args.optimizer_step)}"
        )
    elif (
        args.basis
        == "residual_transition_replay_cartesian_action_pair_single_stage"
    ):
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError(
                "Single-stage replay Cartesian action-pair basis mismatch"
            )
        candidate = (
            "pusht_motion_damping_replay_cartesian_action_pair_"
            "single_stage_v1"
        )
        comparison_label = (
            "residual_transition_replay_cartesian_action_pair_single_stage_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == (
        "residual_transition_replay_cartesian_action_pair_"
        "single_stage_matched_budget"
    ):
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError(
                "Matched-budget single-stage replay basis mismatch"
            )
        candidate = (
            "pusht_motion_damping_replay_cartesian_action_pair_"
            "single_stage_matched_budget_v1"
        )
        comparison_label = (
            "residual_transition_replay_cartesian_action_pair_"
            "single_stage_matched_budget_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "absolute_replay_cartesian_action_pair_single_stage":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None) != "absolute"
            or getattr(model, "temporal_output_basis", None) != "absolute"
        ):
            raise RuntimeError("Absolute replay single-stage basis mismatch")
        candidate = (
            "pusht_motion_damping_replay_cartesian_action_pair_"
            "absolute_single_stage_v1"
        )
        comparison_label = (
            "absolute_replay_cartesian_action_pair_single_stage_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == (
        "absolute_replay_cartesian_action_pair_single_stage_step2048"
    ):
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None) != "absolute"
            or getattr(model, "temporal_output_basis", None) != "absolute"
        ):
            raise RuntimeError("Absolute replay step2048 basis mismatch")
        candidate = (
            "pusht_motion_damping_replay_cartesian_action_pair_"
            "absolute_single_stage_step2048_v1"
        )
        comparison_label = (
            "absolute_replay_cartesian_action_pair_single_stage_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_canonical_response_function_anchor":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError(
                "Canonical-response function-anchor basis mismatch"
            )
        candidate = (
            "pusht_motion_damping_canonical_response_function_anchor_v1"
        )
        comparison_label = (
            "residual_transition_canonical_response_function_anchor_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_action_intervention_anchor":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Action-intervention anchor basis mismatch")
        candidate = "pusht_motion_damping_action_intervention_anchor_v1"
        comparison_label = (
            "residual_transition_action_intervention_anchor_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "temporal_homotopy_exact_future":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Temporal-homotopy exact-future basis mismatch")
        candidate = "pusht_motion_damping_temporal_homotopy_exact_future_v1"
        comparison_label = (
            f"temporal_homotopy_exact_future_step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_exact_future_weight003":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Weight-0.03 exact-future basis mismatch")
        candidate = (
            "pusht_motion_damping_residual_transition_"
            "exact_future_weight003_v1"
        )
        comparison_label = (
            "residual_transition_exact_future_weight003_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_native_consolidation":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Native-consolidation basis mismatch")
        candidate = (
            "pusht_motion_damping_residual_transition_"
            "native_consolidation_v1"
        )
        comparison_label = (
            "residual_transition_native_consolidation_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_function_anchor":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Function-anchor basis mismatch")
        candidate = (
            "pusht_motion_damping_residual_transition_"
            "function_anchor_v1"
        )
        comparison_label = (
            "residual_transition_function_anchor_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "residual_transition_original_only_consolidation":
        checkpoint_config = json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        model.temporal_input_basis = checkpoint_config.get(
            "temporal_input_basis"
        )
        model.temporal_output_basis = checkpoint_config.get(
            "temporal_output_basis"
        )
        if (
            getattr(model, "temporal_input_basis", None)
            != "causal_transition"
            or getattr(model, "temporal_output_basis", None) != "residual"
        ):
            raise RuntimeError("Original-only consolidation basis mismatch")
        candidate = (
            "pusht_motion_damping_residual_transition_"
            "original_only_consolidation_v1"
        )
        comparison_label = (
            "residual_transition_original_only_consolidation_"
            f"step{int(args.optimizer_step)}"
        )
    elif args.basis == "terminal_aligned_transition":
        terminal_aligned._install_model_predict(model)
        candidate = terminal_aligned.CANDIDATE
        comparison_label = "terminal_aligned_transition_step1024"
    else:
        prefix_aligned._install_model_predict(model)
        candidate = prefix_aligned.CANDIDATE
        comparison_label = "prefix_aligned_transition_step1024"
    metrics = response.evaluate_with_explicit_context(
        model,
        evaluation,
        controller=_NoContextIntervention(),
        pilot_module=motion.trainer.mixed.pilot,
        device=device,
        batch_size=int(args.batch_size),
    )

    baseline_report = json.loads(
        args.baseline.expanduser().resolve().read_text(encoding="utf-8")
    )
    baseline_rows = [
        row["hidden_evaluation"]
        for row in baseline_report["result"]["snapshots"]
        if int(row["optimizer_step"]) == int(args.optimizer_step)
    ]
    if not baseline_rows and args.basis != (
        "residual_transition_replay_cartesian_action_pair_single_stage"
    ):
        raise RuntimeError(
            "Native baseline has no exact requested optimizer-step snapshot"
        )
    baseline = baseline_rows[0] if baseline_rows else None
    keys = (
        "two_real_future_target_selection_rate",
        "correct_history_preference_rate",
        "correct_rule_switch_rate",
        "worst_mode_target_selection_rate",
    )
    comparison = (
        {
            key: {
                f"native_absolute_step{int(args.optimizer_step)}": float(
                    baseline[key]
                ),
                comparison_label: float(metrics[key]),
                "absolute_delta": float(metrics[key]) - float(baseline[key]),
            }
            for key in keys
        }
        if baseline is not None
        else {
            "status": "unavailable_no_exact_native_snapshot",
            "requested_optimizer_step": int(args.optimizer_step),
            "available_native_optimizer_steps": [
                int(row["optimizer_step"])
                for row in baseline_report["result"]["snapshots"]
            ],
            "note": (
                "The one-stage MVE is judged by its frozen direct response "
                "metrics; no interpolated or nearest-step baseline is used."
            ),
        }
    )
    result = {
        "schema_version": 1,
        "status": "completed_development_only",
        "candidate": candidate,
        "temporal_input_basis": getattr(
            model, "temporal_input_basis", "absolute"
        ),
        "temporal_output_basis": getattr(
            model, "temporal_output_basis", "absolute"
        ),
        "optimizer_step": int(args.optimizer_step),
        "training_seed": 14321,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_report": str(training_report),
        "training_report_sha256": _sha256(training_report),
        "model_load": load_receipt,
        "metrics": metrics,
        f"matched_native_step{int(args.optimizer_step)}_comparison": comparison,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
        "claim_boundary": {
            "development_only": True,
            "public_test_opened": False,
            "cem_opened": False,
            "optimizer_steps": 0,
            "single_seed_discovery": True,
            "matched_training_budget_comparison": (
                args.basis
                not in {
                    "residual_transition_action_intervention_anchor",
                    "residual_transition_cartesian_action_pair",
                    "residual_transition_cartesian_action_pair_legacy_scale",
                    "residual_transition_contact_cartesian_action_pair",
                    "residual_transition_replay_cartesian_action_pair",
                    "residual_transition_replay_cartesian_action_pair_single_stage",
                    "residual_transition_native_consolidation",
                    "residual_transition_function_anchor",
                    "residual_transition_original_only_consolidation",
                }
            ),
            "total_parameter_update_exposure": (
                3072
                if args.basis
                in {
                    "residual_transition_action_intervention_anchor",
                    "residual_transition_cartesian_action_pair",
                    "residual_transition_cartesian_action_pair_legacy_scale",
                    "residual_transition_contact_cartesian_action_pair",
                    "residual_transition_replay_cartesian_action_pair",
                    "residual_transition_replay_cartesian_action_pair_single_stage",
                    "residual_transition_native_consolidation",
                    "residual_transition_function_anchor",
                    "residual_transition_original_only_consolidation",
                }
                else int(args.optimizer_step)
            ),
        },
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "future": metrics["two_real_future_target_selection_rate"],
                "history": metrics["correct_history_preference_rate"],
                "switch": metrics["correct_rule_switch_rate"],
                "worst": metrics["worst_mode_target_selection_rate"],
                "gain": metrics["latent_response"]["response_gain"],
                "nre": metrics["latent_response"]["normalized_response_error"],
                "screen_passed": metrics["mechanism_screen"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
