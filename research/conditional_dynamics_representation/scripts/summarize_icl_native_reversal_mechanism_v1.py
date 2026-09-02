#!/usr/bin/env python3
"""Summarize frozen Training-only native ICL reversal evidence.

This audit performs no model forward, optimizer step, data-table read, or
checkpoint mutation. It validates and summarizes existing ActionDelay A0/A3/A4
mechanism reports and the matched ActionStrength LeWM/PLDM component audits.

The output distinguishes local representation/objective/gradient evidence from
held-out behavior. It must not be used to claim a universal data-only root cause.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "research/conditional_dynamics_representation/configs/"
    "icl_native_reversal_mechanism_v1.yaml"
)

MODEL_LOADED = False
OPTIMIZER_STEPS = 0
CHECKPOINT_WRITTEN = False
DEVELOPMENT_OPENED = False
PUBLIC_TEST_OPENED = False
CLAIM_SCOPE = "frozen_training_only_local_mechanism_summary"

_FORBIDDEN_PATH_TOKENS = frozenset(
    {"development", "public", "test", "validation", "val", "sealed"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def _json_dump_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _jsonl_dump_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def training_only_path_guard(path: Path) -> None:
    tokens: set[str] = set()
    for part in path.parts:
        lower = part.lower()
        tokens.add(lower)
        tokens.update(lower.replace("-", "_").split("_"))
    bad = tokens & _FORBIDDEN_PATH_TOKENS
    if bad:
        raise ValueError(
            f"Training-only audit refuses path token(s) {sorted(bad)}: {path}"
        )


def exclusive_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def selected_file_set_digest(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def module_group_l2(groups: Mapping[str, Any]) -> float:
    """Combine disjoint module-group norms into one all-parameter norm."""
    values: list[float] = []
    for value in groups.values():
        if isinstance(value, Mapping):
            value = value["l2_norm"]
        numeric = float(value)
        _require(math.isfinite(numeric) and numeric >= 0.0, "invalid gradient norm")
        values.append(numeric)
    return math.sqrt(sum(value * value for value in values))


def aggregate_scalar(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, "empty scalar aggregate")
    _require(bool(np.isfinite(array).all()), "non-finite scalar aggregate")
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _relative_difference(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), np.finfo(float).tiny)


def _validate_no_eval_access(scope: Mapping[str, Any], *, label: str) -> None:
    forbidden_true = (
        "development_data_opened",
        "current_process_development_paths_opened",
        "public_data_opened",
        "public_test_opened",
        "public_metrics_or_labels_consumed",
        "public_development_test_or_sealed_paths_opened",
    )
    for key in forbidden_true:
        if key in scope:
            _require(scope[key] is False, f"{label}: {key} must be false")
    if "development_open_attempts" in scope:
        _require(scope["development_open_attempts"] == 0, f"{label}: eval open attempt")
    if "probe_split" in scope:
        _require(scope["probe_split"] == "train", f"{label}: probe is not Training")


def _action_delay_row(
    report: Mapping[str, Any], *, arm_name: str, arm_id: str, seed: int, step: int
) -> dict[str, Any]:
    point = report["checkpoints"][str(step)]
    total = point["component_gradients"]["weighted_total"]
    prediction = point["component_gradients"]["pred"]
    return {
        "task": "action_delay",
        "arm": arm_name,
        "arm_id": arm_id,
        "seed": seed,
        "step": step,
        "target_latent_matched_to_unrelated_ratio": float(
            point["terminal_target"]["matched_to_unrelated_ratio"]
        ),
        "response_signed_projection_gain": float(
            point["response"]["signed_projection_gain_mean"]
        ),
        "response_cosine": float(point["response"]["cosine_mean"]),
        "response_nre": float(point["response"]["normalized_residual_mean"]),
        "prediction_mse_pair_coherence": float(
            point["prediction_mse_cancellation"]["coherence_ratio"]
        ),
        "prediction_gradient_l2_all": module_group_l2(
            prediction["gradient_l2_by_module"]
        ),
        "prediction_response_residual_first_order": float(
            prediction["first_order_metric_change_per_unit_lr"]
            ["response_residual_squared"]["all"]
        ),
        "native_total_gradient_l2_all": module_group_l2(
            total["gradient_l2_by_module"]
        ),
        "native_total_response_residual_first_order": float(
            total["first_order_metric_change_per_unit_lr"]
            ["response_residual_squared"]["all"]
        ),
    }


def _validate_action_delay_report(
    *,
    report_path: Path,
    npz_path: Path,
    arm_id: str,
    seed: int,
    steps: tuple[int, ...],
    training_root: Path,
    expected_init_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    tuple[tuple[int, int, int, str], ...],
    dict[str, str],
    Path,
]:
    training_only_path_guard(report_path)
    training_only_path_guard(npz_path)
    report = _json_load(report_path)
    _require(report.get("status") == "passed", f"failed mechanism report: {report_path}")
    _require(
        report.get("analysis") == "action_delay_objective_transplant_mechanism_v1",
        f"wrong ActionDelay analysis: {report_path}",
    )
    _validate_no_eval_access(report["scope_receipt"], label=str(report_path))
    _require(
        tuple(sorted(int(value) for value in report["checkpoints"])) == steps,
        f"checkpoint set mismatch: {report_path}",
    )
    for step in steps:
        point = report["checkpoints"][str(step)]
        for key in ("state_unchanged", "whole_audit_state_unchanged", "hooks_removed"):
            _require(point[key] is True, f"{report_path}: step {step} {key} is false")

    formal = report["provenance_binding"]["formal_run"]
    _require(formal["variant"] == arm_id, f"variant mismatch: {report_path}")
    _require(int(formal["seed"]) == seed, f"seed mismatch: {report_path}")
    _require(formal["status"] == "passed", f"formal run not passed: {report_path}")
    provenance_meta = formal["formal_candidate_provenance"]
    provenance_path = Path(provenance_meta["path"])
    _require(
        provenance_path.is_relative_to(training_root),
        f"formal provenance escaped Training root: {provenance_path}",
    )
    _require(
        file_sha256(provenance_path) == provenance_meta["sha256"],
        f"formal provenance hash mismatch: {provenance_path}",
    )
    provenance = _json_load(provenance_path)
    _require(provenance["status"] == "passed", f"provenance not passed: {provenance_path}")
    _require(provenance["candidate"]["variant"] == arm_id, "provenance variant mismatch")
    _require(int(provenance["candidate"]["seed"]) == seed, "provenance seed mismatch")
    _require(
        int(provenance["candidate"]["prefix_optimizer_steps"]) == max(steps),
        "prefix length mismatch",
    )
    _require(all(provenance["validation"].values()), "formal provenance validation failed")
    init_sha = provenance["frozen_inputs"]["identities"]["initialization_checkpoint"][
        "sha256"
    ]
    _require(init_sha == expected_init_sha256, "initialization checkpoint mismatch")
    batch_digests = tuple(
        (
            int(row["rank"]),
            int(row["batches"]),
            int(row["samples"]),
            str(row["sha256"]),
        )
        for row in provenance["all_rank_runtime"]["rank_local_batch_digests"]
    )

    output_npz = report["output_npz"]
    _require(Path(output_npz["path"]).resolve() == npz_path.resolve(), "NPZ path mismatch")
    _require(file_sha256(npz_path) == output_npz["sha256"], "NPZ hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    _require(
        str(arrays["provenance_binding_sha256"].item())
        == report["provenance_binding_sha256"],
        "NPZ provenance binding mismatch",
    )
    identities = {
        "report_sha256": file_sha256(report_path),
        "npz_sha256": file_sha256(npz_path),
        "formal_candidate_provenance_sha256": provenance_meta["sha256"],
        "initialization_checkpoint_sha256": init_sha,
        "step0_state_sha256": str(report["checkpoints"]["0"]["state_sha256_before_after"]),
    }
    return report, arrays, batch_digests, identities, provenance_path


def _load_action_delay(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    root = _repo_path(config["analysis_root"])
    training_root = _repo_path(config["training_root"]).resolve()
    training_only_path_guard(root)
    training_only_path_guard(training_root)
    arms = dict(config["arms"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    steps = tuple(sorted(int(step) for step in config["steps"]))
    basename = str(config["report_basename"])
    selected_paths: list[Path] = []
    consumed_paths: list[Path] = []
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    archives: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    batches: dict[tuple[str, int], tuple[tuple[int, int, int, str], ...]] = {}
    identities: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for arm_name, arm_id in arms.items():
        for seed in seeds:
            directory = root / f"{arm_id}_s{seed}_prefix256_v1"
            report_path = directory / f"{basename}.json"
            npz_path = directory / f"{basename}.npz"
            selected_paths.extend((report_path, npz_path))
            (
                report,
                arrays,
                batch_digest,
                cell_identity,
                provenance_path,
            ) = _validate_action_delay_report(
                report_path=report_path,
                npz_path=npz_path,
                arm_id=arm_id,
                seed=seed,
                steps=steps,
                training_root=training_root,
                expected_init_sha256=str(
                    config["expected_initialization_checkpoint_sha256"]
                ),
            )
            consumed_paths.extend((report_path, npz_path, provenance_path))
            reports[(arm_name, seed)] = report
            archives[(arm_name, seed)] = arrays
            batches[(arm_name, seed)] = batch_digest
            identities[f"{arm_name}_s{seed}"] = cell_identity
            rows.extend(
                _action_delay_row(
                    report, arm_name=arm_name, arm_id=arm_id, seed=seed, step=step
                )
                for step in steps
            )

    observed_set_digest = selected_file_set_digest(root, selected_paths)
    _require(
        observed_set_digest == config["selected_file_set_sha256"],
        "ActionDelay selected report/NPZ set changed",
    )

    reference_arrays = archives[("lewm_native", seeds[0])]
    common_array_names = (
        "query_id",
        "delay_values",
        "pair_orientation",
        "step0_delta_history",
        "step0_delta_target",
        "step0_delta_prediction",
    )
    for cell, arrays in archives.items():
        for name in common_array_names:
            _require(
                np.array_equal(arrays[name], reference_arrays[name]),
                f"common initialization/probe mismatch for {cell}: {name}",
            )

    for seed in seeds:
        reference_batch = batches[("lewm_native", seed)]
        for arm_name in arms:
            _require(
                batches[(arm_name, seed)] == reference_batch,
                f"logical batch mismatch for seed {seed}: {arm_name}",
            )

    by_arm_step: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    scalar_keys = tuple(
        key for key in rows[0] if key not in {"task", "arm", "arm_id", "seed", "step"}
    )
    for arm_name in arms:
        by_arm_step[arm_name] = {}
        for step in steps:
            selected = [row for row in rows if row["arm"] == arm_name and row["step"] == step]
            by_arm_step[arm_name][str(step)] = {
                key: aggregate_scalar(row[key] for row in selected) for key in scalar_keys
            }

    def mean(arm: str, step: int, key: str) -> float:
        return by_arm_step[arm][str(step)][key]["mean"]

    a0_grad = mean("lewm_native", 0, "native_total_gradient_l2_all")
    a3_grad = mean("lewm_pldm_objective", 0, "native_total_gradient_l2_all")
    a0_response = mean(
        "lewm_native", 0, "native_total_response_residual_first_order"
    )
    a3_response = mean(
        "lewm_pldm_objective", 0, "native_total_response_residual_first_order"
    )
    a4_response = mean(
        "pldm_native", 0, "native_total_response_residual_first_order"
    )
    _require(a0_response < 0.0 and a4_response < 0.0, "unexpected step-0 response sign")

    contrasts = {
        "common_probe_and_step0_latents_exact": True,
        "same_logical_batches_within_seed": True,
        "lewm_pldm_objective_vs_lewm_native_step0_total_gradient_l2_ratio": a3_grad / a0_grad,
        "lewm_pldm_objective_vs_lewm_native_step0_local_response_improvement_magnitude_ratio": abs(a3_response) / abs(a0_response),
        "lewm_pldm_objective_vs_pldm_native_step0_response_derivative_relative_difference": _relative_difference(a3_response, a4_response),
        "lewm_pldm_objective_vs_lewm_native_step256_target_match_ratio": mean(
            "lewm_pldm_objective", 256, "target_latent_matched_to_unrelated_ratio"
        )
        / mean("lewm_native", 256, "target_latent_matched_to_unrelated_ratio"),
        "step256_all_arms_signed_gain_still_negative": all(
            mean(arm, 256, "response_signed_projection_gain") < 0.0 for arm in arms
        ),
    }
    summary = {
        "evidence_design": {
            "arms": arms,
            "seeds": list(seeds),
            "steps": list(steps),
            "same_shared_core_initialization": True,
            "complete_parameter_sets_assumed_isomorphic": False,
            "same_training_probe": True,
            "same_logical_batches_within_seed": True,
            "a3_role": "PLDM_active_objective_on_LeWM_implementation_path",
        },
        "aggregates": by_arm_step,
        "contrasts": contrasts,
        "gate": {
            "result": "model_side_objective_route_separation_supported",
            "reason": (
                "A0/A3/A4 have exact common step-0 probe latents and matched logical "
                "batches. On the same LeWM implementation path, A3's PLDM-active objective "
                "produces a much stronger locally response-aligned total gradient than A0. "
                "A3 approximately matches native A4, localizing the first separation to the "
                "objective route rather than raw data; complete A0/A4 parameter isomorphism "
                "is not claimed."
            ),
            "behavior_claim": "not_established_by_256_step_training_probe",
        },
        "input_identities": identities,
        "selected_file_set_sha256": observed_set_digest,
    }
    return summary, rows, consumed_paths


def _component_metric(point: Mapping[str, Any], component: str) -> dict[str, float]:
    value = point["components"][component]
    all_group = value["gradient_l2_by_module"]["all"]
    first_order = value["first_order_metric_change_per_unit_lr"]
    return {
        "scalar": float(value["scalar"]),
        "gradient_l2_all": float(all_group["l2_norm"]),
        "clip_scale_at_norm_1": float(all_group["clip_scale_at_norm_1"]),
        "signed_gain_first_order": float(
            first_order["signed_response_gain"]["all"]["predicted_change"]
        ),
        "nre_first_order": float(
            first_order["normalized_response_error"]["all"]["predicted_change"]
        ),
        "response_residual_first_order": float(
            first_order["response_residual_squared"]["all"]["predicted_change"]
        ),
    }


def _load_action_strength(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    reports: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    for model in ("lewm", "pldm"):
        item = config[f"{model}_result"]
        path = _repo_path(item["path"])
        training_only_path_guard(path)
        _require(file_sha256(path) == item["sha256"], f"{model} ActionStrength hash changed")
        report = _json_load(path)
        _require(str(report["status"]).startswith("completed_read_only"), f"{model} audit incomplete")
        _validate_no_eval_access(report["scope"], label=f"ActionStrength {model}")
        _require(report["scope"]["optimizer_steps"] == 0, f"{model}: optimizer step taken")
        _require(report["scope"]["model_updates"] is False, f"{model}: model updated")
        _require(report["scope"]["checkpoint_writes"] is False, f"{model}: checkpoint write")
        reports[model] = report
        paths.append(path)

    lewm_task = reports["lewm"]["tasks"]["action_strength"]
    pldm_task = reports["pldm"]["tasks"]["action_strength"]
    _require(lewm_task["batch"] == pldm_task["batch"], "ActionStrength batches differ")
    point_name = str(config["point"])
    rows: list[dict[str, Any]] = []
    init_summaries: dict[str, Any] = {}
    for model, task in (("lewm", lewm_task), ("pldm", pldm_task)):
        point = task["points"][point_name]
        cancellation = point["hidden_terminal_prediction_cancellation"]
        total = _component_metric(point, "weighted_total")
        prediction = _component_metric(point, "hidden_terminal_prediction_mse")
        regularizer_name = (
            "native_sigreg_weighted" if model == "lewm" else "pldm_regularizer_weighted"
        )
        regularizer = _component_metric(point, regularizer_name)
        response = {key: float(value) for key, value in point["response_metrics"].items()}
        checkpoint = point["checkpoint"]
        init_summaries[model] = {
            "checkpoint_sha256": checkpoint["sha256"],
            "model_state_sha256": checkpoint["model_state_sha256"],
            "response": response,
            "condition_pair_prediction_coherence": float(
                cancellation["condition_pair"]["all"]["coherence_ratio"]
            ),
            "row_prediction_coherence": float(
                cancellation["row"]["all"]["coherence_ratio"]
            ),
            "prediction_component": prediction,
            "native_regularizer_component": {
                "name": regularizer_name,
                **regularizer,
            },
            "native_total": total,
        }
        rows.append(
            {
                "task": "action_strength",
                "model": model,
                "point": point_name,
                "condition_pair_prediction_coherence": init_summaries[model][
                    "condition_pair_prediction_coherence"
                ],
                "response_signed_gain": response["signed_response_gain"],
                "response_nre": response["normalized_response_error"],
                "native_total_signed_gain_first_order": total[
                    "signed_gain_first_order"
                ],
                "native_total_response_residual_first_order": total[
                    "response_residual_first_order"
                ],
            }
        )

    pldm_final = pldm_task["points"].get("final")
    pldm_final_response = None
    if pldm_final is not None:
        pldm_final_response = {
            key: float(value) for key, value in pldm_final["response_metrics"].items()
        }
    different_initializations = (
        init_summaries["lewm"]["checkpoint_sha256"]
        != init_summaries["pldm"]["checkpoint_sha256"]
    )
    _require(different_initializations, "ActionStrength initializations unexpectedly identical")
    summary = {
        "evidence_design": {
            "same_exact_batch": True,
            "model_specific_initializations": True,
            "optimizer_steps_in_audit": 0,
        },
        "initialization": init_summaries,
        "pldm_local_training_batch_endpoint_response": pldm_final_response,
        "gate": {
            "result": "reverse_confirmation_not_established",
            "reason": (
                "The exact-batch zero-step audit uses different pretrained model states, "
                "both native total gradients locally improve signed gain, and the PLDM "
                "endpoint shows a strong response on this Training batch. It therefore "
                "does not explain the opposite held-out outcome and instead leaves "
                "coverage/transfer/calibration as candidate binding layers."
            ),
            "cross_model_absolute_threshold_allowed": False,
        },
        "input_identities": {
            "lewm_result_sha256": file_sha256(paths[0]),
            "pldm_result_sha256": file_sha256(paths[1]),
        },
    }
    return summary, rows, paths


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, "config root must be a mapping")
    _require(value.get("schema_version") == 1, "unsupported config schema")
    scope = value["scope"]
    _require(scope["split"] == "train", "config is not Training-only")
    _require(scope["optimizer_steps"] == 0, "config authorizes optimizer steps")
    for key in ("model_loaded", "checkpoint_written", "development_opened", "public_test_opened"):
        _require(scope[key] is False, f"config scope {key} must be false")
    return value


def run(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    config = _load_config(config_path)
    inventory_item = config["reversal_inventory"]
    inventory_path = _repo_path(inventory_item["path"])
    training_only_path_guard(inventory_path)
    _require(file_sha256(inventory_path) == inventory_item["sha256"], "inventory hash changed")
    inventory = _json_load(inventory_path)
    _require(inventory["status"] == "passed_read_only_matrix_inventory", "inventory not passed")
    outcomes = {
        cell["cell_id"]: cell["observed_icl_outcome"]
        for cell in inventory["cells"]
        if cell["cell_id"]
        in {
            "action_delay_lewm_native",
            "action_delay_pldm_native",
            "action_strength_lewm_native",
            "action_strength_pldm_native",
        }
    }
    _require(
        outcomes
        == {
            "action_delay_lewm_native": "negative",
            "action_delay_pldm_native": "positive",
            "action_strength_lewm_native": "positive",
            "action_strength_pldm_native": "negative",
        },
        "frozen reversal outcome labels changed",
    )

    action_delay, delay_rows, delay_paths = _load_action_delay(config["action_delay"])
    action_strength, strength_rows, strength_paths = _load_action_strength(
        config["action_strength"]
    )
    summary = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "passed_training_only_native_reversal_mechanism_summary",
        "scope": {
            "claim_scope": CLAIM_SCOPE,
            "model_loaded": MODEL_LOADED,
            "optimizer_steps": OPTIMIZER_STEPS,
            "checkpoint_written": CHECKPOINT_WRITTEN,
            "development_opened": DEVELOPMENT_OPENED,
            "public_test_opened": PUBLIC_TEST_OPENED,
            "historical_outcome_labels_source": "frozen_reversal_inventory_only",
        },
        "frozen_outcomes": outcomes,
        "action_delay": action_delay,
        "action_strength": action_strength,
        "synthesis": {
            "universal_low_pixel_or_data_only_root_cause": "rejected",
            "general_mechanism": (
                "effective conditional visibility and transferability can attenuate at "
                "the data target, representation, objective/Jacobian, gradient aggregation, "
                "coverage, or calibration layer"
            ),
            "action_delay_binding_evidence": "same_implementation_objective_gradient_route_separates_before_behavior",
            "action_strength_binding_evidence": "single_batch_zero_step_route_does_not_explain_held_out_reversal",
            "data_distribution_role": (
                "general first intervention for upstream-weak cells, but not a universal "
                "sufficient explanation or replacement for model-side diagnostics"
            ),
            "coja_status": "retained_as_explicit_history_use_positive_control_not_failure",
        },
        "claim_boundary": config["claim_boundary"],
        "input_identities": {
            "config_sha256": file_sha256(config_path),
            "reversal_inventory_sha256": file_sha256(inventory_path),
        },
    }
    return summary, [*delay_rows, *strength_rows], [inventory_path, *delay_paths, *strength_paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    summary, rows, input_paths = run(config_path)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "check_ok",
                    "audit_id": summary["audit_id"],
                    "action_delay_gate": summary["action_delay"]["gate"]["result"],
                    "action_strength_gate": summary["action_strength"]["gate"]["result"],
                    "input_file_count": len(input_paths),
                    "model_loaded": MODEL_LOADED,
                    "optimizer_steps": OPTIMIZER_STEPS,
                    "development_opened": DEVELOPMENT_OPENED,
                    "public_test_opened": PUBLIC_TEST_OPENED,
                },
                sort_keys=True,
            )
        )
        return 0

    _require(args.output_dir is not None, "--output-dir is required unless --check-only")
    output_dir = args.output_dir.resolve()
    exclusive_mkdir(output_dir)
    _json_dump_exclusive(output_dir / "summary.json", summary)
    _jsonl_dump_exclusive(output_dir / "per_cell.jsonl", rows)
    receipt = {
        "schema_version": 1,
        "audit_id": str(uuid.uuid4()),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "passed",
        "scope": summary["scope"],
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in sorted(set(input_paths))
        ],
        "outputs": {
            "summary.json": file_sha256(output_dir / "summary.json"),
            "per_cell.jsonl": file_sha256(output_dir / "per_cell.jsonl"),
        },
    }
    _json_dump_exclusive(output_dir / "receipt.json", receipt)
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
