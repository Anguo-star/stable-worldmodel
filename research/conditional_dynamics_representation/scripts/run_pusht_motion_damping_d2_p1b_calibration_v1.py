#!/usr/bin/env python3
"""Run the frozen Training-only Motion D2 P1b calibration split.

This executable applies only the structural admission algorithm committed by
the rollout-free design artifact.  Gamma and robustness quantities are
computed strictly after the admission decision and are never used for ranking
or replacement.  It does not freeze thresholds, open the sealed holdout, load
a model, or take an optimizer step.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Iterator, Sequence
import zipfile

import numpy as np

from pusht_motion_damping_d2_p1b_common_v1 import (
    ADDENDUM_PATH,
    ADMISSION_MAY_USE_ONLY,
    ADMISSION_MUST_NOT_USE,
    AUDIT_ONLY_ACTION_ROTATION_DEGREES,
    AUDIT_ONLY_AGENT_TANGENT_SHIFT_PX,
    CANDIDATE_WINDOWS,
    COVERAGE_CELLS,
    DESIGN_ID,
    DIRECTIONS,
    GROUPS_PER_SPLIT,
    REPO_ROOT,
    STRATUM_IDS,
    action_norm_and_box,
    build_query_actions,
    canonical_sha256,
    ensure_contextworld_on_path,
    local_tangent_unit,
    quota_plan,
    reject_forbidden_path,
    require,
    sha256_file,
    verify_declared_identities,
    write_json_exclusive,
)


THIS_SOURCE = Path(__file__).resolve()
RESOLUTION = 224
SCHEMA_VERSION = 1
RUNNER_ID = "pusht_motion_damping_d2_p1b_calibration_v1"
DEFAULT_DESIGN_DIR = REPO_ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d2_p1b_design_v1/"
    "pre_outcome_v1_contract_clarified"
)
DEFAULT_PIXEL_BASELINE = REPO_ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d2_pixel_baseline_v1/"
    "training_only_v1_contract_clarified/pixel_baseline_v1.json"
)

# Rendering must remain headless and deterministic on worker machines.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@dataclass(frozen=True)
class RuntimeModules:
    motion: Any
    friction: Any
    p1a: Any


class JsonlWriter:
    """Exclusive deterministic JSONL writer with an incremental digest."""

    def __init__(self, path: Path):
        self.path = reject_forbidden_path(path, name="output JSONL").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")
        self._digest = hashlib.sha256()
        self.count = 0

    def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n"
        self._stream.write(line)
        self._digest.update(line.encode("utf-8"))
        self.count += 1

    def close(self) -> str:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()
        return self._digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runtime() -> RuntimeModules:
    """Import simulator code lazily so check-only tests stay lightweight."""

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    ensure_contextworld_on_path()
    from contextworld.evaluation import pusht_contact_friction_h3 as friction
    from contextworld.evaluation import pusht_motion_damping_h3 as motion
    import probe_pusht_motion_damping_d2_p1a_v1 as p1a

    return RuntimeModules(motion=motion, friction=friction, p1a=p1a)


def _candidate_payload_hash(row: dict[str, Any]) -> str:
    payload = dict(row)
    recorded = str(payload.pop("candidate_sha256"))
    observed = canonical_sha256(payload)
    require(observed == recorded, "candidate payload SHA256 changed")
    return observed


def _verify_candidate_row(row: dict[str, Any], expected_index: int) -> None:
    require(row.get("split") == "calibration", "runner refuses non-calibration row")
    require(
        int(row.get("group_index", -1)) == expected_index,
        "calibration candidate indices are not the frozen contiguous window",
    )
    cell = int(row["coverage_cell"])
    require(0 <= cell < COVERAGE_CELLS, "candidate coverage cell is out of range")
    stratum = str(row["action_stratum"])
    require(stratum in STRATUM_IDS, "candidate action stratum changed")
    _candidate_payload_hash(row)
    for direction in DIRECTIONS:
        template = row["templates"][direction]
        require(
            "-cal-" in str(template["template_id"])
            and "-seal-" not in str(template["template_id"]),
            "runner refuses a non-calibration template identity",
        )
        actions = np.asarray(row["query_actions"][direction], dtype=np.float64)
        require(
            np.array_equal(actions, np.asarray(template["query_actions"], dtype=np.float64)),
            "candidate action and template action differ",
        )
        expected = build_query_actions(
            template["expected_natural_query_snapshot"],
            stratum_id=stratum,
            coverage_cell=cell,
            split="calibration",
        )
        require(np.array_equal(actions, expected), "frozen action formula changed")
        require(action_norm_and_box(actions) == row["action_audit"][direction],
                "candidate action audit changed")


def verify_inputs(design_dir: Path, pixel_baseline_path: Path) -> dict[str, Any]:
    """Verify all identities before any calibration outcome is opened."""

    design_dir = reject_forbidden_path(design_dir, name="design directory").resolve()
    pixel_baseline_path = reject_forbidden_path(
        pixel_baseline_path, name="pixel baseline"
    ).resolve()
    receipt_path = design_dir / "design_receipt.json"
    require(receipt_path.is_file(), "design receipt is missing")
    receipt = _json(receipt_path)
    identities = verify_declared_identities()
    addendum_sha = sha256_file(ADDENDUM_PATH)
    require(receipt.get("design_id") == DESIGN_ID, "wrong design identity")
    require(
        receipt.get("status") == "pre_outcome_candidate_streams_committed",
        "design is not a committed pre-outcome artifact",
    )
    require(receipt.get("addendum_sha256") == addendum_sha,
            "design does not bind the current prospective addendum")
    require(
        receipt.get("identity_verification", {}).get("addendum_sha256")
        == addendum_sha,
        "nested design identity has a stale addendum",
    )
    require(
        receipt.get("builder_sha256")
        == sha256_file(THIS_SOURCE.parent / "build_pusht_motion_damping_d2_p1b_design_v1.py"),
        "design builder source changed after commitment",
    )
    require(
        receipt.get("identity_verification", {}).get("common_module_sha256")
        == identities["common_module_sha256"],
        "P1b common module changed after design commitment",
    )
    require(receipt.get("quota_plan") == quota_plan(), "design quota plan changed")

    calibration = receipt.get("splits", {}).get("calibration", {})
    require(
        calibration.get("candidate_window_inclusive")
        == list(CANDIDATE_WINDOWS["calibration"]),
        "calibration window changed",
    )
    require(
        int(calibration.get("candidate_count", -1))
        == CANDIDATE_WINDOWS["calibration"][1]
        - CANDIDATE_WINDOWS["calibration"][0]
        + 1,
        "calibration stream length changed",
    )
    stream_path = design_dir / str(calibration["candidate_stream_path"])
    require(stream_path.name == "calibration_candidate_stream.jsonl",
            "runner refuses a non-calibration stream name")
    require(stream_path.is_file(), "calibration candidate stream is missing")
    require(
        sha256_file(stream_path) == calibration.get("candidate_stream_sha256"),
        "calibration candidate stream SHA256 changed",
    )
    cuts_path = design_dir / str(receipt["coverage_cut_points_path"])
    require(
        sha256_file(cuts_path) == receipt["coverage_cut_points_file_sha256"],
        "coverage cut-point file changed",
    )

    low, high = CANDIDATE_WINDOWS["calibration"]
    count = 0
    with stream_path.open("r", encoding="utf-8") as stream:
        for expected_index, line in zip(range(low, high + 1), stream, strict=True):
            _verify_candidate_row(json.loads(line), expected_index)
            count += 1
    require(count == int(calibration["candidate_count"]), "candidate row count changed")

    require(pixel_baseline_path.is_file(), "Training-only pixel baseline is missing")
    pixel = _json(pixel_baseline_path)
    require(pixel.get("status") == "ok", "pixel baseline is not complete")
    require(pixel.get("pixels_decoded") is True, "pixel baseline has no decoded pixels")
    require(
        pixel.get("input_sha256", {}).get("addendum") == addendum_sha,
        "pixel baseline does not bind the current addendum",
    )
    require(
        int(pixel.get("panel_identity", {}).get("n_queries", -1)) == 512,
        "pixel baseline panel size changed",
    )
    p1a_result_path = REPO_ROOT / (
        "research/conditional_dynamics_representation/artifacts/"
        "pusht_motion_damping_d2_p1a_v1/training_only_cpu_probe_v2.json"
    )
    p1a_inputs = _json(p1a_result_path)["inputs"]
    contextworld_sources = {
        "motion": REPO_ROOT.parent
        / "ContextWorld/contextworld/evaluation/pusht_motion_damping_h3.py",
        "friction": REPO_ROOT.parent
        / "ContextWorld/contextworld/evaluation/pusht_contact_friction_h3.py",
    }
    observed_contextworld = {
        name: sha256_file(path) for name, path in contextworld_sources.items()
    }
    require(
        observed_contextworld["motion"]
        == p1a_inputs["contextworld_motion_source_sha256"],
        "ContextWorld motion source changed after the frozen P1a route result",
    )
    require(
        observed_contextworld["friction"]
        == p1a_inputs["contextworld_friction_source_sha256"],
        "ContextWorld friction source changed after the frozen P1a route result",
    )
    return {
        "identities": identities,
        "addendum_sha256": addendum_sha,
        "design_receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
        "design_receipt_sha256": sha256_file(receipt_path),
        "candidate_stream_path": str(stream_path.relative_to(REPO_ROOT)),
        "candidate_stream_sha256": sha256_file(stream_path),
        "candidate_count": count,
        "pixel_baseline_path": str(pixel_baseline_path.relative_to(REPO_ROOT)),
        "pixel_baseline_sha256": sha256_file(pixel_baseline_path),
        "pixel_panel_sha256": pixel["panel_identity"]["panel_sha256"],
        "contextworld_source_sha256": observed_contextworld,
    }


def iter_calibration_candidates(design_dir: Path) -> Iterator[dict[str, Any]]:
    receipt = _json(Path(design_dir).resolve() / "design_receipt.json")
    path = Path(design_dir).resolve() / receipt["splits"]["calibration"][
        "candidate_stream_path"
    ]
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def template_from_payload(motion: Any, payload: dict[str, Any]) -> Any:
    return motion.MotionDampingTemplate(
        template_id=str(payload["template_id"]),
        faster_decay_reset_snapshot=tuple(payload["faster_decay_reset_snapshot"]),
        no_extra_decay_reset_snapshot=tuple(payload["no_extra_decay_reset_snapshot"]),
        goal_state=tuple(payload["goal_state"]),
        history_actions=tuple(tuple(x) for x in payload["history_actions"]),
        query_actions=tuple(tuple(x) for x in payload["query_actions"]),
        expected_natural_query_snapshot=tuple(
            payload["expected_natural_query_snapshot"]
        ),
        simulator_seed=int(payload["simulator_seed"]),
        visible_shape_id=int(payload.get("visible_shape_id", 4)),
        visible_shape_name=str(payload.get("visible_shape_name", "square")),
    )


def _first_positive_raw_step(counts: Sequence[int], history_steps: int) -> int | None:
    relative = next((i for i, value in enumerate(counts) if int(value) > 0), None)
    return None if relative is None else int(history_steps + relative)


def _max_snapshot_gap(friction: Any, left: Any, right: Any) -> float:
    return float(
        np.max(
            np.abs(
                friction._snapshot_delta(
                    np.asarray(left, dtype=np.float64),
                    np.asarray(right, dtype=np.float64),
                )
            ),
            initial=0.0,
        )
    )


def _native_instrumented_equivalence(
    runtime: RuntimeModules, native: dict[str, Any], traced: dict[str, Any]
) -> dict[str, Any]:
    state_gap = _max_snapshot_gap(
        runtime.friction, native["future_snapshot"], traced["future_snapshot"]
    )
    query_gap = _max_snapshot_gap(
        runtime.friction, native["query_snapshot"], traced["query_snapshot"]
    )
    history_counts_equal = np.array_equal(
        np.asarray(native["history_contacts"], dtype=np.int64),
        np.asarray(traced["history_contacts"], dtype=np.int64),
    )
    query_counts_equal = np.array_equal(
        np.asarray(native["query_contacts"], dtype=np.int64),
        np.asarray(traced["query_contacts_by_raw_step"], dtype=np.int64),
    )
    result = {
        "future_state_max_abs_gap": state_gap,
        "query_state_max_abs_gap": query_gap,
        "query_pixels_identical": bool(
            np.array_equal(native["query_pixels"], traced["query_pixel"])
        ),
        "future_pixels_identical": bool(
            np.array_equal(native["future_pixels"], traced["future_pixel"])
        ),
        "history_raw_contact_counters_identical": bool(history_counts_equal),
        "query_raw_contact_counters_identical": bool(query_counts_equal),
    }
    result["passed"] = bool(
        state_gap == 0.0
        and query_gap == 0.0
        and result["query_pixels_identical"]
        and result["future_pixels_identical"]
        and history_counts_equal
        and query_counts_equal
    )
    return result


def evaluate_active_direction(
    runtime: RuntimeModules, template: Any, direction: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Measure all allowed structural fields for one direction."""

    motion, friction, p1a = runtime.motion, runtime.friction, runtime.p1a
    actions = np.asarray(template.query_actions, dtype=np.float64)
    native: dict[str, dict[str, Any]] = {}
    traced: dict[str, dict[str, Any]] = {}
    equivalence: dict[str, dict[str, Any]] = {}
    for mode in motion.ENDPOINT_MODES:
        native[mode] = motion._simulate_continuous_causal_chain(
            template, mode=mode, resolution=RESOLUTION, render_pixels=True
        )
        traced[mode] = p1a.rollout(
            template,
            mode=mode,
            query_actions=actions,
            label=f"{template.template_id}-active-{mode}",
        )
        equivalence[mode] = _native_instrumented_equivalence(
            runtime, native[mode], traced[mode]
        )
        require(equivalence[mode]["passed"],
                "research-local substep replay diverged from native env.step")
        require(
            traced[mode]["friction_before"]["passed"]
            and traced[mode]["friction_after"]["passed"],
            "motion/contact-friction identity changed",
        )

    fast, slow = motion.ENDPOINT_MODES
    history_gap = friction._visible_response_gap(
        native[fast]["snapshots"][1],
        native[slow]["snapshots"][1],
        angular_radius_px=60.0,
    )
    future_gap = friction._future_gap(
        native[fast]["future_snapshot"], native[slow]["future_snapshot"]
    )
    expected = np.asarray(template.expected_natural_query_snapshot, dtype=np.float64)
    reference_deviation = {
        mode: _max_snapshot_gap(friction, native[mode]["query_snapshot"], expected)
        for mode in motion.ENDPOINT_MODES
    }
    pair_query_gap = _max_snapshot_gap(
        friction, native[fast]["query_snapshot"], native[slow]["query_snapshot"]
    )
    query_pixel_difference = int(
        np.max(
            np.abs(
                native[fast]["query_pixels"].astype(np.int16)
                - native[slow]["query_pixels"].astype(np.int16)
            ),
            initial=0,
        )
    )
    first_counter_raw = {
        mode: _first_positive_raw_step(
            traced[mode]["query_contacts_by_raw_step"], motion.HISTORY_RAW_STEPS
        )
        for mode in motion.ENDPOINT_MODES
    }
    first_physics = {mode: traced[mode]["first_contact"] for mode in motion.ENDPOINT_MODES}

    checks = {
        "history_contact_free": all(
            not any(traced[mode]["history_contacts"])
            and traced[mode]["history_minimum_clearance"] > 0.0
            and not np.any(native[mode]["history_arbiter_counts"])
            for mode in motion.ENDPOINT_MODES
        ),
        "canonical_query_match": bool(
            pair_query_gap <= motion.QUERY_STATE_TOLERANCE
            and query_pixel_difference == 0
            and all(
                value <= motion.QUERY_REFERENCE_TOLERANCE
                for value in reference_deviation.values()
            )
        ),
        "query_start_clear": all(
            traced[mode]["query_start_arbiter_count"] == 0
            and traced[mode]["history_minimum_clearance"] > 0.0
            for mode in motion.ENDPOINT_MODES
        ),
        "both_conditions_contact": all(
            any(traced[mode]["query_contacts_by_raw_step"])
            for mode in motion.ENDPOINT_MODES
        ),
        "equal_first_contact_raw_step": bool(
            first_counter_raw[fast] is not None
            and first_counter_raw[fast] == first_counter_raw[slow]
        ),
        "inherited_history_and_future_separation": bool(
            history_gap["px_equivalent"] >= motion.MINIMUM_HISTORY_GAP_PX
            and future_gap["block_position_px"] >= motion.MINIMUM_FUTURE_GAP_PX
            and not np.array_equal(
                native[fast]["future_pixels"], native[slow]["future_pixels"]
            )
        ),
        "playfield_bounds": all(
            friction._bounds_inside_playfield(bounds)
            for mode in motion.ENDPOINT_MODES
            for bounds in native[mode]["bounds"]
        ),
    }
    return (
        {
            "direction": direction,
            "passed": bool(all(checks.values())),
            "checks": checks,
            "history_visible_response_gap": history_gap,
            "future_gap": future_gap,
            "query_reference_deviation": reference_deviation,
            "pair_query_state_max_abs_gap": pair_query_gap,
            "pair_query_pixel_max_abs_difference": query_pixel_difference,
            "history_minimum_clearance": {
                mode: float(traced[mode]["history_minimum_clearance"])
                for mode in motion.ENDPOINT_MODES
            },
            "query_start_arbiter_count": {
                mode: int(traced[mode]["query_start_arbiter_count"])
                for mode in motion.ENDPOINT_MODES
            },
            "query_contacts_by_raw_step": {
                mode: list(map(int, traced[mode]["query_contacts_by_raw_step"]))
                for mode in motion.ENDPOINT_MODES
            },
            "first_positive_contact_counter_raw_step": first_counter_raw,
            "first_physics_contact": first_physics,
            "native_step_replay_equivalence": equivalence,
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
        },
        native,
        traced,
    )


def _zero_reference_rollouts(
    runtime: RuntimeModules, template: Any, label: str
) -> dict[str, dict[str, Any]]:
    zeros = np.zeros((runtime.motion.QUERY_RAW_STEPS, 2), dtype=np.float64)
    return {
        mode: runtime.p1a.rollout(
            template,
            mode=mode,
            query_actions=zeros,
            label=f"{label}-a-ref-{mode}",
        )
        for mode in runtime.motion.ENDPOINT_MODES
    }


def _gamma_with_axes(
    runtime: RuntimeModules,
    active: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    actions: np.ndarray,
) -> dict[str, Any]:
    report = runtime.p1a.gamma_report(active, reference)
    vector = np.asarray(actions[0], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        approach = vector / norm
        tangent = np.asarray([-approach[1], approach[0]], dtype=np.float64)
        report["local_axis_signed"] = {
            "approach_unit": approach.tolist(),
            "tangent_unit": tangent.tolist(),
            "block_position_approach": float(np.dot(report["block_position"], approach)),
            "block_position_tangent": float(np.dot(report["block_position"], tangent)),
            "block_velocity_approach": float(np.dot(report["block_velocity"], approach)),
            "block_velocity_tangent": float(np.dot(report["block_velocity"], tangent)),
        }
    return report


def shift_agent_geometry(template: Any, shift_px: float, tangent: Sequence[float]) -> Any:
    """Shift only the pusher geometry throughout x0/x2; keep the action fixed."""

    offset = float(shift_px) * np.asarray(tangent, dtype=np.float64)
    require(offset.shape == (2,), "geometry tangent must be planar")

    def shifted_snapshot(value: Sequence[float]) -> tuple[float, ...]:
        array = np.asarray(value, dtype=np.float64).copy()
        require(array.shape == (12,), "geometry audit snapshot must be 12-D")
        array[0:2] += offset
        return tuple(array.tolist())

    goal = np.asarray(template.goal_state, dtype=np.float64).copy()
    goal[0:2] += offset
    suffix = "plus" if shift_px > 0 else "minus"
    return replace(
        template,
        template_id=f"{template.template_id}-geom-{suffix}-1px",
        faster_decay_reset_snapshot=shifted_snapshot(
            template.faster_decay_reset_snapshot
        ),
        no_extra_decay_reset_snapshot=shifted_snapshot(
            template.no_extra_decay_reset_snapshot
        ),
        expected_natural_query_snapshot=shifted_snapshot(
            template.expected_natural_query_snapshot
        ),
        goal_state=tuple(goal.tolist()),
    )


def _variant_summary(
    runtime: RuntimeModules,
    template: Any,
    actions: np.ndarray,
    reference: dict[str, dict[str, Any]],
    nominal_gamma: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    active = {
        mode: runtime.p1a.rollout(
            template,
            mode=mode,
            query_actions=actions,
            label=f"{label}-{mode}",
        )
        for mode in runtime.motion.ENDPOINT_MODES
    }
    gamma = _gamma_with_axes(runtime, active, reference, actions)
    fast, slow = runtime.motion.ENDPOINT_MODES
    first = {
        mode: _first_positive_raw_step(
            active[mode]["query_contacts_by_raw_step"],
            runtime.motion.HISTORY_RAW_STEPS,
        )
        for mode in runtime.motion.ENDPOINT_MODES
    }
    nominal = np.asarray(
        nominal_gamma["block_position"] + nominal_gamma["block_velocity"],
        dtype=np.float64,
    )
    variant = np.asarray(
        gamma["block_position"] + gamma["block_velocity"], dtype=np.float64
    )
    denominator = float(np.linalg.norm(nominal) * np.linalg.norm(variant))
    cosine = None if denominator == 0.0 else float(np.dot(nominal, variant) / denominator)
    dominant = int(np.argmax(np.abs(nominal))) if nominal.size else 0
    dominant_sign_same = None
    if nominal.size and nominal[dominant] != 0.0 and variant[dominant] != 0.0:
        dominant_sign_same = bool(
            math.copysign(1.0, nominal[dominant])
            == math.copysign(1.0, variant[dominant])
        )
    return {
        "contact_both_conditions": bool(
            all(any(active[mode]["query_contacts_by_raw_step"])
                for mode in runtime.motion.ENDPOINT_MODES)
        ),
        "equal_first_contact_raw_step": bool(
            first[fast] is not None and first[fast] == first[slow]
        ),
        "first_contact_raw_step": first,
        "gamma": gamma,
        "gamma_block_response_cosine_to_nominal": cosine,
        "nominal_dominant_component_index": dominant,
        "nominal_dominant_component_sign_stable": dominant_sign_same,
    }


def post_admission_audit(
    runtime: RuntimeModules,
    templates: dict[str, Any],
    active: dict[str, dict[str, dict[str, Any]]],
    references: dict[str, dict[str, dict[str, Any]]],
    nominal_gamma: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Run only audit quantities that cannot affect structural admission."""

    action_robustness: dict[str, Any] = {}
    geometry_robustness: dict[str, Any] = {}
    same_damping_null: dict[str, Any] = {}
    fast = runtime.motion.ENDPOINT_MODES[0]
    for direction in DIRECTIONS:
        template = templates[direction]
        cell = int(candidate["coverage_cell"])
        stratum = str(candidate["action_stratum"])
        action_robustness[direction] = {}
        for degrees in AUDIT_ONLY_ACTION_ROTATION_DEGREES:
            actions = build_query_actions(
                template.expected_natural_query_snapshot,
                stratum_id=stratum,
                coverage_cell=cell,
                split="calibration",
                extra_rotation_degrees=degrees,
            )
            variant = replace(
                template,
                template_id=f"{template.template_id}-action-rot-{degrees:+.0f}",
                query_actions=tuple(map(tuple, actions.tolist())),
            )
            action_robustness[direction][f"{degrees:+.0f}_degrees"] = _variant_summary(
                runtime,
                variant,
                actions,
                references[direction],
                nominal_gamma[direction],
                f"{template.template_id}-action-rot-{degrees:+.0f}",
            )

        geometry_robustness[direction] = {}
        tangent = local_tangent_unit(template.expected_natural_query_snapshot)
        actions = np.asarray(template.query_actions, dtype=np.float64)
        for shift in AUDIT_ONLY_AGENT_TANGENT_SHIFT_PX:
            variant = shift_agent_geometry(template, shift, tangent)
            variant_reference = _zero_reference_rollouts(
                runtime, variant, f"{variant.template_id}-geometry-audit"
            )
            geometry_robustness[direction][f"{shift:+.0f}_px"] = _variant_summary(
                runtime,
                variant,
                actions,
                variant_reference,
                nominal_gamma[direction],
                f"{variant.template_id}-active",
            )

        repeated_active = runtime.p1a.rollout(
            template,
            mode=fast,
            query_actions=np.asarray(template.query_actions, dtype=np.float64),
            label=f"{template.template_id}-same-damping-active-repeat",
        )
        repeated_ref = runtime.p1a.rollout(
            template,
            mode=fast,
            query_actions=np.zeros((runtime.motion.QUERY_RAW_STEPS, 2), dtype=np.float64),
            label=f"{template.template_id}-same-damping-ref-repeat",
        )
        same_damping_null[direction] = runtime.p1a.same_damping_null(
            active[direction][fast],
            repeated_active,
            references[direction][fast],
            repeated_ref,
        )
        require(
            same_damping_null[direction]["passed"],
            "same-damping Gamma null failed; calibration execution is invalid",
        )
    return {
        "selection_use": "audit_only_after_structural_admission",
        "action_rotation": action_robustness,
        "agent_tangent_geometry_shift": geometry_robustness,
        "same_damping_gamma_null": same_damping_null,
    }


def _array_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write an NPZ with fixed member ordering and timestamps."""

    path = reject_forbidden_path(path, name="output NPZ").resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in sorted(arrays):
            require(name and "/" not in name, "invalid NPZ member name")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _array_bytes(np.asarray(arrays[name])))
    return sha256_file(path)


def _accepted_sidecar_arrays(
    runtime: RuntimeModules,
    candidate: dict[str, Any],
    native: dict[str, dict[str, dict[str, Any]]],
    references: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, np.ndarray]:
    modes = runtime.motion.ENDPOINT_MODES
    active_pixels = []
    active_states = []
    reference_query_pixels = []
    reference_future_pixels = []
    reference_query_states = []
    reference_future_states = []
    for direction in DIRECTIONS:
        direction_pixels = []
        direction_states = []
        direction_ref_q_pixels = []
        direction_ref_f_pixels = []
        direction_ref_q_states = []
        direction_ref_f_states = []
        for mode in modes:
            value = native[direction][mode]
            direction_pixels.append(
                np.concatenate(
                    [np.asarray(value["pixels"], dtype=np.uint8),
                     np.asarray(value["future_pixels"], dtype=np.uint8)[None]],
                    axis=0,
                )
            )
            direction_states.append(
                np.concatenate(
                    [np.asarray(value["snapshots"], dtype=np.float64),
                     np.asarray(value["future_snapshot"], dtype=np.float64)[None]],
                    axis=0,
                )
            )
            ref = references[direction][mode]
            direction_ref_q_pixels.append(np.asarray(ref["query_pixel"], dtype=np.uint8))
            direction_ref_f_pixels.append(np.asarray(ref["future_pixel"], dtype=np.uint8))
            direction_ref_q_states.append(np.asarray(ref["query_snapshot"], dtype=np.float64))
            direction_ref_f_states.append(np.asarray(ref["future_snapshot"], dtype=np.float64))
        active_pixels.append(direction_pixels)
        active_states.append(direction_states)
        reference_query_pixels.append(direction_ref_q_pixels)
        reference_future_pixels.append(direction_ref_f_pixels)
        reference_query_states.append(direction_ref_q_states)
        reference_future_states.append(direction_ref_f_states)
    return {
        "active_model_physics_states": np.asarray(active_states, dtype=np.float64),
        "active_model_pixels": np.asarray(active_pixels, dtype=np.uint8),
        "coverage_cell": np.asarray([candidate["coverage_cell"]], dtype=np.int16),
        "group_index": np.asarray([candidate["group_index"]], dtype=np.int64),
        "query_actions": np.asarray(
            [candidate["query_actions"][direction] for direction in DIRECTIONS],
            dtype=np.float64,
        ),
        "reference_future_physics_states": np.asarray(
            reference_future_states, dtype=np.float64
        ),
        "reference_future_pixels": np.asarray(reference_future_pixels, dtype=np.uint8),
        "reference_query_physics_states": np.asarray(
            reference_query_states, dtype=np.float64
        ),
        "reference_query_pixels": np.asarray(reference_query_pixels, dtype=np.uint8),
    }


def _candidate_identity(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_index": int(candidate["group_index"]),
        "candidate_sha256": str(candidate["candidate_sha256"]),
        "coverage_cell": int(candidate["coverage_cell"]),
        "orientation_bin": int(candidate["orientation_bin"]),
        "speed_bin": int(candidate["speed_bin"]),
        "goal_distance_bin": int(candidate["goal_distance_bin"]),
        "action_stratum": str(candidate["action_stratum"]),
        "within_cell_candidate_encounter_rank": int(
            candidate["within_cell_candidate_encounter_rank"]
        ),
        "template_ids": {
            direction: candidate["templates"][direction]["template_id"]
            for direction in DIRECTIONS
        },
    }


def _evaluate_candidate(
    runtime: RuntimeModules, candidate: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    templates = {
        direction: template_from_payload(runtime.motion, candidate["templates"][direction])
        for direction in DIRECTIONS
    }
    direction_summary: dict[str, Any] = {}
    native: dict[str, dict[str, dict[str, Any]]] = {}
    active: dict[str, dict[str, dict[str, Any]]] = {}
    for direction in DIRECTIONS:
        summary, native[direction], active[direction] = evaluate_active_direction(
            runtime, templates[direction], direction
        )
        direction_summary[direction] = summary
    group_checks = {
        "action_norm_and_box": bool(candidate["pre_outcome_passed"]),
        **{
            key: all(direction_summary[d]["checks"][key] for d in DIRECTIONS)
            for key in (
                "history_contact_free",
                "canonical_query_match",
                "query_start_clear",
                "both_conditions_contact",
                "equal_first_contact_raw_step",
                "inherited_history_and_future_separation",
                "playfield_bounds",
            )
        },
    }
    # Freeze the admission decision before reference/Gamma is computed.
    structurally_admitted = bool(all(group_checks.values()))

    references: dict[str, dict[str, dict[str, Any]]] = {}
    gamma: dict[str, Any] = {}
    for direction in DIRECTIONS:
        references[direction] = _zero_reference_rollouts(
            runtime, templates[direction], templates[direction].template_id
        )
        gamma[direction] = _gamma_with_axes(
            runtime,
            active[direction],
            references[direction],
            np.asarray(templates[direction].query_actions, dtype=np.float64),
        )
    summary = {
        **_candidate_identity(candidate),
        "admission_decision_frozen_before_post_outcome_audit": True,
        "structurally_admitted": structurally_admitted,
        "structural_checks": group_checks,
        "structural_by_direction": direction_summary,
        "post_decision_gamma_not_used_for_selection": gamma,
    }
    return summary, templates, native, active, references


def _run_frozen_slot(
    slot: tuple[int, str],
    candidates: list[dict[str, Any]],
    sidecar_dir_text: str,
) -> dict[str, Any]:
    """Scan one independent cell/action slot in frozen index order.

    Selecting the earliest structural pass independently in every preassigned
    slot is exactly equivalent to the committed global scan because slots do
    not share quota.  No cross-slot outcome is visible here.
    """

    cell, stratum = int(slot[0]), str(slot[1])
    require(candidates, f"frozen slot {slot!r} has no candidates")
    require(
        all(
            int(row["coverage_cell"]) == cell
            and str(row["action_stratum"]) == stratum
            for row in candidates
        ),
        "slot worker received a cross-slot candidate",
    )
    indices = [int(row["group_index"]) for row in candidates]
    require(indices == sorted(indices) and len(indices) == len(set(indices)),
            "slot candidate order is not strictly increasing")
    runtime = load_runtime()
    rejected: list[dict[str, Any]] = []
    simulator_evaluated = 0
    pre_outcome_rejected = 0
    for candidate in candidates:
        if not bool(candidate["pre_outcome_passed"]):
            pre_outcome_rejected += 1
            rejected.append({
                **_candidate_identity(candidate),
                "structurally_admitted": False,
                "rejection_reasons": ["action_norm_and_box"],
                "post_decision_gamma_not_computed": "pre_outcome_action_rejection",
            })
            continue
        simulator_evaluated += 1
        summary, templates, native, active, references = _evaluate_candidate(
            runtime, candidate
        )
        if not summary["structurally_admitted"]:
            summary["rejection_reasons"] = [
                key for key, value in summary["structural_checks"].items() if not value
            ]
            rejected.append(summary)
            continue

        # The structural decision above is now immutable.  These calls are
        # audit/materialization only and cannot replace this candidate.
        summary["post_admission_audit"] = post_admission_audit(
            runtime,
            templates,
            active,
            references,
            summary["post_decision_gamma_not_used_for_selection"],
            candidate,
        )
        sidecar_dir = Path(sidecar_dir_text)
        sidecar_path = sidecar_dir / f"group_{int(candidate['group_index']):06d}.npz"
        sidecar_sha = write_npz_deterministic(
            sidecar_path,
            _accepted_sidecar_arrays(runtime, candidate, native, references),
        )
        summary["sidecar"] = {
            "path": f"accepted_sidecars/{sidecar_path.name}",
            "sha256": sidecar_sha,
        }
        return {
            "slot": [cell, stratum],
            "accepted": summary,
            "rejected": rejected,
            "simulator_evaluated": simulator_evaluated,
            "pre_outcome_rejected": pre_outcome_rejected,
        }
    return {
        "slot": [cell, stratum],
        "accepted": None,
        "rejected": rejected,
        "simulator_evaluated": simulator_evaluated,
        "pre_outcome_rejected": pre_outcome_rejected,
    }


def run_calibration(
    *,
    design_dir: Path,
    pixel_baseline_path: Path,
    output_dir: Path,
    workers: int = 32,
    progress_every: int = 8,
) -> tuple[dict[str, Any], int]:
    """Execute exactly one full calibration opening."""

    verified = verify_inputs(design_dir, pixel_baseline_path)
    output_dir = reject_forbidden_path(output_dir, name="output directory").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sidecar_dir = output_dir / "accepted_sidecars"
    sidecar_dir.mkdir()

    try:
        slots: dict[tuple[int, str], list[dict[str, Any]]] = {
            (cell, stratum): []
            for cell in range(COVERAGE_CELLS)
            for stratum in STRATUM_IDS
        }
        all_candidates: list[dict[str, Any]] = []
        for candidate in iter_calibration_candidates(design_dir):
            slot = (int(candidate["coverage_cell"]), str(candidate["action_stratum"]))
            require(slot in slots, "candidate has an unregistered quota slot")
            slots[slot].append(candidate)
            all_candidates.append(candidate)
        require(len(all_candidates) == verified["candidate_count"],
                "formal run did not load the complete verified calibration stream")
        require(all(slots.values()), "at least one frozen quota slot is empty")

        max_workers = max(1, min(int(workers), len(slots), os.cpu_count() or 1))
        slot_results: list[dict[str, Any]] = []
        completed_slots = 0
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
            futures = {
                pool.submit(
                    _run_frozen_slot, slot, candidates, str(sidecar_dir)
                ): slot
                for slot, candidates in slots.items()
            }
            for future in as_completed(futures):
                result = future.result()
                slot_results.append(result)
                completed_slots += 1
                if progress_every > 0 and (
                    completed_slots % progress_every == 0
                    or completed_slots == len(slots)
                ):
                    print(
                        json.dumps({
                            "completed_slots": completed_slots,
                            "total_slots": len(slots),
                            "accepted_slots": sum(
                                value["accepted"] is not None for value in slot_results
                            ),
                            "simulator_evaluated": sum(
                                int(value["simulator_evaluated"])
                                for value in slot_results
                            ),
                        }, sort_keys=True),
                        file=sys.stderr,
                        flush=True,
                    )

        accepted = sorted(
            [result["accepted"] for result in slot_results
             if result["accepted"] is not None],
            key=lambda row: int(row["group_index"]),
        )
        rejected = sorted(
            [row for result in slot_results for row in result["rejected"]],
            key=lambda row: int(row["group_index"]),
        )
        accepted_indices = {int(row["group_index"]) for row in accepted}
        rejected_indices = {int(row["group_index"]) for row in rejected}
        require(accepted_indices.isdisjoint(rejected_indices),
                "candidate was both accepted and rejected")
        require(len(rejected_indices) == len(rejected), "duplicate evaluated rejection")
        filled = {
            (int(row["coverage_cell"]), str(row["action_stratum"]))
            for row in accepted
        }
        quota_complete = len(filled) == COVERAGE_CELLS * len(STRATUM_IDS)
        low, high = CANDIDATE_WINDOWS["calibration"]
        last_scanned_index = max(accepted_indices) if quota_complete else high
        quota_skips = [
            {
                **_candidate_identity(candidate),
                "rejection_reason": "coverage_action_quota_already_full",
                "simulator_rollouts_executed": 0,
            }
            for candidate in all_candidates
            if int(candidate["group_index"]) <= last_scanned_index
            and int(candidate["group_index"]) not in accepted_indices
            and int(candidate["group_index"]) not in rejected_indices
        ]
        scanned_count = last_scanned_index - low + 1
        require(
            scanned_count
            == len(accepted) + len(rejected) + len(quota_skips),
            "global candidate accounting does not close",
        )

        accepted_writer = JsonlWriter(output_dir / "accepted_groups.jsonl")
        rejected_writer = JsonlWriter(output_dir / "evaluated_rejections.jsonl")
        skipped_writer = JsonlWriter(output_dir / "quota_skips.jsonl")
        for row in accepted:
            accepted_writer.write(row)
        for row in rejected:
            rejected_writer.write(row)
        for row in quota_skips:
            skipped_writer.write(row)
        accepted_sha = accepted_writer.close()
        rejected_sha = rejected_writer.close()
        skipped_sha = skipped_writer.close()
        unscanned_count = (high - low + 1) - scanned_count
        evaluated_count = sum(
            int(result["simulator_evaluated"]) for result in slot_results
        )
        pre_outcome_rejected_count = sum(
            int(result["pre_outcome_rejected"]) for result in slot_results
        )
        rejection_reason_counts: dict[str, int] = {}
        for row in rejected:
            for reason in row["rejection_reasons"]:
                rejection_reason_counts[reason] = (
                    rejection_reason_counts.get(reason, 0) + 1
                )
        filled_by_cell = {
            str(cell): {
                stratum: int((cell, stratum) in filled) for stratum in STRATUM_IDS
            }
            for cell in range(COVERAGE_CELLS)
        }
        sidecars = [
            {
                "group_index": int(row["group_index"]),
                "path": row["sidecar"]["path"],
                "sha256": row["sidecar"]["sha256"],
            }
            for row in accepted
        ]
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "runner_id": RUNNER_ID,
            "status": (
                "calibration_structural_quota_complete"
                if quota_complete
                else "valid_negative_p1b_calibration_structural_failure"
            ),
            "scope": "training_only_p1b_calibration_only",
            "runner_sha256": sha256_file(THIS_SOURCE),
            "verified_inputs": verified,
            "calibration_outcome_opened": True,
            "sealed_holdout_opened": False,
            "sealed_holdout_open_count": 0,
            "development_opened": False,
            "public_test_opened": False,
            "model_loaded": False,
            "optimizer_steps": 0,
            "gpu_used": False,
            "quota": {
                "expected_groups": GROUPS_PER_SPLIT,
                "expected_slots": COVERAGE_CELLS * len(STRATUM_IDS),
                "complete": quota_complete,
                "accepted_groups": len(accepted),
                "filled_by_cell_and_stratum": filled_by_cell,
                "underfilled_slots": [
                    {"coverage_cell": cell, "action_stratum": stratum}
                    for cell in range(COVERAGE_CELLS)
                    for stratum in STRATUM_IDS
                    if (cell, stratum) not in filled
                ],
            },
            "candidate_accounting": {
                "window": list(CANDIDATE_WINDOWS["calibration"]),
                "last_scanned_group_index": last_scanned_index,
                "scanned_candidates": scanned_count,
                "unscanned_after_quota_completion": unscanned_count,
                "simulator_evaluated_candidates": evaluated_count,
                "pre_outcome_action_rejections": pre_outcome_rejected_count,
                "simulator_structural_rejections": (
                    rejected_writer.count - pre_outcome_rejected_count
                ),
                "all_evaluated_rejection_rows": rejected_writer.count,
                "quota_already_full_skips": skipped_writer.count,
                "rejection_reason_counts": rejection_reason_counts,
            },
            "execution": {
                "parallelization": "independent_preassigned_cell_action_slots",
                "slot_internal_order": "strictly_increasing_group_index",
                "equivalence_to_global_scan": (
                    "each quota slot is independent, so its earliest structural pass "
                    "equals the candidate selected by the committed global scan"
                ),
                "worker_processes": max_workers,
                "process_start_method": "spawn",
            },
            "selection_contract": {
                "admission_may_use_only": list(ADMISSION_MAY_USE_ONLY),
                "admission_must_not_use": list(ADMISSION_MUST_NOT_USE),
                "gamma_and_robustness_computed_after_decision": True,
                "gamma_or_robustness_used_for_selection": False,
                "candidate_order": "strictly_increasing_frozen_group_index",
                "slot_rule": "first_structurally_passing_candidate_per_cell_action_stratum",
            },
            "outputs": {
                "accepted_groups": {
                    "path": "accepted_groups.jsonl",
                    "rows": accepted_writer.count,
                    "sha256": accepted_sha,
                },
                "evaluated_rejections": {
                    "path": "evaluated_rejections.jsonl",
                    "rows": rejected_writer.count,
                    "sha256": rejected_sha,
                },
                "quota_skips": {
                    "path": "quota_skips.jsonl",
                    "rows": skipped_writer.count,
                    "sha256": skipped_sha,
                },
                "accepted_sidecars": sidecars,
                "accepted_sidecar_schema": {
                    "direction_axis": list(DIRECTIONS),
                    "hidden_mode_axis": ["faster_decay", "no_extra_decay"],
                    "active_model_frame_axis": ["x0", "x1", "x2_query", "x3_future"],
                    "pixel_dtype_and_scale": "uint8_rgb_0_255_224x224",
                    "physics_state_dtype": "float64_12d",
                    "reference_frames": ["x2_query", "x3_future"],
                },
            },
            "claim_boundary": (
                "A complete structural quota only licenses calibration analysis and "
                "threshold freezing. It does not establish an effect gate, holdout pass, "
                "D2-0 pass, representation transfer, or native ICL improvement."
            ),
        }
        receipt_sha = write_json_exclusive(output_dir / "calibration_receipt.json", receipt)
        receipt["calibration_receipt_sha256"] = receipt_sha
        return receipt, (0 if quota_complete else 2)
    except Exception:
        invalid = {
            "schema_version": SCHEMA_VERSION,
            "runner_id": RUNNER_ID,
            "status": "invalid_execution",
            "scientific_result": "not_interpretable",
            "calibration_outcome_may_have_been_partially_opened": True,
            "sealed_holdout_opened": False,
            "sealed_holdout_open_count": 0,
            "last_scanned_group_index": None,
            "simulator_evaluated_candidates": "unknown_if_worker_failed",
            "exception_type": sys.exc_info()[0].__name__ if sys.exc_info()[0] else None,
            "exception_message": str(sys.exc_info()[1]),
            "traceback": traceback.format_exc(),
            "repair_rule": (
                "Repair implementation only, emit a deviation receipt, and use a new "
                "output identity without changing the frozen scientific design."
            ),
        }
        try:
            write_json_exclusive(output_dir / "invalid_execution_receipt.json", invalid)
        except Exception:
            pass
        raise


def check_only(design_dir: Path, pixel_baseline_path: Path) -> dict[str, Any]:
    verified = verify_inputs(design_dir, pixel_baseline_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_id": RUNNER_ID,
        "status": "check_ok_no_outcome_opened",
        "verified_inputs": verified,
        "calibration_outcome_opened": False,
        "sealed_holdout_opened": False,
        "development_opened": False,
        "public_test_opened": False,
        "model_loaded": False,
        "optimizer_steps": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--pixel-baseline", type=Path, default=DEFAULT_PIXEL_BASELINE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=8)
    args = parser.parse_args(argv)
    if args.check_only:
        print(json.dumps(check_only(args.design_dir, args.pixel_baseline),
                         indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-only is used")
    receipt, code = run_calibration(
        design_dir=args.design_dir,
        pixel_baseline_path=args.pixel_baseline,
        output_dir=args.output_dir,
        workers=max(1, int(args.workers)),
        progress_every=max(0, int(args.progress_every)),
    )
    print(json.dumps({
        "status": receipt["status"],
        "output_dir": str(args.output_dir),
        "accepted_groups": receipt["quota"]["accepted_groups"],
    }, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
