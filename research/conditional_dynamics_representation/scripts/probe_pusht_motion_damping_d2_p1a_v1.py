#!/usr/bin/env python3
"""Run the deterministic Training-only Motion D2 P1a physics probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from pymunk.vec2d import Vec2d


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = REPO_ROOT.parent / "ContextWorld"
if str(CONTEXTWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEXTWORLD_ROOT))

from contextworld.evaluation import pusht_contact_friction_h3 as friction  # noqa: E402
from contextworld.evaluation import pusht_motion_damping_h3 as motion  # noqa: E402


SCHEMA_VERSION = 1
RESOLUTION = 224
PLANNER_SCALES = (0.0, 0.25, 0.625, 1.0)
EXPECTED_RECEIPT_SHA256 = (
    "747c290954cc53703124893aaa64b1e0898c07ed34e29c5dfb7b3af929af1450"
)
EXPECTED_PAYLOAD_SHA256 = (
    "aa8ac1f0e3077fecc9ff92ba521a35778fc236d8c69f25d69f275af34f8fddc7"
)
EXPECTED_ACTIONS_SHA256 = (
    "f2a15416f9ed8b7682aa3419dafdb5a65e1a6f56d1e50c9f9b98e6fc43876a86"
)
PLANNER_RECEIPT = REPO_ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_planner_curve_cartesian_action_overlay_half4096_v1/"
    "train_templates4096_v1.pt.json"
)
CONTRACT = REPO_ROOT / (
    "research/conditional_dynamics_representation/configs/"
    "pusht_motion_damping_d2_preregistration_v1.yaml"
)
PLAN = REPO_ROOT / (
    "research/conditional_dynamics_representation/D2_CONSTRUCTION_PLAN_ZH.md"
)
PHYSICAL_ZERO_ATOL = 1.0e-10
PIXEL_ZERO_ATOL = 0.0
NONZERO_ATOL = 1.0e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def verify_frozen_inputs() -> dict[str, Any]:
    required = (PLANNER_RECEIPT, CONTRACT, PLAN, THIS_SOURCE)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    receipt_hash = sha256_file(PLANNER_RECEIPT)
    if receipt_hash != EXPECTED_RECEIPT_SHA256:
        raise RuntimeError("Frozen planner-support receipt SHA256 changed")
    receipt = json.loads(PLANNER_RECEIPT.read_text(encoding="utf-8"))
    expected_fields = {
        "overlay_sha256": EXPECTED_PAYLOAD_SHA256,
        "raw_action_blocks_sha256": EXPECTED_ACTIONS_SHA256,
        "resolution": RESOLUTION,
        "scale_grid": list(PLANNER_SCALES),
        "action_source": "fixed_deployment_planner_proposal_curve",
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"Frozen planner-support field changed: {key}")
    return {
        "planner_support_receipt": str(PLANNER_RECEIPT.relative_to(REPO_ROOT)),
        "planner_support_receipt_sha256": receipt_hash,
        "planner_support_payload_sha256_recorded": receipt["overlay_sha256"],
        "planner_support_raw_actions_sha256": receipt["raw_action_blocks_sha256"],
        "contract_sha256": sha256_file(CONTRACT),
        "plan_sha256": sha256_file(PLAN),
        "script_sha256": sha256_file(THIS_SOURCE),
        "contextworld_motion_source_sha256": sha256_file(Path(motion.__file__)),
        "contextworld_friction_source_sha256": sha256_file(Path(friction.__file__)),
    }


def planner_action(template: motion.MotionDampingTemplate, scale: float) -> np.ndarray:
    query = np.asarray(template.expected_natural_query_snapshot, dtype=np.float64)
    base = np.clip((query[6:8] + query[8:10] - query[0:2]) / 100.0, -1.0, 1.0)
    return np.repeat((float(scale) * base)[None], motion.QUERY_RAW_STEPS, axis=0)


def action_audit(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    norms = np.linalg.norm(actions, axis=1)
    return {
        "per_step_l2_norm": norms.tolist(),
        "maximum_l2_norm": float(norms.max(initial=0.0)),
        "inside_component_box": bool(np.all(np.abs(actions) <= 1.0)),
        "inside_l2_unit_ball": bool(np.all(norms <= 1.0 + 1.0e-12)),
    }


def rotation_audit(
    template: motion.MotionDampingTemplate,
    actions: np.ndarray,
) -> dict[str, Any]:
    candidate = replace(template, query_actions=tuple(map(tuple, actions.tolist())))
    rotated = motion.rigid_transform_template(
        candidate,
        template_id=f"{candidate.template_id}-rotation-audit",
        angle_rad=np.pi / 4,
        tangential_offset=0.0,
        simulator_seed=candidate.simulator_seed,
    )
    transformed = np.asarray(rotated.query_actions, dtype=np.float64)
    before = np.linalg.norm(actions, axis=1)
    after = np.linalg.norm(transformed, axis=1)
    try:
        friction._action_array(
            transformed,
            expected_steps=motion.QUERY_RAW_STEPS,
            name="rotation_audit_query_actions",
        )
        accepted_without_clipping = True
    except ValueError:
        accepted_without_clipping = False
    return {
        "angle_rad": float(np.pi / 4),
        "maximum_norm_difference": float(np.max(np.abs(before - after), initial=0.0)),
        "transformed_inside_component_box": bool(np.all(np.abs(transformed) <= 1.0)),
        "no_clipping_required": bool(
            accepted_without_clipping
            and np.all(np.abs(transformed) <= 1.0)
            and np.allclose(before, after, atol=1.0e-12, rtol=0.0)
        ),
    }


def motion_friction_audit(env: Any, mode: str) -> dict[str, Any]:
    agent = sorted(float(shape.friction) for shape in env.agent.shapes)
    block = sorted(float(shape.friction) for shape in env.block.shapes)
    walls = sorted(
        float(shape.friction)
        for shape in env.space.shapes
        if shape.body not in (env.agent, env.block)
    )
    products = sorted(left * right for left in agent for right in block)
    damping = float(env.space.damping)
    passed = bool(
        products
        and all(np.isclose(x, motion.EFFECTIVE_CONTACT_FRICTION, atol=1.0e-12, rtol=0.0) for x in products)
        and all(x == 0.0 for x in walls)
        and damping == float(motion.DAMPING_VALUES[mode])
    )
    return {
        "agent_shape_friction": agent,
        "block_shape_friction": block,
        "agent_block_products": products,
        "wall_shape_friction": walls,
        "space_damping": damping,
        "expected_space_damping": float(motion.DAMPING_VALUES[mode]),
        "passed": passed,
    }


def agent_block_clearance(env: Any) -> float:
    values: list[float] = []
    for agent_shape in env.agent.shapes:
        radius = getattr(agent_shape, "radius", None)
        offset = getattr(agent_shape, "offset", None)
        if radius is None or offset is None:
            raise RuntimeError("P1a clearance requires the registered circular PushT agent")
        center = agent_shape.body.local_to_world(offset)
        for block_shape in env.block.shapes:
            values.append(float(block_shape.point_query(center).distance - radius))
    if not values or not np.all(np.isfinite(values)):
        raise RuntimeError("Unable to compute agent-block clearance")
    return float(min(values))


def _contact_event(arbiter: Any, raw_step: int, substep: int) -> dict[str, Any]:
    contact_set = arbiter.contact_point_set
    points = []
    for point in contact_set.points:
        points.append(
            {
                "point_a": [float(x) for x in point.point_a],
                "point_b": [float(x) for x in point.point_b],
                "distance": float(point.distance),
            }
        )
    return {
        "raw_step": int(raw_step),
        "physics_substep": int(substep),
        "normal": [float(x) for x in contact_set.normal],
        "points": points,
        "point_count": len(points),
        "total_impulse": [float(x) for x in arbiter.total_impulse],
        "minimum_point_distance": float(min((p["distance"] for p in points), default=0.0)),
    }


def _step_with_substep_trace(
    env: Any,
    action: np.ndarray,
    *,
    raw_step: int,
    trace_state: dict[str, Any],
) -> tuple[int, list[int]]:
    env._contextworld_agent_block_contact_points = 0
    action = np.asarray(action, dtype=np.float64)
    env.latest_action = action.copy()
    target = env.agent.position + action * env.action_scale if env.relative else action
    n_steps = int(1 / (env.dt * env.control_hz))
    arbiter_counts: list[int] = []
    for substep in range(n_steps):
        trace_state["raw_step"] = int(raw_step)
        trace_state["physics_substep"] = int(substep)
        acceleration = env.k_p * (target - env.agent.position) + env.k_v * (
            Vec2d(0, 0) - env.agent.velocity
        )
        env.agent.velocity += acceleration * env.dt
        env.space.step(env.dt)
        arbiter_counts.append(int(len(env.space._get_arbiters())))
    return int(env._contextworld_agent_block_contact_points), arbiter_counts


def rollout(
    template: motion.MotionDampingTemplate,
    *,
    mode: str,
    query_actions: np.ndarray,
    label: str,
) -> dict[str, Any]:
    branch = replace(template, query_actions=tuple(map(tuple, query_actions.tolist())))
    env, _ = motion.make_motion_damping_env(branch, mode=mode, resolution=RESOLUTION)
    trace_state: dict[str, Any] = {"raw_step": -1, "physics_substep": -1}
    events: list[dict[str, Any]] = []

    def post_solve(arbiter: Any, _space: Any, _data: Any) -> None:
        points = len(arbiter.contact_point_set.points)
        env._contextworld_agent_block_contact_points += points
        events.append(
            _contact_event(
                arbiter,
                trace_state["raw_step"],
                trace_state["physics_substep"],
            )
        )

    env.space.on_collision(
        friction.AGENT_COLLISION_TYPE,
        friction.BLOCK_COLLISION_TYPE,
        post_solve=post_solve,
    )
    history_contacts: list[int] = []
    query_contacts: list[int] = []
    query_arbiter_by_substep: list[list[int]] = []
    history_clearances = [agent_block_clearance(env)]
    initial_pixel = np.asarray(env.render(), dtype=np.uint8).copy()
    friction_before = motion_friction_audit(env, mode)
    try:
        for raw_step, action in enumerate(np.asarray(branch.history_actions, dtype=np.float64)):
            contacts, _ = _step_with_substep_trace(
                env, action, raw_step=raw_step, trace_state=trace_state
            )
            history_contacts.append(contacts)
            history_clearances.append(agent_block_clearance(env))
        query_snapshot = friction.body_snapshot(env)
        query_pixel = np.asarray(env.render(), dtype=np.uint8).copy()
        query_start_arbiter_count = int(len(env.space._get_arbiters()))
        for offset, action in enumerate(query_actions):
            raw_step = motion.HISTORY_RAW_STEPS + offset
            contacts, arbiter_counts = _step_with_substep_trace(
                env, action, raw_step=raw_step, trace_state=trace_state
            )
            query_contacts.append(contacts)
            query_arbiter_by_substep.append(arbiter_counts)
        future_snapshot = friction.body_snapshot(env)
        future_pixel = np.asarray(env.render(), dtype=np.uint8).copy()
        friction_after = motion_friction_audit(env, mode)
    finally:
        env.close()

    query_events = [event for event in events if event["raw_step"] >= motion.HISTORY_RAW_STEPS]
    first = query_events[0] if query_events else None
    return {
        "label": label,
        "mode": mode,
        "initial_pixel_sha256": array_sha256(initial_pixel),
        "query_snapshot": query_snapshot,
        "query_pixel": query_pixel,
        "future_snapshot": future_snapshot,
        "future_pixel": future_pixel,
        "history_contacts": history_contacts,
        "history_minimum_clearance": float(min(history_clearances)),
        "query_start_arbiter_count": query_start_arbiter_count,
        "query_contacts_by_raw_step": query_contacts,
        "query_arbiter_counts_by_physics_substep": query_arbiter_by_substep,
        "query_contact_events": query_events,
        "first_contact": first,
        "friction_before": friction_before,
        "friction_after": friction_after,
    }


def snapshot_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return friction._snapshot_delta(
        np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    )


def pixel_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(left, dtype=np.float64) / 255.0 - np.asarray(right, dtype=np.float64) / 255.0


def gamma_report(pair: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fast, slow = motion.ENDPOINT_MODES
    delta_phys = snapshot_delta(pair[slow]["future_snapshot"], pair[fast]["future_snapshot"])
    delta_ref_phys = snapshot_delta(
        reference[slow]["future_snapshot"], reference[fast]["future_snapshot"]
    )
    gamma_phys = delta_phys - delta_ref_phys
    delta_pix = pixel_delta(pair[slow]["future_pixel"], pair[fast]["future_pixel"])
    delta_ref_pix = pixel_delta(
        reference[slow]["future_pixel"], reference[fast]["future_pixel"]
    )
    gamma_pix = delta_pix - delta_ref_pix
    return {
        "physical_components": gamma_phys.tolist(),
        "physical_max_abs": float(np.max(np.abs(gamma_phys), initial=0.0)),
        "block_position": gamma_phys[6:8].tolist(),
        "block_velocity": gamma_phys[8:10].tolist(),
        "delta_active_pixel_energy": float(np.mean(delta_pix**2)),
        "delta_a_ref_pixel_energy": float(np.mean(delta_ref_pix**2)),
        "gamma_pixel_energy": float(np.mean(gamma_pix**2)),
        "gamma_pixel_max_abs": float(np.max(np.abs(gamma_pix), initial=0.0)),
        "gamma_pixel_sha256": array_sha256(gamma_pix),
    }


def same_damping_null(
    original_active: dict[str, Any],
    repeated_active: dict[str, Any],
    original_ref: dict[str, Any],
    repeated_ref: dict[str, Any],
) -> dict[str, Any]:
    delta_active = snapshot_delta(
        repeated_active["future_snapshot"], original_active["future_snapshot"]
    )
    delta_ref = snapshot_delta(
        repeated_ref["future_snapshot"], original_ref["future_snapshot"]
    )
    gamma_phys = delta_active - delta_ref
    gamma_pix = pixel_delta(
        repeated_active["future_pixel"], original_active["future_pixel"]
    ) - pixel_delta(repeated_ref["future_pixel"], original_ref["future_pixel"])
    return {
        "physical_max_abs": float(np.max(np.abs(gamma_phys), initial=0.0)),
        "pixel_max_abs": float(np.max(np.abs(gamma_pix), initial=0.0)),
        "passed": bool(
            np.max(np.abs(gamma_phys), initial=0.0) <= PHYSICAL_ZERO_ATOL
            and np.max(np.abs(gamma_pix), initial=0.0) <= PIXEL_ZERO_ATOL
        ),
    }


def serializable_rollout(rollout_value: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": rollout_value["label"],
        "mode": rollout_value["mode"],
        "initial_pixel_sha256": rollout_value["initial_pixel_sha256"],
        "query_snapshot_sha256": array_sha256(rollout_value["query_snapshot"]),
        "query_pixel_sha256": array_sha256(rollout_value["query_pixel"]),
        "future_snapshot": rollout_value["future_snapshot"].tolist(),
        "future_pixel_sha256": array_sha256(rollout_value["future_pixel"]),
        "history_contacts": rollout_value["history_contacts"],
        "history_minimum_clearance": rollout_value["history_minimum_clearance"],
        "query_start_arbiter_count": rollout_value["query_start_arbiter_count"],
        "query_contacts_by_raw_step": rollout_value["query_contacts_by_raw_step"],
        "query_arbiter_counts_by_physics_substep": rollout_value[
            "query_arbiter_counts_by_physics_substep"
        ],
        "query_contact_events": rollout_value["query_contact_events"],
        "first_contact": rollout_value["first_contact"],
        "friction_before": rollout_value["friction_before"],
        "friction_after": rollout_value["friction_after"],
    }


def probe_direction(name: str, template: motion.MotionDampingTemplate) -> dict[str, Any]:
    candidate_rows = []
    accepted: dict[float, np.ndarray] = {}
    for scale in PLANNER_SCALES:
        actions = planner_action(template, scale)
        audit = action_audit(actions)
        rotation = rotation_audit(template, actions)
        is_accepted = bool(
            audit["inside_component_box"]
            and audit["inside_l2_unit_ball"]
            and rotation["no_clipping_required"]
        )
        candidate_rows.append(
            {
                "scale": scale,
                "action": actions[0].tolist(),
                "accepted": is_accepted,
                "rejection_reason": None if is_accepted else "l2_or_rotation_box_gate",
                "action_audit": audit,
                "rotation_audit": rotation,
            }
        )
        if is_accepted:
            accepted[scale] = actions
    required_scales = (0.0, 0.25, 0.625)
    if not all(scale in accepted for scale in required_scales):
        raise RuntimeError("Frozen base templates do not retain the preregistered P1a scales")

    rollouts: dict[float, dict[str, dict[str, Any]]] = {}
    for scale in required_scales:
        rollouts[scale] = {
            mode: rollout(
                template,
                mode=mode,
                query_actions=accepted[scale],
                label=f"{name}-scale-{scale:.3f}",
            )
            for mode in motion.ENDPOINT_MODES
        }
    reference = rollouts[0.0]
    contact_free = rollouts[0.25]
    active = rollouts[0.625]
    contact_free_gamma = gamma_report(contact_free, reference)
    contact_gamma = gamma_report(active, reference)
    approach = accepted[0.625][0] / np.linalg.norm(accepted[0.625][0])
    tangent = np.asarray([-approach[1], approach[0]], dtype=np.float64)
    contact_gamma["local_axis_signed"] = {
        "approach_unit": approach.tolist(),
        "tangent_unit": tangent.tolist(),
        "block_position_approach": float(np.dot(contact_gamma["block_position"], approach)),
        "block_position_tangent": float(np.dot(contact_gamma["block_position"], tangent)),
        "block_velocity_approach": float(np.dot(contact_gamma["block_velocity"], approach)),
        "block_velocity_tangent": float(np.dot(contact_gamma["block_velocity"], tangent)),
    }

    fast = motion.ENDPOINT_MODES[0]
    repeated_active = rollout(
        template,
        mode=fast,
        query_actions=accepted[0.625],
        label=f"{name}-gamma-null-active-repeat",
    )
    repeated_ref = rollout(
        template,
        mode=fast,
        query_actions=accepted[0.0],
        label=f"{name}-gamma-null-ref-repeat",
    )
    null = same_damping_null(active[fast], repeated_active, reference[fast], repeated_ref)

    native_replay: dict[str, dict[str, Any]] = {}
    active_template = replace(
        template,
        query_actions=tuple(map(tuple, accepted[0.625].tolist())),
    )
    for mode in motion.ENDPOINT_MODES:
        native = motion.simulate_motion_damping_clip(
            active_template,
            mode=mode,
            resolution=RESOLUTION,
        )
        native_counts = [int(float(row[0])) for row in native["rows"]["n_contacts"][10:15]]
        native_replay[mode] = {
            "future_state_max_abs_gap": float(
                np.max(
                    np.abs(
                        snapshot_delta(
                            active[mode]["future_snapshot"],
                            native["natural_future_snapshot"],
                        )
                    ),
                    initial=0.0,
                )
            ),
            "future_pixels_identical": bool(
                np.array_equal(active[mode]["future_pixel"], native["model_pixels"][3])
            ),
            "raw_contact_counters_identical": native_counts
            == active[mode]["query_contacts_by_raw_step"],
            "native_raw_contact_counters": native_counts,
        }

    query_states = [value[mode]["query_snapshot"] for value in rollouts.values() for mode in motion.ENDPOINT_MODES]
    query_pixels = [value[mode]["query_pixel"] for value in rollouts.values() for mode in motion.ENDPOINT_MODES]
    base_query = query_states[0]
    query_gap = float(
        max(np.max(np.abs(snapshot_delta(state, base_query)), initial=0.0) for state in query_states)
    )
    query_pixels_identical = all(np.array_equal(pixel, query_pixels[0]) for pixel in query_pixels)
    active_first = {mode: active[mode]["first_contact"] for mode in motion.ENDPOINT_MODES}
    first_physics_raw = [active_first[mode]["raw_step"] if active_first[mode] else None for mode in motion.ENDPOINT_MODES]
    first_sub = [active_first[mode]["physics_substep"] if active_first[mode] else None for mode in motion.ENDPOINT_MODES]
    first_counter_raw = []
    for mode in motion.ENDPOINT_MODES:
        relative = next(
            (index for index, count in enumerate(active[mode]["query_contacts_by_raw_step"]) if count > 0),
            None,
        )
        first_counter_raw.append(
            None if relative is None else motion.HISTORY_RAW_STEPS + relative
        )

    all_rollouts = [value for by_mode in rollouts.values() for value in by_mode.values()] + [
        repeated_active,
        repeated_ref,
    ]
    checks = {
        "zero_prefix_actions": bool(np.all(np.asarray(template.history_actions) == 0.0)),
        "canonical_query_full_state": query_gap <= motion.QUERY_STATE_TOLERANCE,
        "canonical_query_pixels": query_pixels_identical,
        "history_contact_free": all(not any(value["history_contacts"]) for value in all_rollouts),
        "history_clearance_positive": all(value["history_minimum_clearance"] > 0.0 for value in all_rollouts),
        "query_start_arbiter_zero": all(value["query_start_arbiter_count"] == 0 for value in all_rollouts),
        "contact_free_candidate_has_no_contact": all(not any(contact_free[mode]["query_contacts_by_raw_step"]) for mode in motion.ENDPOINT_MODES),
        "contact_free_gamma_componentwise_zero": contact_free_gamma["physical_max_abs"] <= PHYSICAL_ZERO_ATOL,
        "active_contacts_both_conditions": all(any(active[mode]["query_contacts_by_raw_step"]) for mode in motion.ENDPOINT_MODES),
        "active_first_positive_contact_counter_raw_step_equal": first_counter_raw[0] is not None and first_counter_raw[0] == first_counter_raw[1],
        "active_first_physics_contact_raw_step_equal": first_physics_raw[0] is not None and first_physics_raw[0] == first_physics_raw[1],
        "active_first_contact_physics_substep_equal": first_sub[0] is not None and first_sub[0] == first_sub[1],
        "active_contact_trace_has_normal_point_impulse_penetration": all(
            active_first[mode]
            and active_first[mode]["point_count"] > 0
            and len(active_first[mode]["normal"]) == 2
            and len(active_first[mode]["total_impulse"]) == 2
            and np.isfinite(active_first[mode]["minimum_point_distance"])
            for mode in motion.ENDPOINT_MODES
        ),
        "contact_gamma_nonzero_block_response": max(
            np.max(np.abs(contact_gamma["block_position"]), initial=0.0),
            np.max(np.abs(contact_gamma["block_velocity"]), initial=0.0),
        ) > NONZERO_ATOL,
        "contact_gamma_pixel_nonzero": contact_gamma["gamma_pixel_energy"] > 0.0,
        "motion_friction_identity_all_rollouts": all(
            value["friction_before"]["passed"] and value["friction_after"]["passed"]
            for value in all_rollouts
        ),
        "same_damping_gamma_null": null["passed"],
        "instrumented_substep_replay_matches_native_env_step": all(
            value["future_state_max_abs_gap"] == 0.0
            and value["future_pixels_identical"]
            and value["raw_contact_counters_identical"]
            for value in native_replay.values()
        ),
        "planner_candidates_have_accept_reject_accounting": sum(row["accepted"] for row in candidate_rows) + sum(not row["accepted"] for row in candidate_rows) == len(PLANNER_SCALES),
    }
    return {
        "direction": name,
        "template_id": template.template_id,
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_accounting": {
            "total": len(candidate_rows),
            "accepted": sum(row["accepted"] for row in candidate_rows),
            "rejected": sum(not row["accepted"] for row in candidate_rows),
            "rows": candidate_rows,
        },
        "query_state_max_abs_gap": query_gap,
        "contact_free_gamma": contact_free_gamma,
        "contact_gamma": contact_gamma,
        "same_damping_gamma_null": null,
        "native_step_replay_equivalence": native_replay,
        "first_positive_contact_counter_raw_steps_global": first_counter_raw,
        "first_positive_contact_counter_raw_steps_query_relative": [
            None if value is None else value - motion.HISTORY_RAW_STEPS
            for value in first_counter_raw
        ],
        "first_physics_contact_raw_steps_global": first_physics_raw,
        "first_physics_contact_substeps_within_raw_step": first_sub,
        "rollouts": {
            f"scale_{scale:.3f}": {
                mode: serializable_rollout(by_mode[mode]) for mode in motion.ENDPOINT_MODES
            }
            for scale, by_mode in rollouts.items()
        },
        "gamma_null_repeats": {
            "active": serializable_rollout(repeated_active),
            "a_ref": serializable_rollout(repeated_ref),
        },
    }


def build_report() -> dict[str, Any]:
    inputs = verify_frozen_inputs()
    directions = [
        probe_direction("forward", motion.make_base_template()),
        probe_direction("reverse", motion.make_mirrored_base_template()),
    ]
    passed = all(direction["passed"] for direction in directions)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_p1a_physical_route" if passed else "blocked_p1a_physical_route",
        "passed": passed,
        "scope": "training_only_cpu_p1a_no_model_no_optimizer",
        "resolution": RESOLUTION,
        "optimizer_steps": 0,
        "gpu_used": False,
        "development_opened": False,
        "public_test_opened": False,
        "legacy_query_contact_free_validator_called": False,
        "physics_substep_instrumentation": "research_local_exact_replay_of_PushT_step_10_substeps",
        "pixel_protocol": "env.render uint8 -> float64 / 255 before subtraction; no resize/crop",
        "inputs": inputs,
        "directions": directions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    payload = json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
