#!/usr/bin/env python3
"""Commit the deterministic, rollout-free Motion D2 P1b candidate design."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pusht_motion_damping_d2_p1b_common_v1 import (
    ADDENDUM_PATH,
    CANDIDATE_WINDOWS,
    COVERAGE_CELLS,
    DESIGN_ID,
    DIRECTIONS,
    GROUPS_PER_SPLIT,
    PILOT_CATALOG_SEED,
    STRATUM_IDS,
    action_norm_and_box,
    assign_d0_cell,
    assign_quota,
    build_query_actions,
    candidate_window,
    canonical_sha256,
    cut_points_reproduce_d0,
    ensure_contextworld_on_path,
    infer_d0_cut_points,
    load_d0_coverage_rows,
    orientation_bin_from_block_angle,
    quota_plan,
    reject_forbidden_path,
    rename_to_seed_bearing_identity,
    require,
    seed_bearing_template_id,
    sha256_file,
    stratum_for_encounter,
    verify_declared_identities,
    windows_are_disjoint,
    write_json_exclusive,
)


THIS_SOURCE = Path(__file__).resolve()


def _wrapped_abs(left: float, right: float) -> float:
    return abs((float(left) - float(right) + math.pi) % (2.0 * math.pi) - math.pi)


def candidate_geometry(forward: Any, reverse: Any, cut_points: dict[str, Any]) -> dict[str, Any]:
    """Return outcome-free twin descriptors used for frozen cell assignment."""

    q_forward = np.asarray(forward.expected_natural_query_snapshot, dtype=np.float64)
    q_reverse = np.asarray(reverse.expected_natural_query_snapshot, dtype=np.float64)
    goal_forward = np.asarray(forward.goal_state, dtype=np.float64)
    goal_reverse = np.asarray(reverse.goal_state, dtype=np.float64)
    require(q_forward.shape == q_reverse.shape == (12,), "query snapshot is not 12-D")
    require(goal_forward.shape == goal_reverse.shape == (7,), "goal state is not 7-D")
    require(forward.simulator_seed == reverse.simulator_seed, "twin simulator seeds differ")
    require(np.array_equal(goal_forward, goal_reverse), "twin goal states differ")
    require(_wrapped_abs(q_forward[10], q_reverse[10]) <= 1.0e-12, "twin block angles differ")
    speed = 0.5 * (
        float(np.linalg.norm(q_forward[8:10]))
        + float(np.linalg.norm(q_reverse[8:10]))
    )
    goal_distance = 0.5 * (
        float(np.linalg.norm(goal_forward[2:4] - q_forward[6:8]))
        + float(np.linalg.norm(goal_reverse[2:4] - q_reverse[6:8]))
    )
    orientation = orientation_bin_from_block_angle(float(q_forward[10]))
    assigned = assign_d0_cell(
        orientation_bin=orientation,
        query_speed=speed,
        goal_distance=goal_distance,
        cut_points=cut_points,
    )
    return {
        **assigned,
        "query_speed_forward_reverse_mean": speed,
        "goal_distance_forward_reverse_mean": goal_distance,
        "simulator_seed": int(forward.simulator_seed),
    }


def _template_with_action(template: Any, template_id: str, actions: np.ndarray) -> Any:
    renamed = rename_to_seed_bearing_identity(template, template_id)
    return replace(renamed, query_actions=tuple(map(tuple, actions.tolist())))


def build_candidate_stream(
    *, split: str, cut_points: dict[str, Any], motion_module: Any
) -> list[dict[str, Any]]:
    """Build the complete frozen candidate window without simulator rollout."""

    low, high = candidate_window(split)
    encounter_by_cell = np.zeros(COVERAGE_CELLS, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for group_index in range(low, high + 1):
        source = {
            "forward": motion_module.make_catalog_template(
                split="train", catalog_index=2 * group_index,
                catalog_seed=PILOT_CATALOG_SEED,
            ),
            "reverse": motion_module.make_catalog_template(
                split="train", catalog_index=2 * group_index + 1,
                catalog_seed=PILOT_CATALOG_SEED,
            ),
        }
        geometry = candidate_geometry(source["forward"], source["reverse"], cut_points)
        cell = int(geometry["coverage_cell"])
        encounter_rank = int(encounter_by_cell[cell])
        encounter_by_cell[cell] += 1
        stratum = stratum_for_encounter(encounter_rank)
        templates: dict[str, dict[str, Any]] = {}
        actions_payload: dict[str, list[list[float]]] = {}
        audits: dict[str, dict[str, Any]] = {}
        for direction in DIRECTIONS:
            template_id = seed_bearing_template_id(
                split=split, group_index=group_index, direction=direction
            )
            actions = build_query_actions(
                source[direction].expected_natural_query_snapshot,
                stratum_id=stratum,
                coverage_cell=cell,
                split=split,
            )
            audit = action_norm_and_box(actions)
            frozen_template = _template_with_action(
                source[direction], template_id, actions
            )
            templates[direction] = asdict(frozen_template)
            actions_payload[direction] = actions.tolist()
            audits[direction] = audit
        passed = all(value["passed"] for value in audits.values())
        row = {
            "split": split,
            "group_index": group_index,
            "source_catalog_indices": {
                "forward": 2 * group_index,
                "reverse": 2 * group_index + 1,
            },
            "within_cell_candidate_encounter_rank": encounter_rank,
            "action_stratum": stratum,
            **geometry,
            "query_actions": actions_payload,
            "action_audit": audits,
            "pre_outcome_passed": bool(passed),
            "rejection_reason": None if passed else "action_norm_and_box",
            "templates": templates,
        }
        row["candidate_sha256"] = canonical_sha256(row)
        rows.append(row)
    require(len(rows) == high - low + 1, "candidate window length changed")
    return rows


def _write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path = reject_forbidden_path(path, name="candidate output").resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def build_design(output_dir: Path) -> dict[str, Any]:
    output_dir = reject_forbidden_path(output_dir, name="output directory").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    identities = verify_declared_identities()
    require(windows_are_disjoint(), "pilot candidate windows overlap")
    d0_rows = load_d0_coverage_rows()
    cuts = infer_d0_cut_points(d0_rows)
    require(cut_points_reproduce_d0(d0_rows, cuts), "fixed cuts do not reproduce D0")
    cuts_path = output_dir / "coverage_cut_points.json"
    cuts_sha = write_json_exclusive(cuts_path, cuts)

    ensure_contextworld_on_path()
    from contextworld.evaluation import pusht_motion_damping_h3 as motion

    split_receipts: dict[str, Any] = {}
    planned_ids: set[str] = set()
    for split in CANDIDATE_WINDOWS:
        candidates = build_candidate_stream(
            split=split, cut_points=cuts, motion_module=motion
        )
        plan = assign_quota(candidates, split=split)
        require(plan["quota_complete"], f"pre-outcome quota plan incomplete: {split}")
        require(plan["accepted_count"] == GROUPS_PER_SPLIT, "planned group count changed")
        require(not plan["structural_admission_supplied"], "design claimed rollout admission")
        for row in plan["accepted"]:
            for direction in DIRECTIONS:
                identity = row["templates"][direction]["template_id"]
                require(identity not in planned_ids, f"duplicate template id {identity}")
                planned_ids.add(identity)
        candidates_path = output_dir / f"{split}_candidate_stream.jsonl"
        candidates_sha = _write_jsonl_exclusive(candidates_path, candidates)
        plan_path = output_dir / f"{split}_pre_outcome_quota_plan.json"
        plan_sha = write_json_exclusive(plan_path, plan)
        split_receipts[split] = {
            "candidate_window_inclusive": list(candidate_window(split)),
            "candidate_count": len(candidates),
            "candidate_stream_path": candidates_path.name,
            "candidate_stream_sha256": candidates_sha,
            "planned_group_count": int(plan["accepted_count"]),
            "pre_outcome_quota_plan_path": plan_path.name,
            "pre_outcome_quota_plan_sha256": plan_sha,
            "rollouts_executed": 0,
            "structural_admission_claimed": False,
        }

    receipt = {
        "schema_version": 1,
        "design_id": DESIGN_ID,
        "status": "pre_outcome_candidate_streams_committed",
        "scope": "training_only_rollout_free_design",
        "addendum_path": str(ADDENDUM_PATH.relative_to(THIS_SOURCE.parents[3])),
        "addendum_sha256": sha256_file(ADDENDUM_PATH),
        "builder_sha256": sha256_file(THIS_SOURCE),
        "identity_verification": identities,
        "catalog_seed": PILOT_CATALOG_SEED,
        "source_catalog_distribution_tag": "train",
        "source_contextworld_modified": False,
        "coverage_cut_points_path": cuts_path.name,
        "coverage_cut_points_file_sha256": cuts_sha,
        "coverage_cut_points_payload_sha256": cuts["sha256"],
        "quota_plan": quota_plan(),
        "splits": split_receipts,
        "cross_split_template_ids_disjoint": True,
        "model_loaded": False,
        "optimizer_steps": 0,
        "gpu_used": False,
        "pixels_rendered": 0,
        "development_opened": False,
        "public_test_opened": False,
        "claim_boundary": (
            "Commits candidate order, cells, actions and quota algorithm only; "
            "it does not claim contact, structural admission, Gamma, rho or P1b success."
        ),
    }
    write_json_exclusive(output_dir / "design_receipt.json", receipt)
    return receipt


def check_only() -> dict[str, Any]:
    identities = verify_declared_identities()
    d0_rows = load_d0_coverage_rows()
    cuts = infer_d0_cut_points(d0_rows)
    require(cut_points_reproduce_d0(d0_rows, cuts), "fixed cuts do not reproduce D0")
    require(windows_are_disjoint(), "pilot candidate windows overlap")
    return {
        "status": "check_ok",
        "identity_verification": identities,
        "d0_rows": len(d0_rows),
        "coverage_cut_points_sha256": cuts["sha256"],
        "candidate_windows": {key: list(value) for key, value in CANDIDATE_WINDOWS.items()},
        "rollouts_executed": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        print(json.dumps(check_only(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-only is used")
    result = build_design(args.output_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
