from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = RESEARCH_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load("motion_d2_p1b_common", "pusht_motion_damping_d2_p1b_common_v1.py")
builder = _load("motion_d2_p1b_design", "build_pusht_motion_damping_d2_p1b_design_v1.py")


def synthetic_coverage_rows() -> list[dict]:
    rows = []
    twin = 0
    for orientation in range(4):
        for speed_bin in range(4):
            for goal_bin in range(4):
                rows.append(
                    {
                        "twin_id": twin,
                        "coverage_cell": orientation * 16 + speed_bin * 4 + goal_bin,
                        "orientation_bin": orientation,
                        "speed_bin": speed_bin,
                        "goal_distance_bin": goal_bin,
                        "query_speed": 10.0 * orientation + 2.0 * speed_bin + 0.1 * goal_bin,
                        "goal_distance": 100.0 * orientation + 20.0 * speed_bin + 3.0 * goal_bin,
                    }
                )
                twin += 1
    return rows


def test_fixed_cut_points_reproduce_all_64_cells() -> None:
    rows = synthetic_coverage_rows()
    cuts = common.infer_d0_cut_points(rows)
    assert common.cut_points_reproduce_d0(rows, cuts)
    assert cuts["cells"] == 64
    assert len(cuts["speed_cut_points_by_orientation"]) == 4
    assert len(cuts["goal_distance_cut_points_by_orientation_and_speed_bin"]) == 16


def test_action_rotation_and_frozen_strata_preserve_norm() -> None:
    query = np.array([0, 0, 0, 0, 0, 0, 40, 30, 10, -5, 0, 0], dtype=float)
    base = common.planner_base_vector(query)
    for split in common.CANDIDATE_WINDOWS:
        for cell in range(64):
            for stratum in common.STRATUM_IDS:
                actions = common.build_query_actions(
                    query, stratum_id=stratum, coverage_cell=cell, split=split
                )
                audit = common.action_norm_and_box(actions)
                assert audit["passed"]
                expected = common.PLANNER_SCALE_BY_STRATUM[stratum] * np.linalg.norm(base)
                np.testing.assert_allclose(np.linalg.norm(actions, axis=1), expected, atol=1e-12)
    assert common.tangent_angle_degrees(
        stratum_id="mid_tangent_assisted", coverage_cell=0, split="calibration"
    ) == -common.tangent_angle_degrees(
        stratum_id="mid_tangent_assisted", coverage_cell=0, split="sealed_holdout"
    )


def test_stratum_is_assigned_before_structural_admission_and_quota_is_exact() -> None:
    candidates = []
    admitted = []
    group_index = 0
    for cell in range(64):
        for encounter in range(6):
            candidates.append(
                {
                    "group_index": group_index,
                    "coverage_cell": cell,
                    "action_stratum": common.stratum_for_encounter(encounter),
                    "pre_outcome_passed": True,
                }
            )
            admitted.append(encounter >= 3)
            group_index += 1
    result = common.assign_quota(
        candidates, split="calibration", structurally_admitted=admitted
    )
    assert result["quota_complete"]
    assert result["accepted_count"] == 192
    assert all(value == 3 for value in result["per_cell_counts"].values())
    assert all(
        set(row) == set(common.STRATUM_IDS)
        and all(value == 1 for value in row.values())
        for row in result["per_cell_action_counts"].values()
    )
    assert all(item["within_cell_candidate_encounter_rank"] >= 3
               for item in result["accepted"] if "within_cell_candidate_encounter_rank" in item)


def test_identity_windows_and_seed_bearing_ids_are_disjoint() -> None:
    assert common.windows_are_disjoint()
    calibration = common.seed_bearing_template_id(
        split="calibration", group_index=0, direction="forward"
    )
    holdout = common.seed_bearing_template_id(
        split="sealed_holdout", group_index=8192, direction="forward"
    )
    assert calibration != holdout
    assert str(common.PILOT_CATALOG_SEED) in calibration


def test_candidate_geometry_uses_forward_reverse_mean() -> None:
    rows = synthetic_coverage_rows()
    cuts = common.infer_d0_cut_points(rows)
    qf = np.array([0, 0, 0, 0, 0, 0, 10, 10, 2.0, 0, 0, 0], dtype=float)
    qr = qf.copy(); qr[6] = 12; qr[8] = -2.0
    goal = (0, 0, 13, 10, 0, 0, 0)
    forward = SimpleNamespace(expected_natural_query_snapshot=tuple(qf), goal_state=goal,
                              simulator_seed=7)
    reverse = SimpleNamespace(expected_natural_query_snapshot=tuple(qr), goal_state=goal,
                              simulator_seed=7)
    value = builder.candidate_geometry(forward, reverse, cuts)
    assert value["query_speed_forward_reverse_mean"] == 2.0
    assert value["goal_distance_forward_reverse_mean"] == 2.0


def test_forbidden_split_and_exclusive_output_guards(tmp_path: Path) -> None:
    with pytest.raises(common.ForbiddenSplitError):
        common.reject_forbidden_path(tmp_path / "development" / "x.json")
    output = tmp_path / "receipt.json"
    common.write_json_exclusive(output, {"ok": True})
    with pytest.raises(FileExistsError):
        common.write_json_exclusive(output, {"ok": False})


def test_source_does_not_import_model_or_optimizer_paths() -> None:
    source = (SCRIPTS / "build_pusht_motion_damping_d2_p1b_design_v1.py").read_text()
    assert "optimizer.step" not in source
    assert "torch" not in source
    assert '"rollouts_executed": 0' in source
