from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = RESEARCH_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "run_pusht_motion_damping_d2_p1b_calibration_v1",
    SCRIPTS / "run_pusht_motion_damping_d2_p1b_calibration_v1.py",
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_check_only_binds_clarified_design_without_opening_outcome():
    report = runner.check_only(runner.DEFAULT_DESIGN_DIR, runner.DEFAULT_PIXEL_BASELINE)
    assert report["status"] == "check_ok_no_outcome_opened"
    assert report["calibration_outcome_opened"] is False
    assert report["sealed_holdout_opened"] is False
    assert report["verified_inputs"]["candidate_count"] == 8192
    assert (
        report["verified_inputs"]["addendum_sha256"]
        == "2415d5765b3d26bf59fc33290a1c9e4b59b450af6887b27813526b77f5ed9ffa"
    )


def test_candidate_row_guard_refuses_sealed_identity():
    candidate = next(runner.iter_calibration_candidates(runner.DEFAULT_DESIGN_DIR))
    candidate["split"] = "sealed_holdout"
    with pytest.raises(RuntimeError, match="refuses non-calibration"):
        runner._verify_candidate_row(candidate, 0)


def test_first_positive_raw_step():
    assert runner._first_positive_raw_step([0, 0, 2, 0], 10) == 12
    assert runner._first_positive_raw_step([0, 0, 0], 10) is None


def test_deterministic_npz_is_byte_identical(tmp_path: Path):
    arrays = {
        "z": np.arange(12, dtype=np.int16).reshape(3, 4),
        "a": np.asarray([[0.25, -0.5]], dtype=np.float64),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_hash = runner.write_npz_deterministic(first, arrays)
    second_hash = runner.write_npz_deterministic(second, arrays)
    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as loaded:
        assert np.array_equal(loaded["z"], arrays["z"])
        assert np.array_equal(loaded["a"], arrays["a"])


@dataclass(frozen=True)
class DummyTemplate:
    template_id: str
    faster_decay_reset_snapshot: tuple[float, ...]
    no_extra_decay_reset_snapshot: tuple[float, ...]
    expected_natural_query_snapshot: tuple[float, ...]
    goal_state: tuple[float, ...]


def test_geometry_audit_shifts_only_agent_and_goal_agent_position():
    snapshot = tuple(float(x) for x in range(12))
    template = DummyTemplate(
        template_id="candidate",
        faster_decay_reset_snapshot=snapshot,
        no_extra_decay_reset_snapshot=snapshot,
        expected_natural_query_snapshot=snapshot,
        goal_state=tuple(float(x) for x in range(7)),
    )
    shifted = runner.shift_agent_geometry(template, 1.0, [0.0, 1.0])
    for field in (
        "faster_decay_reset_snapshot",
        "no_extra_decay_reset_snapshot",
        "expected_natural_query_snapshot",
    ):
        before = np.asarray(getattr(template, field))
        after = np.asarray(getattr(shifted, field))
        assert np.array_equal(after[2:], before[2:])
        assert np.array_equal(after[:2], before[:2] + np.asarray([0.0, 1.0]))
    before_goal = np.asarray(template.goal_state)
    after_goal = np.asarray(shifted.goal_state)
    assert np.array_equal(after_goal[2:], before_goal[2:])
    assert np.array_equal(after_goal[:2], before_goal[:2] + np.asarray([0.0, 1.0]))


def test_jsonl_writer_refuses_overwrite_and_hashes_bytes(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    writer = runner.JsonlWriter(path)
    writer.write({"b": 2, "a": 1})
    digest = writer.close()
    assert path.read_text() == '{"a":1,"b":2}\n'
    assert digest == runner.sha256_file(path)
    with pytest.raises(FileExistsError):
        runner.JsonlWriter(path)


def test_slot_scan_stops_at_earliest_structural_pass(monkeypatch, tmp_path: Path):
    calls: list[int] = []

    def candidate(index: int) -> dict:
        return {
            "group_index": index,
            "candidate_sha256": f"sha-{index}",
            "coverage_cell": 7,
            "orientation_bin": 0,
            "speed_bin": 1,
            "goal_distance_bin": 3,
            "action_stratum": "mid_approach",
            "within_cell_candidate_encounter_rank": index,
            "pre_outcome_passed": True,
            "templates": {
                "forward": {"template_id": f"f-{index}"},
                "reverse": {"template_id": f"r-{index}"},
            },
        }

    def fake_evaluate(_runtime, row):
        index = int(row["group_index"])
        calls.append(index)
        passed = index == 2
        summary = {
            **runner._candidate_identity(row),
            "structurally_admitted": passed,
            "structural_checks": {"both_conditions_contact": passed},
            "post_decision_gamma_not_used_for_selection": {},
        }
        return summary, {}, {}, {}, {}

    monkeypatch.setattr(runner, "load_runtime", lambda: object())
    monkeypatch.setattr(runner, "_evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(runner, "post_admission_audit", lambda *args: {"ok": True})
    monkeypatch.setattr(
        runner,
        "_accepted_sidecar_arrays",
        lambda *args: {"x": np.asarray([1], dtype=np.int8)},
    )
    result = runner._run_frozen_slot(
        (7, "mid_approach"),
        [candidate(index) for index in range(4)],
        str(tmp_path),
    )
    assert calls == [0, 1, 2]
    assert result["accepted"]["group_index"] == 2
    assert [row["group_index"] for row in result["rejected"]] == [0, 1]
    assert (tmp_path / "group_000002.npz").is_file()
