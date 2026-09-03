from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RESEARCH_ROOT / "scripts/probe_pusht_motion_damping_d2_p1a_v1.py"
SPEC = importlib.util.spec_from_file_location("motion_d2_p1a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


@pytest.fixture(scope="module")
def report() -> dict:
    return module.build_report()


def test_frozen_planner_support_and_action_gate() -> None:
    inputs = module.verify_frozen_inputs()
    assert inputs["planner_support_receipt_sha256"] == module.EXPECTED_RECEIPT_SHA256
    template = module.motion.make_base_template()
    accepted = []
    for scale in module.PLANNER_SCALES:
        actions = module.planner_action(template, scale)
        action = module.action_audit(actions)
        rotation = module.rotation_audit(template, actions)
        accepted.append(action["inside_l2_unit_ball"] and rotation["no_clipping_required"])
    assert accepted == [True, True, True, False]


def test_p1a_route_passes_with_exact_nulls_and_native_replay(report: dict) -> None:
    assert report["status"] == "passed_p1a_physical_route"
    assert report["passed"] is True
    assert report["optimizer_steps"] == 0
    assert report["gpu_used"] is False
    assert report["development_opened"] is False
    assert report["public_test_opened"] is False
    for direction in report["directions"]:
        assert direction["passed"] is True
        assert direction["contact_free_gamma"]["physical_max_abs"] == 0.0
        assert direction["contact_free_gamma"]["gamma_pixel_energy"] == 0.0
        assert direction["contact_gamma"]["gamma_pixel_energy"] > 0.0
        assert max(np.abs(direction["contact_gamma"]["block_velocity"])) > 1.0
        assert direction["same_damping_gamma_null"]["physical_max_abs"] == 0.0
        assert direction["same_damping_gamma_null"]["pixel_max_abs"] == 0.0
        assert direction["first_positive_contact_counter_raw_steps_query_relative"] == [2, 2]
        assert direction["first_physics_contact_raw_steps_global"] == [12, 12]
        assert direction["first_physics_contact_substeps_within_raw_step"] == [8, 8]
        assert all(
            value["future_state_max_abs_gap"] == 0.0
            and value["future_pixels_identical"]
            and value["raw_contact_counters_identical"]
            for value in direction["native_step_replay_equivalence"].values()
        )


def test_output_refuses_overwrite_without_rerunning_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report: dict,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(module, "build_report", lambda: report)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.main()
    assert output.read_text(encoding="utf-8") == "keep"


def test_probe_source_has_no_training_or_split_access() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert "validate_motion_damping_pair(" not in source
    assert "evaluate_template(" not in source
    assert '"optimizer_steps": 0' in source
    assert '"development_opened": False' in source
    assert '"public_test_opened": False' in source
