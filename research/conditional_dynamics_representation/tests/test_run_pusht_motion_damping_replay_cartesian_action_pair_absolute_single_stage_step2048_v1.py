from __future__ import annotations

from pathlib import Path

import pytest

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_step2048_v1 as method,
)


def test_wrapper_changes_only_candidate_source_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}
    original_candidate = method.base.CANDIDATE
    original_steps = method.base.OPTIMIZER_STEPS
    original_source = method.base.THIS_SOURCE

    def fake_main(argv):
        observed.update(
            candidate=method.base.CANDIDATE,
            steps=method.base.OPTIMIZER_STEPS,
            source=method.base.THIS_SOURCE,
            argv=argv,
        )
        return 0

    monkeypatch.setattr(method.base, "main", fake_main)
    assert method.main(["--dry-run"]) == 0
    assert observed == {
        "candidate": method.CANDIDATE,
        "steps": 2048,
        "source": method.THIS_SOURCE,
        "argv": ["--dry-run"],
    }
    assert method.base.CANDIDATE == original_candidate
    assert method.base.OPTIMIZER_STEPS == original_steps
    assert method.base.THIS_SOURCE == original_source
