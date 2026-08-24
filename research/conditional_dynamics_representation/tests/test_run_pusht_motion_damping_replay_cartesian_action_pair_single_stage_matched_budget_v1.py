from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_replay_cartesian_action_pair_single_stage_matched_budget_v1 as method,
)


def test_wrapper_changes_only_candidate_source_and_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}
    original_candidate = method.base.CANDIDATE
    original_steps = method.base.OPTIMIZER_STEPS
    original_source = method.base.THIS_SOURCE

    def fake_main(argv):
        observed.update(
            {
                "argv": argv,
                "candidate": method.base.CANDIDATE,
                "steps": method.base.OPTIMIZER_STEPS,
                "source": method.base.THIS_SOURCE,
            }
        )
        return 0

    monkeypatch.setattr(method.base, "main", fake_main)
    assert method.main(["--dry-run"]) == 0
    assert observed == {
        "argv": ["--dry-run"],
        "candidate": method.CANDIDATE,
        "steps": 1024,
        "source": method.THIS_SOURCE,
    }
    assert method.base.CANDIDATE == original_candidate
    assert method.base.OPTIMIZER_STEPS == original_steps
    assert method.base.THIS_SOURCE == original_source
