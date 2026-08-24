#!/usr/bin/env python3
"""One justified 2,048-step extension of the absolute single-stage MVE."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_v1 as base,
)


CANDIDATE = (
    "pusht_motion_damping_replay_cartesian_action_pair_"
    "absolute_single_stage_step2048_v1"
)
OPTIMIZER_STEPS = 2048


def main(argv: Sequence[str] | None = None) -> int:
    original = {
        "candidate": base.CANDIDATE,
        "optimizer_steps": base.OPTIMIZER_STEPS,
        "source": base.THIS_SOURCE,
    }
    base.CANDIDATE = CANDIDATE
    base.OPTIMIZER_STEPS = OPTIMIZER_STEPS
    base.THIS_SOURCE = THIS_SOURCE
    try:
        return base.main(argv)
    finally:
        base.CANDIDATE = original["candidate"]
        base.OPTIMIZER_STEPS = original["optimizer_steps"]
        base.THIS_SOURCE = original["source"]


if __name__ == "__main__":
    raise SystemExit(main())
