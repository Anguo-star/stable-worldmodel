#!/usr/bin/env python3
"""Legacy-scale coverage qualification for the Cartesian action-pair MVE.

Only the frozen training overlay expands from 256 to 2,048 templates, matching
the complete legacy Motion training scale.  Model, objective, source,
optimizer, budget, action construction and evaluation gates are unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_cartesian_action_pair_v1 as base,
)


CANDIDATE = "pusht_motion_damping_cartesian_action_pair_legacy_scale_v2"
OVERLAY_SHA256 = "71e50c7922d48f245421ade13539ced27ff672f2bf9d9e23ba062cfb7d3c34d2"
OVERLAY_TEMPLATE_COUNT = 2048
OVERLAY_CONDITION_PAIR_COUNT = 4096


def main() -> int:
    base.THIS_SOURCE = THIS_SOURCE
    base.CANDIDATE = CANDIDATE
    base.OVERLAY_SHA256 = OVERLAY_SHA256
    base.OVERLAY_TEMPLATE_COUNT = OVERLAY_TEMPLATE_COUNT
    base.OVERLAY_CONDITION_PAIR_COUNT = OVERLAY_CONDITION_PAIR_COUNT
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
