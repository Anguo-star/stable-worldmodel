#!/usr/bin/env python3
"""Run ContextWorld's frozen PushT CEM wrapper on the current SWM layout."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys
import types

# ``stable_pretraining`` imports optional Kornia feature modules while Hydra
# resolves the ViT factory.  The host flash-attn extension targets a different
# PyTorch ABI and is not used by LeWM's ViT-tiny encoder; fail it closed before
# that optional import path is traversed, matching the current-runtime direct
# response evaluators in this research directory.
sys.modules.setdefault("flash_attn", None)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_worldmodel.planning import (
    GoalMSE,
    ShootingCostEvaluator,
    solver as planning_solver,
)


# ContextWorld's frozen evaluator predates both the package move and the
# constructor rename from ``model=`` to ``cost=``.  Adapt only that API seam;
# the solver class and every numerical hyperparameter remain current SWM.
class _LegacyCEMSolver(planning_solver.CEMSolver):
    def __init__(self, *, model=None, cost=None, **kwargs):
        if (model is None) == (cost is None):
            raise TypeError("exactly one of model or cost is required")
        if cost is None:
            cost = ShootingCostEvaluator(model, GoalMSE())
        super().__init__(cost=cost, **kwargs)


legacy_solver = types.ModuleType("stable_worldmodel.solver")
for name in planning_solver.__all__:
    setattr(legacy_solver, name, getattr(planning_solver, name))
legacy_solver.CEMSolver = _LegacyCEMSolver
sys.modules["stable_worldmodel.solver"] = legacy_solver

CONTEXTWORLD_EVALUATOR = (
    REPO_ROOT.parent
    / "ContextWorld/scripts/eval_pusht_standard_cem_retention.py"
)
runpy.run_path(str(CONTEXTWORLD_EVALUATOR), run_name="__main__")
