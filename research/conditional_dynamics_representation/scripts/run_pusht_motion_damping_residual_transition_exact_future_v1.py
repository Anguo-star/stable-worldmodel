#!/usr/bin/env python3
"""Run residual-transition LeWM with one normalized joint pair objective.

The auxiliary is the existing pair-normalized exact terminal-future residual.
It is exactly CCRM response error plus normalized common-center error, so it
tests the one missing component exposed by the CCRM arm without adding a new
module, parameter, inference branch, or component-balance hyperparameter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    pair_normalized_exact_future_residual_v1 as exact_future,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_causal_transition_basis_v1 as causal,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_native_twin_sampler_v1 as native_twin,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_residual_output_basis_v1 as residual,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_residual_transition_ccrm_v1 as paired,
)


CANDIDATE = "pusht_motion_damping_residual_transition_exact_future_v1"
OBJECTIVE = "pair_normalized_exact_future"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_training_report(
    output: Path,
    *,
    release_state: dict[str, Any],
    residual_state: dict[str, Any],
    auxiliary_state: dict[str, Any],
) -> Path:
    report = residual._rewrite_training_report(
        output,
        input_basis="causal_transition",
        release_state=release_state,
        method_state=residual_state,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    checks = {
        "one_auxiliary_call_per_optimizer_step": (
            int(auxiliary_state["loss_calls"])
            == int(result["optimizer_steps"])
        ),
        "residual_transition_contract_passed": all(
            result["residual_output_basis_contract"]["checks"].values()
        ),
        "objective_exact": auxiliary_state["objective_name"] == OBJECTIVE,
        "parameter_count_unchanged": (
            auxiliary_state["model"]["parameter_count"]
            == residual_state["model"]["parameter_count_after"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Residual exact-future contract failed: {checks}")
    result["residual_transition_exact_future_contract"] = {
        "checks": checks,
        "auxiliary": "pair_normalized_exact_future_residual_v1",
        "auxiliary_weight": paired.CCRM_WEIGHT,
        "analytic_identity": (
            "normalized_exact_future = CCRM_response + "
            "normalized_common_center_MSE"
        ),
        "first_loss_components": auxiliary_state["first_components"],
        "last_loss_components": auxiliary_state["last_components"],
        "predictor_inputs_detached": True,
        "targets_detached_inside_objective": True,
        "learned_parameters_added": 0,
        "model_modules_added": 0,
        "component_balance_hyperparameters_added": 0,
    }
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = residual._discovery_args(effective)
    if args.input_basis != "causal_transition" or args.optimizer_steps != 2048:
        raise RuntimeError("Residual-transition exact-future is fixed to 2048 steps")
    trainer = causal._load_trainer()
    release_state = native_twin._install_runtime(trainer)
    residual_state = residual._install_residual_output(
        trainer,
        input_basis="causal_transition",
    )
    auxiliary_state = paired._install_paired_auxiliary(
        trainer,
        objective=exact_future.pair_normalized_exact_future_residual,
        objective_name=OBJECTIVE,
        response_metric_key="normalized_exact_future_residual_by_group",
    )
    previous = list(sys.argv)
    try:
        sys.argv = [str(THIS_SOURCE), *residual._trainer_argv(effective)]
        trainer.main()
    finally:
        sys.argv = previous

    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_training_report(
            output,
            release_state=release_state,
            residual_state=residual_state,
            auxiliary_state=auxiliary_state,
        )
        sidecar = output / "residual_transition_exact_future_method_v1.json"
        if sidecar.exists():
            raise FileExistsError(sidecar)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": CANDIDATE,
                    "source": str(THIS_SOURCE),
                    "source_sha256": _sha256(THIS_SOURCE),
                    "training_report": str(report),
                    "training_report_sha256": _sha256(report),
                    "objective": (
                        "native_mse + 0.09*SIGReg + "
                        "0.09*pair_normalized_exact_future"
                    ),
                    "temporal_input_basis": "causal_transition",
                    "temporal_output_basis": "residual",
                    "learned_parameters_added": 0,
                    "model_modules_added": 0,
                    "auxiliary_terms_added": 1,
                    "component_balance_hyperparameters_added": 0,
                    "claim_boundary": {
                        "development_only": True,
                        "single_seed_discovery": True,
                        "public_test_opened": False,
                        "cem_opened": False,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
