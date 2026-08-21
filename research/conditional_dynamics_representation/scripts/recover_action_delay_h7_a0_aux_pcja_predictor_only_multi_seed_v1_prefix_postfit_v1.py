#!/usr/bin/env python3
"""Seed overlay for the frozen predictor-only exact-256 prefix post-fit recovery.

The frozen recipe does not exit cleanly out of the ContextWorld trainer.  With
`expected_optimizer_steps=1024` and an intentional stop at optimizer step 256,
the trainer's LossTrace callback targets 1024 and therefore never writes a row
at step 256, while the trainer's post-training validator compares the last trace
row against `trainer.global_step`.  240 != 256, so the trainer raises

    Loss trace is incomplete or has duplicate/out-of-order steps: [1, 20, ... 240]

*after* every training artifact is already on disk.  This is a pre-existing
trainer/runner contract mismatch, already classified on the record as
`infrastructure_NOGO_not_method_failure`, and the frozen seed-3072 prefix run
hit it too: its `contextworld_report.json` carries the status
`postfit_recovered_after_exact_prefix_native_loss_trace_terminal_omission`.

The sanctioned remedy is the frozen recovery runner, which audits the on-disk
evidence and writes the phase report without imputing the missing terminal loss.
This file is the seed overlay over that runner, and nothing more.

What it does NOT do:
  * it does not edit, relax or reimplement any frozen check;
  * it does not touch the frozen seed-3072 artifacts;
  * it does not merge, mutate or fabricate a loss/gradient trace row;
  * it does not resume, re-execute or extend training (0 optimizer steps);
  * it does not instantiate a model for inference, and computes no score.

Only two surfaces in the frozen recovery stack are pinned to seed 3072:
  (a) `SEED` in the predictor-only recovery runner, which the parent uses to
      assert `config.seed == SEED`;
  (b) `_paths()` in the parent recovery, which hard-codes the literal `s3072`
      twice - once in the phase directory and once in the run-name suffix.
`CANDIDATE_ID` is deliberately NOT rebound: the candidate, the objective and the
whole recipe are unchanged, only the training seed differs.  The seed-4096 and
seed-5120 runs record `training_objective = action_delay_h7_a0_aux_pcja_
predictor_only_v1` in their own artifacts, so the frozen identity assertions
pass on their own terms rather than by being widened.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"

FROZEN_RECOVERY = THIS_SOURCE.with_name(
    "recover_action_delay_h7_a0_aux_pcja_predictor_only_v1_prefix_postfit_v1.py"
)

MULTI_SEED_ID = "action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1"
ARTIFACT_ROOT = ROOT / "artifacts" / MULTI_SEED_ID
AMENDMENT = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1"
    "_postfit_recovery_amendment_v1.yaml"
)

FROZEN_SEED = 3072
AUTHORIZED_SEEDS = (4096, 5120)
STEP = 256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    _require(
        specification is not None and specification.loader is not None,
        f"cannot import {path}",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


child = _load(FROZEN_RECOVERY, "_multi_seed_prefix_postfit_child_v1")
parent = child.parent


def _frozen_contract() -> dict[str, Any]:
    """Re-validate the frozen recovery stack on its own terms."""

    implementation = child._validate_identity()
    frozen = {
        "child_seed": child.SEED == FROZEN_SEED,
        "child_step": child.STEP == STEP,
        "child_candidate": child.CANDIDATE_ID
        == "action_delay_h7_a0_aux_pcja_predictor_only_v1",
        "child_artifact_root": child.ARTIFACT_ROOT
        == ROOT / "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_v1",
        "parent_paths_are_s3072": "prefix256_s3072_v1"
        in str(parent._paths()["root"]),
        "expected_loss_steps": tuple(child.EXPECTED_LOSS_STEPS)
        == (1, *range(20, 241, 20)),
        "expected_gradient_steps": tuple(child.EXPECTED_GRADIENT_STEPS)
        == (1, 4, 16, 64, 256),
    }
    _require(all(frozen.values()), f"frozen recovery contract changed: {frozen}")
    return {"implementation": implementation, "checks": frozen}


def _install(seed: int):
    """Rebind the two seed-pinned surfaces, snapshotting BEFORE any rebinding."""

    saved = {
        "child_SEED": child.SEED,
        "child_ARTIFACT_ROOT": child.ARTIFACT_ROOT,
        "parent_SEED": parent.SEED,
        "parent_ARTIFACT_ROOT": parent.ARTIFACT_ROOT,
        "parent_CANDIDATE_ID": parent.CANDIDATE_ID,
        "parent_paths": parent._paths,
    }

    root = ARTIFACT_ROOT / f"training/prefix256_s{seed}_v1"
    run = root / f"checkpoints/{MULTI_SEED_ID}_s{seed}_prefix256_s{seed}"

    def _seed_paths() -> dict[str, Path]:
        return {
            "root": root,
            "run": run,
            "runtime": root / "native_sampler_aux_pcja_runtime_audit_v1.json",
            "report": root / "contextworld_report.json",
            "full_state": run / "last.ckpt",
            "weights": run / "weights_step_256.pt",
            "config": run / "config.json",
            "train_config": run / "train_config.yaml",
            "loss_trace": run / "loss_trace.jsonl",
            "gradient_trace": run / "gradient_trace.jsonl",
        }

    child.SEED = seed
    child.ARTIFACT_ROOT = ARTIFACT_ROOT
    parent._paths = _seed_paths

    install = {
        "seed": seed,
        "phase_root": str(root),
        "run_dir": str(run),
        "candidate_id_rebound": False,
        "rebound_surfaces": ["child.SEED", "child.ARTIFACT_ROOT", "parent._paths"],
    }
    return install, saved


def _restore(saved: dict[str, Any]) -> None:
    child.SEED = saved["child_SEED"]
    child.ARTIFACT_ROOT = saved["child_ARTIFACT_ROOT"]
    parent.SEED = saved["parent_SEED"]
    parent.ARTIFACT_ROOT = saved["parent_ARTIFACT_ROOT"]
    parent.CANDIDATE_ID = saved["parent_CANDIDATE_ID"]
    parent._paths = saved["parent_paths"]


def _authorized() -> dict[str, Any]:
    _require(
        AMENDMENT.is_file(),
        "the append-only post-fit recovery amendment is missing; the recovery "
        "layer is not preregistered without it",
    )
    text = AMENDMENT.read_text(encoding="utf-8")
    _require(
        "postfit_recovery_authorized: true" in text,
        "the amendment does not authorize the post-fit recovery",
    )
    return {"path": str(AMENDMENT), "sha256": parent._sha256(AMENDMENT)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=AUTHORIZED_SEEDS)
    parser.add_argument("--check-only", action="store_true")
    options = parser.parse_args(argv)

    before = _frozen_contract()
    amendment = None if options.check_only else _authorized()

    install, saved = _install(options.seed)
    try:
        evidence = child.audit()
        report = child._report(evidence, before["implementation"])
        _require(
            report["seed"] == options.seed,
            f"report seed is not the overlaid seed: {report['seed']}",
        )
        report_path = evidence["paths"]["report"]
        if not options.check_only:
            child._write_exclusive(report_path, report)
    finally:
        _restore(saved)

    after = _frozen_contract()
    _require(
        after["checks"] == before["checks"]
        and after["implementation"] == before["implementation"],
        "the frozen recovery contract does not revalidate after restore",
    )

    print(
        json.dumps(
            {
                "status": "validated" if options.check_only else "recovered",
                "seed": options.seed,
                "report": str(report_path),
                "report_written": not options.check_only,
                "content_sha256": report["content_sha256"],
                "optimizer_steps_added": 0,
                "install": install,
                "amendment": amendment,
                "frozen_contract_revalidated_after_restore": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
