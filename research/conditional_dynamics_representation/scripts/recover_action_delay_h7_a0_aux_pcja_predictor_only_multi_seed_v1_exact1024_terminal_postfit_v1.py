#!/usr/bin/env python3
"""Seed overlay for the frozen predictor-only exact-1024 terminal post-fit recovery.

Like the prefix phase, the continuation phase does not exit cleanly out of the
ContextWorld trainer, and the frozen seed-3072 run was completed by a frozen
recovery runner whose report carries the status
`postfit_recovered_after_predictor_only_v1_exact1024_prefix_diagnostic_namespace_omission`.
This file is the seed overlay over that runner, and nothing more.  It adds zero
optimizer steps, instantiates no model, computes no score, and writes into no
frozen path.

Four kinds of seed pinning exist in the frozen continuation-recovery stack:

  (a) the four phase/run directories (`PREFIX_ROOT`, `PREFIX_RUN`,
      `TERMINAL_ROOT`, `TERMINAL_RUN`), held as module globals on both the child
      and the parent, and simply rebound here;

  (b) `EXPECTED_PLAN_SHA256` and `EXPECTED_CONTINUATION_DIGESTS` on the child,
      which are the seed-3072 data-order plan.  These are NOT copied from the
      run being audited - that would make the check vacuous.  They are DERIVED
      for the new seed by the frozen `_standard_sampler_digest` and the frozen
      `_validate_primary_plan`, reached through the multi-seed training runner's
      own install/restore, so the check still binds;

  (c) `CANDIDATE_ID`, deliberately NOT rebound - the candidate, objective and
      recipe are unchanged, only the training seed differs;

  (d) two bare `3072` integer literals inlined in the parent's `audit` and
      `_report`.  The same semantic quantity that the prefix recovery exposes as
      a rebindable `SEED` global is, here, a constant in a code object.  It is
      re-targeted by rebuilding the function with one entry of `co_consts`
      replaced, after asserting that `co_code` and every other code attribute
      are byte-identical.  No frozen logic is copied, retyped, or weakened: the
      executed bytecode is the same bytecode, pointed at the seed under test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"

FROZEN_RECOVERY = THIS_SOURCE.with_name(
    "recover_action_delay_h7_a0_aux_pcja_predictor_only_v1_"
    "exact1024_terminal_postfit_v1.py"
)
TRAINING_RUNNER = THIS_SOURCE.with_name(
    "run_action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1.py"
)

MULTI_SEED_ID = "action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1"
ARTIFACT_ROOT = ROOT / "artifacts" / MULTI_SEED_ID
AMENDMENT = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1"
    "_postfit_recovery_amendment_v1.yaml"
)

FROZEN_SEED = 3072
AUTHORIZED_SEEDS = (4096, 5120)

_CODE_ATTRIBUTES = (
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_code",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_filename",
    "co_name",
    "co_firstlineno",
)


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


child = _load(FROZEN_RECOVERY, "_multi_seed_terminal_postfit_child_v1")
parent = child.parent


def _nested_holds_seed(code: types.CodeType) -> bool:
    for value in code.co_consts:
        if isinstance(value, types.CodeType) and (
            FROZEN_SEED in [c for c in value.co_consts if isinstance(c, int)]
            or _nested_holds_seed(value)
        ):
            return True
    return False


def _retarget_seed_literal(function: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    """Rebuild `function` with its single 3072 constant replaced by `seed`."""

    code = function.__code__
    indices = [
        index
        for index, value in enumerate(code.co_consts)
        if type(value) is int and value == FROZEN_SEED
    ]
    _require(
        len(indices) == 1,
        f"{function.__name__}: expected exactly one {FROZEN_SEED} constant, "
        f"found {len(indices)}",
    )
    _require(
        not _nested_holds_seed(code),
        f"{function.__name__}: a nested code object also holds {FROZEN_SEED}",
    )
    constants = list(code.co_consts)
    constants[indices[0]] = int(seed)
    rebuilt = code.replace(co_consts=tuple(constants))

    identical = {
        attribute: getattr(rebuilt, attribute) == getattr(code, attribute)
        for attribute in _CODE_ATTRIBUTES
    }
    _require(
        all(identical.values()),
        f"{function.__name__}: code object changed beyond the constant: "
        f"{ {k: v for k, v in identical.items() if not v} }",
    )
    differing = [
        index
        for index in range(len(code.co_consts))
        if rebuilt.co_consts[index] is not code.co_consts[index]
        and rebuilt.co_consts[index] != code.co_consts[index]
    ]
    _require(
        differing == indices,
        f"{function.__name__}: more than one constant changed: {differing}",
    )

    replacement = types.FunctionType(
        rebuilt,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    replacement.__kwdefaults__ = function.__kwdefaults__
    replacement.__dict__.update(function.__dict__)
    proof = {
        "function": function.__name__,
        "constant_index": indices[0],
        "old_value": FROZEN_SEED,
        "new_value": int(seed),
        "co_code_identical": True,
        "other_code_attributes_identical": True,
        "constants_changed": 1,
    }
    return replacement, proof


def _derive_plan(seed: int) -> dict[str, Any]:
    """Derive the seed's data-order plan with the FROZEN plan validator."""

    executor = _load(TRAINING_RUNNER, "_multi_seed_training_runner_for_recovery")
    predictor_only, native_parent, install, saved = executor._install(seed, "continue")
    try:
        plan = native_parent._validate_primary_plan()
    finally:
        executor._restore(native_parent, saved)
    _require(
        native_parent.SEED == FROZEN_SEED,
        "the training runner did not restore the frozen seed after derivation",
    )
    return {
        "plan_sha256": plan["plan_sha256"],
        "continuation": tuple(plan["continuation"]),
        "prefix": tuple(plan["prefix"]),
        "derived_by": "frozen _standard_sampler_digest + frozen _validate_primary_plan",
        "config": install.get("config"),
    }


def _frozen_contract() -> dict[str, Any]:
    frozen = {
        "child_candidate": child.CANDIDATE_ID
        == "action_delay_h7_a0_aux_pcja_predictor_only_v1",
        "child_prefix_root_is_s3072": "prefix256_s3072_v1" in str(child.PREFIX_ROOT),
        "child_terminal_root_is_s3072": "continue1024_s3072_v1"
        in str(child.TERMINAL_ROOT),
        "parent_prefix_root_is_child_bound": parent.PREFIX_ROOT == child.PREFIX_ROOT,
        "parent_terminal_root_is_child_bound": parent.TERMINAL_ROOT
        == child.TERMINAL_ROOT,
        "loss_steps": tuple(child.LOSS_STEPS) == (*range(260, 1021, 20), 1024),
        "diagnostic_steps": tuple(child.DIAGNOSTIC_STEPS) == (1, 4, 16, 64, 256),
        "expected_plan_sha_is_frozen": child.EXPECTED_PLAN_SHA256
        == "1cd6cbc60d1fc2756c8ad79b3a9fafb89ba7f1654a694a81277f2d66196a482b",
        "expected_digests_count": len(child.EXPECTED_CONTINUATION_DIGESTS) == 8,
    }
    _require(all(frozen.values()), f"frozen recovery contract changed: {frozen}")
    return frozen


def _install(seed: int, plan: dict[str, Any]):
    saved = {
        "child_PREFIX_ROOT": child.PREFIX_ROOT,
        "child_PREFIX_RUN": child.PREFIX_RUN,
        "child_TERMINAL_ROOT": child.TERMINAL_ROOT,
        "child_TERMINAL_RUN": child.TERMINAL_RUN,
        "child_EXPECTED_PLAN_SHA256": child.EXPECTED_PLAN_SHA256,
        "child_EXPECTED_CONTINUATION_DIGESTS": child.EXPECTED_CONTINUATION_DIGESTS,
        "parent_PREFIX_ROOT": parent.PREFIX_ROOT,
        "parent_PREFIX_RUN": parent.PREFIX_RUN,
        "parent_TERMINAL_ROOT": parent.TERMINAL_ROOT,
        "parent_TERMINAL_RUN": parent.TERMINAL_RUN,
        "parent_audit": parent.audit,
        "parent_report": parent._report,
    }

    prefix_root = ARTIFACT_ROOT / f"training/prefix256_s{seed}_v1"
    prefix_run = prefix_root / f"checkpoints/{MULTI_SEED_ID}_s{seed}_prefix256_s{seed}"
    terminal_root = ARTIFACT_ROOT / f"training/continue1024_s{seed}_v1"
    terminal_run = (
        terminal_root / f"checkpoints/{MULTI_SEED_ID}_s{seed}_continue1024_s{seed}"
    )

    for module in (child, parent):
        module.PREFIX_ROOT = prefix_root
        module.PREFIX_RUN = prefix_run
        module.TERMINAL_ROOT = terminal_root
        module.TERMINAL_RUN = terminal_run

    child.EXPECTED_PLAN_SHA256 = plan["plan_sha256"]
    child.EXPECTED_CONTINUATION_DIGESTS = tuple(plan["continuation"])

    audit_function, audit_proof = _retarget_seed_literal(parent.audit, seed)
    report_function, report_proof = _retarget_seed_literal(parent._report, seed)
    parent.audit = audit_function
    parent._report = report_function

    install = {
        "seed": seed,
        "prefix_root": str(prefix_root),
        "terminal_root": str(terminal_root),
        "terminal_run": str(terminal_run),
        "candidate_id_rebound": False,
        "plan_sha256": plan["plan_sha256"],
        "continuation_digest_rank0": plan["continuation"][0],
        "seed_literal_retargeting": [audit_proof, report_proof],
    }
    return install, saved


def _restore(saved: dict[str, Any]) -> None:
    child.PREFIX_ROOT = saved["child_PREFIX_ROOT"]
    child.PREFIX_RUN = saved["child_PREFIX_RUN"]
    child.TERMINAL_ROOT = saved["child_TERMINAL_ROOT"]
    child.TERMINAL_RUN = saved["child_TERMINAL_RUN"]
    child.EXPECTED_PLAN_SHA256 = saved["child_EXPECTED_PLAN_SHA256"]
    child.EXPECTED_CONTINUATION_DIGESTS = saved["child_EXPECTED_CONTINUATION_DIGESTS"]
    parent.PREFIX_ROOT = saved["parent_PREFIX_ROOT"]
    parent.PREFIX_RUN = saved["parent_PREFIX_RUN"]
    parent.TERMINAL_ROOT = saved["parent_TERMINAL_ROOT"]
    parent.TERMINAL_RUN = saved["parent_TERMINAL_RUN"]
    parent.audit = saved["parent_audit"]
    parent._report = saved["parent_report"]


def _authorized() -> dict[str, Any]:
    _require(
        AMENDMENT.is_file(),
        "the append-only post-fit recovery amendment is missing; the recovery "
        "layer is not preregistered without it",
    )
    _require(
        "postfit_recovery_authorized: true"
        in AMENDMENT.read_text(encoding="utf-8"),
        "the amendment does not authorize the post-fit recovery",
    )
    return {"path": str(AMENDMENT), "sha256": parent._reference(AMENDMENT)["sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=AUTHORIZED_SEEDS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--publish", action="store_true")
    options = parser.parse_args(argv)

    before = _frozen_contract()
    amendment = _authorized() if options.publish else None
    plan = _derive_plan(options.seed)

    install, saved = _install(options.seed, plan)
    try:
        evidence = child.audit(require_report_absent=True)
        report = child._report(evidence, child._validate_implementation())
        _require(
            report["seed"] == options.seed,
            f"report seed is not the overlaid seed: {report['seed']}",
        )
        report_path = evidence["paths"]["report"]
        if options.publish:
            child._write_exclusive(report_path, report)
    finally:
        _restore(saved)

    after = _frozen_contract()
    _require(
        after == before,
        "the frozen recovery contract does not revalidate after restore",
    )

    print(
        json.dumps(
            {
                "status": "published" if options.publish else "validated",
                "seed": options.seed,
                "report": str(report_path),
                "report_written": bool(options.publish),
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
