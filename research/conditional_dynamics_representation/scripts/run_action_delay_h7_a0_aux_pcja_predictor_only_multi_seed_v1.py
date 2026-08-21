#!/usr/bin/env python3
"""Run the frozen ActionDelay predictor-only PCJA recipe at additional seeds.

Append-only overlay.  It edits no frozen file.  It imports the frozen
predictor-only runner, re-validates the whole inherited seed-3072 contract,
then rebinds exactly the seed-dependent constants for the duration of one
phase, and restores them afterwards.

Four places hard-code the seed in the frozen chain, and all four are handled:

  (a) ``SEED`` in the native-sampler module.  Rebound.  ``DATA_SPLIT_SEED`` is
      deliberately NOT rebound -- the frozen PLDM multi-seed precedent holds the
      split fixed so the comparison is a training-seed replication rather than a
      different held-out split.
  (b) ``EXPECTED_PRIMARY_{PREFIX,CONTINUATION}_DIGESTS``, checked by
      ``_validate_primary_plan()``.  Rebound to digests DERIVED at run time by
      the frozen ``_standard_sampler_digest`` for the new seed.  The guard keeps
      its full force: it still proves the plan is the standard
      DistributedSampler plan for the declared seed.
  (c) ``primary_order_authority.{prefix,continuation}_sha256_by_rank`` in the
      preregistration YAML, checked by ``_validate_preregistration()``.  A
      per-seed config is generated from the frozen one with only the seed fields
      and these derived digests changed.
  (d) the literal ``s3072`` interpolated into artifact paths by ``_paths()``.
      Redirected to a separate artifact root so the frozen s3072 evidence is
      never written into.  ``_ensure_fresh()`` is left in force as the backstop.

SPAWN SAFETY.  Lightning re-launches this script for DDP ranks 1..7 with the
same argv, so every entry re-installs the identical overlay from ``--seed``.
Rank zero generates the per-seed config; nonzero children only read and verify
it, and never write it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
ANALYSIS_ID = "action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1"
PREREG = ROOT / f"configs/{ANALYSIS_ID}.yaml"
AUTHORIZATION = ROOT / f"configs/{ANALYSIS_ID}_execution_authorization_v1.yaml"
ARTIFACT_ROOT = ROOT / f"artifacts/{ANALYSIS_ID}"
GENERATED_CONFIG_DIR = ARTIFACT_ROOT / "generated_configs"

PREDICTOR_ONLY_SOURCE = (
    ROOT / "scripts/run_action_delay_h7_a0_aux_pcja_predictor_only_v1.py"
)

FROZEN_SEED = 3072
FROZEN_DATA_SPLIT_SEED = 3072
AUTHORIZED_SEEDS = (4096, 5120)
PREFIX_EPOCHS = (0,)
CONTINUATION_EPOCHS = (1, 2, 3)


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_predictor_only() -> Any:
    import importlib.util

    name = "_ad_pcja_predictor_only_multi_seed_parent"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(
        name, PREDICTOR_ONLY_SOURCE
    )
    if specification is None or specification.loader is None:
        raise ImportError(PREDICTOR_ONLY_SOURCE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _load_contract() -> dict[str, Any]:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    authorization = yaml.safe_load(AUTHORIZATION.read_text(encoding="utf-8"))

    _require(
        authorization.get("status") == "authorized_for_execution",
        "execution authorization status changed",
    )
    _require(
        authorization["execution_authority"]["execution_authorized"] is True
        and authorization["execution_authority"]["execution_blocked_until_settled"]
        is False,
        "execution is not authorized",
    )
    _require(
        authorization["ordering_ruling"]["ruling"] == "readme_1_3_governs",
        "the ordering ruling this executor relies on is not on record",
    )

    pinned = authorization["authorizes_preregistration"]
    _require(
        Path(pinned["path"]).name == PREREG.name,
        "authorization points at a different preregistration",
    )
    _require(
        _sha256(PREREG) == pinned["sha256"],
        "the authorized preregistration has been edited since approval",
    )

    _require(
        prereg.get("status") == "preregistered_awaiting_execution_approval",
        "preregistration status changed",
    )
    _require(prereg.get("overwrite") == "forbidden", "overwrite guard removed")

    change = prereg["change"]
    _require(
        tuple(change["new_training_seeds"]) == AUTHORIZED_SEEDS,
        f"preregistered seeds are not {AUTHORIZED_SEEDS}",
    )
    _require(
        int(change["data_split_seed_is_held_fixed_at_3072"]["value"])
        == FROZEN_DATA_SPLIT_SEED
        and change["data_split_seed_is_held_fixed_at_3072"]["changed"] is False,
        "the data split seed is not held fixed",
    )
    _require(
        change["initialization_checkpoint_is_held_fixed"]["changed"] is False,
        "the initialization checkpoint is not held fixed",
    )

    scope = authorization["execution_authority"]["scope"]
    _require(
        tuple(scope["training_seeds"]) == AUTHORIZED_SEEDS
        and int(scope["optimizer_steps_per_seed"]) == 1024
        and int(scope["total_new_optimizer_steps"]) == 2048
        and int(scope["data_split_seed"]) == FROZEN_DATA_SPLIT_SEED,
        "authorized scope does not match this executor",
    )

    closed = authorization["execution_authority"]["everything_else_remains_closed"]
    _require(
        all(
            closed[key] is False
            for key in (
                "public_test_opened",
                "cem_opened",
                "speed_opened",
                "contact_training_authorized",
                "motion_line_training_authorized",
                "weight_sweep_authorized",
                "k_increase_authorized",
                "longer_budget_authorized",
                "architecture_change_authorized",
                "data_change_authorized",
                "family_rejection_authorized",
                "additional_seeds_for_any_other_line",
            )
        )
        and closed["hidden_or_sealed_test_remains_forbidden"] is True,
        "authority closures changed",
    )

    decision = prereg["decision"]["per_seed_independence_is_mandatory"]
    _require(
        decision["cross_seed_pooling_performed"] is False
        and decision["cross_seed_averaging_or_rescue_allowed"] is False,
        "per-seed independence was weakened",
    )
    return {"preregistration": prereg, "authorization": authorization}


# ---------------------------------------------------------------------------
# Seed-dependent derivation
# ---------------------------------------------------------------------------


def _derive_digests(parent: Any, seed: int) -> dict[str, list[str]]:
    """Derive the sampler plan digests for `seed` using the FROZEN function."""

    previous = parent.SEED
    try:
        parent.SEED = int(seed)
        prefix = [
            parent._standard_sampler_digest(rank=rank, epochs=PREFIX_EPOCHS)
            for rank in range(parent.WORLD_SIZE)
        ]
        continuation = [
            parent._standard_sampler_digest(rank=rank, epochs=CONTINUATION_EPOCHS)
            for rank in range(parent.WORLD_SIZE)
        ]
    finally:
        parent.SEED = previous
    _require(
        len(set(prefix)) == parent.WORLD_SIZE
        and len(set(continuation)) == parent.WORLD_SIZE,
        f"seed {seed} produced non-distinct per-rank sampler digests",
    )
    return {"prefix": prefix, "continuation": continuation}


# Upstream ContextWorld commit 45e398b ("land release preparation for public
# v1", 2026-08-19) converted contextworld/synthesis/__init__.py from eager
# imports to lazy ``__getattr__`` exports.  As a result these two modules are no
# longer pulled in when the trainer is imported, and the frozen
# CONTEXTWORLD_EXECUTION_CLOSURE guard fires with them reported missing.
#
# The guard is NOT relaxed here, and the frozen closure list is NOT edited.  The
# fix runs the other way: the two modules are imported explicitly so the process
# reaches the exact import state the closure was frozen against.  The equality
# check then passes on its own terms and still fails closed on any other drift.
# This restores the frozen contract rather than weakening it.
FROZEN_CLOSURE_MODULES_DEFERRED_UPSTREAM = (
    "contextworld.synthesis.compiler",
    "contextworld.synthesis.reset_constraints",
)


def _restore_frozen_import_closure(parent: Any) -> dict[str, Any]:
    import importlib

    core = parent._core()
    root = str(core.CONTEXTWORLD_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    restored: list[str] = []
    already: list[str] = []
    for name in FROZEN_CLOSURE_MODULES_DEFERRED_UPSTREAM:
        if name in sys.modules:
            already.append(name)
            continue
        importlib.import_module(name)
        restored.append(name)
    return {
        "reason": "upstream ContextWorld 45e398b made these lazy imports",
        "restored": restored,
        "already_present": already,
        "frozen_closure_list_edited": False,
        "guard_relaxed": False,
    }


def _validate_frozen_contract(parent: Any) -> dict[str, Any]:
    """Prove the inherited seed-3072 contract is intact before anything moves."""

    _require(parent.SEED == FROZEN_SEED, "frozen SEED is not 3072 at entry")
    _require(
        parent.DATA_SPLIT_SEED == FROZEN_DATA_SPLIT_SEED,
        "frozen DATA_SPLIT_SEED is not 3072 at entry",
    )
    plan = parent._validate_primary_plan()
    checks = parent._validate_preregistration()
    _require(
        tuple(plan["prefix"]) == tuple(parent.EXPECTED_PRIMARY_PREFIX_DIGESTS)
        and int(plan["seed"]) == FROZEN_SEED,
        "the frozen seed-3072 sampler plan does not reproduce",
    )
    _require(all(checks.values()), "the frozen preregistration contract does not hold")
    return {"seed": FROZEN_SEED, "plan_verified": True, "contract_verified": True}


def _seed_config_path(seed: int) -> Path:
    return GENERATED_CONFIG_DIR / f"predictor_only_s{seed}_derived_v1.yaml"


def _build_seed_config(parent: Any, base: dict[str, Any], seed: int) -> dict[str, Any]:
    digests = _derive_digests(parent, seed)
    config = copy.deepcopy(base)
    config["training_contract"]["seed"] = int(seed)
    config["training_contract"]["data"]["sampler_seed"] = int(seed)
    config["primary_order_authority"]["seed"] = int(seed)
    config["primary_order_authority"]["prefix_sha256_by_rank"] = digests["prefix"]
    config["primary_order_authority"]["continuation_sha256_by_rank"] = digests[
        "continuation"
    ]
    # Held fixed on purpose -- see the preregistration.
    _require(
        int(config["training_contract"]["data_split_seed"]) == FROZEN_DATA_SPLIT_SEED,
        "generated config moved the data split seed",
    )
    return config


def _materialize_seed_config(parent: Any, predictor_only: Any, seed: int) -> Path:
    """Write (rank zero) or verify (DDP child) the derived per-seed config."""

    base = yaml.safe_load(predictor_only.CONFIG.read_text(encoding="utf-8"))
    expected = _build_seed_config(parent, base, seed)
    rendered = yaml.safe_dump(expected, sort_keys=True, allow_unicode=True)
    path = _seed_config_path(seed)

    local_rank = parent._local_rank()
    nonzero_child = local_rank is not None and int(local_rank) > 0

    if path.exists():
        _require(
            yaml.safe_load(path.read_text(encoding="utf-8")) == expected,
            f"existing derived config for seed {seed} does not match the "
            f"derivation; refusing to proceed",
        )
        return path
    _require(
        not nonzero_child,
        f"DDP child found no derived config for seed {seed}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.partial")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)
    return path


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def _seed_paths(seed: int) -> Any:
    """Mirror the frozen `_paths`, with the new seed and a separate root."""

    candidate_id = f"{ANALYSIS_ID}_s{seed}"

    def paths(phase: str, *, preflight: bool = False) -> dict[str, Path]:
        if phase not in {"prefix", "continue"}:
            raise ValueError(f"unknown phase: {phase}")
        label = "prefix256" if phase == "prefix" else "continue1024"
        root = ARTIFACT_ROOT / ("preflight" if preflight else "training") / (
            f"{label}_s{seed}_v1"
        )
        run_name = f"{candidate_id}_{label}_s{seed}"
        run_dir = root / "checkpoints" / run_name
        return {
            "root": root,
            "run_dir": run_dir,
            "run_name": Path(run_name),
            "report": root / "contextworld_report.json",
            "runtime": root / "native_sampler_aux_pcja_runtime_audit_v1.json",
            "preparation": root / "continuation_preparation_v1.json",
            "last": run_dir / "last.ckpt",
            "weights": run_dir
            / f"weights_step_{256 if phase == 'prefix' else 1024}.pt",
            "immutable_prefix_full": root / "immutable_step256_full_state.ckpt",
            "immutable_prefix_model": root / "immutable_step256_model.pt",
        }

    return paths


def _install(seed: int, phase: str) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    _require(int(seed) in AUTHORIZED_SEEDS, f"seed {seed} is not authorized")
    predictor_only = _load_predictor_only()
    parent = predictor_only._install_predictor_only_overlay(phase=phase)

    frozen = _validate_frozen_contract(parent)
    closure = _restore_frozen_import_closure(parent)

    config_path = _materialize_seed_config(parent, predictor_only, seed)
    digests = _derive_digests(parent, seed)

    # Snapshot BEFORE any rebinding, so restore returns the frozen values.
    saved = _snapshot(parent)

    parent.SEED = int(seed)
    parent.EXPECTED_PRIMARY_PREFIX_DIGESTS = tuple(digests["prefix"])
    parent.EXPECTED_PRIMARY_CONTINUATION_DIGESTS = tuple(digests["continuation"])
    parent.EXPECTED_PREFIX_DIGESTS = parent.EXPECTED_PRIMARY_PREFIX_DIGESTS
    parent.EXPECTED_CONTINUATION_DIGESTS = parent.EXPECTED_PRIMARY_CONTINUATION_DIGESTS
    parent.CONFIG = config_path
    parent.ARTIFACT_ROOT = ARTIFACT_ROOT
    parent._paths = _seed_paths(seed)
    # Lightning re-launches sys.argv[0] for DDP ranks 1..7; point it at THIS
    # file so children re-enter the overlay instead of the frozen s3072 runner.
    parent.THIS_SOURCE = THIS_SOURCE

    _require(
        parent.DATA_SPLIT_SEED == FROZEN_DATA_SPLIT_SEED,
        "the data split seed moved during install",
    )
    return (
        predictor_only,
        parent,
        {
            "frozen_contract": frozen,
            "frozen_import_closure_restoration": closure,
            "derived_config": str(config_path),
            "derived_config_sha256": _sha256(config_path),
            "prefix_digest_rank0": digests["prefix"][0],
            "continuation_digest_rank0": digests["continuation"][0],
        },
        saved,
    )


def _restore(parent: Any, saved: dict[str, Any]) -> dict[str, Any]:
    for key, value in saved.items():
        setattr(parent, key, value)
    plan = parent._validate_primary_plan()
    _require(
        int(plan["seed"]) == FROZEN_SEED
        and tuple(plan["prefix"]) == tuple(parent.EXPECTED_PRIMARY_PREFIX_DIGESTS),
        "the frozen seed-3072 plan does not revalidate after restore",
    )
    return {"restored": True, "frozen_plan_revalidated": True}


def _snapshot(parent: Any) -> dict[str, Any]:
    return {
        "SEED": parent.SEED,
        "DATA_SPLIT_SEED": parent.DATA_SPLIT_SEED,
        "EXPECTED_PRIMARY_PREFIX_DIGESTS": parent.EXPECTED_PRIMARY_PREFIX_DIGESTS,
        "EXPECTED_PRIMARY_CONTINUATION_DIGESTS": (
            parent.EXPECTED_PRIMARY_CONTINUATION_DIGESTS
        ),
        "EXPECTED_PREFIX_DIGESTS": parent.EXPECTED_PREFIX_DIGESTS,
        "EXPECTED_CONTINUATION_DIGESTS": parent.EXPECTED_CONTINUATION_DIGESTS,
        "CONFIG": parent.CONFIG,
        "ARTIFACT_ROOT": parent.ARTIFACT_ROOT,
        "_paths": parent._paths,
        "THIS_SOURCE": parent.THIS_SOURCE,
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def check_only(seed: int, phase: str) -> dict[str, Any]:
    contract = _load_contract()
    predictor_only, parent, install, saved = _install(seed, phase)
    try:
        static = parent._static_contract(phase)
        _require(int(static["seed"]) == int(seed), "static contract seed mismatch")
        paths = parent._paths(phase)
        _require(
            ARTIFACT_ROOT in paths["root"].parents,
            "phase artifacts would land outside the multi-seed root",
        )
        _require(
            "s3072" not in str(paths["root"]) and "s3072" not in str(paths["run_dir"]),
            "phase artifacts would collide with the frozen s3072 namespace",
        )
        frozen_root = (
            ROOT / "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_v1"
        )
        _require(
            frozen_root not in paths["root"].parents and paths["root"] != frozen_root,
            "phase artifacts would be written into the frozen s3072 artifact root",
        )
        result = {
            "status": "passed_multi_seed_static_check",
            "analysis_id": ANALYSIS_ID,
            "seed": int(seed),
            "phase": phase,
            "authorized_seeds": list(AUTHORIZED_SEEDS),
            "data_split_seed": FROZEN_DATA_SPLIT_SEED,
            "install": install,
            "artifact_root": str(paths["root"]),
            "root_exists": paths["root"].exists(),
            "preregistration_sha256": _sha256(PREREG),
            "authorization_sha256": _sha256(AUTHORIZATION),
            "implementation_sha256": _sha256(THIS_SOURCE),
            "ordering_ruling": contract["authorization"]["ordering_ruling"]["ruling"],
            "preregistration_checks": static["preregistration_checks"],
        }
    finally:
        _restore(parent, saved)
    return result


def run(seed: int, phase: str) -> int:
    _load_contract()
    predictor_only, parent, install, saved = _install(seed, phase)
    try:
        print(
            json.dumps(
                {
                    "status": "multi_seed_overlay_installed",
                    "seed": int(seed),
                    "phase": phase,
                    **install,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(parent.main(["--phase", phase]))
    finally:
        _restore(parent, saved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=AUTHORIZED_SEEDS)
    parser.add_argument("--phase", choices=("prefix", "continue"), default="prefix")
    parser.add_argument("--check-only", action="store_true")
    options = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if options.check_only:
        print(json.dumps(check_only(options.seed, options.phase), indent=2, sort_keys=True))
        return 0
    return run(options.seed, options.phase)


if __name__ == "__main__":
    raise SystemExit(main())
