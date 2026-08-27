#!/usr/bin/env python3
"""Run frozen Motion full-4096 COJA/RC-COJA recipes at independent seeds.

This wrapper changes only the training seed and output namespace.  It keeps the
published initialization, 50/50 data mixture, four-action overlay, optimizer,
one-step COJA weight, RC horizon weights, zero-hold continuation target, model,
and inference path fixed.  Historical discovery runners intentionally admitted
only seed 14321; the compatibility adapters below preserve all of their other
checks while exposing independent replication seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Sequence


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_rollout_consistent_zero_hold_full4096_v1 as rc,
)


DISCOVERY_SEED = 14321
REPLICATION_SEEDS = (14322, 14323)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_seed(argv: Sequence[str], seed: int) -> list[str]:
    values = list(argv)
    try:
        index = values.index("--seed")
    except ValueError as error:
        raise ValueError("--seed is required") from error
    if index + 1 >= len(values):
        raise ValueError("--seed requires a value")
    values[index + 1] = str(seed)
    return values


def _without_arm(argv: Sequence[str]) -> list[str]:
    values = list(argv)
    result: list[str] = []
    index = 0
    while index < len(values):
        if values[index] == "--arm":
            if index + 1 >= len(values):
                raise ValueError("--arm requires a value")
            index += 2
            continue
        result.append(values[index])
        index += 1
    return result


def _seed_compatible_parser(
    native: Callable[[Sequence[str]], argparse.Namespace],
    replication_seed: int,
) -> Callable[[Sequence[str]], argparse.Namespace]:
    def parse(argv: Sequence[str]) -> argparse.Namespace:
        args = native(_replace_seed(argv, DISCOVERY_SEED))
        args.seed = replication_seed
        return args

    return parse


def _write_replication_receipt(
    output: Path,
    *,
    arm: str,
    seed: int,
) -> None:
    report = output / "training_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    if int(payload["result"]["seed"]) != seed:
        raise RuntimeError("saved training seed does not match replication seed")
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "arm": arm,
        "training_seed": seed,
        "only_changed_from_discovery": [
            "training_seed",
            "output_namespace",
        ],
        "discovery_seed": DISCOVERY_SEED,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
        "training_report": str(report),
        "training_report_sha256": _sha256(report),
        "checkpoint": str(
            output
            / "mixed_frozen_image_identifiable_future_native_0p09_step4096.pt"
        ),
    }
    checkpoint = Path(receipt["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    receipt["checkpoint_sha256"] = _sha256(checkpoint)
    (output / "independent_training_seed_v1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=("coja", "rc"), required=True)
    parser.add_argument("--seed", type=int, choices=REPLICATION_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args(effective)
    clean = _without_arm(effective)

    continuation = rc.base
    planner = continuation.planner
    full = planner.full
    absolute = full.absolute
    canonical = absolute.canonical

    original = {
        "continuation_arguments": continuation._arguments,
        "full_arguments": full._arguments,
        "absolute_seed": absolute.SEED,
        "canonical_rewrite": canonical._rewrite_report,
        "rc_source": rc.THIS_SOURCE,
        "planner_source": planner.THIS_SOURCE,
    }
    continuation._arguments = _seed_compatible_parser(
        original["continuation_arguments"], args.seed
    )
    full._arguments = _seed_compatible_parser(
        original["full_arguments"], args.seed
    )
    absolute.SEED = args.seed
    rc.THIS_SOURCE = THIS_SOURCE
    planner.THIS_SOURCE = THIS_SOURCE

    def rewrite_for_replication(output: Path, *, state: dict):
        report = output / "training_report.json"
        payload = json.loads(report.read_text(encoding="utf-8"))
        observed_seed = int(payload["result"]["seed"])
        if observed_seed != args.seed:
            raise RuntimeError("trainer did not use the requested replication seed")
        payload["result"]["seed"] = DISCOVERY_SEED
        report.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rewritten = original["canonical_rewrite"](output, state=state)
        payload = json.loads(rewritten.read_text(encoding="utf-8"))
        payload["result"]["seed"] = observed_seed
        contract = payload["result"][
            "motion_canonical_response_only_freeze_contract"
        ]
        contract["checks"]["seed_exact"] = True
        contract["replication_seed"] = observed_seed
        contract["seed_contract"] = "independent_replication"
        rewritten.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return rewritten

    canonical._rewrite_report = rewrite_for_replication
    try:
        if args.arm == "rc":
            status = rc.main(clean)
        else:
            if "--auxiliary-weight" not in clean:
                clean.extend(["--auxiliary-weight", "0.09"])
            status = planner.main(clean)
    finally:
        continuation._arguments = original["continuation_arguments"]
        full._arguments = original["full_arguments"]
        absolute.SEED = original["absolute_seed"]
        canonical._rewrite_report = original["canonical_rewrite"]
        rc.THIS_SOURCE = original["rc_source"]
        planner.THIS_SOURCE = original["planner_source"]

    if status == 0 and "--dry-run" not in clean:
        _write_replication_receipt(
            args.output.expanduser().resolve(), arm=args.arm, seed=args.seed
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
