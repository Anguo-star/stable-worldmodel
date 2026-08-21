#!/usr/bin/env python3
"""Freeze the three hash-only terminal identities after the v2 release build."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
BUILDER = THIS_SOURCE.with_name(
    "build_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2.py"
)
TERMINAL = ROOT / (
    "configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_"
    "development_v2_terminal_identity_addendum_v1.json"
)
V1_RECEIPT = ROOT / (
    "artifacts/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1/"
    "consumption_receipt.json"
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _builder() -> Any:
    name = "_predictor_only_multiseed_v2_terminal_builder"
    specification = importlib.util.spec_from_file_location(name, BUILDER)
    require(
        specification is not None and specification.loader is not None,
        "cannot import builder",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _reference(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    metadata = os.lstat(path)
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"unsafe terminal artifact: {path}",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"terminal artifact SHA changed: {path}")
    if expected_bytes is not None:
        require(metadata.st_size == int(expected_bytes), f"terminal size changed: {path}")
    return {"path": str(path), "bytes": metadata.st_size, "sha256": digest}


def _candidate_cell(builder: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    terminal = row["terminal_contract"]
    checkpoint = terminal["checkpoint"]
    checkpoint_config = terminal["checkpoint_config"]
    report = terminal["contextworld_report"]
    return {
        "candidate_id": row["candidate_id"],
        "role": row["role"],
        "model_family": row["model_family"],
        "objective": row["objective"],
        "sampler": "standard DistributedSampler",
        "seed": int(row["seed"]),
        "optimizer_step": int(row["optimizer_step"]),
        "variant": "persistent_pcja_predictor_only_gradient_route",
        "artifacts": {
            "checkpoint": _reference(
                _path(checkpoint["path"]),
                expected_sha256=checkpoint["sha256"],
                expected_bytes=checkpoint.get("bytes"),
            ),
            "checkpoint_config": _reference(
                _path(checkpoint_config["path"]),
                expected_sha256=checkpoint_config["sha256"],
                expected_bytes=checkpoint_config.get("bytes"),
            ),
            "training_receipt": _reference(
                _path(report["path"]), expected_sha256=report["sha256"]
            ),
        },
        "checkpoint_deserialized_or_model_loaded": False,
        "prediction_latent_or_score_computed": False,
    }


def _calibration_cell(builder: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = row["checkpoint"]
    checkpoint_config = row["checkpoint_config"]
    return {
        "candidate_id": row["control_id"],
        "role": row["role"],
        "model_family": row["model_family"],
        "objective": (
            "native weighted live MSE plus 0.09 native SIGReg plus 0.09 "
            "auxiliary PCJA through step1024 with predictor-only gradient routing"
        ),
        "sampler": "standard DistributedSampler",
        "seed": int(row["seed"]),
        "optimizer_step": int(row["optimizer_step"]),
        "variant": "historical_positive_calibration_same_new_release",
        "artifacts": {
            "checkpoint": _reference(
                _path(checkpoint["path"]),
                expected_sha256=checkpoint["sha256"],
                expected_bytes=checkpoint.get("bytes"),
            ),
            "checkpoint_config": _reference(
                _path(checkpoint_config["path"]),
                expected_sha256=checkpoint_config["sha256"],
                expected_bytes=checkpoint_config.get("bytes"),
            ),
            "historical_evidence": _reference(V1_RECEIPT),
        },
        "checkpoint_deserialized_or_model_loaded": False,
        "prediction_latent_or_score_computed": False,
    }


def expected_payload() -> dict[str, Any]:
    builder = _builder()
    config = builder.load_config()
    release = builder.validate_release_identity()
    cells = [
        *(_candidate_cell(builder, row) for row in config["candidates"]),
        _calibration_cell(builder, config["positive_calibration"]),
    ]
    require(
        tuple(row["candidate_id"] for row in cells) == builder.AUTHORIZED_IDS,
        "terminal cell order changed",
    )
    identity_path = builder.EXPECTED_OUTPUT_ROOT / "release_identity.json"
    content = {
        "schema_version": 1,
        "status": (
            "frozen_hash_only_terminal_identity_after_release_build_before_claim_or_score"
        ),
        "release_id": builder.EXPECTED_RELEASE_ID,
        "release_identity": {
            **_reference(identity_path),
            "identity_sha256": release["release_identity"]["identity_sha256"],
        },
        "cells": cells,
        "training_identity": {
            "objective": builder.effective_config()["training_contract"]["objective"],
            "sampler": builder.effective_config()["training_contract"]["sampler"],
            "terminal_optimizer_step": builder.TERMINAL_STEP,
            "training_steps_added_by_this_release": 0,
        },
        "scope": {
            "audit_mode": "regular_file_byte_sha256_and_json_structure_only",
            "release_built_before_terminal_freeze": True,
            "checkpoint_deserialized_or_model_loaded": False,
            "prediction_latent_or_score_computed": False,
            "exclusive_claim_committed": False,
            "public_test_or_sealed_content_opened": False,
        },
    }
    return {**content, "content_sha256": builder.canonical_sha256(content)}


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    expected = expected_payload()
    if TERMINAL.exists():
        observed = json.loads(TERMINAL.read_text(encoding="utf-8"))
        require(observed == expected, "existing terminal manifest differs")
        action = "validated_existing"
    else:
        require(not arguments.check, "terminal manifest absent")
        _write_exclusive(TERMINAL, expected)
        action = "created"
    print(json.dumps({"status": "passed", "action": action}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
