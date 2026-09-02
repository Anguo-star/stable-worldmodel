#!/usr/bin/env python3
"""Validate and inventory the cross-task ICL root-cause reversal matrix.

This audit is intentionally read-only. Historical published scoreboards may
label a model-recipe cell as positive or negative, but they are never counted
as new root-cause evidence. Root-cause evidence must come from Training or
Development and must declare its support boundary explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


THIS_SOURCE = Path(__file__).resolve()
ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD_ROOT = ROOT.parent / "ContextWorld"
ANALYSIS_ID = "icl_root_cause_reversal_matrix_v1"
DEFAULT_CONFIG = (
    ROOT
    / "research/conditional_dynamics_representation/configs"
    / f"{ANALYSIS_ID}.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "research/conditional_dynamics_representation/artifacts"
    / ANALYSIS_ID
    / "inventory_v1.json"
)

LAYERS = ("data", "representation", "optimization", "behavior")
LAYER_STATUSES = {"available", "partial", "missing", "not_applicable"}
OUTCOMES = {"positive", "negative", "mixed", "unknown"}
EVIDENCE_TIERS = {
    "historical_frozen_outcome",
    "development_single_seed",
    "development_multi_seed",
    "training_diagnostic",
    "analytic",
}
SOURCE_KINDS = {
    "historical_published_outcome",
    "training_diagnostic",
    "development_on_support",
    "development_intervention",
    "analytic_identity",
    "off_support_auxiliary",
    "configuration_provenance",
}
ROOT_CAUSE_SOURCE_KINDS = {
    "training_diagnostic",
    "development_on_support",
    "development_intervention",
    "analytic_identity",
}
PUBLIC_RAW_TOKENS = (
    "/public_test/raw/",
    "/public_score_v1/",
    "/validation.lance",
    "public_access_started",
    "public_manifest",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(row: Mapping[str, Any], *, root: Path, contextworld_root: Path) -> Path:
    repository = row.get("repository")
    _require(
        repository in {"stable-worldmodel", "ContextWorld"},
        f"unknown source repository: {repository!r}",
    )
    base = root if repository == "stable-worldmodel" else contextworld_root
    path = Path(str(row.get("path", "")))
    _require(not path.is_absolute(), "source paths must be repository-relative")
    resolved = (base / path).resolve()
    expected_base = base.resolve()
    _require(
        resolved == expected_base or expected_base in resolved.parents,
        f"source escapes repository root: {path}",
    )
    return resolved


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return None


def _deep_get(payload: Any, path: Sequence[str | int]) -> Any:
    value = payload
    for key in path:
        if isinstance(key, int):
            _require(isinstance(value, Sequence), f"cannot index non-sequence with {key}")
            value = value[key]
        else:
            _require(isinstance(value, Mapping), f"cannot read field {key!r}")
            _require(key in value, f"missing field {key!r}")
            value = value[key]
    return value


def _equal(actual: Any, expected: Any, tolerance: float | None) -> bool:
    if tolerance is not None and isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def _verify_outcome(
    cell: Mapping[str, Any],
    payloads: Mapping[str, Any],
) -> dict[str, Any]:
    check = cell.get("outcome_check")
    _require(isinstance(check, Mapping), f"{cell['cell_id']}: missing outcome_check")
    source_id = str(check.get("source"))
    _require(source_id in payloads, f"{cell['cell_id']}: unknown outcome source {source_id}")
    payload = payloads[source_id]
    check_type = check.get("type")

    if check_type == "scoreboard_row":
        rows = _deep_get(payload, ["component_results"])
        _require(isinstance(rows, list), "scoreboard component_results must be a list")
        component_id = str(check.get("component_id"))
        method_contains = str(check.get("method_contains", "")).lower()
        matches = [
            row
            for row in rows
            if row.get("component_id") == component_id
            and method_contains in str(row.get("method_name", "")).lower()
        ]
        _require(
            len(matches) == 1,
            f"{cell['cell_id']}: expected one scoreboard row, found {len(matches)}",
        )
        selected = matches[0]
        checks = [
            {
                "path": ["icl_ability", "result"],
                "expected": str(check.get("expected_result")),
            },
            {
                "path": ["icl_ability", "primary_metric", "mean"],
                "expected": check.get("expected_metric_mean"),
                "tolerance": float(check.get("metric_tolerance", 1.0e-12)),
            },
        ]
    elif check_type == "fields":
        selected = payload
        checks = check.get("checks")
        _require(isinstance(checks, list) and checks, f"{cell['cell_id']}: empty field checks")
    else:
        raise ValueError(f"{cell['cell_id']}: unsupported outcome check {check_type!r}")

    verified = []
    for row in checks:
        path = row.get("path")
        _require(isinstance(path, list) and path, f"{cell['cell_id']}: invalid check path")
        actual = _deep_get(selected, path)
        expected = row.get("expected")
        tolerance = row.get("tolerance")
        tolerance_float = float(tolerance) if tolerance is not None else None
        _require(
            _equal(actual, expected, tolerance_float),
            f"{cell['cell_id']}: outcome mismatch at {path}: {actual!r} != {expected!r}",
        )
        verified.append({"path": path, "actual": actual, "expected": expected})
    return {"source": source_id, "type": check_type, "checks": verified}


def validate_and_inventory(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
    contextworld_root: Path = CONTEXTWORLD_ROOT,
) -> dict[str, Any]:
    _require(config.get("schema_version") == 1, "schema_version must equal 1")
    _require(config.get("analysis_id") == ANALYSIS_ID, "analysis_id changed")
    _require(config.get("status") == "frozen_discovery_matrix", "matrix is not frozen")

    authority = config.get("authority", {})
    _require(authority.get("optimizer_steps_authorized") == 0, "matrix authorizes training")
    _require(authority.get("public_test_raw_access_authorized") is False, "Public Test raw access opened")
    _require(authority.get("public_test_rerun_authorized") is False, "Public Test rerun opened")
    _require(authority.get("checkpoint_mutation_authorized") is False, "checkpoint mutation opened")

    comparability = config.get("comparability", {})
    _require(
        comparability.get("cross_family_absolute_metric_comparison") == "forbidden",
        "cross-family absolute metric comparison must be forbidden",
    )
    _require(
        comparability.get("universal_numeric_threshold") == "forbidden",
        "universal numeric threshold must be forbidden",
    )
    _require(
        comparability.get("allowed_basis")
        == ["within_cell_relative_delta", "paired_uncertainty", "intervention_direction_consistency"],
        "allowed comparison basis changed",
    )

    support = config.get("support_contract", {})
    _require(
        support.get("primary_behavior_evidence") == "on_support_correct_vs_swapped_g_swap",
        "primary behavior evidence must be on-support G_swap",
    )
    _require(
        support.get("removed_history") == "auxiliary_off_support_only",
        "removed-history boundary changed",
    )

    sources = config.get("sources")
    _require(isinstance(sources, list) and sources, "sources must be non-empty")
    source_rows: dict[str, Mapping[str, Any]] = {}
    payloads: dict[str, Any] = {}
    source_inventory: list[dict[str, Any]] = []
    for row in sources:
        _require(isinstance(row, Mapping), "source row must be a mapping")
        source_id = str(row.get("source_id", ""))
        _require(source_id and source_id not in source_rows, f"duplicate source_id: {source_id}")
        kind = row.get("kind")
        _require(kind in SOURCE_KINDS, f"{source_id}: unsupported source kind {kind!r}")
        root_cause_eligible = bool(row.get("root_cause_eligible"))
        _require(
            root_cause_eligible == (kind in ROOT_CAUSE_SOURCE_KINDS),
            f"{source_id}: root_cause_eligible conflicts with source kind",
        )
        path = _resolve_source(row, root=root, contextworld_root=contextworld_root)
        normalized = "/" + str(row.get("path", "")).lower().strip("/")
        if kind == "historical_published_outcome":
            _require(
                not any(token in normalized for token in PUBLIC_RAW_TOKENS),
                f"{source_id}: historical outcome points at raw Public Test material",
            )
        else:
            _require(
                "public" not in normalized and "validation.lance" not in normalized,
                f"{source_id}: root-cause source crosses Public Test boundary",
            )
        _require(path.is_file(), f"{source_id}: missing source {path}")
        observed_sha = _sha256(path)
        _require(observed_sha == row.get("sha256"), f"{source_id}: sha256 changed")
        source_rows[source_id] = row
        payloads[source_id] = _load_payload(path)
        source_inventory.append(
            {
                "source_id": source_id,
                "kind": kind,
                "repository": row["repository"],
                "path": row["path"],
                "sha256": observed_sha,
                "root_cause_eligible": root_cause_eligible,
            }
        )

    cells = config.get("cells")
    _require(isinstance(cells, list) and cells, "cells must be non-empty")
    seen_cells: set[str] = set()
    cell_inventory: list[dict[str, Any]] = []
    role_members: dict[str, list[str]] = {}
    for cell in cells:
        _require(isinstance(cell, Mapping), "cell row must be a mapping")
        cell_id = str(cell.get("cell_id", ""))
        _require(cell_id and cell_id not in seen_cells, f"duplicate cell_id: {cell_id}")
        seen_cells.add(cell_id)
        _require(cell.get("observed_icl_outcome") in OUTCOMES, f"{cell_id}: invalid outcome")
        _require(cell.get("evidence_tier") in EVIDENCE_TIERS, f"{cell_id}: invalid evidence tier")
        _require(cell.get("task") and cell.get("model_family") and cell.get("recipe"), f"{cell_id}: incomplete axes")
        role = str(cell.get("matrix_role", ""))
        _require(role, f"{cell_id}: missing matrix role")
        role_members.setdefault(role, []).append(cell_id)

        outcome_receipt = _verify_outcome(cell, payloads)
        layers = cell.get("layers")
        _require(isinstance(layers, Mapping), f"{cell_id}: layers must be a mapping")
        _require(set(layers) == set(LAYERS), f"{cell_id}: layers must equal {LAYERS}")
        missing: list[str] = []
        available: list[str] = []
        layer_receipts: dict[str, Any] = {}
        for layer in LAYERS:
            layer_row = layers[layer]
            _require(isinstance(layer_row, Mapping), f"{cell_id}/{layer}: invalid layer row")
            status = layer_row.get("status")
            _require(status in LAYER_STATUSES, f"{cell_id}/{layer}: invalid status")
            source_ids = layer_row.get("sources", [])
            _require(isinstance(source_ids, list), f"{cell_id}/{layer}: sources must be a list")
            _require(all(source_id in source_rows for source_id in source_ids), f"{cell_id}/{layer}: unknown source")
            if status in {"available", "partial"}:
                _require(source_ids, f"{cell_id}/{layer}: evidence status without source")
                _require(
                    all(
                        source_rows[source_id]["root_cause_eligible"]
                        for source_id in source_ids
                    ),
                    f"{cell_id}/{layer}: outcome-only source entered a root-cause layer",
                )
                available.append(layer)
            elif status == "missing":
                _require(not source_ids, f"{cell_id}/{layer}: missing layer has sources")
                missing.append(layer)
            if bool(layer_row.get("primary")):
                _require(layer == "behavior", f"{cell_id}/{layer}: only behavior can be primary")
                _require(status == "available", f"{cell_id}: primary behavior must be available")
                primary_kinds = {
                    source_rows[source_id]["kind"] for source_id in source_ids
                }
                _require(
                    "development_on_support" in primary_kinds,
                    f"{cell_id}: primary behavior must include on-support Development evidence",
                )
                _require(
                    primary_kinds
                    <= {"development_on_support", "development_intervention"},
                    f"{cell_id}: primary behavior contains an ineligible source kind",
                )
            _require(
                not any(source_rows[source_id]["kind"] == "off_support_auxiliary" for source_id in source_ids),
                f"{cell_id}/{layer}: off-support evidence cannot enter the matrix",
            )
            layer_receipts[layer] = {
                "status": status,
                "sources": source_ids,
                "root_cause_eligible_sources": [
                    source_id for source_id in source_ids if source_rows[source_id]["root_cause_eligible"]
                ],
            }

        cell_inventory.append(
            {
                "cell_id": cell_id,
                "task": cell["task"],
                "model_family": cell["model_family"],
                "recipe": cell["recipe"],
                "matrix_role": role,
                "observed_icl_outcome": cell["observed_icl_outcome"],
                "evidence_tier": cell["evidence_tier"],
                "binding_cause_status": cell.get("binding_cause_status", "unresolved"),
                "outcome_verification": outcome_receipt,
                "layers": layer_receipts,
                "available_or_partial_layers": available,
                "missing_layers": missing,
                "four_layer_complete": not missing and all(
                    layer_receipts[layer]["status"] == "available" for layer in LAYERS
                ),
            }
        )

    required_roles = set(config.get("required_matrix_roles", []))
    _require(required_roles, "required_matrix_roles must be non-empty")
    _require(required_roles <= set(role_members), "one or more required matrix roles are absent")

    completed_cells = [row["cell_id"] for row in cell_inventory if row["four_layer_complete"]]
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "passed_read_only_matrix_inventory",
        "claim_status": {
            "universal_native_vulnerability": "analytic_identity_established",
            "single_low_rho_phys_universal_root_cause": "not_supported",
            "effective_conditional_visibility_general_mechanism": (
                "evidence_inventory_only_pending_matched_interventions"
            ),
            "data_distribution_universal_binding_cause": "not_supported",
        },
        "matrix_readiness": {
            "outcome_labels_verified": True,
            "four_layer_complete_cells": completed_cells,
            "four_layer_complete_cell_count": len(completed_cells),
            "all_cells_four_layer_complete": len(completed_cells) == len(cell_inventory),
            "required_roles": sorted(required_roles),
            "role_members": role_members,
        },
        "sources": source_inventory,
        "cells": cell_inventory,
        "authority": authority,
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = validate_and_inventory(config)
    if args.check_only:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        _write_exclusive(args.output, report)
        print(args.output)


if __name__ == "__main__":
    main()
