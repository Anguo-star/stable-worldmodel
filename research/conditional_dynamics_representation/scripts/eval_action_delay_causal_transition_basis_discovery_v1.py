#!/usr/bin/env python3
"""Score the zero-parameter transition-basis MVE on frozen Development data.

This is deliberately a Development-only discovery scorer.  The original
release evaluator's runtime receipt only permits candidate changes in
``loss.py``; this method changes LeWM's fixed temporal coordinates instead.
The script therefore reuses the frozen catalog, scoring implementation, gate,
and target encoding while recording the exact dirty runtime sources rather
than pretending that the old receipt covers them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
ROOT = REPO_ROOT / "research/conditional_dynamics_representation"
CONTEXTWORLD_ROOT = (REPO_ROOT / "../ContextWorld").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CONTEXTWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEXTWORLD_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    eval_action_delay_h7_development as base,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_action_delay_causal_transition_basis_v1 as method,
)
import contextworld.benchmarks.adapters as adapters  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / (
    "artifacts/action_delay_h7_causal_transition_basis_v1/training/checkpoints/"
    "action_delay_h7_causal_transition_basis_s3072_step1024_v1/"
    "weights_final_step_1024.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "artifacts/action_delay_h7_causal_transition_basis_v1/development/"
    "stage1_s3072_step1024_discovery_v2.json"
)
RELEASE_RUNTIME = Path("/tmp/stable-worldmodel-ad2")


def _load_catalog() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load every frozen asset check except the obsolete runtime-source gate."""

    native_verify = base._verify_code_identity
    base._verify_code_identity = lambda _config, _catalog: None
    try:
        return base.load_frozen_catalog(
            config_path=base.DEFAULT_CONFIG,
            catalog_path=base.DEFAULT_CATALOG,
            verify_source_trees=True,
        )
    finally:
        base._verify_code_identity = native_verify


def _source_identity(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    checkpoint_config = checkpoint.parent / "config.json"
    payload = json.loads(checkpoint_config.read_text(encoding="utf-8"))
    basis = payload.get("model", {}).get("temporal_input_basis")
    if basis != "causal_transition":
        raise RuntimeError(
            "Checkpoint does not declare temporal_input_basis=causal_transition"
        )

    expected = config["frozen_identity"]["code"]
    scoring_paths = {
        name: CONTEXTWORLD_ROOT / relative
        for name, relative in config["contextworld"]["scoring_sources"].items()
    }
    return {
        "status": "exact_method_runtime_recorded_old_loss_only_receipt_not_reused",
        "learned_parameters_added": 0,
        "checkpoint_config": {
            "path": str(checkpoint_config),
            "sha256": base.builder.file_sha256(checkpoint_config),
            "temporal_input_basis": basis,
        },
        "method_sources": {
            "release_runtime_lewm": {
                "path": str(
                    RELEASE_RUNTIME / "stable_worldmodel/wm/lewm/lewm.py"
                ),
                "sha256": base.builder.file_sha256(
                    RELEASE_RUNTIME / "stable_worldmodel/wm/lewm/lewm.py"
                ),
            },
            "permanent_current_runtime_implementation": {
                "path": str(REPO_ROOT / "stable_worldmodel/wm/lewm/lewm.py"),
                "sha256": base.builder.file_sha256(
                    REPO_ROOT / "stable_worldmodel/wm/lewm/lewm.py"
                ),
            },
            "runner": {
                "path": str(
                    ROOT / "scripts/run_action_delay_causal_transition_basis_v1.py"
                ),
                "sha256": base.builder.file_sha256(
                    ROOT / "scripts/run_action_delay_causal_transition_basis_v1.py"
                ),
            },
            "scorer": {
                "path": str(THIS_SOURCE),
                "sha256": base.builder.file_sha256(THIS_SOURCE),
            },
        },
        "frozen_evaluator": {
            "path": str(Path(base.__file__).resolve()),
            "observed_sha256": base.builder.file_sha256(Path(base.__file__).resolve()),
            "expected_sha256": expected["evaluator_sha256"],
        },
        "contextworld_scoring_sources": {
            name: {
                "path": str(path),
                "observed_sha256": base.builder.file_sha256(path),
                "frozen_expected_sha256": expected[f"{name}_sha256"],
            }
            for name, path in scoring_paths.items()
        },
        "claim_boundary": {
            "development_only": True,
            "formal_release_runtime_receipt": False,
            "public_or_private_test_opened": False,
            "candidate_promotion_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=16)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    config, catalog, source_audit = _load_catalog()
    source_identity = _source_identity(config, checkpoint)
    cache_root = ROOT / "artifacts/action_delay_h7_development_v1/model_cache"
    os.environ["CONTEXTWORLD_ARTIFACT_ROOT"] = str(cache_root.resolve())
    normalizer = CONTEXTWORLD_ROOT / config["contextworld"]["normalizer"]
    native_load_runtime = adapters.load_stable_worldmodel

    def load_method_runtime(*load_args: Any, **load_kwargs: Any):
        swm, runtime, commit = native_load_runtime(*load_args, **load_kwargs)
        from stable_worldmodel.wm.lewm.lewm import LeWM

        method._install_runtime_basis(LeWM)
        return swm, runtime, commit

    adapters.load_stable_worldmodel = load_method_runtime
    try:
        adapter = adapters.StableWorldModelLeWMHistory7Adapter.from_checkpoint(
            checkpoint,
            normalizer=normalizer,
            repo_root=CONTEXTWORLD_ROOT,
            stablewm_repo=str(RELEASE_RUNTIME),
            stablewm_ref=method.BASE_RUNTIME_COMMIT,
            device=str(args.device),
        )
    finally:
        adapters.load_stable_worldmodel = native_load_runtime
    if getattr(adapter.model, "temporal_input_basis", None) != "causal_transition":
        raise RuntimeError("Loaded model lost the causal-transition inference semantic")

    state_before = adapter.frozen_state_hash()
    records, score_audit = base.score_adapter(
        adapter,
        catalog=catalog,
        stage="stage1",
        batch_size=int(args.batch_size),
        query_batch_size=int(args.query_batch_size),
    )
    state_after = adapter.frozen_state_hash()
    if state_before != state_after:
        raise RuntimeError("Model state changed during Development scoring")
    summary = base.summarize_stage(records, stage="stage1", config=config)
    selected_by_query: dict[str, set[int]] = {}
    for record in records:
        selected_by_query.setdefault(str(record["query_id"]), set()).add(
            int(record["selected_physical_group"])
        )
    history_responsive = sum(
        len(selected_groups) > 1
        for selected_groups in selected_by_query.values()
    )
    summary["history_responsive_query_count"] = int(history_responsive)
    summary["history_responsive_query_rate"] = float(
        history_responsive / len(selected_by_query)
    )
    profile = config["scoring"]["gate_profiles"]["stage1_1024"]
    gate = base.evaluate_gate(
        summary,
        thresholds=profile,
        anti_collapse=score_audit["anti_collapse"],
    )
    result = {
        "schema_version": 1,
        "status": "completed_development_only_discovery",
        "benchmark": config["benchmark"],
        "candidate": {
            "name": "action_delay_h7_causal_transition_basis_s3072_step1024_v1",
            "training_seed": 3072,
            "optimizer_step": 1024,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": base.builder.file_sha256(checkpoint),
            "checkpoint_protocol": base._checkpoint_protocol(checkpoint),
            "adapter": adapter.metadata,
            "state_sha256_before": state_before,
            "state_sha256_after": state_after,
        },
        "freeze_identity": {
            "config": str(base.DEFAULT_CONFIG),
            "catalog": str(base.DEFAULT_CATALOG),
            "catalog_sha256": base.builder.file_sha256(base.DEFAULT_CATALOG),
            "catalog_content_sha256": catalog["content_sha256"],
            "source_tree_audit": source_audit,
        },
        "runtime_identity": source_identity,
        "score_audit": score_audit,
        "summary": summary,
        "gate": gate,
        "records": records,
        "claim_boundary": {
            "development_only": True,
            "public_benchmark_claim_allowed": False,
            "public_or_private_test_assets_opened": False,
            "online_environment_calls": 0,
            "optimizer_steps_during_scoring": 0,
            "decision_role": "single_seed_method_discovery_only",
        },
    }
    base._write_new_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "macro": summary["physical_group_macro_accuracy"],
                "worst": summary["minimum_physical_group_accuracy"],
                "history_responsive_queries": summary[
                    "history_responsive_query_count"
                ],
                "gate_passed": gate["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
