#!/usr/bin/env python3
"""Localize LeWM contraction with read-only checkpoint module swaps.

The diagnostic never writes a checkpoint.  It loads the original History-3
LeWM once, splices selected tensors from a continued-training checkpoint into
the in-memory model, and evaluates the frozen eight-query door-rule batch.

Two factorials answer different questions:

1. Representation factorial:
   Encoder parameters, Projector parameters, and Projector BatchNorm buffers
   are independently selected from the original or trained checkpoint.
2. Dynamics factorial:
   For either the original or trained representation, Predictor,
   ActionEncoder, PredProj parameters, and PredProj BatchNorm buffers are
   independently selected.  Cached history embeddings keep this factorial
   cheap and ensure that only the named dynamics modules change.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import analyze_checkpoint_geometry as geometry


ORIGINAL = "original"
TRAINED = "trained"
SOURCES = (ORIGINAL, TRAINED)
STATE_GROUPS = (
    "encoder_parameters",
    "projector_parameters",
    "projector_buffers",
    "predictor_parameters",
    "action_encoder_parameters",
    "pred_proj_parameters",
    "pred_proj_buffers",
)
REPRESENTATION_GROUPS = (
    "encoder_parameters",
    "projector_parameters",
    "projector_buffers",
)
DYNAMICS_GROUPS = (
    "predictor_parameters",
    "action_encoder_parameters",
    "pred_proj_parameters",
    "pred_proj_buffers",
)


def classify_state_key(name: str, parameter_names: set[str]) -> str:
    """Map a model state key to the independently swappable group."""

    if name.startswith("encoder."):
        return "encoder_parameters"
    if name.startswith("projector."):
        return (
            "projector_parameters"
            if name in parameter_names
            else "projector_buffers"
        )
    if name.startswith("predictor."):
        return "predictor_parameters"
    if name.startswith("action_encoder."):
        return "action_encoder_parameters"
    if name.startswith("pred_proj."):
        return (
            "pred_proj_parameters"
            if name in parameter_names
            else "pred_proj_buffers"
        )
    return "unclassified"


def validate_compatible_states(
    original_state: Mapping[str, Any],
    trained_state: Mapping[str, Any],
) -> None:
    original_keys = set(original_state)
    trained_keys = set(trained_state)
    if original_keys != trained_keys:
        raise ValueError(
            "Checkpoint state keys differ: "
            f"missing={sorted(original_keys - trained_keys)}, "
            f"extra={sorted(trained_keys - original_keys)}"
        )
    mismatched = [
        name
        for name in original_state
        if (
            tuple(original_state[name].shape)
            != tuple(trained_state[name].shape)
            or original_state[name].dtype != trained_state[name].dtype
        )
    ]
    if mismatched:
        raise ValueError(
            "Checkpoint tensor metadata differs for: "
            + ", ".join(mismatched)
        )


def splice_state(
    *,
    original_state: Mapping[str, Any],
    trained_state: Mapping[str, Any],
    parameter_names: set[str],
    sources: Mapping[str, str],
) -> OrderedDict[str, Any]:
    """Construct an in-memory hybrid state without modifying either source."""

    unknown_groups = set(sources) - set(STATE_GROUPS)
    if unknown_groups:
        raise ValueError(f"Unknown state groups: {sorted(unknown_groups)}")
    invalid_sources = set(sources.values()) - set(SOURCES)
    if invalid_sources:
        raise ValueError(f"Unknown checkpoint sources: {sorted(invalid_sources)}")

    output: OrderedDict[str, Any] = OrderedDict()
    for name, original_value in original_state.items():
        group = classify_state_key(name, parameter_names)
        source = sources.get(group, ORIGINAL)
        output[name] = (
            trained_state[name] if source == TRAINED else original_value
        )
    return output


def source_map(**overrides: str) -> dict[str, str]:
    output = {group: ORIGINAL for group in STATE_GROUPS}
    output.update(overrides)
    return output


def source_code(sources: Mapping[str, str], groups: tuple[str, ...]) -> str:
    return "".join("T" if sources[group] == TRAINED else "O" for group in groups)


def state_group_counts(
    state: Mapping[str, Any], parameter_names: set[str]
) -> dict[str, int]:
    counts = {group: 0 for group in (*STATE_GROUPS, "unclassified")}
    for name in state:
        counts[classify_state_key(name, parameter_names)] += 1
    return counts


def load_hybrid(
    adapter: Any,
    *,
    original_state: Mapping[str, Any],
    trained_state: Mapping[str, Any],
    parameter_names: set[str],
    sources: Mapping[str, str],
) -> None:
    state = splice_state(
        original_state=original_state,
        trained_state=trained_state,
        parameter_names=parameter_names,
        sources=sources,
    )
    incompatible = adapter.model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict hybrid load failed: {incompatible}")
    adapter.model.eval()


def representation_cache(
    adapter: Any,
    batch: geometry.FrozenBatch,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Encode all frozen targets and histories in one representation state."""

    images = [
        batch.targets["passable"],
        batch.targets["blocked"],
    ]
    for condition in geometry.HISTORY_CONDITIONS:
        history = batch.histories[condition]
        images.append(history.reshape(-1, *history.shape[2:]))
    raw, projected = geometry.encode_raw_and_projected(
        adapter,
        np.concatenate(images, axis=0),
        batch_size=batch_size,
    )

    query_count = len(batch.query_ids)
    target_count = 2 * query_count
    raw_targets = {
        "passable": raw[:query_count],
        "blocked": raw[query_count:target_count],
    }
    projected_targets = {
        "passable": projected[:query_count],
        "blocked": projected[query_count:target_count],
    }
    projected_histories: dict[str, np.ndarray] = {}
    offset = target_count
    for condition in geometry.HISTORY_CONDITIONS:
        history = batch.histories[condition]
        count = int(history.shape[0] * history.shape[1])
        projected_histories[condition] = projected[
            offset : offset + count
        ].reshape(history.shape[0], history.shape[1], -1)
        offset += count
    if offset != len(projected):
        raise RuntimeError("Frozen representation cache split is inconsistent")
    return {
        "raw_targets": raw_targets,
        "projected_targets": projected_targets,
        "projected_histories": projected_histories,
    }


def target_geometry(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_encoder": geometry.paired_and_unrelated_mse(
            cache["raw_targets"]["passable"],
            cache["raw_targets"]["blocked"],
        ),
        "prediction_space": geometry.paired_and_unrelated_mse(
            cache["projected_targets"]["passable"],
            cache["projected_targets"]["blocked"],
        ),
    }


def normalized_actions(
    adapter: Any,
    batch: geometry.FrozenBatch,
) -> dict[str, np.ndarray]:
    return {
        condition: adapter._normalize_actions(batch.actions[condition])
        for condition in geometry.HISTORY_CONDITIONS
    }


def predict_from_cached_histories(
    adapter: Any,
    histories: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run only ActionEncoder + Predictor + PredProj on cached embeddings."""

    import torch

    output: dict[str, np.ndarray] = {}
    dtype = next(adapter.model.parameters()).dtype
    with torch.inference_mode():
        for condition in geometry.HISTORY_CONDITIONS:
            history_values = np.asarray(histories[condition], dtype=np.float32)
            action_values = np.asarray(actions[condition], dtype=np.float32)
            if history_values.shape[:2] != action_values.shape[:2]:
                raise ValueError(
                    f"History/action token mismatch for {condition}: "
                    f"{history_values.shape} vs {action_values.shape}"
                )
            chunks: list[np.ndarray] = []
            for start in range(0, len(history_values), batch_size):
                embeddings = torch.from_numpy(
                    history_values[start : start + batch_size]
                ).to(device=adapter.device, dtype=dtype)
                action_tensor = torch.from_numpy(
                    action_values[start : start + batch_size]
                ).to(device=adapter.device, dtype=dtype)
                action_embeddings = adapter.model.action_encoder(action_tensor)
                prediction = adapter.model.predict(
                    embeddings, action_embeddings
                )[:, -1]
                chunks.append(prediction.detach().float().cpu().numpy())
            output[condition] = np.concatenate(chunks, axis=0)
    return output


def representation_factorial(
    adapter: Any,
    *,
    batch: geometry.FrozenBatch,
    batch_size: int,
    original_state: Mapping[str, Any],
    trained_state: Mapping[str, Any],
    parameter_names: set[str],
    original_cache: Mapping[str, Any],
    trained_cache: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected in itertools.product(SOURCES, repeat=len(REPRESENTATION_GROUPS)):
        sources = source_map(**dict(zip(REPRESENTATION_GROUPS, selected)))
        code = source_code(sources, REPRESENTATION_GROUPS)
        if code == "OOO":
            cache = original_cache
        elif code == "TTT":
            cache = trained_cache
        else:
            load_hybrid(
                adapter,
                original_state=original_state,
                trained_state=trained_state,
                parameter_names=parameter_names,
                sources=sources,
            )
            cache = representation_cache_targets_only(
                adapter, batch, batch_size=batch_size
            )
        rows.append(
            {
                "case": code,
                "sources": {
                    group: sources[group] for group in REPRESENTATION_GROUPS
                },
                "target_geometry": target_geometry(cache),
            }
        )
    return rows


def representation_cache_targets_only(
    adapter: Any,
    batch: geometry.FrozenBatch,
    *,
    batch_size: int,
) -> dict[str, Any]:
    images = np.concatenate(
        [batch.targets["passable"], batch.targets["blocked"]], axis=0
    )
    raw, projected = geometry.encode_raw_and_projected(
        adapter, images, batch_size=batch_size
    )
    count = len(batch.query_ids)
    return {
        "raw_targets": {
            "passable": raw[:count],
            "blocked": raw[count:],
        },
        "projected_targets": {
            "passable": projected[:count],
            "blocked": projected[count:],
        },
    }


def dynamics_factorial(
    adapter: Any,
    *,
    batch_size: int,
    original_state: Mapping[str, Any],
    trained_state: Mapping[str, Any],
    parameter_names: set[str],
    representation_caches: Mapping[str, Mapping[str, Any]],
    actions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation_source in SOURCES:
        cache = representation_caches[representation_source]
        representation_overrides = {
            group: representation_source for group in REPRESENTATION_GROUPS
        }
        for selected in itertools.product(SOURCES, repeat=len(DYNAMICS_GROUPS)):
            overrides = {
                **representation_overrides,
                **dict(zip(DYNAMICS_GROUPS, selected)),
            }
            sources = source_map(**overrides)
            load_hybrid(
                adapter,
                original_state=original_state,
                trained_state=trained_state,
                parameter_names=parameter_names,
                sources=sources,
            )
            predictions = predict_from_cached_histories(
                adapter,
                cache["projected_histories"],
                actions,
                batch_size=batch_size,
            )
            targets = cache["projected_targets"]
            rows.append(
                {
                    "case": (
                        f"R={representation_source[0].upper()};"
                        f"D={source_code(sources, DYNAMICS_GROUPS)}"
                    ),
                    "representation_source": representation_source,
                    "sources": {
                        group: sources[group] for group in DYNAMICS_GROUPS
                    },
                    "prediction": geometry.prediction_summary(
                        predicted_passable_history=predictions[
                            "observed_passable"
                        ],
                        predicted_blocked_history=predictions[
                            "observed_blocked"
                        ],
                        predicted_no_attempt_history=predictions[
                            "did_not_attempt_crossing"
                        ],
                        target_passable=targets["passable"],
                        target_blocked=targets["blocked"],
                    ),
                }
            )
    return rows


def analyze_trained_checkpoint(
    adapter: Any,
    *,
    label: str,
    path: Path,
    batch: geometry.FrozenBatch,
    batch_size: int,
    original_state: Mapping[str, Any],
    parameter_names: set[str],
    original_cache: Mapping[str, Any],
    actions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    trained_state = geometry._load_state(path)
    validate_compatible_states(original_state, trained_state)

    all_trained_representation = source_map(
        **{group: TRAINED for group in REPRESENTATION_GROUPS}
    )
    load_hybrid(
        adapter,
        original_state=original_state,
        trained_state=trained_state,
        parameter_names=parameter_names,
        sources=all_trained_representation,
    )
    trained_cache = representation_cache(
        adapter, batch, batch_size=batch_size
    )
    representation_caches = {
        ORIGINAL: original_cache,
        TRAINED: trained_cache,
    }

    return {
        "label": label,
        "trained_checkpoint": str(path),
        "trained_checkpoint_sha256": geometry.file_sha256(path),
        "representation_factorial_group_order": list(REPRESENTATION_GROUPS),
        "representation_factorial": representation_factorial(
            adapter,
            batch=batch,
            batch_size=batch_size,
            original_state=original_state,
            trained_state=trained_state,
            parameter_names=parameter_names,
            original_cache=original_cache,
            trained_cache=trained_cache,
        ),
        "dynamics_factorial_group_order": list(DYNAMICS_GROUPS),
        "dynamics_factorial": dynamics_factorial(
            adapter,
            batch_size=batch_size,
            original_state=original_state,
            trained_state=trained_state,
            parameter_names=parameter_names,
            representation_caches=representation_caches,
            actions=actions,
        ),
    }


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "--trained-checkpoint must be LABEL=/path/to/weights.pt"
        )
    return label.strip(), Path(raw_path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextworld-repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--stable-repo", type=Path, default=None)
    parser.add_argument("--stable-ref", default=geometry.STABLE_COMMIT)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--normalizer", type=Path, default=None)
    parser.add_argument("--original-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--trained-checkpoint",
        action="append",
        type=parse_labeled_path,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contextworld_repo = args.contextworld_repo.expanduser().resolve()
    artifact_root = (
        args.artifact_root.expanduser().resolve()
        if args.artifact_root is not None
        else (
            contextworld_repo.parents[1]
            / "data/world_model/context_world"
        ).resolve()
    )
    stable_repo = (
        args.stable_repo.expanduser().resolve()
        if args.stable_repo is not None
        else (contextworld_repo.parent / "stable-worldmodel").resolve()
    )
    catalog = (
        args.catalog.expanduser().resolve()
        if args.catalog is not None
        else artifact_root
        / "evaluation/history3/"
        "hidden_passage_frozen_representation_diagnostic_v1/catalog.json"
    )
    normalizer = (
        args.normalizer.expanduser().resolve()
        if args.normalizer is not None
        else artifact_root
        / "splits/tworoom_original_train_s3072_normalizer.json"
    )
    original_checkpoint = (
        args.original_checkpoint.expanduser().resolve()
        if args.original_checkpoint is not None
        else artifact_root
        / "training/runs/checkpoints/h3_origheldout_s3072/"
        "weights_final_step_6420.pt"
    )
    trained_checkpoints = [
        (label, path) for label, path in args.trained_checkpoint
    ]
    required_paths = [
        contextworld_repo,
        stable_repo,
        catalog,
        normalizer,
        original_checkpoint,
        *(path for _, path in trained_checkpoints),
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    batch = geometry.load_frozen_batch(catalog, artifact_root)
    adapter = geometry._stable_adapter(
        checkpoint=original_checkpoint,
        training_method="lewm",
        contextworld_repo=contextworld_repo,
        stable_repo=stable_repo,
        stable_ref=args.stable_ref,
        normalizer=normalizer,
        device=args.device,
    )
    original_state = geometry._load_state(original_checkpoint)
    parameter_names = geometry._parameter_name_set(adapter)
    validate_compatible_states(adapter.model.state_dict(), original_state)
    original_cache = representation_cache(
        adapter, batch, batch_size=args.batch_size
    )
    actions = normalized_actions(adapter, batch)
    original_hash_before = adapter.frozen_state_hash()

    results = []
    for index, (label, path) in enumerate(trained_checkpoints, start=1):
        print(
            f"[{index}/{len(trained_checkpoints)}] module swaps for "
            f"{label}: {path}",
            flush=True,
        )
        results.append(
            analyze_trained_checkpoint(
                adapter,
                label=label,
                path=path,
                batch=batch,
                batch_size=args.batch_size,
                original_state=original_state,
                parameter_names=parameter_names,
                original_cache=original_cache,
                actions=actions,
            )
        )

    payload = {
        "schema_version": 1,
        "status": "retrospective_read_only_module_swap_diagnostic",
        "question": (
            "LeWM 联合训练中，门规则相关表示收缩和历史切换失败分别由"
            " Encoder、Projector/BN 还是 Predictor 路径的哪一部分携带？"
        ),
        "provenance": {
            "contextworld_repo": str(contextworld_repo),
            "stable_worldmodel_repo": str(stable_repo),
            "stable_worldmodel_commit": args.stable_ref,
            "artifact_root": str(artifact_root),
            "catalog": str(catalog),
            "catalog_sha256": geometry.file_sha256(catalog),
            "normalizer": str(normalizer),
            "normalizer_sha256": geometry.file_sha256(normalizer),
            "original_checkpoint": str(original_checkpoint),
            "original_checkpoint_sha256": geometry.file_sha256(
                original_checkpoint
            ),
            "device": args.device,
            "batch_size": args.batch_size,
            "checkpoints_modified": False,
            "environment_calls": 0,
        },
        "frozen_batch": {
            "queries": len(batch.query_ids),
            "history_conditions": list(geometry.HISTORY_CONDITIONS),
        },
        "state_group_tensor_counts": state_group_counts(
            original_state, parameter_names
        ),
        "original_model_state_hash_after_baseline_encoding": (
            original_hash_before
        ),
        "trained_checkpoints": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            geometry._json_safe(payload),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
