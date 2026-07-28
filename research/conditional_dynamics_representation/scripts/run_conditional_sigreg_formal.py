#!/usr/bin/env python3
"""Run paired replay50 controls through ContextWorld's formal trainer.

This file is a narrow, read-only overlay on the existing ContextWorld
History-3 runner.  It deliberately reuses that runner's data audits, DDP
topology, optimizer schedule, checkpointing, loss trace, and external logger.
Only two experiment-controlled boundaries change:

1. every train microbatch contains 50% original samples and 50% adjacent
   passage pairs with the same visible condition/action sequence;
2. ``conditional_sigreg`` evaluates the native SIGReg statistic on the
   active pair contrasts, whereas ``paired_native`` keeps native SIGReg.

The overlay is kept in StableWorldModel so the already-modified ContextWorld
checkout does not need to be rewritten merely to test this objective.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Sequence

import torch


DATA_STABLEWM_COMMIT = "5864b74980f6ed328fd0045e777b3865962eff43"
PAIRING_PROTOCOL = "visible_condition_replay50_adjacent_v1"
PAIR_BATCH_MARKER = "__visible_condition_paired_batch__"
_CONDITIONAL_REGULARIZER: torch.nn.Module | None = None


def _fetch_many(dataset: Any, indices: list[int]) -> list[Any]:
    getter = getattr(dataset, "__getitems__", None)
    values = (
        list(getter(indices))
        if getter is not None
        else [dataset[index] for index in indices]
    )
    if len(values) != len(indices):
        raise RuntimeError(
            "Batched child read returned an unexpected sample count: "
            f"requested={len(indices)}, returned={len(values)}"
        )
    return values


class VisibleConditionReplay50Dataset:
    """Make every full microbatch replay-balanced and condition-paired.

    The underlying formal passage group is factor-balanced as
    ``blocked_0, passable_0, blocked_1, passable_1, ...``.  The equality
    contract for those adjacent records is independently checked at the
    model-visible tensor boundary on every train batch.  The rule value is
    never sent to the objective, and the active time mask is derived only
    from pixel inequality.

    Non-full reads retain the native logical mapping.  This preserves the
    ContextWorld sample-contract audit, while DataLoader's batched read path
    receives the paired mapping used for optimization.
    """

    def __init__(self, dataset: Any, *, batch_size: int) -> None:
        if batch_size <= 0 or batch_size % 4:
            raise ValueError(
                "paired replay batch_size must be a positive multiple of 4"
            )
        required_names = {"original", "passage_mixed"}
        names = list(getattr(dataset, "names", []))
        if set(names) != required_names:
            raise ValueError(
                "paired replay requires exactly original and passage_mixed; "
                f"observed={names}"
            )
        weights = dict(getattr(dataset, "normalized_weights", {}))
        if weights != {"original": 0.5, "passage_mixed": 0.5}:
            raise ValueError(
                "paired replay requires exact 50/50 logical weights; "
                f"observed={weights}"
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.original = dataset.groups[names.index("original")]
        self.passage = dataset.groups[names.index("passage_mixed")]
        if len(self.passage) <= 0 or len(self.passage) % 2:
            raise ValueError(
                "passage_mixed must contain an even number of virtual slots"
            )
        self.passage_pair_count = len(self.passage) // 2
        self._full_batch_reads = 0

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def column_names(self) -> list[str]:
        return list(getattr(self.dataset, "column_names", []))

    def __getitem__(self, index: int):
        return self.dataset[index]

    @staticmethod
    def _pair_unit(left_token: int, right_token: int, count: int) -> int:
        # Stable integer mixing; unlike Python hash(), this is independent of
        # PYTHONHASHSEED and process identity.
        value = (
            (int(left_token) * 0x9E3779B1)
            ^ (int(right_token) * 0x85EBCA77)
        )
        return value % count

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:
        tokens = [int(index) for index in indices]
        if len(tokens) != self.batch_size:
            return _fetch_many(self.dataset, tokens)

        half = self.batch_size // 2
        original_tokens = tokens[:half]
        passage_tokens = tokens[half:]
        # A native 50/50 logical epoch exposes original occurrence
        # ``global_index // 2``.  Preserve that exact support instead of
        # widening the original-data population as a side effect of pairing.
        original_indices = [
            (token // 2) % len(self.original)
            for token in original_tokens
        ]

        pair_units: list[int] = []
        used: set[int] = set()
        orientations: list[bool] = []
        for offset in range(0, len(passage_tokens), 2):
            left_token = passage_tokens[offset]
            right_token = passage_tokens[offset + 1]
            unit = self._pair_unit(
                left_token,
                right_token,
                self.passage_pair_count,
            )
            while unit in used:
                unit = (unit + 1) % self.passage_pair_count
            used.add(unit)
            pair_units.append(unit)
            orientations.append(bool(right_token & 1))

        passage_indices = []
        for unit, reverse in zip(pair_units, orientations, strict=True):
            pair = (2 * unit, 2 * unit + 1)
            if reverse:
                pair = pair[::-1]
            passage_indices.extend(pair)

        originals = _fetch_many(self.original, original_indices)
        passages = _fetch_many(self.passage, passage_indices)
        self._full_batch_reads += 1
        output = []
        for sample in originals + passages:
            value = dict(sample)
            value[PAIR_BATCH_MARKER] = True
            output.append(value)
        return output

    def audit(self) -> dict[str, Any]:
        return {
            "protocol": PAIRING_PROTOCOL,
            "batch_size": self.batch_size,
            "original_samples_per_batch": self.batch_size // 2,
            "passage_samples_per_batch": self.batch_size // 2,
            "passage_pairs_per_batch": self.batch_size // 4,
            "passage_virtual_slots": len(self.passage),
            "passage_pair_units": self.passage_pair_count,
            "original_virtual_slot_mapping": (
                "native logical occurrence global_index_div_2"
            ),
            "pair_key": (
                "same initial visible pixels plus same full action sequence"
            ),
            "pair_storage_mapping": (
                "adjacent factor-balanced virtual slots, checked from "
                "model-visible tensors in every optimization batch"
            ),
            "active_mask": "per-time exact pixel inequality",
            "uses_rule_value_in_loss": False,
            "uses_pair_id_in_loss": False,
            "full_batch_reads_in_current_process": self._full_batch_reads,
        }

    def audit_visible_batch(self) -> dict[str, Any]:
        """Read one deterministic full batch and verify its visible contract."""

        samples = self.__getitems__(list(range(self.batch_size)))
        collated = torch.utils.data.default_collate(samples)
        metadata = conditional_pair_metadata(
            collated["pixels"],
            collated["action"],
        )
        if metadata is None:
            raise RuntimeError(
                "paired replay audit did not recover condition-matched pairs"
            )
        pairs, active = metadata
        expected_pairs = self.batch_size // 4
        if pairs.size(0) != expected_pairs:
            raise RuntimeError(
                "paired replay audit recovered the wrong pair count: "
                f"{pairs.size(0)} != {expected_pairs}"
            )
        return {
            "passed": True,
            "raw_batch_keys": sorted(collated),
            "pixels_shape": list(collated["pixels"].shape),
            "actions_shape": list(collated["action"].shape),
            "pair_count": int(pairs.size(0)),
            "pair_indices_start": int(pairs.min()),
            "pair_indices_stop_exclusive": int(pairs.max()) + 1,
            "same_initial_pixels_for_every_pair": True,
            "same_full_action_sequence_for_every_pair": True,
            "active_pair_counts_by_time": [
                int(value) for value in active.sum(dim=1)
            ],
            "uses_only_pixels_and_actions": True,
        }


def conditional_pair_metadata(
    pixels: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Recover formal train pairs from the paired microbatch layout.

    Returns ``None`` for native validation batches.  A train batch fails
    closed if its declared paired half does not satisfy the visible
    conditioning contract.
    """

    if pixels.ndim != 5 or actions.ndim != 3:
        raise ValueError("conditional pairing expects batched pixels/actions")
    batch_size, time_steps = pixels.shape[:2]
    if batch_size <= 0 or batch_size % 4:
        return None
    passage_start = batch_size // 2
    passage_count = batch_size - passage_start
    pair_pixels = pixels[passage_start:].reshape(
        passage_count // 2,
        2,
        *pixels.shape[1:],
    )
    pair_actions = torch.nan_to_num(
        actions[passage_start:],
        0.0,
    ).reshape(
        passage_count // 2,
        2,
        *actions.shape[1:],
    )
    same_initial_pixels = torch.eq(
        pair_pixels[:, 0, 0],
        pair_pixels[:, 1, 0],
    ).flatten(1).all(dim=1)
    same_actions = torch.eq(
        pair_actions[:, 0],
        pair_actions[:, 1],
    ).flatten(1).all(dim=1)
    if not bool((same_initial_pixels & same_actions).all()):
        return None

    active = torch.ne(
        pair_pixels[:, 0],
        pair_pixels[:, 1],
    ).flatten(2).any(dim=-1).transpose(0, 1)
    if tuple(active.shape) != (time_steps, passage_count // 2):
        raise RuntimeError("conditional active mask has an invalid shape")
    if not bool(active.any()):
        raise RuntimeError("condition-matched pairs contain no distinct frame")

    pairs = torch.arange(
        passage_start,
        batch_size,
        device=pixels.device,
        dtype=torch.long,
    ).reshape(-1, 2)
    return pairs, active


def _conditional_lewm_forward(self, batch, stage, cfg):
    """LeWM forward with one 0.09-weight conditional SIGReg statistic."""

    global _CONDITIONAL_REGULARIZER

    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    if any(
        bool(cfg.loss.get(name).enabled)
        for name in ("std", "std_t", "cov", "cov_t")
        if cfg.loss.get(name) is not None
    ):
        raise ValueError(
            "formal conditional SIGReg must not include VCReg components"
        )

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    pairs = batch.pop("conditional_pairs", None)
    active = batch.pop("conditional_active", None)
    if (pairs is None) != (active is None):
        raise RuntimeError("incomplete conditional pair metadata")

    output = self.model.encode(batch)
    embeddings = output["emb"]
    action_embeddings = output["act_emb"]
    prediction = self.model.predict(
        embeddings[:, :ctx_len],
        action_embeddings[:, :ctx_len],
    )
    target = embeddings[:, n_preds:]
    output["pred_loss"] = (prediction - target).pow(2).mean()

    if _CONDITIONAL_REGULARIZER is None:
        # Import only after ContextWorld has applied each rank's CPU-affinity
        # contract and pinned the requested StableWorldModel checkout.
        from stable_worldmodel.wm.loss import ConditionalSIGReg

        _CONDITIONAL_REGULARIZER = ConditionalSIGReg(
            **dict(cfg.loss.sigreg.kwargs)
        )
    _CONDITIONAL_REGULARIZER = _CONDITIONAL_REGULARIZER.to(
        embeddings.device
    )
    output["conditional_sigreg_loss"] = _CONDITIONAL_REGULARIZER(
        embeddings.transpose(0, 1),
        pairs=pairs,
        active=active,
    )
    output["loss"] = (
        output["pred_loss"]
        + float(cfg.loss.sigreg.weight)
        * output["conditional_sigreg_loss"]
    )
    losses = {
        f"{stage}/{key}": value.detach()
        for key, value in output.items()
        if "loss" in key
    }
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


def _load_contextworld_train(
    contextworld_repo: Path,
) -> ModuleType:
    contextworld_repo = contextworld_repo.resolve()
    if str(contextworld_repo) not in sys.path:
        sys.path.insert(0, str(contextworld_repo))
    path = contextworld_repo / "scripts/train_tworoom_step1.py"
    specification = importlib.util.spec_from_file_location(
        "contextworld_conditional_sigreg_train",
        path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load ContextWorld trainer from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def install_overlay(
    train_module: ModuleType,
    *,
    objective: str,
    batch_size: int,
) -> None:
    if objective not in {"paired_native", "conditional_sigreg"}:
        raise ValueError(f"unsupported paired objective: {objective}")

    native_build = train_module.build_tworoom_grouped_data

    def build_paired_data(*args, **kwargs):
        # The immutable synthesis artifacts retain their original code pin;
        # the objective implementation is separately pinned by the runtime
        # StableWorldModel commit and source hashes in the training report.
        kwargs["expected_stablewm_commit"] = DATA_STABLEWM_COMMIT
        grouped = native_build(*args, **kwargs)
        paired = VisibleConditionReplay50Dataset(
            grouped.train,
            batch_size=batch_size,
        )
        grouped.train = paired
        visible_batch_audit = paired.audit_visible_batch()
        grouped.metadata["paired_batch_execution"] = {
            **paired.audit(),
            "visible_batch_audit": visible_batch_audit,
        }
        return grouped

    train_module.build_tworoom_grouped_data = build_paired_data

    if objective == "paired_native":
        return

    native_project = train_module._project_lewm_model_batch

    def project_conditional_batch(batch, *, sequence_steps=4):
        declared_paired_batch = batch.get(PAIR_BATCH_MARKER)
        visible = native_project(
            batch,
            sequence_steps=sequence_steps,
        )
        if declared_paired_batch is None:
            return visible
        if not torch.is_tensor(declared_paired_batch) or not bool(
            declared_paired_batch.all()
        ):
            raise RuntimeError("invalid paired-train-batch marker")
        metadata = conditional_pair_metadata(
            visible["pixels"],
            visible["action"],
        )
        if metadata is None:
            raise RuntimeError(
                "declared train batch violated the visible pair contract"
            )
        pairs, active = metadata
        visible["conditional_pairs"] = pairs
        visible["conditional_active"] = active
        return visible

    train_module._project_lewm_model_batch = project_conditional_batch

    native_load_train = train_module._load_pinned_train_module

    def load_train_overlay(stable_repo, training_method):
        pinned = native_load_train(stable_repo, training_method)
        if training_method != "lewm":
            raise ValueError("conditional SIGReg overlay requires LeWM")
        return SimpleNamespace(lejepa_forward=_conditional_lewm_forward)

    train_module._load_pinned_train_module = load_train_overlay

    native_objective_spec = train_module._training_objective_spec

    def conditional_objective_spec(training_method, cfg):
        specification = native_objective_spec(training_method, cfg)
        if training_method != "lewm":
            raise ValueError("conditional SIGReg overlay requires LeWM")
        specification.update(
            {
                "name": "lewm_conditional_sigreg_0p09",
                "representation_regularizer": "conditional_sigreg",
                "regularizer_weight": float(cfg.loss.sigreg.weight),
                "regularizer_kwargs": dict(cfg.loss.sigreg.kwargs),
                "sigreg_weight": 0.0,
                "conditional_sigreg_weight": float(
                    cfg.loss.sigreg.weight
                ),
                "conditional_pairing_protocol": PAIRING_PROTOCOL,
                "single_scalar_regularizer": True,
                "extra_vcreg_components": False,
            }
        )
        return specification

    train_module._training_objective_spec = conditional_objective_spec


def _parse_overlay_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--paired-objective",
        required=True,
        choices=("paired_native", "conditional_sigreg"),
    )
    parser.add_argument(
        "--contextworld-repo",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--paired-batch-size",
        type=int,
        default=128,
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    overlay_args, context_args = _parse_overlay_args(argv)
    train_module = _load_contextworld_train(
        overlay_args.contextworld_repo
    )
    install_overlay(
        train_module,
        objective=overlay_args.paired_objective,
        batch_size=overlay_args.paired_batch_size,
    )

    prior_argv = sys.argv
    try:
        sys.argv = [prior_argv[0], *context_args]
        args = train_module.parse_args()
    finally:
        sys.argv = prior_argv
    if int(args.batch_size) != int(overlay_args.paired_batch_size):
        raise ValueError(
            "ContextWorld and paired overlay batch sizes differ: "
            f"context={args.batch_size}, "
            f"overlay={overlay_args.paired_batch_size}"
        )
    if args.lewm_regularizer != "sigreg":
        raise ValueError(
            "overlay CLI must retain the base runner's native sigreg setting"
        )
    if float(args.lewm_sigreg_weight) != 0.09:
        raise ValueError("conditional formal screen freezes SIGReg weight 0.09")

    result = train_module.run(args)
    if train_module._process_is_global_zero():
        summary = {
            "passed": result["passed"],
            "run_kind": result["run_kind"],
            "paired_objective": overlay_args.paired_objective,
        }
        if "training" in result:
            summary.update(
                {
                    "global_step": result["training"]["global_step"],
                    "pretrained": result["artifacts"]["pretrained"],
                }
            )
        else:
            summary["training_plan"] = result["training_plan"]
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
