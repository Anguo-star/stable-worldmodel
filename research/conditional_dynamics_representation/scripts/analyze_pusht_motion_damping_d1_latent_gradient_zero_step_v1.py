#!/usr/bin/env python3
"""Audit whether D1-MS50 reaches frozen latents and native gradients.

The audit takes zero optimizer steps and opens Training data only.  It first
encodes all 4,096 Motion twins to evaluate latent conditional energy on the
physical neighbour graph frozen by D1-0.  It then compares D0 and D1 on 16
pre-fixed, schedule-spanning batches.  Both arms reuse the same original PushT
rows, initialization, model state, train mode, and RNG state; only hidden twin
indices differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD = ROOT.parent / "ContextWorld"
for source_root in (ROOT, CONTEXTWORLD, CONTEXTWORLD / "scripts"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    conditional_signal_metrics as signal_metrics,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_motion_damping_native_gradient_snr_diagnostic_v1 as native_gradient,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_d1_energy_stratified_native_v1 as d1_runner,
)


ANALYSIS_ID = "pusht_motion_damping_d1_latent_gradient_zero_step_v1"
ANALYSIS_VARIANT = "analysis_motion_d1_latent_gradient_zero_step_v1"
ALLOWED_PARAMETER_GROUPS = native_gradient.ALLOWED_PARAMETER_GROUPS
EXPECTED_TWINS = 4096
EXPECTED_PAIRS = 8192
EXPECTED_HIDDEN_ROWS = 16384
ORIGINAL_ROWS = 64
HIDDEN_ROWS_PER_BATCH = 64
TWINS_PER_BATCH = 16
AUDIT_BATCH_COUNT = 16
SCHEDULE_BATCH_COUNT = d1_runner.EXPECTED_BATCHES
AUDIT_SCHEDULE_INDICES = tuple(
    int((index + 0.5) * SCHEDULE_BATCH_COUNT / AUDIT_BATCH_COUNT)
    for index in range(AUDIT_BATCH_COUNT)
)
LATENT_NEIGHBOUR_SCALES = (32, 64, 128)
LATENT_ENCODE_BATCH_SIZE = 64
EXPECTED_CHECKPOINT_SHA256 = (
    "9f13b2c28bb909338047e08d62bf6eb16ea3616cb53e57cfa9b5072b43db1c59"
)
EXPECTED_MODEL_STATE_SHA256 = (
    "c352c343c0ef7fa0a964d77e4bf418176448c1cebaa5d080b3e870f5ad6ff3d3"
)
EXPECTED_V1_CATALOG_SHA256 = (
    "c84df85632c4f4d81728393e22ca553773e1a5992cccc79b5b798f288c5dbb99"
)
EXPECTED_MULTIPLICITY_SHA256 = d1_runner.EXPECTED_SCHEDULE_SHA256[
    "multiplicity.jsonl"
]
EXPECTED_D1_RUNNER_SHA256 = (
    "bafcd5406d70efcc2133014b287cc86f4543b5f615860fbcdf18a57b0ea7bab3"
)
DEFAULT_CHECKPOINT = native_gradient.DEFAULT_ORIGINAL_H5.parent / (
    "ckpt/pusht_lewm_baseline_seed3073/"
    "pusht_lewm_baseline_seed3073_weights.ckpt"
)
DEFAULT_CATALOG = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_metric_v1/d1_0_training_only_v1_final/"
    "per_twin_catalog.jsonl"
)
DEFAULT_MULTIPLICITY = d1_runner.DEFAULT_SCHEDULE_DIR / "multiplicity.jsonl"
DEFAULT_OUTPUT = ROOT / (
    "research/conditional_dynamics_representation/artifacts/"
    "pusht_motion_damping_d1_latent_gradient_zero_step_v1/"
    "initialization_16batch_v2_final/report.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _group(name: str) -> str:
    return name.split(".", 1)[0]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def audit_schedule_indices() -> dict[str, Any]:
    expected = tuple(256 + 512 * index for index in range(AUDIT_BATCH_COUNT))
    checks = {
        "count_exactly_16": len(AUDIT_SCHEDULE_INDICES) == AUDIT_BATCH_COUNT,
        "midpoint_formula_exact": AUDIT_SCHEDULE_INDICES == expected,
        "strictly_increasing": all(
            left < right
            for left, right in zip(
                AUDIT_SCHEDULE_INDICES, AUDIT_SCHEDULE_INDICES[1:]
            )
        ),
        "inside_schedule": (
            min(AUDIT_SCHEDULE_INDICES) >= 0
            and max(AUDIT_SCHEDULE_INDICES) < SCHEDULE_BATCH_COUNT
        ),
        "one_batch_per_two_cycle_stratum": all(
            value // 512 == index
            for index, value in enumerate(AUDIT_SCHEDULE_INDICES)
        ),
    }
    _require(all(checks.values()), f"audit batch schedule changed: {checks}")
    return {
        "indices": list(AUDIT_SCHEDULE_INDICES),
        "selection": (
            "midpoint of each of 16 equal 512-batch strata; frozen before "
            "latent or gradient evaluation"
        ),
        "checks": checks,
    }


def validate_twin_rows(rows: torch.Tensor) -> list[int]:
    values = rows.detach().cpu().to(dtype=torch.long).reshape(-1, 4)
    _require(values.shape == (TWINS_PER_BATCH, 4), "hidden batch shape changed")
    twins: list[int] = []
    for group in values.tolist():
        twin = int(group[0]) // 4
        _require(group == [4 * twin + offset for offset in range(4)], "split twin")
        _require(0 <= twin < EXPECTED_TWINS, "twin id out of range")
        twins.append(twin)
    _require(len(set(twins)) == TWINS_PER_BATCH, "duplicate twin within batch")
    return twins


def load_catalog(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    path = Path(path).expanduser().resolve()
    _require(_sha256(path) == EXPECTED_V1_CATALOG_SHA256, "D1-0 catalog changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(len(rows) == EXPECTED_TWINS, "D1-0 catalog twin count changed")
    neighbours = np.empty((EXPECTED_PAIRS, max(LATENT_NEIGHBOUR_SCALES)), dtype=np.int64)
    for twin, row in enumerate(rows):
        _require(int(row["twin_id"]) == twin, "catalog twin order changed")
        _require(row["pair_indices"] == [2 * twin, 2 * twin + 1], "pair identity changed")
        for direction_offset, direction in enumerate(("forward", "reverse")):
            values = row["neighbors"][direction]["pair_indices_k128"]
            _require(len(values) == max(LATENT_NEIGHBOUR_SCALES), "neighbour width changed")
            _require(len(set(values)) == len(values), "duplicate physical neighbour")
            _require(
                all(0 <= int(value) < EXPECTED_PAIRS for value in values),
                "physical neighbour out of range",
            )
            _require(
                2 * twin not in values and 2 * twin + 1 not in values,
                "current twin leaked into physical neighbours",
            )
            neighbours[2 * twin + direction_offset] = np.asarray(values, dtype=np.int64)
    return rows, neighbours


def load_realized_weights(path: Path) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    _require(_sha256(path) == EXPECTED_MULTIPLICITY_SHA256, "multiplicity changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(len(rows) == EXPECTED_TWINS, "multiplicity twin count changed")
    counts = np.empty(EXPECTED_TWINS, dtype=np.int64)
    for twin, row in enumerate(rows):
        _require(int(row["twin_id"]) == twin, "multiplicity twin order changed")
        counts[twin] = int(row["realized_total_count"])
    _require(np.all(counts > 0), "D1 has zero-support twin")
    _require(int(counts.sum()) == SCHEDULE_BATCH_COUNT * TWINS_PER_BATCH, "slot total changed")
    return counts.astype(np.float64) / float(counts.sum())


def latent_local_energy(
    query_latents: torch.Tensor,
    future_latents: torch.Tensor,
    neighbours: np.ndarray,
    weights: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute ratio-of-means latent energy on the frozen physical graph."""

    query = query_latents.detach().cpu().float().reshape(EXPECTED_PAIRS, 2, -1)
    future = future_latents.detach().cpu().float().reshape(EXPECTED_PAIRS, 2, -1)
    _require(bool(torch.isfinite(query).all()), "non-finite query latent")
    _require(bool(torch.isfinite(future).all()), "non-finite future latent")
    query_identity_error = float((query[:, 0] - query[:, 1]).abs().max())
    query_center = query.mean(dim=1)
    future_center = future.mean(dim=1)
    directed_mean_displacement = future_center - query_center
    directed_conditional = 0.25 * (future[:, 1] - future[:, 0]).square().mean(dim=1)
    twin_conditional = directed_conditional.reshape(EXPECTED_TWINS, 2).mean(dim=1)

    background_by_k: dict[int, torch.Tensor] = {}
    neighbour_tensor = torch.from_numpy(neighbours)
    for scale in LATENT_NEIGHBOUR_SCALES:
        values = torch.empty(EXPECTED_PAIRS, dtype=torch.float64)
        for start in range(0, EXPECTED_PAIRS, 128):
            stop = min(start + 128, EXPECTED_PAIRS)
            selected = neighbour_tensor[start:stop, :scale]
            local = directed_mean_displacement[selected]
            centered = local - local.mean(dim=1, keepdim=True)
            values[start:stop] = centered.double().square().mean(dim=(1, 2))
        background_by_k[scale] = values.reshape(EXPECTED_TWINS, 2).mean(dim=1)

    arms: dict[str, Any] = {}
    for arm, raw_weight in weights.items():
        weight = torch.from_numpy(np.asarray(raw_weight, dtype=np.float64))
        _require(weight.shape == (EXPECTED_TWINS,), f"{arm} weight shape changed")
        _require(
            bool(torch.isfinite(weight).all()) and bool((weight > 0).all()),
            f"{arm} invalid weights",
        )
        _require(abs(float(weight.sum()) - 1.0) <= 1.0e-12, f"{arm} weights do not sum to one")
        conditional = float((weight * twin_conditional.double()).sum())
        by_k = {}
        for scale in LATENT_NEIGHBOUR_SCALES:
            background = float((weight * background_by_k[scale]).sum())
            rho = _safe_ratio(conditional, conditional + background)
            _require(rho is not None, f"{arm} latent rho denominator is zero")
            by_k[str(scale)] = {
                "weighted_conditional_energy": conditional,
                "weighted_background_energy": background,
                "rho_lat": rho,
            }
        arms[arm] = {"by_k": by_k}

    comparison = {}
    for scale in LATENT_NEIGHBOUR_SCALES:
        d0 = arms["D0"]["by_k"][str(scale)]
        d1 = arms["D1-MS50"]["by_k"][str(scale)]
        comparison[str(scale)] = {
            "rho_lat_absolute_delta": d1["rho_lat"] - d0["rho_lat"],
            "rho_lat_relative_delta": _safe_ratio(
                d1["rho_lat"] - d0["rho_lat"], d0["rho_lat"]
            ),
            "conditional_energy_relative_delta": _safe_ratio(
                d1["weighted_conditional_energy"]
                - d0["weighted_conditional_energy"],
                d0["weighted_conditional_energy"],
            ),
        }
    return {
        "representation": "frozen initialization image embedding, eval mode",
        "physical_neighbour_graph_reused": True,
        "query_mode_identity_max_abs_error": query_identity_error,
        "arms": arms,
        "comparison": comparison,
    }


class DeviceGradientAccumulator:
    """Accumulate gradient population moments without retaining samples."""

    def __init__(
        self,
        parameters: Sequence[torch.nn.Parameter],
        parameter_groups: Sequence[str],
    ) -> None:
        _require(len(parameters) == len(parameter_groups) > 0, "gradient route empty")
        self.parameter_groups = tuple(parameter_groups)
        self.sums = [torch.zeros_like(value, dtype=torch.float32) for value in parameters]
        self.sample_squared = {name: 0.0 for name in self.scope_names}
        self.sample_norm_sum = {name: 0.0 for name in self.scope_names}
        self.nonzero = {name: 0 for name in self.scope_names}
        self.count = 0

    @property
    def scope_names(self) -> tuple[str, ...]:
        return ("all", *tuple(dict.fromkeys(self.parameter_groups)))

    def add(self, gradients: Sequence[torch.Tensor | None]) -> None:
        _require(len(gradients) == len(self.sums), "gradient width changed")
        squared = {name: 0.0 for name in self.scope_names}
        for destination, value, group in zip(
            self.sums, gradients, self.parameter_groups, strict=True
        ):
            if value is None:
                continue
            _require(bool(torch.isfinite(value).all()), "non-finite parameter gradient")
            converted = value.detach().float()
            destination.add_(converted)
            magnitude = float(converted.square().sum())
            squared["all"] += magnitude
            squared[group] += magnitude
        for scope, magnitude in squared.items():
            self.sample_squared[scope] += magnitude
            norm = math.sqrt(max(magnitude, 0.0))
            self.sample_norm_sum[scope] += norm
            self.nonzero[scope] += int(norm > 0.0)
        self.count += 1

    def _sum_norm_squared(self, scope: str) -> float:
        return sum(
            float(value.square().sum())
            for value, group in zip(self.sums, self.parameter_groups, strict=True)
            if scope == "all" or group == scope
        )

    def summary(self, *, snr_batch_sizes: Sequence[int]) -> dict[str, Any]:
        _require(self.count > 0, "gradient population empty")
        scopes = {}
        for scope in self.scope_names:
            mean_squared = self._sum_norm_squared(scope) / (self.count * self.count)
            second = self.sample_squared[scope] / self.count
            noise = max(0.0, second - mean_squared)
            bcrit = _safe_ratio(noise, mean_squared)
            scopes[scope] = {
                "sample_count": self.count,
                "nonzero_sample_count": self.nonzero[scope],
                "mean_gradient_norm": math.sqrt(max(mean_squared, 0.0)),
                "mean_sample_gradient_norm": self.sample_norm_sum[scope] / self.count,
                "rms_noise": math.sqrt(noise),
                "critical_batch_size": bcrit,
                "snr_by_batch_size": {
                    str(int(size)): (
                        None if bcrit is None or bcrit == 0.0 else math.sqrt(int(size) / bcrit)
                    )
                    for size in snr_batch_sizes
                },
            }
        return {"scopes": scopes}


def _gradient_tuple(
    scalar: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        torch.autograd.grad(
            scalar,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
    )


def _linear_combination(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    *,
    left_weight: float,
    right_weight: float,
) -> tuple[torch.Tensor | None, ...]:
    output: list[torch.Tensor | None] = []
    for one, two in zip(left, right, strict=True):
        value = None
        if one is not None:
            value = float(left_weight) * one
        if two is not None:
            term = float(right_weight) * two
            value = term if value is None else value + term
        output.append(value)
    return tuple(output)


def gradient_relation(
    left: DeviceGradientAccumulator,
    right: DeviceGradientAccumulator,
) -> dict[str, Any]:
    _require(left.count > 0 and right.count > 0, "empty gradient relation")
    _require(left.parameter_groups == right.parameter_groups, "gradient routes differ")
    report = {}
    for scope in left.scope_names:
        dot = 0.0
        left_squared = 0.0
        right_squared = 0.0
        for one, two, group in zip(
            left.sums, right.sums, left.parameter_groups, strict=True
        ):
            if scope != "all" and group != scope:
                continue
            left_mean = one / left.count
            right_mean = two / right.count
            dot += float((left_mean * right_mean).sum())
            left_squared += float(left_mean.square().sum())
            right_squared += float(right_mean.square().sum())
        denominator = math.sqrt(left_squared * right_squared)
        report[scope] = {
            "dot": dot,
            "cosine": dot / denominator if denominator > 0.0 else None,
            "left_mean_gradient_norm": math.sqrt(left_squared),
            "right_mean_gradient_norm": math.sqrt(right_squared),
        }
    return report


def static_identity(
    *,
    checkpoint: Path,
    catalog: Path,
    multiplicity: Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    _require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
    _require(_sha256(checkpoint) == EXPECTED_CHECKPOINT_SHA256, "checkpoint changed")
    _require(
        _sha256(Path(d1_runner.__file__).resolve()) == EXPECTED_D1_RUNNER_SHA256,
        "D1 training runner changed",
    )
    frozen_schedule = d1_runner.verify_schedule_artifact()
    _, neighbours = load_catalog(catalog)
    weights = load_realized_weights(multiplicity)
    schedule = audit_schedule_indices()
    return {
        "checkpoint": {"path": str(checkpoint), "sha256": EXPECTED_CHECKPOINT_SHA256},
        "catalog": {
            "path": str(Path(catalog).expanduser().resolve()),
            "sha256": EXPECTED_V1_CATALOG_SHA256,
            "directed_neighbour_shape": list(neighbours.shape),
        },
        "multiplicity": {
            "path": str(Path(multiplicity).expanduser().resolve()),
            "sha256": EXPECTED_MULTIPLICITY_SHA256,
            "positive_support": bool(np.all(weights > 0.0)),
        },
        "schedule": {
            "dir": str(frozen_schedule["schedule_dir"]),
            "sha256": frozen_schedule["observed_sha256"],
            "runner_sha256": EXPECTED_D1_RUNNER_SHA256,
            "gradient_audit_batches": schedule,
        },
        "authority": {
            "training_only": True,
            "optimizer_steps": 0,
            "development_opened": False,
            "public_test_opened": False,
            "full_training_authorized": False,
        },
    }


def _collect_audit_rows(motion: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    selected = set(AUDIT_SCHEDULE_INDICES)
    d0_stream = motion.CompleteTwinPairedBatchStream(
        EXPECTED_PAIRS,
        batch_size=HIDDEN_ROWS_PER_BATCH,
        seed=d1_runner.EXPECTED_SEED,
    )
    d1_stream = d1_runner.EnergyStratifiedTwinBatchStream(
        EXPECTED_PAIRS,
        batch_size=HIDDEN_ROWS_PER_BATCH,
        seed=d1_runner.EXPECTED_SEED,
    )
    d0_rows: list[torch.Tensor] = []
    d1_rows: list[torch.Tensor] = []
    for index, (d0, d1) in enumerate(zip(d0_stream, d1_stream, strict=True)):
        if index in selected:
            validate_twin_rows(d0)
            validate_twin_rows(d1)
            d0_rows.append(d0.detach().cpu().long())
            d1_rows.append(d1.detach().cpu().long())
        if index >= max(AUDIT_SCHEDULE_INDICES):
            break
    _require(len(d0_rows) == len(d1_rows) == AUDIT_BATCH_COUNT, "audit rows missing")
    return d0_rows, d1_rows


def _materialize_training_inputs(
    *,
    motion: Any,
    original_h5: Path,
    original_lance: Path,
) -> tuple[Any, dict[str, torch.Tensor], dict[str, Any]]:
    trainer = motion.trainer
    mixed = trainer.mixed
    action_stats = trainer.ACTION_STATS_LOADER(original_h5)
    hidden = trainer._training_split(
        native_gradient.replay.replay.TRAIN_TABLE,
        expected_pairs=EXPECTED_PAIRS,
        action_stats=action_stats,
    )
    _require(hidden.pair_count == EXPECTED_PAIRS, "hidden pair count changed")
    _require(hidden.pixels.shape[0] == EXPECTED_HIDDEN_ROWS, "hidden row count changed")
    _, original_loader = mixed.original_loader(
        original_lance,
        batch_size=ORIGINAL_ROWS,
        seed=d1_runner.EXPECTED_SEED,
        num_workers=0,
    )
    original = next(iter(original_loader))
    original_actions = mixed.pilot.normalize_action_blocks(
        torch.nan_to_num(original["action"].float(), 0.0), action_stats
    )
    anchor = {
        "pixels": original["pixels"].detach().cpu().contiguous(),
        "actions": original_actions.detach().cpu().contiguous(),
    }
    d0_rows, d1_rows = _collect_audit_rows(motion)
    receipt = {
        "hidden_pair_count": int(hidden.pair_count),
        "hidden_row_count": int(hidden.pixels.shape[0]),
        "original_anchor_rows": int(anchor["pixels"].shape[0]),
        "original_anchor_pixels_sha256": _tensor_sha256(anchor["pixels"]),
        "original_anchor_actions_sha256": _tensor_sha256(anchor["actions"]),
        "d0_audit_row_stream_sha256": _tensor_sha256(torch.stack(d0_rows)),
        "d1_audit_row_stream_sha256": _tensor_sha256(torch.stack(d1_rows)),
        "same_original_anchor_for_both_arms_and_all_batches": True,
        "audit_schedule_indices": list(AUDIT_SCHEDULE_INDICES),
    }
    return hidden, anchor, {"receipt": receipt, "D0": d0_rows, "D1-MS50": d1_rows}


def _encode_all_latents(
    *,
    model: torch.nn.Module,
    mixed: Any,
    hidden: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_parts: list[torch.Tensor] = []
    future_parts: list[torch.Tensor] = []
    modes_before = native_gradient.replay.exact_audit._module_modes(model)
    buffers_before = native_gradient.replay.live_ccrm._buffer_snapshot(model)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, EXPECTED_HIDDEN_ROWS, batch_size):
                stop = min(start + batch_size, EXPECTED_HIDDEN_ROWS)
                pixels = mixed.pilot.preprocess_pixels(hidden.pixels[start:stop], device)
                actions = hidden.action[start:stop].to(device=device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    encoded = model.encode({"pixels": pixels, "action": actions})
                embeddings = encoded["emb"].detach().float().cpu()
                _require(embeddings.shape[1] >= 4, "encoded history length changed")
                query_parts.append(embeddings[:, 2].contiguous())
                future_parts.append(embeddings[:, 3].contiguous())
    finally:
        native_gradient.replay.exact_audit._restore_buffers(model, buffers_before)
        native_gradient.replay.exact_audit._restore_module_modes(model, modes_before)
    query = torch.cat(query_parts, dim=0)
    future = torch.cat(future_parts, dim=0)
    _require(query.shape[0] == future.shape[0] == EXPECTED_HIDDEN_ROWS, "latent rows missing")
    return query, future


def _build_batch(
    hidden: Any,
    anchor: Mapping[str, torch.Tensor],
    rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_twin_rows(rows)
    hidden_pixels = hidden.pixels[rows]
    hidden_actions = hidden.action[rows]
    paired_pixels = hidden_pixels.reshape(32, 2, *hidden_pixels.shape[1:])
    paired_actions = hidden_actions.reshape(32, 2, *hidden_actions.shape[1:])
    checks = {
        "same_action_within_pair": bool(
            torch.eq(paired_actions[:, 0], paired_actions[:, 1]).flatten(1).all()
        ),
        "same_query_rgb_within_pair": bool(
            torch.eq(paired_pixels[:, 0, 2], paired_pixels[:, 1, 2]).flatten(1).all()
        ),
        "different_history_within_pair": bool(
            torch.ne(paired_pixels[:, 0, :2], paired_pixels[:, 1, :2]).flatten(1).any(dim=1).all()
        ),
        "different_future_within_pair": bool(
            torch.ne(paired_pixels[:, 0, 3], paired_pixels[:, 1, 3]).flatten(1).any(dim=1).all()
        ),
    }
    _require(all(checks.values()), f"hidden batch invariant failed: {checks}")
    return {
        "pixels": torch.cat([anchor["pixels"], hidden_pixels], dim=0),
        "actions": torch.cat([anchor["actions"], hidden_actions], dim=0),
    }


def _arm_forward_and_gradients(
    *,
    batch: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    mixed: Any,
    device: torch.device,
    parameters: Sequence[torch.nn.Parameter],
    response_accumulator: DeviceGradientAccumulator,
    nonconditional_accumulator: DeviceGradientAccumulator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    pixels = mixed.pilot.preprocess_pixels(batch["pixels"], device)
    actions = batch["actions"].to(device=device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        encoded = model.encode({"pixels": pixels, "action": actions})
        embeddings = encoded["emb"]
        prediction = model.predict(embeddings[:, :3], encoded["act_emb"][:, :3])
        target = embeddings[:, 1:].detach()
        error = (prediction - target).square().mean(dim=-1)
        original_mean = error[:ORIGINAL_ROWS].mean(dim=1).float().mean()

    groups = native_gradient.replay.live_ccrm.binary_hidden_groups(
        original_batch_size=ORIGINAL_ROWS,
        batch_size=int(embeddings.shape[0]),
        device=device,
    )
    selected_prediction = prediction[groups, -1].float()
    selected_target = target[groups, -1].float()
    parts = signal_metrics.paired_signal_components(selected_prediction, selected_target)
    center_mean = parts["center_loss"].mean()
    response_by_twin = parts["response_loss"].reshape(TWINS_PER_BATCH, 2).mean(dim=1)
    original_gradient = _gradient_tuple(original_mean, parameters)
    center_gradient = _gradient_tuple(center_mean, parameters)
    nonconditional_accumulator.add(
        _linear_combination(
            original_gradient,
            center_gradient,
            left_weight=0.5,
            right_weight=0.5,
        )
    )
    for loss in response_by_twin:
        response_accumulator.add(
            tuple(
                None if value is None else 0.5 * value
                for value in _gradient_tuple(loss, parameters)
            )
        )
    losses = {
        "original": float(original_mean.detach().cpu()),
        "hidden_center": float(center_mean.detach().cpu()),
        "hidden_response": float(parts["response_loss"].mean().detach().cpu()),
    }
    return (
        selected_prediction.detach().cpu(),
        selected_target.detach().cpu(),
        losses,
    )


def _gradient_audit(
    *,
    model: torch.nn.Module,
    mixed: Any,
    hidden: Any,
    anchor: Mapping[str, torch.Tensor],
    rows_by_arm: Mapping[str, Sequence[torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    modes_before = native_gradient.replay.exact_audit._module_modes(model)
    buffers_before = native_gradient.replay.live_ccrm._buffer_snapshot(model)
    parameter_hash = native_gradient.replay.live_ccrm.parameter_value_sha256(model)
    model.train()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and _group(name) in ALLOWED_PARAMETER_GROUPS
    ]
    _require(bool(named_parameters), "no predictor parameters selected")
    parameters = [parameter for _, parameter in named_parameters]
    parameter_groups = [_group(name) for name, _ in named_parameters]
    _require(set(parameter_groups) == set(ALLOWED_PARAMETER_GROUPS), "parameter route changed")
    _require(
        sum(parameter.numel() for parameter in parameters)
        == native_gradient.EXPECTED_TRAINABLE_PARAMETER_COUNT,
        "trainable parameter count changed",
    )
    accumulators = {
        arm: {
            "response": DeviceGradientAccumulator(parameters, parameter_groups),
            "nonconditional": DeviceGradientAccumulator(parameters, parameter_groups),
        }
        for arm in ("D0", "D1-MS50")
    }
    predictions = {arm: [] for arm in accumulators}
    targets = {arm: [] for arm in accumulators}
    losses = {arm: [] for arm in accumulators}
    paired_rng_checks: list[bool] = []
    for batch_index in range(AUDIT_BATCH_COUNT):
        pair_buffers = native_gradient.replay.live_ccrm._buffer_snapshot(model)
        rng = native_gradient.replay.gradient_core._rng_snapshot()
        for arm in ("D0", "D1-MS50"):
            native_gradient.replay.gradient_core._restore_rng(rng)
            native_gradient.replay.exact_audit._restore_buffers(model, pair_buffers)
            before = native_gradient.replay.gradient_core._rng_snapshot()
            batch = _build_batch(hidden, anchor, rows_by_arm[arm][batch_index])
            prediction, target, loss = _arm_forward_and_gradients(
                batch=batch,
                model=model,
                mixed=mixed,
                device=device,
                parameters=parameters,
                response_accumulator=accumulators[arm]["response"],
                nonconditional_accumulator=accumulators[arm]["nonconditional"],
            )
            after = native_gradient.replay.gradient_core._rng_snapshot()
            predictions[arm].append(prediction)
            targets[arm].append(target)
            losses[arm].append(loss)
            if arm == "D0":
                d0_before = before
                d0_after = after
            else:
                paired_rng_checks.append(
                    native_gradient.replay.gradient_core._rng_equal(d0_before, before)
                    and native_gradient.replay.gradient_core._rng_equal(d0_after, after)
                )
        native_gradient.replay.exact_audit._restore_buffers(model, pair_buffers)

    summaries = {}
    for arm in ("D0", "D1-MS50"):
        response = accumulators[arm]["response"].summary(snr_batch_sizes=(16, 256))
        nonconditional = accumulators[arm]["nonconditional"].summary(
            snr_batch_sizes=(1, 16)
        )
        relation = gradient_relation(
            accumulators[arm]["response"], accumulators[arm]["nonconditional"]
        )
        strength = {}
        for scope in ("all", *ALLOWED_PARAMETER_GROUPS):
            response_norm = response["scopes"][scope]["mean_gradient_norm"]
            nonconditional_norm = nonconditional["scopes"][scope]["mean_gradient_norm"]
            strength[scope] = {
                "weighted_response_mean_gradient_norm": response_norm,
                "weighted_nonconditional_mean_gradient_norm": nonconditional_norm,
                "response_to_nonconditional_norm_ratio": _safe_ratio(
                    response_norm, nonconditional_norm
                ),
                "response_nonconditional_cosine": relation[scope]["cosine"],
            }
        summaries[arm] = {
            "response_twin_population": response,
            "nonconditional_batch_population": nonconditional,
            "gradient_strength": strength,
            "paired_output_signal": signal_metrics.paired_signal_summary(
                torch.cat(predictions[arm], dim=0),
                torch.cat(targets[arm], dim=0),
                batch_sizes=(16, 256),
            ),
            "loss_means": {
                name: float(np.mean([row[name] for row in losses[arm]]))
                for name in ("original", "hidden_center", "hidden_response")
            },
        }
    cross_arm_response = gradient_relation(
        accumulators["D0"]["response"], accumulators["D1-MS50"]["response"]
    )
    comparison = {}
    for scope in ("all", *ALLOWED_PARAMETER_GROUPS):
        d0 = summaries["D0"]
        d1 = summaries["D1-MS50"]
        d0_norm = d0["gradient_strength"][scope]["weighted_response_mean_gradient_norm"]
        d1_norm = d1["gradient_strength"][scope]["weighted_response_mean_gradient_norm"]
        d0_ratio = d0["gradient_strength"][scope][
            "response_to_nonconditional_norm_ratio"
        ]
        d1_ratio = d1["gradient_strength"][scope][
            "response_to_nonconditional_norm_ratio"
        ]
        d0_snr = d0["response_twin_population"]["scopes"][scope]["snr_by_batch_size"]["16"]
        d1_snr = d1["response_twin_population"]["scopes"][scope]["snr_by_batch_size"]["16"]
        comparison[scope] = {
            "response_mean_gradient_norm_relative_delta": _safe_ratio(d1_norm - d0_norm, d0_norm),
            "response_to_nonconditional_ratio_relative_delta": (
                None
                if d0_ratio is None or d1_ratio is None
                else _safe_ratio(d1_ratio - d0_ratio, d0_ratio)
            ),
            "response_snr16_relative_delta": (
                None if d0_snr is None or d1_snr is None else _safe_ratio(d1_snr - d0_snr, d0_snr)
            ),
            "d0_d1_response_mean_gradient_cosine": cross_arm_response[scope]["cosine"],
        }
    native_gradient.replay.exact_audit._restore_buffers(model, buffers_before)
    native_gradient.replay.exact_audit._restore_module_modes(model, modes_before)
    _require(
        native_gradient.replay.live_ccrm.parameter_value_sha256(model) == parameter_hash,
        "zero-step audit changed parameters",
    )
    _require(
        all(parameter.grad is None for parameter in model.parameters()),
        "parameter .grad populated",
    )
    _require(all(paired_rng_checks), "D0/D1 RNG consumption differed")
    return {
        "batch_count_per_arm": AUDIT_BATCH_COUNT,
        "twin_gradient_units_per_arm": AUDIT_BATCH_COUNT * TWINS_PER_BATCH,
        "same_original_anchor": True,
        "paired_rng_before_and_after_each_arm": True,
        "arms": summaries,
        "comparison": comparison,
        "state_restoration": {
            "parameters_unchanged": True,
            "parameter_grad_slots_remain_none": True,
            "buffers_restored": True,
            "module_modes_restored": True,
        },
    }


def _decision(latent: Mapping[str, Any], gradient: Mapping[str, Any]) -> dict[str, Any]:
    latent_checks = {
        f"rho_lat_k{scale}_strictly_above_d0": (
            latent["comparison"][str(scale)]["rho_lat_absolute_delta"] > 0.0
        )
        for scale in LATENT_NEIGHBOUR_SCALES
    }
    all_comparison = gradient["comparison"]["all"]
    norm_delta = all_comparison["response_mean_gradient_norm_relative_delta"]
    snr_delta = all_comparison["response_snr16_relative_delta"]
    gradient_visible = (norm_delta is not None and norm_delta > 0.0) or (
        snr_delta is not None and snr_delta > 0.0
    )
    checks = {
        **latent_checks,
        "response_mean_norm_or_snr16_strictly_improves": gradient_visible,
        "same_16_batch_and_256_twin_budget": (
            gradient["batch_count_per_arm"] == AUDIT_BATCH_COUNT
            and gradient["twin_gradient_units_per_arm"]
            == AUDIT_BATCH_COUNT * TWINS_PER_BATCH
        ),
        "state_restoration_passed": all(gradient["state_restoration"].values()),
    }
    return {
        "status": (
            "passed_go_for_single_d1_native_training"
            if all(checks.values())
            else "failed_no_go"
        ),
        "checks": checks,
        "rule": (
            "All three frozen-graph rho_lat values must increase, and either "
            "the all-parameter mean response-gradient norm or twin-cluster "
            "SNR(16) must strictly increase. No effect size is treated as a "
            "cross-task threshold."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    static = static_identity(
        checkpoint=checkpoint,
        catalog=args.catalog,
        multiplicity=args.multiplicity,
    )
    device = torch.device(args.device)
    _require(device.type == "cuda" and torch.cuda.is_available(), "CUDA required")
    native_gradient._prepare_optional_flash_attention()
    replay = native_gradient.replay
    runtime_path, runtime = replay.completion.load_completion(replay.RUNTIME_COMPLETION)
    release_path, _ = replay.completion.load_source_release(runtime)
    _require(release_path.resolve() == replay.CURRENT_RELEASE.resolve(), "release changed")
    worktree = replay.completion._pinned_stable_worldmodel(runtime)
    trainer = replay.completion._configure_component_trainer("motion_damping", worktree)
    import contextworld.benchmarks.motion_damping_icl_score as motion_score
    import run_pusht_motion_damping_h3_train as motion

    _require(motion.trainer is trainer, "Motion trainer binding changed")
    mixed = trainer.mixed
    missing = object()
    previous_weight = mixed.VARIANT_WEIGHTS.get(ANALYSIS_VARIANT, missing)
    was_twin_variant = ANALYSIS_VARIANT in motion.TWIN_GROUP_VARIANTS
    was_diagnostic_variant = ANALYSIS_VARIANT in trainer.DIAGNOSTIC_VARIANTS["lewm"]
    torch.cuda.get_rng_state_all()
    outer_rng = replay.gradient_core._rng_snapshot()
    try:
        mixed.VARIANT_WEIGHTS[ANALYSIS_VARIANT] = (
            "native",
            replay.WEIGHT,
            "identifiable_future_only",
        )
        motion.TWIN_GROUP_VARIANTS.add(ANALYSIS_VARIANT)
        trainer.DIAGNOSTIC_VARIANTS["lewm"].add(ANALYSIS_VARIANT)
        with replay.replay._install_fail_closed_guards(
            motion, motion_score, allow_training_table=True
        ) as guard_counts:
            hidden, anchor, inputs = _materialize_training_inputs(
                motion=motion,
                original_h5=args.original_h5.expanduser().resolve(),
                original_lance=args.original_lance.expanduser().resolve(),
            )
            mixed.pilot.set_reproducible_seed(d1_runner.EXPECTED_SEED)
            model, load_receipt = mixed.load_model_for_variant(
                checkpoint, variant=ANALYSIS_VARIANT, device=device
            )
            _require(
                load_receipt.get("sha256") == EXPECTED_CHECKPOINT_SHA256
                and load_receipt.get("model_state_sha256")
                == EXPECTED_MODEL_STATE_SHA256
                and load_receipt.get("strict_state_dict_load") is True,
                "checkpoint load identity changed",
            )
            query_latent, future_latent = _encode_all_latents(
                model=model,
                mixed=mixed,
                hidden=hidden,
                device=device,
                batch_size=int(args.latent_batch_size),
            )
            _, neighbours = load_catalog(args.catalog)
            d1_weight = load_realized_weights(args.multiplicity)
            latent = latent_local_energy(
                query_latent,
                future_latent,
                neighbours,
                {
                    "D0": np.full(EXPECTED_TWINS, 1.0 / EXPECTED_TWINS),
                    "D1-MS50": d1_weight,
                },
            )
            gradient = _gradient_audit(
                model=model,
                mixed=mixed,
                hidden=hidden,
                anchor=anchor,
                rows_by_arm={"D0": inputs["D0"], "D1-MS50": inputs["D1-MS50"]},
                device=device,
            )
        expected_guards = {
            "release_loader": 0,
            "release_auditor": 0,
            "optimizer_constructor": 0,
            "optimizer_step": 0,
            "development_scorer": 0,
            "public_scorer": 0,
            "training_table_reads": 1,
            "non_training_benchmark_reads": 0,
        }
        _require(
            all(guard_counts[name] == value for name, value in expected_guards.items()),
            f"forbidden zero-step action: {guard_counts}",
        )
    finally:
        replay.gradient_core._restore_rng(outer_rng)
        if previous_weight is missing:
            mixed.VARIANT_WEIGHTS.pop(ANALYSIS_VARIANT, None)
        else:
            mixed.VARIANT_WEIGHTS[ANALYSIS_VARIANT] = previous_weight
        if was_twin_variant:
            motion.TWIN_GROUP_VARIANTS.add(ANALYSIS_VARIANT)
        else:
            motion.TWIN_GROUP_VARIANTS.discard(ANALYSIS_VARIANT)
        if was_diagnostic_variant:
            trainer.DIAGNOSTIC_VARIANTS["lewm"].add(ANALYSIS_VARIANT)
        else:
            trainer.DIAGNOSTIC_VARIANTS["lewm"].discard(ANALYSIS_VARIANT)
    _require(
        replay.gradient_core._rng_equal(outer_rng, replay.gradient_core._rng_snapshot()),
        "outer RNG restoration failed",
    )
    decision = _decision(latent, gradient)
    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": decision["status"],
        "optimizer_updates": 0,
        "source": {"path": str(THIS_SOURCE), "sha256": _sha256(THIS_SOURCE)},
        "runtime": {
            "completion": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "release": {"path": str(release_path), "sha256": _sha256(release_path)},
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
        },
        "static_identity": static,
        "training_inputs": inputs["receipt"],
        "checkpoint_load": load_receipt,
        "latent_manipulation": latent,
        "gradient_visibility": gradient,
        "decision": decision,
        "guard_counts": dict(guard_counts),
        "evidence_boundary": {
            "training_only": True,
            "development_opened": False,
            "public_test_opened": False,
            "cem_executed": False,
            "optimizer_steps": 0,
            "model_parameters_changed": False,
            "latent_or_gradient_used_for_sample_selection": False,
            "full_training_authorized": decision["status"]
            == "passed_go_for_single_d1_native_training",
            "single_seed_mechanism_gate_not_endpoint_evidence": True,
        },
        "interpretation_boundary": {
            "rho_lat": (
                "Uses the D1-0 physical neighbour graph and therefore is not "
                "numerically comparable to the earlier unconditional-global "
                "rho_lat diagnostic."
            ),
            "gradient": (
                "The 16 batches are fixed schedule-spanning mechanism samples. "
                "A positive delta authorizes one frozen-recipe training test; "
                "it is not a universal effect threshold or a training outcome."
            ),
            "original_anchor": (
                "The same original rows and paired RNG are used in both arms. "
                "Their loss can still change under true train-mode batch-coupled "
                "model semantics when hidden rows change."
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--multiplicity", type=Path, default=DEFAULT_MULTIPLICITY)
    parser.add_argument("--original-h5", type=Path, default=native_gradient.DEFAULT_ORIGINAL_H5)
    parser.add_argument(
        "--original-lance",
        type=Path,
        default=native_gradient.DEFAULT_ORIGINAL_LANCE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-batch-size", type=int, default=LATENT_ENCODE_BATCH_SIZE)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    _require(args.latent_batch_size > 0, "latent batch size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    if args.check_only:
        payload = static_identity(
            checkpoint=args.checkpoint,
            catalog=args.catalog,
            multiplicity=args.multiplicity,
        )
        print(json.dumps({"status": "passed_static_identity", "identity": payload}, sort_keys=True))
        return 0
    payload = run(args)
    _write_exclusive(output, payload)
    receipt = output.parent / "receipt.json"
    _write_exclusive(
        receipt,
        {
            "schema_version": 1,
            "analysis_id": ANALYSIS_ID,
            "status": payload["status"],
            "input_sha256": {
                "checkpoint": EXPECTED_CHECKPOINT_SHA256,
                "d1_v1_catalog": EXPECTED_V1_CATALOG_SHA256,
                "d1_multiplicity": EXPECTED_MULTIPLICITY_SHA256,
                "d1_schedule": d1_runner.EXPECTED_SCHEDULE_SHA256[
                    "schedule.jsonl"
                ],
                "d1_training_runner": EXPECTED_D1_RUNNER_SHA256,
                "source": payload["source"]["sha256"],
            },
            "output_sha256": {"report.json": _sha256(output)},
            "decision_checks": payload["decision"]["checks"],
            **payload["evidence_boundary"],
        },
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "receipt": str(receipt),
                "receipt_sha256": _sha256(receipt),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed_go_for_single_d1_native_training" else 2


if __name__ == "__main__":
    raise SystemExit(main())
