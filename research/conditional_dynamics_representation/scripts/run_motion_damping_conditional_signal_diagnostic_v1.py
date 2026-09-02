#!/usr/bin/env python3
"""Zero-training conditional-signal audit on Motion Development pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch


THIS_SOURCE = Path(__file__).resolve()
ROOT = THIS_SOURCE.parents[3]
CONTEXTWORLD = ROOT.parent / "ContextWorld"
for source_root in (ROOT, CONTEXTWORLD):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.benchmarks.adapters import (  # noqa: E402
    StableWorldModelLeWMMotionDampingAdapter,
)
from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    MotionDampingICLDevelopmentDataset,
    load_motion_damping_icl_release,
)
from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    conditional_signal_metrics as signal,
)


ANALYSIS_ID = "motion_damping_conditional_signal_diagnostic_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().double().reshape(-1).cpu()
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64),
    )
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "minimum": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "maximum": float(quantiles[4]),
    }


def _standardized_state_targets(arrays: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    states = torch.from_numpy(
        np.stack(
            [
                arrays.faster_decay_states[:, 3],
                arrays.no_extra_decay_states[:, 3],
            ],
            axis=1,
        )
    ).float()
    flattened = states.reshape(-1, states.shape[-1])
    standard_deviation = flattened.std(dim=0, unbiased=False)
    active = standard_deviation > 1.0e-8
    _require(bool(active.any()), "physical state has no varying dimensions")
    standardized = (
        states[..., active] - flattened[:, active].mean(dim=0)
    ) / standard_deviation[active]
    return standardized, {
        "raw_dimensions": int(states.shape[-1]),
        "active_standardized_dimensions": int(active.sum()),
        "inactive_dimension_indices": torch.nonzero(
            ~active, as_tuple=False
        ).flatten().tolist(),
        "standardization": "Development targets, population standard deviation",
    }


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_count": summary["pair_count"],
        "target_variance": summary["target_variance"],
        "native_risk": summary["native_risk"],
        "response_geometry": summary["response_geometry"],
        "g_swap": {
            "distribution": summary["g_swap"]["distribution"],
            "positive_fraction": summary["g_swap"]["positive_fraction"],
            "absolute_cancellation_ratio": summary["g_swap"][
                "absolute_cancellation_ratio"
            ],
        },
    }


def _quartile_panel(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    values: torch.Tensor,
) -> list[dict[str, Any]]:
    order = torch.argsort(values.detach().reshape(-1).cpu())
    rows = []
    for index, selected in enumerate(torch.tensor_split(order, 4)):
        summary = signal.paired_signal_summary(
            predictions[selected], targets[selected]
        )
        rows.append(
            {
                "quartile": index + 1,
                "stratifier": _distribution(values[selected]),
                "conditional_signal": _compact_summary(summary),
            }
        )
    return rows


def _random_pairing_baselines(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    random_seed: int,
    resamples: int,
) -> dict[str, Any]:
    prediction_delta = (predictions[:, 1] - predictions[:, 0]).double().cpu()
    target_delta = (targets[:, 1] - targets[:, 0]).double().cpu()
    pair_count = prediction_delta.shape[0]
    generator = torch.Generator().manual_seed(int(random_seed))
    matched = (prediction_delta * target_delta).mean(dim=1)

    cross_query_means = []
    for _ in range(int(resamples)):
        permutation = torch.randperm(pair_count, generator=generator)
        cross_query_means.append(
            (prediction_delta * target_delta[permutation]).mean()
        )
    cross_query = torch.stack(cross_query_means)

    sign_flip_means = []
    for _ in range(int(resamples)):
        signs = torch.randint(
            0, 2, (pair_count,), generator=generator, dtype=torch.int64
        ).double()
        signs = signs.mul_(2.0).sub_(1.0)
        sign_flip_means.append((matched * signs).mean())
    sign_flip = torch.stack(sign_flip_means)
    observed = float(matched.mean())
    cross_std = float(cross_query.std(unbiased=False))
    return {
        "random_seed": int(random_seed),
        "resamples": int(resamples),
        "matched_g_swap_mean": observed,
        "cross_query_pairing": {
            "null_mean": float(cross_query.mean()),
            "null_standard_deviation": cross_std,
            "matched_minus_null_mean": observed - float(cross_query.mean()),
            "z_score": (
                (observed - float(cross_query.mean())) / cross_std
                if cross_std > 0.0
                else None
            ),
            "one_sided_monte_carlo_p": (
                int((cross_query >= observed).sum()) + 1
            )
            / (int(resamples) + 1),
            "interpretation": (
                "Tests query-specific response against globally reusable "
                "response directions."
            ),
        },
        "within_pair_sign_flip": {
            "two_sided_monte_carlo_p": (
                int((sign_flip.abs() >= abs(observed)).sum()) + 1
            )
            / (int(resamples) + 1),
            "interpretation": "Paired randomization null for mean G_swap.",
        },
    }


def _reference_parity(
    path: Path | None,
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = (payload.get("official_recompute") or {}).get("metrics")
    _require(isinstance(metrics, dict), "reference result has no metrics")
    latent = metrics.get("latent_response") or {}
    observed = {
        "response_gain": summary["response_geometry"]["gain"],
        "normalized_response_error": summary["response_geometry"][
            "normalized_response_error"
        ],
        "other_minus_correct_mse_margin_mean": summary["g_swap"][
            "distribution"
        ]["mean"],
    }
    expected = {
        "response_gain": latent.get(
            "response_gain", latent.get("aggregate_response_gain")
        ),
        "normalized_response_error": latent.get("normalized_response_error"),
        "other_minus_correct_mse_margin_mean": metrics.get(
            "other_minus_correct_mse_margin_mean",
            (metrics.get("prediction_mse") or {}).get(
                "incorrect_minus_correct_margin"
            ),
        ),
    }
    differences = {
        name: (
            None
            if expected[name] is None or observed[name] is None
            else abs(float(observed[name]) - float(expected[name]))
        )
        for name in observed
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "expected": expected,
        "observed": observed,
        "absolute_difference": differences,
        "all_available_within_2e_4": all(
            value is None or value <= 2.0e-4 for value in differences.values()
        ),
    }


def _records(
    pair_ids: Sequence[str],
    components: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for index, pair_id in enumerate(pair_ids):
        target_energy = float(components["target_delta_energy"][index])
        cross = float(components["cross_energy"][index])
        rows.append(
            {
                "pair_id": str(pair_id),
                "correct_native_loss": float(components["correct_loss"][index]),
                "swapped_history_native_loss": float(
                    components["swapped_loss"][index]
                ),
                "g_swap": float(components["g_swap"][index]),
                "target_delta_energy": target_energy,
                "prediction_delta_energy": float(
                    components["prediction_delta_energy"][index]
                ),
                "response_gain": (
                    cross / target_energy if target_energy > 0.0 else None
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    release_config = args.release_config.expanduser().resolve()
    _require(checkpoint.is_file(), f"checkpoint not found: {checkpoint}")
    release = load_motion_damping_icl_release(release_config)
    dataset = MotionDampingICLDevelopmentDataset(
        release=release,
        repo_root=CONTEXTWORLD,
    )
    identity = dataset.identity
    _require(identity.get("passed") is True, "Development identity check failed")
    arrays = dataset.arrays
    normalization = release["evaluation"]["action_normalization"]
    runtime = release["runtime"]["stable_worldmodel"]
    adapter = StableWorldModelLeWMMotionDampingAdapter.from_checkpoint(
        checkpoint,
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=CONTEXTWORLD,
        stablewm_repo=str(ROOT),
        stablewm_ref=args.stablewm_ref,
        device=args.device,
    )

    histories = np.concatenate(
        [
            arrays.faster_decay_pixels[:, :3],
            arrays.no_extra_decay_pixels[:, :3],
        ],
        axis=0,
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]],
        axis=0,
    )
    future_pixels = np.concatenate(
        [
            arrays.faster_decay_pixels[:, 3],
            arrays.no_extra_decay_pixels[:, 3],
        ],
        axis=0,
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(
        histories, actions, batch_size=int(args.batch_size)
    )[:, 0]
    target = adapter.encode_pixels(
        future_pixels, batch_size=int(args.batch_size)
    )
    after = adapter.frozen_state_hash()
    _require(before == after, "model state changed during diagnostic")
    pair_count = arrays.pair_count
    predictions = torch.from_numpy(
        np.stack([predicted[:pair_count], predicted[pair_count:]], axis=1)
    )
    targets = torch.from_numpy(
        np.stack([target[:pair_count], target[pair_count:]], axis=1)
    )

    summary = signal.paired_signal_summary(
        predictions,
        targets,
        batch_sizes=args.snr_batch_sizes,
    )
    components = signal.paired_signal_components(predictions, targets)
    standardized_states, state_protocol = _standardized_state_targets(arrays)
    state_energy = (
        standardized_states[:, 1] - standardized_states[:, 0]
    ).square().mean(dim=1)
    latent_energy = components["target_delta_energy"].detach().cpu()
    query_actions = torch.from_numpy(arrays.raw_action_blocks[:, 2]).float()
    query_action_norm = query_actions.reshape(pair_count, -1).norm(dim=1)

    return {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "status": "completed_zero_training_development_diagnostic",
        "claim_scope": "Development_only_not_Public_Test_not_training_endpoint",
        "optimizer_updates": 0,
        "source": {"path": str(THIS_SOURCE), "sha256": _sha256(THIS_SOURCE)},
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "adapter": adapter.metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": {
            "release_config": str(release_config),
            "development_identity": identity,
            "description": dataset.describe(),
            "public_test_opened": False,
        },
        "latent_signal": summary,
        "physical_state_signal": {
            "protocol": state_protocol,
            "target_variance": signal.paired_target_variance_summary(
                standardized_states
            ),
            "target_delta_energy": _distribution(state_energy),
        },
        "stratification": {
            "latent_target_energy_quartiles": _quartile_panel(
                predictions, targets, latent_energy
            ),
            "standardized_state_target_energy_quartiles": _quartile_panel(
                predictions, targets, state_energy
            ),
            "query_action_norm_quartiles": _quartile_panel(
                predictions, targets, query_action_norm
            ),
        },
        "random_pairing_baselines": _random_pairing_baselines(
            predictions,
            targets,
            random_seed=int(args.random_seed),
            resamples=int(args.random_resamples),
        ),
        "reference_result_parity": _reference_parity(
            args.reference_result, summary
        ),
        "records": _records(arrays.pair_ids, components),
        "interpretation_boundary": {
            "rho_cond_data": (
                "Representation-dependent target variance share; compare "
                "checkpoints only when the target encoder is matched."
            ),
            "parameter_gradient_snr": (
                "Not measured here. It requires exact train-mode training "
                "batch replay and must not be inferred from Development."
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    )
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stablewm-ref", default="")
    parser.add_argument("--random-seed", type=int, default=20260830)
    parser.add_argument("--random-resamples", type=int, default=2048)
    parser.add_argument(
        "--snr-batch-sizes",
        type=int,
        nargs="+",
        default=(1, 8, 32, 128),
    )
    args = parser.parse_args(argv)
    _require(args.batch_size > 0, "batch size must be positive")
    _require(args.random_resamples > 0, "random resamples must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    output = args.output.expanduser().resolve()
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "checkpoint_sha256": payload["model"]["checkpoint_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
