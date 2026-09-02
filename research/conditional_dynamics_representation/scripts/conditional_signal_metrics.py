"""Task-neutral diagnostics for paired conditional world-model signals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


Gradient = torch.Tensor | None
GradientSample = Sequence[Gradient]

REDUCTION_SIGNATURE = {
    "pair_rows": "mean_of_two",
    "feature_dimensions": "mean",
    "time_steps": "selected_by_caller",
    "target_gradient": "detached_by_caller",
    "compute_dtype": "float32_or_float64",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _paired(value: torch.Tensor, name: str) -> torch.Tensor:
    _require(torch.is_tensor(value), f"{name} must be a torch.Tensor")
    _require(value.ndim >= 2, f"{name} must have shape [pair, 2, ...]")
    _require(value.shape[0] > 0, f"{name} must contain at least one pair")
    _require(value.shape[1] == 2, f"{name} second dimension must equal 2")
    _require(value.is_floating_point(), f"{name} must be floating point")
    _require(bool(torch.isfinite(value).all()), f"{name} contains non-finite values")
    dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    return value.to(dtype=dtype).reshape(value.shape[0], 2, -1)


def _identity_tolerance(*values: torch.Tensor) -> float:
    scale = max(
        1.0,
        *(
            float(value.detach().abs().max().cpu())
            for value in values
            if value.numel()
        ),
    )
    return 64.0 * torch.finfo(values[0].dtype).eps * scale


def paired_signal_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exact center/response components for width-two causal pairs."""

    prediction = _paired(predictions, "predictions")
    target = _paired(targets, "targets")
    _require(
        prediction.shape == target.shape,
        "predictions and targets must have identical shapes",
    )

    p0, p1 = prediction[:, 0], prediction[:, 1]
    t0, t1 = target[:, 0], target[:, 1]
    prediction_delta = p1 - p0
    target_delta = t1 - t0
    prediction_center = 0.5 * (p0 + p1)
    target_center = 0.5 * (t0 + t1)

    correct_loss = 0.5 * (
        (p0 - t0).square().mean(dim=1)
        + (p1 - t1).square().mean(dim=1)
    )
    swapped_loss = 0.5 * (
        (p1 - t0).square().mean(dim=1)
        + (p0 - t1).square().mean(dim=1)
    )
    target_delta_energy = target_delta.square().mean(dim=1)
    prediction_delta_energy = prediction_delta.square().mean(dim=1)
    cross_energy = (prediction_delta * target_delta).mean(dim=1)
    center_loss = (prediction_center - target_center).square().mean(dim=1)
    response_loss = 0.25 * (
        prediction_delta - target_delta
    ).square().mean(dim=1)
    g_swap = swapped_loss - correct_loss

    tolerance = _identity_tolerance(correct_loss, swapped_loss, cross_energy)
    center_error = float(
        (correct_loss - center_loss - response_loss).detach().abs().max().cpu()
    )
    swap_error = float((g_swap - cross_energy).detach().abs().max().cpu())
    if center_error > tolerance:
        raise RuntimeError(
            "paired loss decomposition failed: "
            f"error={center_error}, tolerance={tolerance}"
        )
    if swap_error > tolerance:
        raise RuntimeError(
            "G_swap identity failed: "
            f"error={swap_error}, tolerance={tolerance}"
        )

    return {
        "correct_loss": correct_loss,
        "swapped_loss": swapped_loss,
        "g_swap": g_swap,
        "target_delta_energy": target_delta_energy,
        "prediction_delta_energy": prediction_delta_energy,
        "cross_energy": cross_energy,
        "center_loss": center_loss,
        "response_loss": response_loss,
    }


def _finite_float(value: torch.Tensor | float) -> float:
    result = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite scalar: {result}")
    return result


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().double().reshape(-1).cpu()
    _require(flat.numel() > 0, "cannot summarize an empty tensor")
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64),
    )
    return {
        "count": int(flat.numel()),
        "mean": _finite_float(flat.mean()),
        "minimum": _finite_float(quantiles[0]),
        "p10": _finite_float(quantiles[1]),
        "p25": _finite_float(quantiles[2]),
        "median": _finite_float(quantiles[3]),
        "p75": _finite_float(quantiles[4]),
        "p90": _finite_float(quantiles[5]),
        "maximum": _finite_float(quantiles[6]),
    }


def _top_absolute_mass(values: torch.Tensor, fraction: float) -> float | None:
    absolute = values.detach().double().abs().reshape(-1).cpu()
    total = _finite_float(absolute.sum())
    if total == 0.0:
        return None
    count = max(1, math.ceil(float(absolute.numel()) * fraction))
    return _finite_float(torch.topk(absolute, count).values.sum()) / total


def _scalar_signal_noise(
    values: torch.Tensor,
    batch_sizes: Sequence[int],
) -> dict[str, Any]:
    flat = values.detach().double().reshape(-1).cpu()
    mean = _finite_float(flat.mean())
    variance = _finite_float(flat.var(unbiased=False))
    standard_deviation = math.sqrt(max(variance, 0.0))
    b_crit = _safe_ratio(variance, mean * mean)
    snr = {}
    for batch_size in batch_sizes:
        _require(int(batch_size) > 0, "batch sizes must be positive")
        if standard_deviation == 0.0:
            value = None
        else:
            value = math.sqrt(int(batch_size)) * mean / standard_deviation
        snr[str(int(batch_size))] = value
    return {
        "mean": mean,
        "population_standard_deviation": standard_deviation,
        "critical_batch_size": b_crit,
        "signed_snr_by_batch_size": snr,
        "noise_free": standard_deviation == 0.0,
    }


def paired_target_variance_summary(targets: torch.Tensor) -> dict[str, Any]:
    """Decompose target variance into within-pair and between-pair energy."""

    target = _paired(targets, "targets").double()
    flat_target = target.reshape(-1, target.shape[-1])
    target_mean = flat_target.mean(dim=0, keepdim=True)
    total_target_variance = _finite_float(
        (flat_target - target_mean).square().mean()
    )
    target_centers = target.mean(dim=1)
    between_pair_variance = _finite_float(
        (target_centers - target_mean).square().mean()
    )
    target_delta = target[:, 1] - target[:, 0]
    within_pair_variance = 0.25 * _finite_float(target_delta.square().mean())
    return {
        "total": total_target_variance,
        "within_pair_conditional": within_pair_variance,
        "between_pair_centers": between_pair_variance,
        "decomposition_absolute_error": abs(
            total_target_variance
            - between_pair_variance
            - within_pair_variance
        ),
        "rho_cond_data": _safe_ratio(
            within_pair_variance, total_target_variance
        ),
    }


def paired_signal_summary(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_sizes: Sequence[int] = (1, 8, 32, 128),
) -> dict[str, Any]:
    """Summarize conditional energy, learned response, and scalar SNR."""

    prediction = _paired(predictions, "predictions")
    target = _paired(targets, "targets")
    components = paired_signal_components(prediction, target)

    target_variance = paired_target_variance_summary(target)
    within_pair_variance = float(target_variance["within_pair_conditional"])

    target_energy = _finite_float(components["target_delta_energy"].sum())
    prediction_energy = _finite_float(
        components["prediction_delta_energy"].sum()
    )
    cross_energy = _finite_float(components["cross_energy"].sum())
    response_error_energy = 4.0 * _finite_float(
        components["response_loss"].sum()
    )
    gain = _safe_ratio(cross_energy, target_energy)
    energy_ratio = _safe_ratio(prediction_energy, target_energy)
    nre = _safe_ratio(response_error_energy, target_energy)
    orthogonal_residual = None
    scale_error = None
    nre_decomposition_error = None
    if gain is not None and energy_ratio is not None:
        orthogonal_residual = max(0.0, energy_ratio - gain * gain)
        scale_error = (gain - 1.0) ** 2
        if nre is not None:
            nre_decomposition_error = abs(
                nre - orthogonal_residual - scale_error
            )

    g_swap = components["g_swap"].detach().double().cpu()
    absolute_total = _finite_float(g_swap.abs().sum())
    g_swap_sum = _finite_float(g_swap.sum())
    correct_risk = _finite_float(components["correct_loss"].double().mean())
    response_risk = _finite_float(components["response_loss"].double().mean())
    center_risk = _finite_float(components["center_loss"].double().mean())

    return {
        "schema_version": 1,
        "reduction_signature": dict(REDUCTION_SIGNATURE),
        "pair_count": int(prediction.shape[0]),
        "feature_dimensions": int(prediction.shape[-1]),
        "target_variance": target_variance,
        "native_risk": {
            "correct": correct_risk,
            "center": center_risk,
            "response": response_risk,
            "conditional_target_energy_over_native_risk": _safe_ratio(
                within_pair_variance, correct_risk
            ),
            "output_response_gradient_energy_share": _safe_ratio(
                response_risk, correct_risk
            ),
        },
        "response_geometry": {
            "gain": gain,
            "prediction_to_target_energy_ratio": energy_ratio,
            "normalized_response_error": nre,
            "orthogonal_residual": orthogonal_residual,
            "scale_error": scale_error,
            "nre_decomposition_absolute_error": nre_decomposition_error,
        },
        "g_swap": {
            "definition": "native_swapped_history_loss_minus_correct_history_loss",
            "identity": "mean_feature_inner_product_prediction_delta_target_delta",
            "distribution": _distribution(g_swap),
            "positive_fraction": _finite_float((g_swap > 0).double().mean()),
            "negative_fraction": _finite_float((g_swap < 0).double().mean()),
            "absolute_cancellation_ratio": (
                abs(g_swap_sum) / absolute_total if absolute_total > 0.0 else None
            ),
            "top_absolute_mass_fraction": {
                "top_1_percent": _top_absolute_mass(g_swap, 0.01),
                "top_5_percent": _top_absolute_mass(g_swap, 0.05),
                "top_10_percent": _top_absolute_mass(g_swap, 0.10),
            },
            "scalar_minibatch_noise": _scalar_signal_noise(
                g_swap, batch_sizes
            ),
        },
    }


def _validate_gradient_samples(
    samples: Sequence[GradientSample],
    parameter_groups: Sequence[str] | None,
) -> tuple[int, tuple[str, ...]]:
    _require(bool(samples), "gradient population is empty")
    width = len(samples[0])
    _require(width > 0, "gradient samples contain no parameter slots")
    _require(all(len(sample) == width for sample in samples), "gradient widths differ")
    groups = tuple(parameter_groups or ("all",) * width)
    _require(len(groups) == width, "parameter_groups width differs")
    reference_shapes: list[torch.Size | None] = [None] * width
    for sample in samples:
        for index, value in enumerate(sample):
            if value is None:
                continue
            _require(torch.is_tensor(value), "gradients must be Tensor or None")
            _require(
                bool(torch.isfinite(value).all()),
                "gradient population contains non-finite values",
            )
            if reference_shapes[index] is None:
                reference_shapes[index] = value.shape
            _require(
                value.shape == reference_shapes[index],
                "gradient tensor shapes differ within a parameter slot",
            )
    return width, groups


def _gradient_scope_summary(
    samples: Sequence[GradientSample],
    selected: Sequence[bool],
    batch_sizes: Sequence[int],
) -> dict[str, Any]:
    count = len(samples)
    sums: list[torch.Tensor | None] = [None] * len(selected)
    sample_norms: list[float] = []
    sum_squared_norms = 0.0
    for sample in samples:
        squared = 0.0
        for index, (value, include) in enumerate(zip(sample, selected, strict=True)):
            if not include or value is None:
                continue
            detached = value.detach().to(device="cpu", dtype=torch.float64)
            squared += _finite_float(detached.square().sum())
            sums[index] = detached.clone() if sums[index] is None else sums[index] + detached
        norm = math.sqrt(max(squared, 0.0))
        sample_norms.append(norm)
        sum_squared_norms += squared

    sum_norm_squared = sum(
        _finite_float(value.square().sum()) for value in sums if value is not None
    )
    mean_norm_squared = sum_norm_squared / (count * count)
    second_moment = sum_squared_norms / count
    noise_energy = max(0.0, second_moment - mean_norm_squared)
    mean_norm = math.sqrt(mean_norm_squared)
    rms_noise = math.sqrt(noise_energy)
    b_crit = _safe_ratio(noise_energy, mean_norm_squared)
    sum_sample_norms = sum(sample_norms)
    snr = {}
    for batch_size in batch_sizes:
        _require(int(batch_size) > 0, "batch sizes must be positive")
        snr[str(int(batch_size))] = (
            None
            if b_crit is None or b_crit == 0.0
            else math.sqrt(int(batch_size) / b_crit)
        )
    return {
        "sample_count": count,
        "nonzero_sample_count": sum(value > 0.0 for value in sample_norms),
        "mean_gradient_norm": mean_norm,
        "mean_sample_gradient_norm": sum_sample_norms / count,
        "rms_noise": rms_noise,
        "coherence": _safe_ratio(mean_norm_squared, second_moment),
        "critical_batch_size": b_crit,
        "cancellation_ratio": _safe_ratio(
            math.sqrt(sum_norm_squared), sum_sample_norms
        ),
        "snr_by_batch_size": snr,
    }


def gradient_population_summary(
    samples: Sequence[GradientSample],
    *,
    parameter_groups: Sequence[str] | None = None,
    batch_sizes: Sequence[int] = (1, 8, 32, 128),
) -> dict[str, Any]:
    """Summarize gradient signal/noise without concatenating parameters."""

    width, groups = _validate_gradient_samples(samples, parameter_groups)
    scopes = {"all": [True] * width}
    if parameter_groups is not None:
        for group in dict.fromkeys(groups):
            scopes[str(group)] = [value == group for value in groups]
    return {
        "definition": {
            "critical_batch_size": (
                "E||g-Eg||^2 / ||Eg||^2 over supplied gradient units"
            ),
            "snr": "sqrt(batch_size / critical_batch_size)",
        },
        "scopes": {
            name: _gradient_scope_summary(samples, selected, batch_sizes)
            for name, selected in scopes.items()
        },
    }


class GradientPopulationAccumulator:
    """Stream parameter-gradient samples into the population estimands."""

    def __init__(
        self,
        *,
        parameter_groups: Sequence[str],
        batch_sizes: Sequence[int] = (1, 8, 32, 128),
    ) -> None:
        _require(bool(parameter_groups), "parameter_groups must not be empty")
        _require(
            all(int(batch_size) > 0 for batch_size in batch_sizes),
            "batch sizes must be positive",
        )
        self.parameter_groups = tuple(str(value) for value in parameter_groups)
        self.batch_sizes = tuple(int(value) for value in batch_sizes)
        self._sums: list[torch.Tensor | None] = [None] * len(self.parameter_groups)
        self._shapes: list[torch.Size | None] = [None] * len(self.parameter_groups)
        self._sample_norm_sums = {
            name: 0.0 for name in self._scope_names()
        }
        self._sample_squared_sums = {
            name: 0.0 for name in self._scope_names()
        }
        self._nonzero_counts = {name: 0 for name in self._scope_names()}
        self.count = 0

    def _scope_names(self) -> tuple[str, ...]:
        return ("all", *tuple(dict.fromkeys(self.parameter_groups)))

    def add(self, sample: GradientSample) -> None:
        _require(
            len(sample) == len(self.parameter_groups),
            "gradient sample width differs from parameter_groups",
        )
        squared_by_scope = {name: 0.0 for name in self._scope_names()}
        for index, (value, group) in enumerate(
            zip(sample, self.parameter_groups, strict=True)
        ):
            if value is None:
                continue
            _require(torch.is_tensor(value), "gradients must be Tensor or None")
            _require(
                bool(torch.isfinite(value).all()),
                "gradient population contains non-finite values",
            )
            if self._shapes[index] is None:
                self._shapes[index] = value.shape
            _require(
                value.shape == self._shapes[index],
                "gradient tensor shapes differ within a parameter slot",
            )
            detached = value.detach().to(device="cpu", dtype=torch.float64)
            squared = _finite_float(detached.square().sum())
            squared_by_scope["all"] += squared
            squared_by_scope[group] += squared
            if self._sums[index] is None:
                self._sums[index] = detached.clone()
            else:
                self._sums[index].add_(detached)

        for scope, squared in squared_by_scope.items():
            self._sample_squared_sums[scope] += squared
            norm = math.sqrt(max(squared, 0.0))
            self._sample_norm_sums[scope] += norm
            self._nonzero_counts[scope] += int(norm > 0.0)
        self.count += 1

    def summed_gradients(self) -> tuple[Gradient, ...]:
        _require(self.count > 0, "gradient population is empty")
        return tuple(
            None if value is None else value.clone() for value in self._sums
        )

    def mean_gradients(self) -> tuple[Gradient, ...]:
        _require(self.count > 0, "gradient population is empty")
        return tuple(
            None if value is None else value / self.count for value in self._sums
        )

    def summary(self) -> dict[str, Any]:
        _require(self.count > 0, "gradient population is empty")
        scopes = {}
        for scope in self._scope_names():
            selected = (
                [True] * len(self.parameter_groups)
                if scope == "all"
                else [value == scope for value in self.parameter_groups]
            )
            sum_norm_squared = sum(
                _finite_float(value.square().sum())
                for value, include in zip(self._sums, selected, strict=True)
                if include and value is not None
            )
            mean_norm_squared = sum_norm_squared / (self.count * self.count)
            second_moment = self._sample_squared_sums[scope] / self.count
            noise_energy = max(0.0, second_moment - mean_norm_squared)
            b_crit = _safe_ratio(noise_energy, mean_norm_squared)
            snr = {
                str(batch_size): (
                    None
                    if b_crit is None or b_crit == 0.0
                    else math.sqrt(batch_size / b_crit)
                )
                for batch_size in self.batch_sizes
            }
            scopes[scope] = {
                "sample_count": self.count,
                "nonzero_sample_count": self._nonzero_counts[scope],
                "mean_gradient_norm": math.sqrt(mean_norm_squared),
                "mean_sample_gradient_norm": (
                    self._sample_norm_sums[scope] / self.count
                ),
                "rms_noise": math.sqrt(noise_energy),
                "coherence": _safe_ratio(mean_norm_squared, second_moment),
                "critical_batch_size": b_crit,
                "cancellation_ratio": _safe_ratio(
                    math.sqrt(sum_norm_squared),
                    self._sample_norm_sums[scope],
                ),
                "snr_by_batch_size": snr,
            }
        return {
            "definition": {
                "critical_batch_size": (
                    "E||g-Eg||^2 / ||Eg||^2 over supplied gradient units"
                ),
                "snr": "sqrt(batch_size / critical_batch_size)",
                "streaming": True,
            },
            "scopes": scopes,
        }


def _gradient_dot_norms(
    left: GradientSample,
    right: GradientSample,
    selected: Sequence[bool],
) -> tuple[float, float, float]:
    left_squared = 0.0
    right_squared = 0.0
    dot = 0.0
    for one, two, include in zip(left, right, selected, strict=True):
        if not include:
            continue
        if one is not None:
            left_squared += _finite_float(one.detach().double().square().sum())
        if two is not None:
            right_squared += _finite_float(two.detach().double().square().sum())
        if one is not None and two is not None:
            _require(one.shape == two.shape, "related gradient shapes differ")
            dot += _finite_float(
                (one.detach().double() * two.detach().double()).sum()
            )
    return dot, math.sqrt(left_squared), math.sqrt(right_squared)


def gradient_relation_summary(
    center: GradientSample,
    response: GradientSample,
    *,
    parameter_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare center and response gradients by parameter block."""

    _require(len(center) == len(response) and len(center) > 0, "gradient widths differ")
    _, groups = _validate_gradient_samples([center, response], parameter_groups)
    scopes = {"all": [True] * len(center)}
    if parameter_groups is not None:
        for group in dict.fromkeys(groups):
            scopes[str(group)] = [value == group for value in groups]
    result = {}
    for name, selected in scopes.items():
        dot, center_norm, response_norm = _gradient_dot_norms(
            center, response, selected
        )
        cosine = (
            dot / (center_norm * response_norm)
            if center_norm > 0.0 and response_norm > 0.0
            else None
        )
        result[name] = {
            "center_gradient_norm": center_norm,
            "response_gradient_norm": response_norm,
            "response_to_center_norm_ratio": _safe_ratio(
                response_norm, center_norm
            ),
            "cosine": cosine,
            "inner_product": dot,
        }
    return {"scopes": result}


def gradients_by_name(
    named_gradients: Mapping[str, Gradient],
) -> tuple[tuple[str, ...], tuple[Gradient, ...], tuple[str, ...]]:
    """Convert named gradients to the tuple contract used by this module."""

    names = tuple(named_gradients)
    gradients = tuple(named_gradients[name] for name in names)
    groups = tuple(name.split(".", 1)[0] for name in names)
    return names, gradients, groups
