"""Optional dataset hooks shared by world-model training entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import lightning as pl
import torch


def initialize_training_seed(seed: int) -> int:
    """Seed process RNGs before any model or data-loader construction."""

    normalized_seed = int(seed)
    if not 0 <= normalized_seed <= 2**32 - 1:
        raise ValueError(
            f'Training seed must be in [0, 2**32 - 1], got {normalized_seed}'
        )
    pl.seed_everything(normalized_seed, workers=True)
    return normalized_seed


def split_training_dataset(
    dataset: Any,
    *,
    train_fraction: float,
    generator: torch.Generator,
    fallback: Callable[..., Any] = torch.utils.data.random_split,
) -> tuple[Any, Any]:
    """Use a dataset-owned structural split when one is available."""

    custom_split = getattr(dataset, 'split_for_training', None)
    if custom_split is not None:
        return custom_split(
            train_fraction=float(train_fraction),
            generator=generator,
        )
    return fallback(
        dataset,
        lengths=[train_fraction, 1.0 - train_fraction],
        generator=generator,
    )


def configure_training_loader(
    dataset: Any,
    loader_config: Mapping[str, Any],
    *,
    seed: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Let a dataset preserve structural batch relations when required."""

    config = dict(loader_config)
    configure = getattr(dataset, 'configure_train_loader', None)
    if configure is not None:
        config = configure(config, seed=int(seed))
    config.setdefault('generator', generator)
    return config


__all__ = [
    'configure_training_loader',
    'initialize_training_seed',
    'split_training_dataset',
]
