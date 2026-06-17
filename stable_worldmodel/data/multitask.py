"""Small helpers for task-id-free multi-dataset training.

The utilities here deliberately stay at the dataset boundary. They do not add
task labels, task embeddings, or model-side routing. Each child dataset owns
its transforms and normalization; this module only checks interface
compatibility and exposes simple composition wrappers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from .utils import load_dataset


class ActionDimPadTransform:
    """Apply a sample transform, then right-pad per-step actions."""

    def __init__(
        self,
        transform: Callable[[dict], dict] | None,
        *,
        source_dim: int,
        target_dim: int,
    ) -> None:
        self.transform = transform
        self.source_dim = source_dim
        self.target_dim = target_dim

    def __call__(self, sample: dict) -> dict:
        if self.transform is not None:
            sample = self.transform(sample)
        if 'action' not in sample:
            return sample

        action = sample['action']
        sample = dict(sample)
        sample['action'] = _pad_action_last_dim(
            action,
            source_dim=self.source_dim,
            target_dim=self.target_dim,
        )
        return sample


class BalancedConcatDataset:
    """Repeat datasets so each contributes equally within an epoch.

    ``torch.utils.data.ConcatDataset`` preserves raw sample counts, which lets
    large datasets dominate mixed-environment training. This wrapper gives each
    child dataset one slot per round and repeats shorter datasets modulo their
    length. It is deterministic and works with DataLoader ``shuffle=True``.
    """

    def __init__(self, datasets: Sequence[Any]) -> None:
        if not datasets:
            raise ValueError('Need at least one dataset')
        lengths = [len(ds) for ds in datasets]
        if any(length <= 0 for length in lengths):
            raise ValueError(f'All datasets must be non-empty, got {lengths}')

        self.datasets = list(datasets)
        self._lengths = lengths
        self._max_len = max(lengths)
        self._len = self._max_len * len(self.datasets)

    @property
    def column_names(self) -> list[str]:
        return list(getattr(self.datasets[0], 'column_names', []))

    def __len__(self) -> int:
        return self._len

    def _loc(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        task_idx = idx % len(self.datasets)
        round_idx = idx // len(self.datasets)
        local_idx = round_idx % self._lengths[task_idx]
        return task_idx, local_idx

    def __getitem__(self, idx: int) -> dict:
        task_idx, local_idx = self._loc(idx)
        return self.datasets[task_idx][local_idx]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        return [self[idx] for idx in indices]


def load_multitask_datasets(
    cfg: Mapping[str, Any],
    *,
    cache_dir: str | None = None,
    transform_factory: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> tuple[list[Any], int]:
    """Load child datasets from a multitask dataset config.

    Expected config shape::

        mode: multitask
        frameskip: 5
        num_steps: 4
        keys_to_load: [pixels, action]
        keys_to_cache: [action]
        action_dim_policy: strict | pad_to_max
        items:
          - name: tworoom.h5
            label: tworoom

    Common keys are inherited by every item; item-level keys override common
    values. ``label`` is metadata only and is never returned in samples.
    With ``action_dim_policy=pad_to_max``, lower-dimensional per-step actions
    are padded after their child transform and before ``Dataset.__getitem__``
    reshapes them into frameskip action blocks.
    """

    items = list(cfg.get('items') or [])
    if not items:
        raise ValueError('multitask dataset config requires non-empty items')

    policy = str(cfg.get('action_dim_policy', 'strict')).lower()
    if policy not in {'strict', 'pad_to_max'}:
        raise ValueError(
            'Unsupported action_dim_policy: '
            f'{policy!r}. Expected strict or pad_to_max.'
        )

    common = _common_dataset_kwargs(cfg)
    datasets = []
    action_dims = []

    for raw_item in items:
        item = dict(raw_item)
        name = item.pop('name', None)
        if not name:
            raise ValueError(f'Multitask item is missing name: {raw_item}')

        label = item.pop('label', name)
        kwargs = deepcopy(common)
        kwargs.update(item)
        item_cache_dir = kwargs.pop('cache_dir', cache_dir)
        fmt = kwargs.pop('format', None)

        print(f'Loading multitask dataset "{label}" from {name}')
        dataset = load_dataset(
            name,
            transform=None,
            cache_dir=item_cache_dir,
            format=fmt,
            **kwargs,
        )
        if transform_factory is not None:
            dataset.transform = transform_factory(dataset, raw_item)
        datasets.append(dataset)
        action_dims.append(dataset.get_dim('action'))

    if policy == 'strict' and len(set(action_dims)) != 1:
        details = ', '.join(
            f'{_item_label(item)}={dim}'
            for item, dim in zip(items, action_dims)
        )
        raise ValueError(
            'Multitask LeWM currently requires identical action dims; '
            f'got {details}'
        )

    if policy == 'pad_to_max':
        target_dim = max(action_dims)
        for dataset, source_dim in zip(datasets, action_dims):
            if source_dim < target_dim:
                dataset.transform = ActionDimPadTransform(
                    dataset.transform,
                    source_dim=source_dim,
                    target_dim=target_dim,
                )
        return datasets, target_dim

    return datasets, action_dims[0]


def _pad_action_last_dim(action: Any, *, source_dim: int, target_dim: int):
    if target_dim < source_dim:
        raise ValueError(
            f'target action dim {target_dim} is smaller than source dim '
            f'{source_dim}'
        )

    if isinstance(action, torch.Tensor):
        if action.ndim == 1 and source_dim == 1:
            action = action.unsqueeze(-1)
        last_dim = action.shape[-1]
        if last_dim > target_dim:
            raise ValueError(
                f'action dim {last_dim} exceeds target dim {target_dim}'
            )
        if last_dim == target_dim:
            return action
        return torch.nn.functional.pad(action, (0, target_dim - last_dim))

    if isinstance(action, np.ndarray):
        if action.ndim == 1 and source_dim == 1:
            action = action[..., None]
        last_dim = action.shape[-1]
        if last_dim > target_dim:
            raise ValueError(
                f'action dim {last_dim} exceeds target dim {target_dim}'
            )
        if last_dim == target_dim:
            return action
        pad_width = [(0, 0)] * action.ndim
        pad_width[-1] = (0, target_dim - last_dim)
        return np.pad(action, pad_width, mode='constant')

    raise TypeError(f'Unsupported action type for padding: {type(action)}')


def _common_dataset_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        'action_dim_policy',
        'balance_val',
        'items',
        'mode',
        'normalizers',
        'sampling',
    }
    return {k: deepcopy(v) for k, v in cfg.items() if k not in excluded}


def _item_label(item: Mapping[str, Any]) -> str:
    return str(item.get('label') or item.get('name') or '<unnamed>')
