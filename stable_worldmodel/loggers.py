"""Training logger construction shared by Stable-WorldModel entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _patch_swanlab_success_finalize(logger_cls, rank_zero_only):
    """Close successful SwanLab runs before distributed workers tear down.

    SwanLab 0.7.13 only closes the experiment from ``finalize`` when
    Lightning reports a failure.  Successful runs are left to an ``atexit``
    handler, which can race with DDP worker teardown.  Patch the installed
    class in place so it remains importable/pickleable for spawned workers.
    """

    marker = '_stable_worldmodel_finalizes_success'
    if getattr(logger_cls, marker, False):
        return logger_cls

    original_finalize = logger_cls.finalize

    @rank_zero_only
    def finalize(self, status: str) -> None:
        original_finalize(self, status)
        if status != 'success':
            return

        experiment = getattr(self, '_experiment', None)
        if experiment is None:
            return

        # Clear the reference first so a later Lightning/atexit callback is
        # idempotent even if the SDK raises while flushing the run.
        self._experiment = None
        experiment.finish()

    logger_cls.finalize = finalize
    setattr(logger_cls, marker, True)
    return logger_cls


def _as_dict(value: Any, *, resolve: bool = True) -> dict[str, Any]:
    """Convert an OmegaConf mapping or a regular mapping to a plain dict."""

    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None

    if OmegaConf is not None and OmegaConf.is_config(value):
        converted = OmegaConf.to_container(value, resolve=resolve)
        if not isinstance(converted, dict):
            raise TypeError('Logger configuration must be a mapping')
        return converted
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError('Logger configuration must be a mapping')


def _import_swanlab():
    try:
        from lightning.pytorch.utilities import rank_zero_only
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        from swanlab.swanlab_settings import Settings as SwanLabSettings
    except ImportError as exc:
        raise ImportError(
            'swanlab is not installed. Run: pip install swanlab'
        ) from exc
    return (
        _patch_swanlab_success_finalize(SwanLabLogger, rank_zero_only),
        SwanLabSettings,
    )


def _import_wandb():
    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError as exc:
        raise ImportError(
            'Lightning WandB support is not installed. '
            'Install stable-worldmodel training dependencies.'
        ) from exc
    return WandbLogger


def build_training_logger(cfg):
    """Build the logger selected by ``cfg.logger_backend``.

    Both Hydra ``DictConfig`` objects and ordinary mappings are accepted.
    Supported backends are ``none``, ``swanlab``, and ``wandb``. A selected
    backend also requires its corresponding ``enabled`` flag.
    """

    backend = str(cfg.get('logger_backend', 'none')).lower()
    if backend in {'', 'none', 'false', 'disabled'}:
        return None

    if backend == 'swanlab':
        swanlab_cfg = cfg.get('swanlab', {}) or {}
        if not swanlab_cfg.get('enabled', False):
            return None
        logger_cls, settings_cls = _import_swanlab()
        logger_kwargs = _as_dict(
            swanlab_cfg.get('config', {}),
            resolve=True,
        )
        if 'settings' not in logger_kwargs:
            logger_kwargs['settings'] = settings_cls(
                collect_hardware=swanlab_cfg.get(
                    'collect_hardware',
                    False,
                ),
                hardware_monitor=swanlab_cfg.get(
                    'hardware_monitor',
                    False,
                ),
            )
        logger = logger_cls(**logger_kwargs)
        if swanlab_cfg.get('log_hyperparams', False):
            logger.log_hyperparams(_as_dict(cfg, resolve=True))
        return logger

    if backend == 'wandb':
        wandb_cfg = cfg.get('wandb', {}) or {}
        if not wandb_cfg.get('enabled', False):
            return None
        logger_cls = _import_wandb()
        logger = logger_cls(
            **_as_dict(wandb_cfg.get('config', {}), resolve=True)
        )
        logger.log_hyperparams(_as_dict(cfg, resolve=True))
        return logger

    raise ValueError(f'Unsupported logger_backend: {backend}')


__all__ = ['build_training_logger']
