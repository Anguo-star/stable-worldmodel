from types import SimpleNamespace

import pytest

from stable_worldmodel import loggers


class _FakeLogger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.hyperparams = None

    def log_hyperparams(self, values):
        self.hyperparams = values


class _FakeExperiment:
    def __init__(self):
        self.finish_calls = 0

    def finish(self):
        self.finish_calls += 1


def test_disabled_logger_returns_none():
    assert loggers.build_training_logger({'logger_backend': 'none'}) is None
    assert (
        loggers.build_training_logger(
            {
                'logger_backend': 'swanlab',
                'swanlab': {'enabled': False},
            }
        )
        is None
    )


def test_swanlab_logger_uses_shared_config(monkeypatch):
    settings = []

    def settings_cls(**kwargs):
        value = SimpleNamespace(**kwargs)
        settings.append(value)
        return value

    monkeypatch.setattr(
        loggers,
        '_import_swanlab',
        lambda: (_FakeLogger, settings_cls),
    )
    cfg = {
        'logger_backend': 'swanlab',
        'run_name': 'diagnostic-s3072',
        'swanlab': {
            'enabled': True,
            'collect_hardware': True,
            'hardware_monitor': False,
            'log_hyperparams': True,
            'config': {
                'project': 'stable-wm',
                'experiment_name': 'diagnostic-s3072',
            },
        },
    }

    logger = loggers.build_training_logger(cfg)

    assert logger.kwargs['project'] == 'stable-wm'
    assert logger.kwargs['experiment_name'] == 'diagnostic-s3072'
    assert logger.kwargs['settings'] is settings[0]
    assert settings[0].collect_hardware is True
    assert settings[0].hardware_monitor is False
    assert logger.hyperparams['run_name'] == 'diagnostic-s3072'


def test_wandb_logger_uses_shared_config(monkeypatch):
    monkeypatch.setattr(loggers, '_import_wandb', lambda: _FakeLogger)
    logger = loggers.build_training_logger(
        {
            'logger_backend': 'wandb',
            'wandb': {
                'enabled': True,
                'config': {'project': 'stable-wm', 'name': 'run'},
            },
        }
    )

    assert logger.kwargs == {'project': 'stable-wm', 'name': 'run'}
    assert logger.hyperparams['logger_backend'] == 'wandb'


def test_unknown_logger_backend_fails_closed():
    with pytest.raises(ValueError, match='Unsupported logger_backend'):
        loggers.build_training_logger({'logger_backend': 'tensorboard'})


def test_swanlab_success_finalize_flushes_once_and_is_idempotent():
    class FakeSwanLabLogger:
        def __init__(self):
            self._experiment = _FakeExperiment()
            self.finalize_statuses = []

        def finalize(self, status):
            self.finalize_statuses.append(status)

    patched = loggers._patch_swanlab_success_finalize(
        FakeSwanLabLogger,
        lambda function: function,
    )
    logger = patched()
    experiment = logger._experiment

    logger.finalize('success')
    logger.finalize('success')

    assert logger.finalize_statuses == ['success', 'success']
    assert experiment.finish_calls == 1
    assert logger._experiment is None


def test_swanlab_failure_finalize_keeps_sdk_failure_path_unchanged():
    class FakeSwanLabLogger:
        def __init__(self):
            self._experiment = _FakeExperiment()
            self.finalize_statuses = []

        def finalize(self, status):
            self.finalize_statuses.append(status)

    patched = loggers._patch_swanlab_success_finalize(
        FakeSwanLabLogger,
        lambda function: function,
    )
    logger = patched()
    experiment = logger._experiment

    logger.finalize('failed')

    assert logger.finalize_statuses == ['failed']
    assert experiment.finish_calls == 0
    assert logger._experiment is experiment
