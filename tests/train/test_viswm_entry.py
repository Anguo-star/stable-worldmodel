from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train.lewm import (
    build_loss_components,
    lejepa_forward,
)
from scripts.train.viswm import VISWM_REGULARIZERS
from stable_worldmodel.wm.loss import VISRegLoss


def _loss_config(regularizer: str):
    return OmegaConf.create(
        {
            'loss': {
                'regularizer': regularizer,
                'sigreg': {
                    'weight': 0.09,
                    'kwargs': {'knots': 17, 'num_proj': 32},
                },
                'visreg': {
                    'weight': 4.5,
                    'kwargs': {
                        'num_projections': 32,
                        'lambda_scale': 1.0,
                        'lambda_shape': 1.0,
                        'lambda_center': 1.0,
                    },
                },
                'std': {'enabled': False},
                'std_t': {'enabled': False},
                'cov': {'enabled': False},
                'cov_t': {'enabled': False},
            },
            'wm': {'history_size': 3, 'num_preds': 1},
        }
    )


def test_viswm_config_inherits_the_lewm_backbone_with_its_own_identity() -> None:
    config_dir = Path('scripts/train/config').resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name='viswm')

    assert cfg.output_model_name == 'viswm'
    assert cfg.model._target_ == 'stable_worldmodel.wm.lewm.LeWM'
    assert cfg.loss.regularizer == 'visreg'
    assert cfg.loss.visreg.weight == 4.5
    assert cfg.optimizer.lr == 1e-4
    assert cfg.loss.conditional_joint.enabled is False

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        lewm_cfg = compose(config_name='lewm')
    assert 'visreg' not in lewm_cfg.loss


def test_viswm_entry_owns_the_visreg_objective() -> None:
    cfg = _loss_config('visreg')

    components = build_loss_components(
        cfg,
        regularizers=VISWM_REGULARIZERS,
    )
    assert set(components) == {'visreg'}
    assert isinstance(components['visreg'], VISRegLoss)

    with pytest.raises(ValueError, match='Unsupported LeWM'):
        build_loss_components(
            _loss_config('sigreg'),
            regularizers=VISWM_REGULARIZERS,
        )


def test_viswm_forward_uses_visreg_not_sigreg() -> None:
    class Model:
        def encode(self, batch):
            return {
                'emb': torch.ones(1, 4, 2),
                'act_emb': torch.zeros(1, 4, 2),
            }

        def predict(self, context, actions):
            return torch.zeros_like(context)

    module = SimpleNamespace(model=Model(), logged=None)
    module.visreg = lambda embeddings: embeddings.new_tensor(2.0)
    module.log_dict = (
        lambda losses, **kwargs: setattr(module, 'logged', losses)
    )
    cfg = _loss_config('visreg')

    output = lejepa_forward(
        module,
        {'action': torch.zeros(1, 4, 2)},
        'fit',
        cfg,
        regularizers=VISWM_REGULARIZERS,
    )

    assert torch.equal(
        output['loss'],
        output['pred_loss'] + 4.5 * output['visreg_loss'],
    )
    assert 'sigreg_loss' not in output
    assert 'fit/visreg_loss' in module.logged


def test_lewm_objective_registry_does_not_contain_visreg() -> None:
    with pytest.raises(ValueError, match='Unsupported LeWM'):
        build_loss_components(_loss_config('visreg'))
