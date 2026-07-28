from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scripts.train.lewm import build_loss_components, lejepa_forward
from stable_worldmodel.wm.loss import (
    ConditionalSIGReg,
    SIGReg,
    VCReg,
    VISRegLoss,
)


class _Model:
    def encode(self, batch):
        self.last_batch_keys = set(batch)
        batch['emb'] = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            ]
        )
        batch['act_emb'] = torch.zeros(1, 4, 2)
        return batch

    def predict(self, context, actions):
        return torch.zeros_like(context)


class _SIGReg:
    def __call__(self, embeddings):
        return embeddings.new_tensor(2.0)


class _VISReg:
    def __call__(self, embeddings):
        return embeddings.new_tensor(13.0)


class _ConditionalSIGReg:
    def __init__(self):
        self.call = None

    def __call__(self, embeddings, *, pairs=None, active=None):
        self.call = {
            'embeddings': embeddings,
            'pairs': pairs,
            'active': active,
        }
        return embeddings.new_tensor(17.0)


class _VCReg:
    def __call__(self, embeddings):
        return {
            'std_loss': embeddings.new_tensor(3.0),
            'std_t_loss': embeddings.new_tensor(5.0),
            'cov_loss': embeddings.new_tensor(7.0),
            'cov_t_loss': embeddings.new_tensor(11.0),
        }


def _module():
    value = SimpleNamespace(
        model=_Model(),
        conditional_sigreg=_ConditionalSIGReg(),
        sigreg=_SIGReg(),
        visreg=_VISReg(),
        vc_reg=_VCReg(),
        logged=None,
    )
    value.log_dict = lambda losses, **kwargs: setattr(value, 'logged', losses)
    return value


def _config(
    *,
    std: bool,
    cov: bool,
    regularizer: str | None = None,
):
    loss = {
        'sigreg': {
            'weight': 0.09,
            'kwargs': {'knots': 17, 'num_proj': 32},
        },
        'conditional_sigreg': {
            'weight': 0.09,
            'kwargs': {
                'knots': 17,
                'num_proj': 32,
                'randomize_pair_orientation': True,
            },
        },
        'visreg': {
            'weight': 0.25,
            'kwargs': {
                'num_projections': 32,
                'lambda_scale': 1.0,
                'lambda_shape': 1.0,
                'lambda_center': 1.0,
            },
        },
        'std': {'enabled': std, 'weight': 18.0},
        'std_t': {'enabled': False, 'weight': 0.7},
        'cov': {'enabled': cov, 'weight': 12.0},
        'cov_t': {'enabled': False, 'weight': 0.0},
    }
    if regularizer is not None:
        loss['regularizer'] = regularizer
    return OmegaConf.create(
        {
            'wm': {'history_size': 3, 'num_preds': 1},
            'loss': loss,
        }
    )


def test_native_lewm_objective_is_unchanged_when_vcreg_disabled() -> None:
    module = _module()
    output = lejepa_forward(
        module,
        {'action': torch.zeros(1, 4, 2)},
        'fit',
        _config(std=False, cov=False),
    )

    expected = output['pred_loss'] + 0.09 * output['sigreg_loss']
    assert torch.equal(output['loss'], expected)
    assert 'std_loss' not in output
    assert 'cov_loss' not in output
    assert 'fit/std_loss' not in module.logged
    assert 'fit/cov_loss' not in module.logged


def test_visreg_replaces_sigreg_in_base_objective() -> None:
    module = _module()
    output = lejepa_forward(
        module,
        {'action': torch.zeros(1, 4, 2)},
        'fit',
        _config(std=False, cov=False, regularizer='visreg'),
    )

    expected = output['pred_loss'] + 0.25 * output['visreg_loss']
    assert torch.equal(output['loss'], expected)
    assert 'sigreg_loss' not in output
    assert 'fit/sigreg_loss' not in module.logged
    assert torch.equal(
        module.logged['fit/visreg_loss'],
        output['visreg_loss'],
    )


def test_std_cov_candidate_adds_only_declared_regularizers() -> None:
    module = _module()
    output = lejepa_forward(
        module,
        {'action': torch.zeros(1, 4, 2)},
        'fit',
        _config(std=True, cov=True),
    )

    expected = (
        output['pred_loss']
        + 0.09 * output['sigreg_loss']
        + 18.0 * output['std_loss']
        + 12.0 * output['cov_loss']
    )
    assert torch.equal(output['loss'], expected)


def test_conditional_sigreg_receives_loss_only_pair_metadata() -> None:
    module = _module()
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    active = torch.tensor([[False], [True], [False], [True]])
    batch = {
        'action': torch.zeros(1, 4, 2),
        'conditional_pairs': pairs,
        'conditional_active': active,
    }
    output = lejepa_forward(
        module,
        batch,
        'fit',
        _config(
            std=False,
            cov=False,
            regularizer='conditional_sigreg',
        ),
    )

    expected = (
        output['pred_loss']
        + 0.09 * output['conditional_sigreg_loss']
    )
    assert torch.equal(output['loss'], expected)
    assert module.conditional_sigreg.call['pairs'] is pairs
    assert module.conditional_sigreg.call['active'] is active
    assert module.model.last_batch_keys == {'action'}
    assert 'conditional_pairs' not in output
    assert 'conditional_active' not in output


def test_conditional_sigreg_requires_complete_pair_metadata() -> None:
    module = _module()
    with pytest.raises(ValueError, match='must be supplied together'):
        lejepa_forward(
            module,
            {
                'action': torch.zeros(1, 4, 2),
                'conditional_pairs': torch.tensor([[0, 1]]),
            },
            'fit',
            _config(
                std=False,
                cov=False,
                regularizer='conditional_sigreg',
            ),
        )


def test_build_loss_components_instantiates_only_active_regularizers() -> None:
    visreg_components = build_loss_components(
        _config(std=False, cov=False, regularizer='visreg')
    )
    assert set(visreg_components) == {'visreg'}
    assert isinstance(visreg_components['visreg'], VISRegLoss)

    sigreg_components = build_loss_components(
        _config(std=True, cov=False, regularizer='sigreg')
    )
    assert set(sigreg_components) == {'sigreg', 'vc_reg'}
    assert isinstance(sigreg_components['sigreg'], SIGReg)
    assert isinstance(sigreg_components['vc_reg'], VCReg)

    conditional_components = build_loss_components(
        _config(
            std=False,
            cov=False,
            regularizer='conditional_sigreg',
        )
    )
    assert set(conditional_components) == {'conditional_sigreg'}
    assert isinstance(
        conditional_components['conditional_sigreg'],
        ConditionalSIGReg,
    )


def test_unknown_representation_regularizer_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match='Unsupported LeWM representation regularizer',
    ):
        build_loss_components(
            _config(std=False, cov=False, regularizer='unknown')
        )
