from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import scripts.train.lewm as lewm_train
from scripts.train.lewm import (
    build_data_loaders,
    build_loss_components,
    lejepa_forward,
    split_dataset,
)
from stable_worldmodel.wm.loss import (
    ConditionalSIGReg,
    GroupBalancedSIGReg,
    JointTemporalCovarianceSIGReg,
    SIGReg,
    TemporallyCenteredSIGReg,
    VCReg,
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


class _PairedModel:
    """Two-row batch so a width-two conditional-joint group can be formed."""

    def __init__(self):
        self.predictor = torch.nn.Identity()
        self.pred_proj = torch.nn.Identity()
        self.predict_grad_flags = []

    def encode(self, batch):
        self.last_batch_keys = set(batch)
        batch['emb'] = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [3.0, 4.0],
                ],
            ],
            requires_grad=True,
        )
        batch['act_emb'] = torch.zeros(2, 4, 2, requires_grad=True)
        return batch

    def predict(self, context, actions):
        self.predict_grad_flags.append(
            (context.requires_grad, actions.requires_grad)
        )
        return context * 0.0


class _SIGReg:
    def __call__(self, embeddings):
        return embeddings.new_tensor(2.0)


class _TemporallyCenteredSIGReg:
    def __call__(self, embeddings):
        return embeddings.new_tensor(23.0)


class _JointTemporalCovarianceSIGReg:
    def __init__(self):
        self.embeddings = None

    def __call__(self, embeddings):
        self.embeddings = embeddings
        return embeddings.new_tensor(29.0)


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


class _GroupBalancedSIGReg(_ConditionalSIGReg):
    def __call__(self, embeddings, *, pairs=None, active=None):
        super().__call__(embeddings, pairs=pairs, active=active)
        return embeddings.new_tensor(19.0)


class _VCReg:
    def __call__(self, embeddings):
        return {
            'std_loss': embeddings.new_tensor(3.0),
            'std_t_loss': embeddings.new_tensor(5.0),
            'cov_loss': embeddings.new_tensor(7.0),
            'cov_t_loss': embeddings.new_tensor(11.0),
        }


class _GradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Parameter(
            torch.arange(1.0, 9.0).reshape(1, 4, 2)
        )
        self.prediction_scale = torch.nn.Parameter(torch.tensor(0.5))

    def encode(self, batch):
        return {
            'emb': self.embeddings,
            'act_emb': torch.zeros_like(self.embeddings),
        }

    def predict(self, context, actions):
        return context * self.prediction_scale


class _SquareRegularizer:
    def __call__(self, embeddings):
        return embeddings.square().mean()


def _module():
    value = SimpleNamespace(
        model=_Model(),
        conditional_sigreg=_ConditionalSIGReg(),
        group_balanced_sigreg=_GroupBalancedSIGReg(),
        joint_temporal_covariance_sigreg=(
            _JointTemporalCovarianceSIGReg()
        ),
        predictive_joint_temporal_covariance_sigreg=(
            _JointTemporalCovarianceSIGReg()
        ),
        sigreg=_SIGReg(),
        temporally_centered_sigreg=_TemporallyCenteredSIGReg(),
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
    conditional_joint: dict | None = None,
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
        'group_balanced_sigreg': {
            'weight': 0.02,
            'kwargs': {
                'knots': 17,
                'num_proj': 32,
                'randomize_pair_orientation': True,
            },
        },
        'temporally_centered_sigreg': {
            'weight': 0.6,
            'kwargs': {'knots': 17, 'num_proj': 32},
        },
        'joint_temporal_covariance_sigreg': {
            'weight': 0.09,
            'kwargs': {
                'knots': 17,
                'num_proj': 32,
                'rho': None,
            },
        },
        'predictive_joint_temporal_covariance_sigreg': {
            'weight': 0.09,
            'kwargs': {
                'knots': 17,
                'num_proj': 32,
                'rho': None,
            },
        },
        'std': {'enabled': std, 'weight': 18.0},
        'std_t': {'enabled': False, 'weight': 0.7},
        'cov': {'enabled': cov, 'weight': 12.0},
        'cov_t': {'enabled': False, 'weight': 0.0},
    }
    if regularizer is not None:
        loss['regularizer'] = regularizer
    if conditional_joint is not None:
        loss['conditional_joint'] = conditional_joint
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


def test_temporally_centered_sigreg_replaces_native_without_metadata() -> None:
    module = _module()
    output = lejepa_forward(
        module,
        {'action': torch.zeros(1, 4, 2)},
        'fit',
        _config(
            std=False,
            cov=False,
            regularizer='temporally_centered_sigreg',
        ),
    )

    expected = (
        output['pred_loss']
        + 0.6 * output['temporally_centered_sigreg_loss']
    )
    assert torch.equal(output['loss'], expected)
    assert 'sigreg_loss' not in output
    assert module.model.last_batch_keys == {'action'}


def test_joint_temporal_covariance_replaces_native_without_metadata() -> None:
    module = _module()
    output = lejepa_forward(
        module,
        {'action': torch.zeros(1, 4, 2)},
        'fit',
        _config(
            std=False,
            cov=False,
            regularizer='joint_temporal_covariance_sigreg',
        ),
    )

    expected = (
        output['pred_loss']
        + 0.09 * output['joint_temporal_covariance_sigreg_loss']
    )
    assert torch.equal(output['loss'], expected)
    assert 'sigreg_loss' not in output
    assert module.model.last_batch_keys == {'action'}


def test_predictive_joint_temporal_covariance_uses_causal_predictions() -> None:
    module = _module()
    output = lejepa_forward(
        module,
        {'action': torch.zeros(1, 4, 2)},
        'fit',
        _config(
            std=False,
            cov=False,
            regularizer='predictive_joint_temporal_covariance_sigreg',
        ),
    )

    key = 'predictive_joint_temporal_covariance_sigreg_loss'
    assert torch.equal(output['loss'], output['pred_loss'] + 0.09 * output[key])
    population = (
        module.predictive_joint_temporal_covariance_sigreg.embeddings
    )
    assert population.shape == (4, 1, 2)
    assert torch.equal(population[0], torch.tensor([[0.0, 0.0]]))
    assert torch.count_nonzero(population[1:]) == 0
    assert 'sigreg_loss' not in output
    assert module.model.last_batch_keys == {'action'}


def test_predictive_joint_temporal_covariance_gradient_is_causal() -> None:
    module = _module()
    module.model = _GradientModel()
    module.predictive_joint_temporal_covariance_sigreg = _SquareRegularizer()

    output = lejepa_forward(
        module,
        {'action': torch.zeros(1, 4, 2)},
        'fit',
        _config(
            std=False,
            cov=False,
            regularizer='predictive_joint_temporal_covariance_sigreg',
        ),
    )
    regularizer = output[
        'predictive_joint_temporal_covariance_sigreg_loss'
    ]
    embedding_gradient, predictor_gradient = torch.autograd.grad(
        regularizer,
        [module.model.embeddings, module.model.prediction_scale],
    )

    assert torch.count_nonzero(predictor_gradient) == 1
    assert torch.count_nonzero(embedding_gradient[:, :3]) > 0
    assert torch.count_nonzero(embedding_gradient[:, 3]) == 0


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


def test_group_balanced_sigreg_receives_loss_only_pair_metadata() -> None:
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
            regularizer='group_balanced_sigreg',
        ),
    )

    expected = (
        output['pred_loss']
        + 0.02 * output['group_balanced_sigreg_loss']
    )
    assert torch.equal(output['loss'], expected)
    assert module.group_balanced_sigreg.call['pairs'] is pairs
    assert module.group_balanced_sigreg.call['active'] is active
    assert module.model.last_batch_keys == {'action'}


def _paired_module():
    module = _module()
    module.model = _PairedModel()
    return module


def _joint_config(*, enabled: bool, group_width: int = 2):
    return _config(
        std=False,
        cov=False,
        conditional_joint={
            'enabled': enabled,
            'weight': 0.25,
            'group_width': group_width,
        },
    )


def _paired_batch(group=None):
    batch = {'action': torch.zeros(2, 4, 2)}
    if group is not None:
        batch['conditional_joint_group'] = group
    return batch


def test_conditional_joint_disabled_leaves_native_objective_unchanged() -> None:
    module = _paired_module()
    output = lejepa_forward(
        module,
        _paired_batch(torch.tensor([0, 0])),
        'train',
        _joint_config(enabled=False),
    )

    expected = output['pred_loss'] + 0.09 * output['sigreg_loss']
    assert torch.equal(output['loss'], expected)
    assert 'conditional_joint_loss' not in output
    # Group metadata is stripped even when the auxiliary is off, so the
    # encoder boundary never depends on the sampling relation.
    assert module.model.last_batch_keys == {'action'}
    assert 'conditional_joint_group' not in output
    assert module.model.predict_grad_flags == [(True, True)]


def test_conditional_joint_adds_weighted_training_term() -> None:
    module = _paired_module()
    output = lejepa_forward(
        module,
        _paired_batch(torch.tensor([0, 0])),
        'fit',
        _joint_config(enabled=True),
    )

    expected = (
        output['pred_loss']
        + 0.09 * output['sigreg_loss']
        + 0.25 * output['conditional_joint_loss']
    )
    assert torch.equal(output['loss'], expected)
    assert output['conditional_joint_loss'].item() > 0.0
    assert torch.equal(
        output['conditional_joint_loss'],
        output['conditional_joint_response_loss']
        + output['conditional_joint_assignment_loss'],
    )
    assert module.model.last_batch_keys == {'action'}
    assert 'conditional_joint_group' not in output
    assert module.model.predict_grad_flags == [(True, True), (False, False)]


def test_conditional_joint_is_skipped_outside_training() -> None:
    module = _paired_module()
    output = lejepa_forward(
        module,
        _paired_batch(torch.tensor([0, 0])),
        'validate',
        _joint_config(enabled=True),
    )

    expected = output['pred_loss'] + 0.09 * output['sigreg_loss']
    assert torch.equal(output['loss'], expected)
    assert 'conditional_joint_loss' not in output


def test_conditional_joint_requires_group_metadata() -> None:
    module = _paired_module()
    with pytest.raises(ValueError, match='conditional_joint_group'):
        lejepa_forward(
            module,
            _paired_batch(),
            'fit',
            _joint_config(enabled=True),
        )


def test_conditional_joint_rejects_mismatched_group_width() -> None:
    module = _paired_module()
    with pytest.raises(ValueError, match='groups changed width'):
        lejepa_forward(
            module,
            _paired_batch(torch.tensor([0, 0])),
            'fit',
            _joint_config(enabled=True, group_width=3),
        )


def test_conditional_joint_requires_an_active_group() -> None:
    module = _paired_module()
    with pytest.raises(ValueError, match='no active group'):
        lejepa_forward(
            module,
            _paired_batch(torch.full((2,), -1, dtype=torch.long)),
            'fit',
            _joint_config(enabled=True),
        )


def test_split_dataset_delegates_to_an_optional_dataset_training_hook() -> None:
    sentinel = (object(), object())

    class _DatasetWithTrainingSplit:
        def split_for_training(self, *, train_fraction, generator):
            self.call = {
                'train_fraction': train_fraction,
                'generator': generator,
            }
            return sentinel

    dataset = _DatasetWithTrainingSplit()
    generator = torch.Generator().manual_seed(3072)

    result = split_dataset(
        dataset,
        SimpleNamespace(train_split=0.9),
        generator,
    )

    assert result is sentinel
    assert dataset.call == {
        'train_fraction': 0.9,
        'generator': generator,
    }


def test_build_data_loaders_applies_optional_train_loader_hook(
    monkeypatch,
) -> None:
    class _Rows(torch.utils.data.Dataset):
        def __init__(self, *, configurable=False):
            self.configurable = configurable
            self.call = None

        def __len__(self):
            return 8

        def __getitem__(self, index):
            return torch.tensor(index)

        def configure_train_loader(self, train_cfg, *, seed):
            assert self.configurable
            self.call = {'train_cfg': dict(train_cfg), 'seed': seed}
            return {**train_cfg, 'batch_size': 1, 'shuffle': False}

    train_set = _Rows(configurable=True)
    val_set = _Rows()

    class _Dataset:
        def split_for_training(self, *, train_fraction, generator):
            assert train_fraction == 0.9
            assert isinstance(generator, torch.Generator)
            return train_set, val_set

    monkeypatch.setattr(
        lewm_train,
        'build_single_dataset',
        lambda cfg, cache_dir: (_Dataset(), 2),
    )
    cfg = OmegaConf.create(
        {
            'seed': 3072,
            'train_split': 0.9,
            'data': {
                'dataset': {
                    'mode': 'single',
                    'frameskip': 5,
                }
            },
            'model': {'action_encoder': {'input_dim': 0}},
            'loader': {
                'batch_size': 4,
                'num_workers': 0,
                'drop_last': True,
                'shuffle': True,
            },
        }
    )

    train, val = build_data_loaders(cfg)

    assert train.dataset is train_set
    assert train.batch_size == 1
    assert train_set.call['seed'] == 3072
    assert train_set.call['train_cfg']['batch_size'] == 4
    assert val.dataset is val_set
    assert val.batch_size == 4
    assert cfg.model.action_encoder.input_dim == 10


def test_build_loss_components_instantiates_only_active_regularizers() -> None:
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

    group_balanced_components = build_loss_components(
        _config(
            std=False,
            cov=False,
            regularizer='group_balanced_sigreg',
        )
    )
    assert set(group_balanced_components) == {'group_balanced_sigreg'}
    assert isinstance(
        group_balanced_components['group_balanced_sigreg'],
        GroupBalancedSIGReg,
    )

    temporal_components = build_loss_components(
        _config(
            std=False,
            cov=False,
            regularizer='temporally_centered_sigreg',
        )
    )
    assert set(temporal_components) == {'temporally_centered_sigreg'}
    assert isinstance(
        temporal_components['temporally_centered_sigreg'],
        TemporallyCenteredSIGReg,
    )

    joint_temporal_components = build_loss_components(
        _config(
            std=False,
            cov=False,
            regularizer='joint_temporal_covariance_sigreg',
        )
    )
    assert set(joint_temporal_components) == {
        'joint_temporal_covariance_sigreg'
    }
    assert isinstance(
        joint_temporal_components['joint_temporal_covariance_sigreg'],
        JointTemporalCovarianceSIGReg,
    )

    predictive_joint_components = build_loss_components(
        _config(
            std=False,
            cov=False,
            regularizer='predictive_joint_temporal_covariance_sigreg',
        )
    )
    assert set(predictive_joint_components) == {
        'predictive_joint_temporal_covariance_sigreg'
    }
    assert isinstance(
        predictive_joint_components[
            'predictive_joint_temporal_covariance_sigreg'
        ],
        JointTemporalCovarianceSIGReg,
    )


def test_unknown_representation_regularizer_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match='Unsupported LeWM representation regularizer',
    ):
        build_loss_components(
            _config(std=False, cov=False, regularizer='unknown')
        )
