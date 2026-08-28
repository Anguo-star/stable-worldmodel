from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from scripts.train.pldm import pldm_forward
from scripts.train.prejepa import dinowm_forward


class _PLDMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = torch.nn.Identity()
        self.pred_proj = torch.nn.Identity()
        self.embeddings = torch.nn.Parameter(
            torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                ]
            )
        )
        self.prediction_scale = torch.nn.Parameter(torch.tensor(0.5))
        self.predict_grad_flags: list[tuple[bool, bool]] = []
        self.last_batch_keys: set[str] = set()

    def encode(self, batch):
        self.last_batch_keys = set(batch)
        return {
            'emb': self.embeddings,
            'act_emb': torch.zeros_like(self.embeddings),
        }

    def predict(self, context, actions):
        self.predict_grad_flags.append(
            (context.requires_grad, actions.requires_grad)
        )
        return context * self.prediction_scale


class _PreJEPAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extra_encoders = torch.nn.ModuleDict(
            {'action': torch.nn.Identity()}
        )
        self.predictor = torch.nn.Identity()
        self.embeddings = torch.nn.Parameter(
            torch.tensor(
                [
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    ],
                ]
            )
        )
        self.prediction_scale = torch.nn.Parameter(torch.tensor(0.5))
        self.predict_grad_flags: list[bool] = []
        self.last_batch_keys: set[str] = set()

    def encode(self, batch, **_kwargs):
        self.last_batch_keys = set(batch)
        output = dict(batch)
        output['emb'] = self.embeddings
        output['pixels_emb'] = self.embeddings[..., :2]
        output['action_emb'] = self.embeddings[..., 2:]
        return output

    def predict(self, embedding):
        self.predict_grad_flags.append(embedding.requires_grad)
        return embedding * self.prediction_scale


def _loss_cfg(enabled: bool = True):
    return {
        'conditional_joint': {
            'enabled': enabled,
            'weight': 0.25,
            'group_width': 2,
        }
    }


def _assert_predictor_only_gradient(output, model) -> None:
    encoder_gradient, predictor_gradient = torch.autograd.grad(
        output['conditional_joint_loss'],
        [model.embeddings, model.prediction_scale],
        allow_unused=True,
    )
    assert encoder_gradient is None
    assert predictor_gradient is not None
    assert torch.count_nonzero(predictor_gradient) == 1


def test_pldm_adapter_strips_metadata_and_routes_coja_to_predictor() -> None:
    model = _PLDMModel()
    module = SimpleNamespace(
        model=model,
        idm=lambda value: value[..., :2] * 0.0,
        path_straight=lambda value: value.new_tensor(0.0),
        pldm=lambda *_args: {},
        log_dict=lambda *_args, **_kwargs: None,
    )
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': 3, 'num_preds': 1},
            'loss': _loss_cfg(),
        }
    )
    batch = {
        'action': torch.zeros(2, 4, 2),
        'conditional_joint_group': torch.tensor([0, 0]),
    }

    output = pldm_forward(module, batch, 'train', cfg)

    assert model.last_batch_keys == {'action'}
    assert model.predict_grad_flags == [(True, False), (False, False)]
    assert torch.equal(
        output['loss'],
        output['pred_loss'] + 0.25 * output['conditional_joint_loss'],
    )
    _assert_predictor_only_gradient(output, model)


def test_prejepa_adapter_supports_patch_latents_without_model_metadata() -> None:
    model = _PreJEPAModel()
    module = SimpleNamespace(
        model=model,
        log_dict=lambda *_args, **_kwargs: None,
    )
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': 3, 'num_preds': 1},
            'backbone': {'is_video_encoder': False},
            'loss': _loss_cfg(),
        }
    )
    batch = {
        'pixels': torch.zeros(2, 4, 3, 2, 2),
        'action': torch.zeros(2, 4, 1),
        'conditional_joint_group': torch.tensor([0, 0]),
    }

    output = dinowm_forward(module, batch, 'train', cfg)

    assert model.last_batch_keys == {'pixels', 'action'}
    assert model.predict_grad_flags == [True, False]
    native_loss = torch.nn.functional.mse_loss(
        output['actionless_pred_emb'], output['actionless_target_emb']
    )
    assert torch.equal(
        output['loss'],
        native_loss + 0.25 * output['conditional_joint_loss'],
    )
    _assert_predictor_only_gradient(output, model)
