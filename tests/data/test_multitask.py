import pytest
import torch

import stable_worldmodel.data.multitask as multitask


class _FakeDataset:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.transform = None

    def get_dim(self, col):
        assert col == 'action'
        return self.action_dim


class _ScaleAction:
    def __call__(self, sample):
        sample = dict(sample)
        sample['action'] = sample['action'] * 2
        return sample


def _load_fake_dataset(name, **kwargs):
    return _FakeDataset(action_dim=int(name))


def test_multitask_strict_rejects_mismatched_action_dims(monkeypatch):
    monkeypatch.setattr(multitask, 'load_dataset', _load_fake_dataset)

    cfg = {
        'action_dim_policy': 'strict',
        'items': [{'name': '2'}, {'name': '5'}],
    }

    with pytest.raises(ValueError, match='requires identical action dims'):
        multitask.load_multitask_datasets(cfg)


def test_multitask_pad_to_max_pads_after_child_transform(monkeypatch):
    monkeypatch.setattr(multitask, 'load_dataset', _load_fake_dataset)

    cfg = {
        'action_dim_policy': 'pad_to_max',
        'items': [{'name': '2'}, {'name': '5'}],
    }

    datasets, action_dim = multitask.load_multitask_datasets(
        cfg,
        transform_factory=lambda dataset, item: _ScaleAction(),
    )

    sample = datasets[0].transform({'action': torch.ones(4, 2)})

    assert action_dim == 5
    torch.testing.assert_close(
        sample['action'][:, :2], torch.full((4, 2), 2.0)
    )
    torch.testing.assert_close(sample['action'][:, 2:], torch.zeros(4, 3))
    unpadded = datasets[1].transform({'action': torch.ones(4, 5)})
    assert unpadded['action'].shape == (4, 5)
