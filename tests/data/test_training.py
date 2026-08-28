import pytest

from stable_worldmodel.data import training


def test_initialize_training_seed_runs_before_construction(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        training.pl,
        'seed_everything',
        lambda seed, *, workers: calls.append((seed, workers)),
    )

    seed = training.initialize_training_seed('3072')

    assert seed == 3072
    assert calls == [(3072, True)]


@pytest.mark.parametrize('seed', [-1, 2**32])
def test_initialize_training_seed_rejects_out_of_range_values(
    monkeypatch, seed
) -> None:
    monkeypatch.setattr(
        training.pl,
        'seed_everything',
        lambda *args, **kwargs: pytest.fail('invalid seed must fail first'),
    )

    with pytest.raises(ValueError, match='Training seed must be in'):
        training.initialize_training_seed(seed)
