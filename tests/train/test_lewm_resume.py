from pathlib import Path

from scripts.train.lewm import (
    SaveCkptCallback,
    get_existing_resume_checkpoint_path,
    get_resume_num_sanity_val_steps,
    get_resume_checkpoint_path,
    get_resume_weights_only,
)


def test_resume_checkpoint_path_does_not_require_existing_file(tmp_path):
    run_dir = tmp_path / 'checkpoints' / 'lewm_mt3_lance'

    path = get_resume_checkpoint_path(run_dir, 'lewm_mt3_lance')

    assert path == Path(run_dir, 'lewm_mt3_lance_weights.ckpt')
    assert not path.exists()


def test_existing_resume_checkpoint_is_none_for_fresh_run(tmp_path):
    run_dir = tmp_path / 'checkpoints' / 'lewm_mt3_lance'

    path = get_existing_resume_checkpoint_path(run_dir, 'lewm_mt3_lance')

    assert path is None


def test_existing_resume_checkpoint_preserves_real_file(tmp_path):
    run_dir = tmp_path / 'checkpoints' / 'lewm_mt3_lance'
    run_dir.mkdir(parents=True)
    expected = run_dir / 'lewm_mt3_lance_weights.ckpt'
    expected.touch()

    path = get_existing_resume_checkpoint_path(run_dir, 'lewm_mt3_lance')

    assert path == expected


def test_existing_resume_checkpoint_restores_full_training_state(tmp_path):
    checkpoint = tmp_path / 'last.ckpt'
    checkpoint.touch()

    assert get_resume_weights_only(None) is True
    assert get_resume_weights_only(checkpoint) is False
    assert get_resume_num_sanity_val_steps(None) == 1
    assert get_resume_num_sanity_val_steps(checkpoint) == 0


def test_epoch_callback_installs_full_state_recovery_checkpoint(
    tmp_path, monkeypatch
):
    saved_pretrained = []
    monkeypatch.setattr(
        'scripts.train.lewm.save_pretrained',
        lambda model, **kwargs: saved_pretrained.append((model, kwargs)),
    )

    class Strategy:
        def __init__(self):
            self.barriers = []

        def barrier(self, name):
            self.barriers.append(name)

    class Trainer:
        current_epoch = 2
        max_epochs = 10
        is_global_zero = True

        def __init__(self):
            self.strategy = Strategy()
            self.checkpoints = []

        def save_checkpoint(self, path, *, weights_only):
            self.checkpoints.append((path, weights_only))

    class Module:
        model = object()

    destination = tmp_path / 'run_weights.ckpt'
    callback = SaveCkptCallback(
        run_name='run',
        cfg={'seed': 3073},
        epoch_interval=1,
        full_state_checkpoint_path=destination,
    )
    trainer = Trainer()

    callback.on_train_epoch_end(trainer, Module())

    assert saved_pretrained[0][1]['filename'] == 'weights_epoch_3.pt'
    assert trainer.strategy.barriers == ['lewm_save_pretrained_complete']
    assert trainer.checkpoints == [(str(destination), False)]


def test_epoch_callback_keeps_full_state_save_opt_in(monkeypatch):
    saved_pretrained = []
    monkeypatch.setattr(
        'scripts.train.lewm.save_pretrained',
        lambda model, **kwargs: saved_pretrained.append((model, kwargs)),
    )

    class Trainer:
        current_epoch = 0
        max_epochs = 2
        is_global_zero = True

        def save_checkpoint(self, path, *, weights_only):
            raise AssertionError('full-state save must remain opt-in')

    class Module:
        model = object()

    callback = SaveCkptCallback(
        run_name='run',
        cfg={'seed': 3073},
        epoch_interval=1,
    )
    callback.on_train_epoch_end(Trainer(), Module())

    assert saved_pretrained[0][1]['filename'] == 'weights_epoch_1.pt'
