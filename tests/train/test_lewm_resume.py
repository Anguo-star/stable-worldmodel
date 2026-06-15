from pathlib import Path

from scripts.train.lewm import get_resume_checkpoint_path


def test_resume_checkpoint_path_does_not_require_existing_file(tmp_path):
    run_dir = tmp_path / 'checkpoints' / 'lewm_mt3_lance'

    path = get_resume_checkpoint_path(run_dir, 'lewm_mt3_lance')

    assert path == Path(run_dir, 'lewm_mt3_lance_weights.ckpt')
    assert not path.exists()
