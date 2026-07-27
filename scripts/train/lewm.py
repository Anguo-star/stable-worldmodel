from functools import partial
import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from torch.utils.data import ConcatDataset as TorchConcatDataset

from stable_worldmodel.data import (
    BalancedConcatDataset,
    column_normalizer as get_column_normalizer,
    load_multitask_datasets,
)
from stable_worldmodel.loggers import build_training_logger
from stable_worldmodel.wm.loss import SIGReg, VCReg, VISRegLoss
from stable_worldmodel.wm.utils import save_pretrained


_REPRESENTATION_REGULARIZERS = {
    'sigreg': SIGReg,
    'visreg': VISRegLoss,
}


def get_representation_regularizer_name(cfg) -> str:
    """Resolve the active marginal representation regularizer."""

    name = str(cfg.loss.get('regularizer', 'sigreg')).strip().lower()
    if name not in _REPRESENTATION_REGULARIZERS:
        supported = ', '.join(sorted(_REPRESENTATION_REGULARIZERS))
        raise ValueError(
            f'Unsupported LeWM representation regularizer {name!r}; '
            f'expected one of: {supported}'
        )
    if cfg.loss.get(name) is None:
        raise ValueError(
            f'Missing loss.{name} configuration for active regularizer'
        )
    return name


def build_loss_components(cfg) -> dict[str, torch.nn.Module]:
    """Instantiate only the loss modules used by the selected objective."""

    regularizer_name = get_representation_regularizer_name(cfg)
    regularizer_cfg = cfg.loss.get(regularizer_name)
    kwargs = regularizer_cfg.get('kwargs', {}) or {}
    components = {
        regularizer_name: _REPRESENTATION_REGULARIZERS[regularizer_name](
            **kwargs
        )
    }
    if any(
        cfg.loss.get(name) is not None and cfg.loss.get(name).enabled
        for name in ('std', 'std_t', 'cov', 'cov_t')
    ):
        components['vc_reg'] = VCReg()
    return components


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    regularizer_name = get_representation_regularizer_name(cfg)
    regularizer_cfg = cfg.loss.get(regularizer_name)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    regularizer_loss_key = f'{regularizer_name}_loss'
    regularizer = getattr(self, regularizer_name)
    output[regularizer_loss_key] = regularizer(emb.transpose(0, 1))
    output['loss'] = (
        output['pred_loss']
        + regularizer_cfg.weight * output[regularizer_loss_key]
    )
    active_vcreg_names = [
        name
        for name in ('std', 'std_t', 'cov', 'cov_t')
        if cfg.loss.get(name) is not None and cfg.loss.get(name).enabled
    ]
    if active_vcreg_names:
        output.update(self.vc_reg(emb))
    for name in active_vcreg_names:
        regularizer_cfg = cfg.loss.get(name)
        loss_key = f'{name}_loss'
        output['loss'] = (
            output['loss'] + regularizer_cfg.weight * output[loss_key]
        )

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def build_sample_transform(dataset, cfg, keys_to_load):
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]
    normalizers_cfg = cfg.data.dataset.get('normalizers', {}) or {}
    for col in keys_to_load:
        if col.startswith('pixels'):
            continue
        method = normalizers_cfg.get(col, 'zscore')
        transforms.append(get_column_normalizer(dataset, col, col, method))
    return spt.data.transforms.Compose(*transforms)


def build_single_dataset(cfg, cache_dir):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    dataset_cfg.pop('mode', None)
    source = 'local cache: ' + cache_dir if cache_dir else 'default location'
    print(f'Loading dataset "{dataset_name}" from {source}')
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    dataset.transform = build_sample_transform(
        dataset, cfg, dataset_cfg.get('keys_to_load', [])
    )
    return dataset, dataset.get_dim('action')


def build_multitask_dataset_list(cfg, cache_dir):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)

    def transform_factory(dataset, item_cfg):
        keys_to_load = item_cfg.get(
            'keys_to_load', dataset_cfg.get('keys_to_load', [])
        )
        return build_sample_transform(dataset, cfg, keys_to_load)

    return load_multitask_datasets(
        dataset_cfg,
        cache_dir=cache_dir,
        transform_factory=transform_factory,
    )


def split_dataset(dataset, cfg, generator):
    # `spt.data.random_split` returns a Subset that accepts trainer
    # back-references from `spt.data.DataModule`. With spawn-based Lance
    # workers, pickling that trainer can traverse into
    # `spt.Module.forward`, which is installed as MethodType(partial(...)).
    # PyTorch's plain Subset avoids that unused back-reference.
    return torch.utils.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )


def build_data_loaders(cfg):
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    generator = torch.Generator().manual_seed(cfg.seed)
    dataset_mode = str(cfg.data.dataset.get('mode', 'single')).lower()

    if dataset_mode == 'multitask':
        datasets, action_dim = build_multitask_dataset_list(cfg, cache_dir)
        train_sets, val_sets = [], []
        for dataset in datasets:
            train_set, val_set = split_dataset(dataset, cfg, generator)
            train_sets.append(train_set)
            val_sets.append(val_set)

        sampling = str(cfg.data.dataset.get('sampling', 'balanced')).lower()
        if sampling == 'balanced':
            train_set = BalancedConcatDataset(train_sets)
        elif sampling == 'concat':
            train_set = TorchConcatDataset(train_sets)
        else:
            raise ValueError(f'Unsupported multitask sampling: {sampling}')

        if cfg.data.dataset.get('balance_val', False):
            val_set = BalancedConcatDataset(val_sets)
        else:
            val_set = TorchConcatDataset(val_sets)
    elif dataset_mode in {'single', 'default'}:
        dataset, action_dim = build_single_dataset(cfg, cache_dir)
        train_set, val_set = split_dataset(dataset, cfg, generator)
    else:
        raise ValueError(f'Unsupported dataset mode: {dataset_mode}')

    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * action_dim
        )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=generator,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)
    return train, val


def build_logger(cfg):
    """Backward-compatible wrapper for callers of the LeWM entry point."""

    return build_training_logger(cfg)


def get_resume_checkpoint_path(run_dir: Path, output_model_name: str) -> Path:
    return run_dir / f'{output_model_name}_weights.ckpt'


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    train, val = build_data_loaders(cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    loss_components = build_loss_components(cfg)
    world_model = spt.Module(
        model=world_model,
        **loss_components,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = build_logger(cfg)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = get_resume_checkpoint_path(run_dir, cfg.output_model_name)
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
    )

    manager()
    return


if __name__ == '__main__':
    run()
