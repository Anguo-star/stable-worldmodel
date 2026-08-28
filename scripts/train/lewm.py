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
    configure_training_loader,
    initialize_training_seed,
    load_multitask_datasets,
    split_training_dataset,
)
from stable_worldmodel.loggers import build_training_logger
from stable_worldmodel.wm.loss import (
    ConditionalSIGReg,
    GroupBalancedSIGReg,
    JointTemporalCovarianceSIGReg,
    SIGReg,
    TemporallyCenteredSIGReg,
    VCReg,
)
from stable_worldmodel.wm.conditional_joint import (
    CONDITIONAL_JOINT_BATCH_KEY,
    conditional_joint_loss_terms,
    temporary_eval_modules,
)
from stable_worldmodel.wm.utils import save_pretrained


_REPRESENTATION_REGULARIZERS = {
    'conditional_sigreg': ConditionalSIGReg,
    'group_balanced_sigreg': GroupBalancedSIGReg,
    'joint_temporal_covariance_sigreg': JointTemporalCovarianceSIGReg,
    'predictive_joint_temporal_covariance_sigreg': (
        JointTemporalCovarianceSIGReg
    ),
    'sigreg': SIGReg,
    'temporally_centered_sigreg': TemporallyCenteredSIGReg,
}

_PAIR_METADATA_BATCH_KEYS = (
    'conditional_pairs',
    'conditional_active',
)

def get_representation_regularizer_name(
    cfg,
    *,
    regularizers=None,
) -> str:
    """Resolve the active representation regularizer."""

    regularizers = regularizers or _REPRESENTATION_REGULARIZERS
    name = str(cfg.loss.get('regularizer', 'sigreg')).strip().lower()
    if name not in regularizers:
        supported = ', '.join(sorted(regularizers))
        raise ValueError(
            f'Unsupported LeWM representation regularizer {name!r}; '
            f'expected one of: {supported}'
        )
    if cfg.loss.get(name) is None:
        raise ValueError(
            f'Missing loss.{name} configuration for active regularizer'
        )
    return name


def build_loss_components(
    cfg,
    *,
    regularizers=None,
) -> dict[str, torch.nn.Module]:
    """Instantiate only the loss modules used by the selected objective."""

    regularizers = regularizers or _REPRESENTATION_REGULARIZERS
    regularizer_name = get_representation_regularizer_name(
        cfg,
        regularizers=regularizers,
    )
    regularizer_cfg = cfg.loss.get(regularizer_name)
    kwargs = regularizer_cfg.get('kwargs', {}) or {}
    components = {
        regularizer_name: regularizers[regularizer_name](
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

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        full_state_checkpoint_path: Path | None = None,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.full_state_checkpoint_path = full_state_checkpoint_path

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch + 1
        save_epoch = epoch % self.epoch_interval == 0
        save_final = epoch == trainer.max_epochs
        if trainer.is_global_zero:
            if save_epoch or save_final:
                self._save(pl_module.model, epoch)

        if self.full_state_checkpoint_path is not None and (
            save_epoch or save_final
        ):
            # Every rank must enter Lightning's distributed checkpoint path;
            # the strategy itself restricts physical I/O to global rank zero.
            # Synchronize first so the lightweight save_pretrained artifact is
            # complete before installing the full-state recovery pointer.
            trainer.strategy.barrier('lewm_save_pretrained_complete')
            trainer.save_checkpoint(
                str(self.full_state_checkpoint_path),
                weights_only=False,
            )

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg, *, regularizers=None):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    regularizers = regularizers or _REPRESENTATION_REGULARIZERS
    regularizer_name = get_representation_regularizer_name(
        cfg,
        regularizers=regularizers,
    )
    regularizer_cfg = cfg.loss.get(regularizer_name)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    regularizer_kwargs = {}
    conditional_joint_group = batch.get(CONDITIONAL_JOINT_BATCH_KEY)
    # Pair and group metadata belong only to the auxiliary losses.  Strip them
    # unconditionally so the encoder/predictor boundary stays identical to
    # native LeWM regardless of which regularizer is active.
    metadata_keys = set(_PAIR_METADATA_BATCH_KEYS)
    metadata_keys.add(CONDITIONAL_JOINT_BATCH_KEY)
    model_batch = {
        key: value for key, value in batch.items() if key not in metadata_keys
    }
    if regularizer_name in {
        'conditional_sigreg',
        'group_balanced_sigreg',
    }:
        pairs = batch.get('conditional_pairs')
        active = batch.get('conditional_active')
        if (pairs is None) != (active is None):
            raise ValueError(
                'conditional_pairs and conditional_active must be supplied '
                'together'
            )
        if pairs is not None:
            regularizer_kwargs = {
                'pairs': pairs,
                'active': active,
            }

    output = self.model.encode(model_batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    regularizer_embeddings = emb
    if regularizer_name == 'predictive_joint_temporal_covariance_sigreg':
        regularizer_embeddings = torch.cat(
            [emb[:, :n_preds], pred_emb],
            dim=1,
        )
        if regularizer_embeddings.shape != emb.shape:
            raise ValueError(
                'Predictive JTCov trajectory must match the encoder '
                f'trajectory shape: predictive={tuple(regularizer_embeddings.shape)}, '
                f'encoder={tuple(emb.shape)}'
            )
    regularizer_loss_key = f'{regularizer_name}_loss'
    regularizer = getattr(self, regularizer_name)
    output[regularizer_loss_key] = regularizer(
        regularizer_embeddings.transpose(0, 1),
        **regularizer_kwargs,
    )
    output['loss'] = (
        output['pred_loss']
        + regularizer_cfg.weight * output[regularizer_loss_key]
    )
    conditional_joint_cfg = cfg.loss.get('conditional_joint')
    conditional_joint_enabled = bool(
        conditional_joint_cfg is not None
        and conditional_joint_cfg.get('enabled', False)
    )
    if conditional_joint_enabled and stage in {'fit', 'train'}:
        if conditional_joint_group is None:
            raise ValueError(
                'loss.conditional_joint.enabled requires a training batch '
                f'with {CONDITIONAL_JOINT_BATCH_KEY!r}'
            )
        # The auxiliary is deliberately predictor-only: the native MSE and
        # SIGReg retain their ordinary end-to-end gradients, while the paired
        # relation cannot reshape the encoder, projector or action encoder.
        with temporary_eval_modules(
            self.model.predictor, self.model.pred_proj
        ):
            joint_pred_emb = self.model.predict(
                ctx_emb.detach(), ctx_act.detach()
            )
        joint_terms = conditional_joint_loss_terms(
            joint_pred_emb,
            tgt_emb,
            conditional_joint_group,
            group_width=int(conditional_joint_cfg.get('group_width', 2)),
        )
        output.update(joint_terms)
        output['loss'] = (
            output['loss']
            + float(conditional_joint_cfg.weight)
            * output['conditional_joint_loss']
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
    return split_training_dataset(
        dataset,
        train_fraction=float(cfg.train_split),
        generator=generator,
    )


def build_data_loaders(cfg, *, seed: int | None = None):
    seed = int(cfg.seed) if seed is None else int(seed)
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    generator = torch.Generator().manual_seed(seed)
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

    train_cfg = configure_training_loader(
        train_set,
        cfg.loader,
        seed=seed,
        generator=generator,
    )
    train = torch.utils.data.DataLoader(train_set, **train_cfg)
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


def get_existing_resume_checkpoint_path(
    run_dir: Path,
    output_model_name: str,
) -> Path | None:
    """Return a resume checkpoint only when a prior run actually saved one."""

    checkpoint = get_resume_checkpoint_path(run_dir, output_model_name)
    return checkpoint if checkpoint.is_file() else None


def get_resume_weights_only(checkpoint: Path | None) -> bool:
    """Use full-state semantics whenever an actual resume checkpoint exists."""

    return checkpoint is None


def get_resume_num_sanity_val_steps(checkpoint: Path | None) -> int:
    """Do not consume additional validation RNG after full-state restoration."""

    return 1 if checkpoint is None else 0


def run_training(
    cfg,
    *,
    regularizers=None,
):
    """Run the shared JEPA trainer with the entry's objective registry."""

    regularizers = regularizers or _REPRESENTATION_REGULARIZERS
    get_representation_regularizer_name(
        cfg,
        regularizers=regularizers,
    )
    seed = initialize_training_seed(cfg.seed)

    #########################
    ##       dataset       ##
    #########################

    train, val = build_data_loaders(cfg, seed=seed)

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
    loss_components = build_loss_components(cfg, regularizers=regularizers)
    world_model = spt.Module(
        model=world_model,
        **loss_components,
        forward=partial(
            lejepa_forward,
            cfg=cfg,
            regularizers=regularizers,
        ),
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

    ckpt_path = get_existing_resume_checkpoint_path(
        run_dir,
        cfg.output_model_name,
    )
    full_state_checkpoint_path = None
    if os.environ.get('LEWM_SAVE_FULL_RESUME_EACH_EPOCH', '0') == '1':
        full_state_checkpoint_path = get_resume_checkpoint_path(
            run_dir,
            cfg.output_model_name,
        )
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=1,
        full_state_checkpoint_path=full_state_checkpoint_path,
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=get_resume_num_sanity_val_steps(ckpt_path),
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=seed,
        ckpt_path=ckpt_path,
        weights_only=get_resume_weights_only(ckpt_path),
    )

    manager()
    return


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    return run_training(cfg)


if __name__ == '__main__':
    run()
