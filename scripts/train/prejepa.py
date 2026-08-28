import logging
import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from functools import partial
from stable_worldmodel.data import (
    column_normalizer as get_column_normalizer,
    configure_training_loader,
    initialize_training_seed,
    split_training_dataset,
)
from stable_worldmodel.loggers import build_training_logger
from stable_worldmodel.wm.conditional_joint import (
    CONDITIONAL_JOINT_BATCH_KEY,
    conditional_joint_loss_terms,
    temporary_eval_modules,
)
from stable_worldmodel.wm.utils import save_pretrained
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoVideoProcessor


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def get_img_preprocessor(source, target, img_size=224):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


class VideoPipeline(spt.data.transforms.Transform):
    def __init__(self, processor, source='image', target='image'):
        super().__init__()
        self.processor, self.source, self.target = processor, source, target

    def __call__(self, x):
        frames = self.nested_get(x, self.source)
        self.nested_set(
            x,
            self.processor(frames, return_tensors='pt')[
                'pixel_values_videos'
            ].squeeze(0),
            self.target,
        )
        return x


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0:
            self._save(pl_module.model, epoch)
        if epoch == trainer.max_epochs:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _strip_action_dims(tensor, action_range):
    """Remove the action dimensions from the last axis."""
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def dinowm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    conditional_joint_group = batch.get(CONDITIONAL_JOINT_BATCH_KEY)
    model_batch = {
        key: value
        for key, value in batch.items()
        if key != CONDITIONAL_JOINT_BATCH_KEY
    }
    for key in self.model.extra_encoders:
        model_batch[key] = torch.nan_to_num(
            model_batch[key], 0.0
        ).squeeze()

    batch = self.model.encode(
        model_batch,
        target='emb',
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    embedding = batch['emb'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch['emb'][:, cfg.wm.num_preds :, ...].detach()

    # Per-modality losses
    pixels_dim = batch['pixels_emb'].size(-1)
    batch['pixels_loss'] = F.mse_loss(
        pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim]
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    # Actionless embeddings (for probes and total loss)
    batch['actionless_emb'] = _strip_action_dims(batch['emb'], action_range)
    batch['actionless_prev_emb'] = _strip_action_dims(embedding, action_range)
    batch['actionless_pred_emb'] = _strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_emb'] = _strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = F.mse_loss(
        batch['actionless_pred_emb'],
        batch['actionless_target_emb'].detach(),
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
        with temporary_eval_modules(self.model.predictor):
            joint_pred_embedding = self.model.predict(embedding.detach())
        joint_prediction = _strip_action_dims(
            joint_pred_embedding, action_range
        )
        batch.update(
            conditional_joint_loss_terms(
                joint_prediction,
                batch['actionless_target_emb'],
                conditional_joint_group,
                group_width=int(
                    conditional_joint_cfg.get('group_width', 2)
                ),
            )
        )
        batch['loss'] = (
            batch['loss']
            + float(conditional_joint_cfg.weight)
            * batch['conditional_joint_loss']
        )

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='./config', config_name='prejepa')
def run(cfg):
    seed = initialize_training_seed(cfg.seed)

    # --- Dataset ---
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    keys_to_load = ['pixels'] + encoding_keys

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{cfg.dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get('encoding', {})
    ]

    if cfg.backbone.get('is_video_encoder', False):
        processor = AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source='pixels', target='pixels'),
            spt.data.transforms.Resize(
                cfg.image_size, source='pixels', target='pixels'
            ),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.wm.get('encoding', {}):
            if key not in dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = (
                dim if key != 'action' else dim * cfg.frameskip
            )

    rnd_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = split_training_dataset(
        dataset,
        train_fraction=float(cfg.train_split),
        generator=rnd_gen,
        fallback=spt.data.random_split,
    )

    train_config = configure_training_loader(
        train_set,
        {
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_workers,
            'drop_last': True,
            'persistent_workers': True,
            'pin_memory': True,
            'shuffle': True,
        },
        seed=seed,
        generator=rnd_gen,
    )
    train_loader = DataLoader(train_set, **train_config)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    encoder.eval()
    encoder.requires_grad_(False)

    is_cnn = hasattr(encoder.config, 'hidden_sizes')
    embed_dim = (
        encoder.config.hidden_sizes[-1]
        if is_cnn
        else encoder.config.hidden_size
    )
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    embed_dim += sum(cfg.wm.get('encoding', {}).values())

    if cfg.backbone.get('is_video_encoder', False):
        num_patches += num_patches * (cfg.n_steps // 4)

    with open_dict(cfg):
        cfg.model.predictor.dim = embed_dim
        cfg.model.predictor.num_patches = num_patches
        cfg.model.extra_encoders = {
            '_target_': 'torch.nn.ModuleDict',
            'modules': {
                key: {
                    '_target_': 'stable_worldmodel.wm.prejepa.module.Embedder',
                    'in_chans': cfg.extra_dims[key],
                    'emb_dim': int(cfg.wm.encoding[key]),
                }
                for key in cfg.wm.get('encoding', {})
            },
        }

    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder)

    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            'model_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer)}
        },
    )

    # --- Training ---
    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Run ID: {run_id}')

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    training_logger = build_training_logger(cfg)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=5,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
        num_sanity_val_steps=1,
        logger=training_logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        seed=seed,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s',
    )
    run()
