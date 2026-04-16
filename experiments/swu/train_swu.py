from dataclasses import asdict
from os import rename
from os.path import exists
from typing import cast

from mipcandy import RandomROIDataset, CTNormalize, JointTransform
from torch.utils.data import DataLoader

from experiments.backbone import UNetNetwork, UNetTrainer
from experiments.vars import ExpConfig
from segwithu import SegWithUTrainer


def train_swu(cfg: ExpConfig) -> None:
    dataset = cfg.dataset("Tr")
    annotations = cfg.annotations(dataset)
    dataset = RandomROIDataset(annotations, cfg.batch_size)
    dataset.roi_shape(roi_shape=cast(tuple[int, int] | tuple[int, int, int], cfg.example_shape[1:]))
    if cfg.fold == "all":
        train = dataset
        val = dataset.fold()[1]
    else:
        train, val = dataset.fold(fold=cfg.fold)
    norm = CTNormalize(*annotations.intensity_stats())
    train.set_transform(JointTransform(image_only=norm))
    val.set_transform(JointTransform(image_only=norm))
    val.preload(f"{cfg.project_weights}/valPreloaded{cfg.fold}")
    train_dl = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                          prefetch_factor=cfg.prefetch_factor, persistent_workers=True)
    val_dl = DataLoader(val, batch_size=1, shuffle=False, num_workers=cfg.val_num_workers,
                        prefetch_factor=cfg.val_prefetch_factor, persistent_workers=cfg.val_num_workers > 0)
    trainer = SegWithUTrainer(cfg.project_weights, train_dl, val_dl, device=cfg.device, profiler=cfg.profiler)
    trainer.num_classes = cfg.num_classes
    trainer.backbone = UNetNetwork(cfg.num_classes, cfg.device).build_network_from_checkpoint(
        cfg.example_shape, f"{cfg.project_weights}/{UNetTrainer.__name__}/fold{cfg.fold}/checkpoint_best.pth"
    )
    trainer.spatial_dims = cfg.spatial_dims
    trainer.tap_module_names = ("upsamples.1.conv_block", "upsamples.2.conv_block", "output_block")
    trainer.tap_channels = (256, 128, 32)
    trainer.feature_ch = 32
    trainer.train(100, note=f"train_swu, cfg={asdict(cfg)}", ema=False, compile_model=False,
                  early_stop_tolerance=10)
    target_folder = f"{trainer.trainer_folder()}/{trainer.trainer_variant()}/fold{cfg.fold}"
    if exists(target_folder):
        print(f"Target folder {target_folder} already exists. Folder {trainer.experiment_folder()} left unchanged")
    else:
        rename(trainer.experiment_folder(), target_folder)
