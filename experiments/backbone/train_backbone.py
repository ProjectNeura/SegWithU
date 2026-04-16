from dataclasses import asdict
from os import rename
from os.path import exists

from mipcandy import JointTransform, CTNormalize, RandomROIDataset
from torch.utils.data import DataLoader

from experiments.backbone.backbone import UNetTrainer
from experiments.vars import ExpConfig, SERVER


def train_backbone(cfg: ExpConfig) -> None:
    dataset = cfg.dataset("Tr")
    annotations = cfg.annotations(dataset)
    dataset = RandomROIDataset(annotations, cfg.batch_size)
    dataset.roi_shape(roi_shape=cfg.example_shape[1:])
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
    trainer = UNetTrainer(cfg.num_classes, cfg.project_weights, train_dl, val_dl, device=cfg.device,
                          profiler=cfg.profiler)
    trainer.train(200, note=f"train_backbone, cfg={asdict(cfg)}", early_stop_tolerance=200,
                  save_preview=not SERVER)
    target_folder = f"{trainer.trainer_folder()}/{trainer.trainer_variant()}/fold{cfg.fold}"
    if exists(target_folder):
        print(f"Target folder {target_folder} already exists. Folder {trainer.experiment_folder()} left unchanged")
    else:
        rename(trainer.experiment_folder(), target_folder)
