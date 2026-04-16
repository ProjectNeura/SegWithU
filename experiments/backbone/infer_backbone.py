from os import makedirs

import torch
from mipcandy import save_image, CTNormalize, JointTransform

from experiments.backbone.backbone import UNetTrainer, UNetPredictor
from experiments.vars import ExpConfig


def infer_backbone(cfg: ExpConfig) -> None:
    predictor = UNetPredictor(cfg.num_classes, f"{cfg.project_weights}/{UNetTrainer.__name__}/fold{cfg.fold}",
                              cfg.example_shape, device=cfg.device)
    dataset = cfg.dataset("Ts")
    annotations = cfg.annotations(dataset)
    norm = CTNormalize(*annotations.intensity_stats())
    dataset.set_transform(JointTransform(image_only=norm))
    makedirs(f"{cfg.project_weights}/out{cfg.fold}")
    dataset.device(device=cfg.device)
    with torch.no_grad():
        for idx, (image, _) in enumerate(dataset):
            pred = predictor.predict_image(image)
            save_image(pred, f"{cfg.project_weights}/out{cfg.fold}/{dataset.paths()[idx][1]}.nii.gz")
