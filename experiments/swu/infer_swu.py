from os import makedirs

from mipcandy import CTNormalize, JointTransform

from experiments.backbone import UNetNetwork, UNetTrainer
from experiments.vars import ExpConfig
from segwithu import SegWithUPredictor


def infer_swu(cfg: ExpConfig) -> None:
    predictor = SegWithUPredictor(cfg.num_classes, f"{cfg.project_weights}/SegWithUTrainer/fold{cfg.fold}",
                                  cfg.example_shape, device=cfg.device)
    backbone = UNetNetwork(cfg.num_classes, cfg.device).build_network_from_checkpoint(
        cfg.example_shape,
        f"{cfg.project_weights}/{UNetTrainer.__name__}/fold{cfg.fold}/checkpoint_best.pth",
    ).to(cfg.device)
    backbone.eval()
    predictor.backbone = backbone
    predictor.spatial_dims = cfg.spatial_dims
    predictor.feature_ch = int(backbone.filters[0])
    predictor.tap_module_names = ("upsamples.1.conv_block", "upsamples.2.conv_block", "output_block")
    predictor.tap_channels = (256, 128, 32)
    dataset = cfg.dataset("Ts")
    annotations = cfg.annotations(dataset)
    norm = CTNormalize(*annotations.intensity_stats())
    dataset.set_transform(JointTransform(image_only=norm))
    output_folder = f"{cfg.project_weights}/swu_out{cfg.fold}"
    makedirs(output_folder, exist_ok=True)
    predictor.predict_to_files(dataset, output_folder)
