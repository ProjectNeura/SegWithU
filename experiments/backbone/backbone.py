from os import PathLike
from typing import override

import torch
from mipcandy import AmbiguousShape, WithNetwork, Device, SegmentationTrainer, Predictor
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from torch import nn
from torch.utils.data import DataLoader


class UNetNetwork(WithNetwork):
    def __init__(self, num_classes: int, device: Device, *, deep_supervision: bool = True) -> None:
        WithNetwork.__init__(self, device)
        self.num_classes: int = num_classes
        self.deep_supervision: bool = deep_supervision

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        kernel_size = [[3, 3, 3]] * 5
        strides = [[1, 1, 1]] + [[2, 2, 2]] * 4
        return DynUNet(
            spatial_dims=3,
            in_channels=example_shape[0],
            out_channels=self.num_classes,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=strides,
            deep_supervision=self.deep_supervision,
            deep_supr_num=2,
            res_block=True
        )


class UNetTrainer(UNetNetwork, SegmentationTrainer):
    deep_supervision_scales = (1, 2, 4)

    def __init__(self, num_classes: int, trainer_folder: str | PathLike[str], dataloader: DataLoader,
                 validation_dataloader: DataLoader, *, device: Device = "cpu", profiler: bool = False) -> None:
        UNetNetwork.__init__(self, num_classes, device)
        SegmentationTrainer.__init__(self, trainer_folder, dataloader, validation_dataloader, recoverable=False,
                                     device=device, profiler=profiler)


class UNetPredictor(UNetNetwork, Predictor):
    def __init__(self, num_classes: int, experiment_folder: str | PathLike[str], example_shape: AmbiguousShape, *,
                 device: Device = "cpu") -> None:
        UNetNetwork.__init__(self, num_classes, device)
        Predictor.__init__(self, experiment_folder, example_shape, device=device)

    @override
    def predict_image(self, image: torch.Tensor, *, batch: bool = False) -> torch.Tensor:
        self.lazy_load_model()
        image = image.to(self._device)
        if not batch:
            image = image.unsqueeze(0)
        padding_module = self.get_padding_module()
        if padding_module:
            image = padding_module(image)
        output = sliding_window_inference(image, self._example_shape[1:], 2, self._model, .5, "gaussian",
                                          device=self._device).squeeze(0)
        restoring_module = self.get_restoring_module()
        if restoring_module:
            output = restoring_module(output)
        return output if batch else output.squeeze(0)
