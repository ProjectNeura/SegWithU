from typing import override, Literal

import torch
from mipcandy import WithNetwork, SegmentationTrainer, AmbiguousShape, TrainerToolbox, Loss, Params
from torch import nn, optim

from segwithu.net import SegWithU
from segwithu.crit import SegWithUCriterion


class SegWithUNetwork(WithNetwork):
    num_classes: int
    backbone: nn.Module
    spatial_dims: Literal[2, 3]
    feature_ch: int
    tap_module_names: tuple[str, ...] | None = None
    tap_channels: tuple[int, ...] | None = None

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        net = SegWithU(
            self.backbone,
            self.spatial_dims,
            self.feature_ch,
            self.num_classes,
            tap_module_names=self.tap_module_names,
            tap_channels=self.tap_channels,
        )
        net.freeze_backbone()
        return net


class SegWithUTrainer(SegWithUNetwork, SegmentationTrainer):
    @override
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        return optim.AdamW(params, 1e-3, betas=(.9, .999), eps=1e-8)

    @override
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(self._dataloader), eta_min=3e-4)

    @override
    def build_criterion(self) -> nn.Module:
        return SegWithUCriterion(self.num_classes)

    @override
    def _build_toolbox(self, num_epochs: int, example_shape: AmbiguousShape, compile_model: bool, ema: bool, *,
                       model: nn.Module | None = None) -> TrainerToolbox:
        if ema:
            raise ValueError("EMA is not compatible with SegWithU")
        if not model:
            model = self.load_model(example_shape, compile_model)
        model.freeze_backbone()
        optimizer = self.build_optimizer([p for p in model.parameters() if p.requires_grad])
        scheduler = self.build_scheduler(optimizer, num_epochs)
        criterion = self.build_criterion().to(self._device)
        return TrainerToolbox(model, optimizer, scheduler, criterion, None)

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        if self.deep_supervision:
            raise ValueError("Deep supervision is not supported for SegWithU")
        model = toolbox.model
        model.backbone.eval()
        outputs, uncertainties = model(images, return_uncertainty=True)
        loss, metrics = toolbox.criterion(model, outputs, uncertainties, labels)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in toolbox.model.parameters() if p.requires_grad], 12)
        return loss.item(), metrics

    @override
    def validate_case(self, idx: int, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        if self.deep_supervision:
            raise ValueError("Deep supervision is not supported for SegWithU")
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        model = toolbox.model
        output, uncertainties = model(image, return_uncertainty=True)
        if isinstance(toolbox.criterion, Loss):
            toolbox.criterion.validation_mode = True
        loss, metrics = toolbox.criterion(model, output, uncertainties, label)
        metrics["loss"] = loss.item()
        if isinstance(toolbox.criterion, Loss):
            toolbox.criterion.validation_mode = False
        return -loss.item(), metrics, uncertainties.voxel_epi.squeeze(0)

    @override
    def save_preview(self, image: torch.Tensor, label: torch.Tensor, output: torch.Tensor, *,
                     quality: float = .75) -> None:
        self._save_preview(image, "input", quality)
        self._save_preview(label.int(), "label", quality, is_label=True)
        self._save_preview(output, "uncertainty", quality)
