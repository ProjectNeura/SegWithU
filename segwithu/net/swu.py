from typing import Sequence, Literal

import torch
from torch import nn

from segwithu.net.blocks import ITTap, MultiITTap, MultiScaleFusion, R1PP, UncertaintyOutputs


class SegWithU(nn.Module):
    def __init__(self, backbone: nn.Module, spatial_dims: Literal[2, 3], feature_ch: int, num_classes: int, *,
                 num_probes: int = 8, use_aleatoric: bool = True, sigma_init: float = .1,
                 margin_gamma: float = 4, tap_module_names: Sequence[str] | None = None,
                 tap_channels: Sequence[int] | None = None) -> None:
        super().__init__()
        if (tap_module_names is None) != (tap_channels is None):
            raise ValueError("Expected `tap_module_names` and `tap_channels` to be provided together")
        self.spatial_dims: Literal[2, 3] = spatial_dims
        self.backbone: nn.Module = backbone
        self.tap_module_names: tuple[str, ...] | None = None if tap_module_names is None else tuple(tap_module_names)
        self.tap_channels: tuple[int, ...] | None = None if tap_channels is None else tuple(tap_channels)
        if self.tap_module_names is None:
            self.tap: ITTap | MultiITTap = ITTap(backbone)
            self.fusion: MultiScaleFusion | None = None
        else:
            tap_channels = self.tap_channels
            if tap_channels is None:
                raise RuntimeError("Expected `tap_channels` when `tap_module_names` are provided")
            if len(self.tap_module_names) < 1:
                raise ValueError("Expected at least 1 tap module name")
            if len(self.tap_module_names) != len(tap_channels):
                raise ValueError("Expected `tap_module_names` and `tap_channels` to have the same length")
            self.tap = MultiITTap(backbone, self.tap_module_names)
            self.fusion = MultiScaleFusion(spatial_dims, tap_channels, feature_ch)
        self.unc_head: R1PP = R1PP(
            spatial_dims=spatial_dims, feature_ch=feature_ch, num_classes=num_classes, num_probes=num_probes,
            use_aleatoric=use_aleatoric, sigma_init=sigma_init, margin_gamma=margin_gamma
        )

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def forward(self, x: torch.Tensor, *, return_uncertainty: bool = True) -> torch.Tensor | tuple[
        torch.Tensor, UncertaintyOutputs]:
        if isinstance(self.tap, MultiITTap):
            self.tap.clear()
        logits = self.backbone(x)
        if not return_uncertainty:
            return logits
        if isinstance(self.tap, MultiITTap):
            if self.fusion is None:
                raise RuntimeError("Expected `fusion` when `MultiITTap` is enabled")
            captured = self.tap.get()
            missing = [
                module_name
                for module_name, feature in zip(self.tap.module_names, captured, strict=True)
                if feature is None
            ]
            if missing:
                raise RuntimeError("MultiITTap did not capture all feature maps, missing "
                                   f"{missing}. Make sure the corresponding backbone modules are called during "
                                   "the forward pass")
            features = [feature for feature in captured if feature is not None]
            features = self.fusion(features)
        else:
            features = self.tap.last_feature_map
            if features is None:
                raise RuntimeError("ITTap did not capture the feature map, make sure "
                                   f"`backbone.{self.tap.output_module_name}` is called during the forward pass")
        return logits, self.unc_head(features, logits)

    def close(self) -> None:
        self.tap.close()

    def __del__(self) -> None:
        self.close()
