from dataclasses import dataclass, asdict
from typing import Sequence, Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class ITTap(nn.Module):
    """
    Intermediate Tensor Tap (ITTap)
    """

    def __init__(self, backbone: nn.Module, *, output_module_name: str = "output_block") -> None:
        super().__init__()
        if not hasattr(backbone, output_module_name):
            raise AttributeError(f"Expected `backbone` to have an attribute `{output_module_name}`")
        out_mod = getattr(backbone, output_module_name)
        self.backbone: nn.Module = backbone
        self.output_module_name: str = output_module_name
        self.last_feature_map: torch.Tensor | None = None

        def _pre_hook(_module: nn.Module, _inputs: Sequence[torch.Tensor]) -> None:
            x = _inputs[0]
            if not torch.is_tensor(x):
                raise TypeError(f"Unexpected non-tensor input to `{output_module_name}`")
            self.last_feature_map = x

        self._hook_handle: RemovableHandle | None = out_mod.register_forward_pre_hook(_pre_hook)

    def close(self) -> None:
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self) -> None:
        self.close()


class MultiITTap(nn.Module):
    """
    Multi Intermediate Tensor Tap (MultiITTap)
    """

    def __init__(self, backbone: nn.Module, module_names: Sequence[str]) -> None:
        super().__init__()
        if len(module_names) < 1:
            raise ValueError("Expected at least 1 module name")
        self.backbone: nn.Module = backbone
        self.module_names: tuple[str, ...] = tuple(module_names)
        if any(not module_name for module_name in self.module_names):
            raise ValueError("Expected non-empty module names")
        if len(set(self.module_names)) != len(self.module_names):
            raise ValueError("Expected `module_names` to be unique")
        self.feature_maps: dict[str, torch.Tensor] = {}
        self._hook_handles: dict[str, RemovableHandle] = {}

        for module_name in self.module_names:
            try:
                module = backbone.get_submodule(module_name)
            except AttributeError as exc:
                raise AttributeError(f"Expected `backbone` to have a submodule `{module_name}`") from exc

            def _pre_hook(_module: nn.Module, _inputs: Sequence[torch.Tensor], hooked_name: str = module_name) -> None:
                if len(_inputs) < 1:
                    raise RuntimeError(f"Expected `{hooked_name}` to receive at least 1 input")
                x = _inputs[0]
                if not torch.is_tensor(x):
                    raise TypeError(f"Unexpected non-tensor input to `{hooked_name}`")
                self.feature_maps[hooked_name] = x

            self._hook_handles[module_name] = module.register_forward_pre_hook(_pre_hook)

    def clear(self) -> None:
        self.feature_maps.clear()

    def get(self) -> list[torch.Tensor | None]:
        return [self.feature_maps.get(module_name) for module_name in self.module_names]

    def close(self) -> None:
        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles.clear()

    def __del__(self) -> None:
        self.close()


class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Fusion (MultiScaleFusion)

    Project, resize, and fuse multiple feature maps into a single feature tensor.
    """

    def __init__(self, spatial_dims: Literal[2, 3], in_channels_list: Sequence[int], out_channels: int) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"`spatial_dims` must be 2 or 3, got {spatial_dims}")
        if len(in_channels_list) < 1:
            raise ValueError("Expected at least 1 input feature map")
        if any(in_channels < 1 for in_channels in in_channels_list):
            raise ValueError(f"Expected positive channel counts, got {tuple(in_channels_list)}")
        if out_channels < 1:
            raise ValueError(f"`out_channels` must be positive, got {out_channels}")
        self.spatial_dims: Literal[2, 3] = spatial_dims
        self.in_channels_list: tuple[int, ...] = tuple(in_channels_list)
        self.out_channels: int = out_channels
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        norm = nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
        self.projections: nn.ModuleList = nn.ModuleList([
            conv(in_channels, out_channels, 1, bias=False)
            for in_channels in self.in_channels_list
        ])
        fused_channels = len(self.in_channels_list) * out_channels
        self.fusion_block: nn.Module = nn.Sequential(
            conv(fused_channels, out_channels, 3, padding=1, bias=False),
            norm(out_channels, affine=True),
            nn.GELU(),
            conv(out_channels, out_channels, 3, padding=1, bias=False),
            norm(out_channels, affine=True),
            nn.GELU(),
        )

    @staticmethod
    def _spatial_volume(shape: Sequence[int]) -> int:
        volume = 1
        for dim in shape:
            volume *= int(dim)
        return volume

    def _resize(self, feature: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        if tuple(feature.shape[2:]) == tuple(target_shape):
            return feature
        mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
        return nn.functional.interpolate(feature, size=tuple(target_shape), mode=mode, align_corners=False)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.in_channels_list):
            raise ValueError(f"Expected {len(self.in_channels_list)} feature maps, got {len(features)}")
        validated_features: list[torch.Tensor] = []
        for i, (feature, expected_in_channels) in enumerate(zip(features, self.in_channels_list, strict=True)):
            if not torch.is_tensor(feature):
                raise TypeError(f"Expected `features[{i}]` to be a tensor")
            if feature.ndim != self.spatial_dims + 2:
                raise ValueError(
                    f"Expected `features[{i}]` to have {self.spatial_dims + 2} dimensions, got {feature.ndim}"
                )
            if feature.shape[1] != expected_in_channels:
                raise ValueError(
                    f"Expected `features[{i}]` to have {expected_in_channels} channels, got {feature.shape[1]}"
                )
            validated_features.append(feature)
        target_shape = max((tuple(feature.shape[2:]) for feature in validated_features), key=self._spatial_volume)
        projected_features = [
            self._resize(projection(feature), target_shape)
            for feature, projection in zip(validated_features, self.projections, strict=True)
        ]
        return self.fusion_block(torch.cat(projected_features, dim=1))


@dataclass
class UncertaintyOutputs(object):
    # probe responses: [B, R, (D), H, W]
    v: torch.Tensor
    # voxel-wise maps: [B, 1, (D), H, W]
    voxel_epi: torch.Tensor
    voxel_ale: torch.Tensor | None
    voxel_cal: torch.Tensor
    voxel_rnk: torch.Tensor
    # optional extras: [B, 1, (D), H, W]
    weight_map: torch.Tensor
    margin_map: torch.Tensor
    delta_logits: torch.Tensor
    voxel_probe: torch.Tensor
    voxel_entropy: torch.Tensor
    residual_map: torch.Tensor
    anchor_map: torch.Tensor

    def dictify(self) -> dict[str, torch.Tensor]:
        return asdict(self)

    def temper_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.sqrt(1 + self.voxel_cal).clamp_min(1)


class R1PP(nn.Module):
    """
    Rank-1 Posterior Probes (R1PP)
    """

    def __init__(self, spatial_dims: Literal[2, 3], feature_ch: int, num_classes: int, *, num_probes: int = 8,
                 use_aleatoric: bool = True, sigma_init: float = .1, sigma_eps: float = 1e-6, margin_gamma: float = 4,
                 detach_weight: bool = True, calib_init_a: float = 1, calib_init_b: float = 0, calib_init_c: float = 0,
                 calib_eps: float = 1e-8) -> None:
        """
        :param spatial_dims: 2D or 3D
        :param feature_ch: number of feature channels (also known as the number of output channels of the last decoder)
        :param num_classes: number of classes, $C_{out}$
        :param num_probes: number of probes, $R$
        :param use_aleatoric: whether to use aleatoric uncertainty
        :param sigma_init: initial value for probe scales
        :param sigma_eps: smoothness of probe scales
        :param margin_gamma: margin gamma for focal loss
        :param detach_weight: whether to detach the weight from the gradient
        :param calib_init_a: initial value for calibration parameter $a$
        :param calib_init_b: initial value for calibration parameter $b$
        :param calib_eps: epsilon for calibration
        """
        super().__init__()
        if num_probes < 1:
            raise ValueError(f"Expected at least 1 probe, got {num_probes}")
        self.spatial_dims: Literal[2, 3] = spatial_dims
        self.feature_ch: int = feature_ch
        self.num_classes: int = num_classes
        self.num_probes: int = num_probes
        self.use_aleatoric: bool = use_aleatoric
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.probe_conv: nn.Module = conv(feature_ch, num_probes, 1, bias=False)
        self.probe_to_class: nn.Module = conv(num_probes, num_classes, 1, bias=False)
        alpha0 = torch.log(torch.exp(torch.tensor(sigma_init)) - 1)
        self.alpha: nn.Parameter = nn.Parameter(alpha0.repeat(num_probes))
        self.sigma_eps: float = sigma_eps
        branch_in_ch = 3 if use_aleatoric else 2
        self.calibration_head: nn.Module = conv(branch_in_ch, 1, 1, bias=True)
        self.ale_head: nn.Module | None = conv(feature_ch, 1, 1, bias=True) if use_aleatoric else None
        self.margin_gamma: float = margin_gamma
        self.detach_weight: bool = detach_weight
        self.err_a: nn.Parameter = nn.Parameter(torch.tensor(float(calib_init_a)))
        self.err_b: nn.Parameter = nn.Parameter(torch.tensor(float(calib_init_b)))
        self.err_c: nn.Parameter = nn.Parameter(torch.tensor(float(calib_init_c)))
        self.calib_eps: float = calib_eps

    def _margin_weight(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a margin map and a weight map.
        :param logits: logits of shape [B, C, (D), H, W]
        :return: (margin map of shape [B, 1, (D), H, W], weight map of shape [B, 1, (D), H, W])
        """
        probs = nn.functional.softmax(logits, dim=1)
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
        margin_for_w = margin.detach() if self.detach_weight else margin
        weight = torch.exp(-self.margin_gamma * margin_for_w)
        return margin, weight

    @staticmethod
    def _weighted_mean(u: torch.Tensor, w: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
        """
        :param u: [B, 1, (D), H, W]
        :param w: [B, 1, (D), H, W]
        :return: [B]
        """
        b = u.shape[0]
        u_flat = u.view(b, -1)
        w_flat = w.view(b, -1)
        num = (u_flat * w_flat).sum(dim=1)
        den = w_flat.sum(dim=1).clamp_min(eps)
        return num / den

    @staticmethod
    def _quantile(u: torch.Tensor, q: float) -> torch.Tensor:
        """
        :param u: [B, 1, (D), H, W]
        :param q: quantile in (0, 1)
        :return: [B]
        """
        u_flat = u.view(u.shape[0], -1)
        return torch.quantile(u_flat, q=q, dim=1)

    def sigma(self) -> torch.Tensor:
        r"""
        :return: $\sigma_r$ of shape [R]
        """
        return nn.functional.softplus(self.alpha) + self.sigma_eps

    def _deterministic_probe_patterns(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample_ids = torch.arange(1, self.num_probes + 1, device=device, dtype=dtype).unsqueeze(1)
        probe_ids = torch.arange(self.num_probes, device=device, dtype=dtype).unsqueeze(0)
        patterns = torch.cos(torch.pi * sample_ids * (probe_ids + .5) / float(self.num_probes))
        return torch.where(patterns >= 0, torch.ones_like(patterns), -torch.ones_like(patterns))

    def class_delta_from_v(self, v: torch.Tensor, *, z: torch.Tensor | None = None, scale: float = 1) -> torch.Tensor:
        """
        :param v: [B, R, (D), H, W]
        :param z: optional noise, same shape as v; ones if None
        :param scale: the scale factor for the noise
        :return: delta logits of shape [B, C, (D), H, W]
        """
        b, num_responses = v.shape[:2]
        spatial_ones = [1] * (v.ndim - 2)
        if z is None:
            z = torch.ones((b, num_responses, *spatial_ones), device=v.device, dtype=v.dtype)
        else:
            if z.ndim == 2:
                z = z.view(b, num_responses, *spatial_ones)
            elif z.ndim != v.ndim:
                raise ValueError("`z` must be [B, R, ...] or [B, R]")
        sigma = self.sigma().view(1, num_responses, *spatial_ones)
        pert = (scale * sigma * z) * v
        return self.probe_to_class(pert)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> UncertaintyOutputs:
        if features.ndim not in (4, 5):
            raise ValueError(f"`features.ndim` must be 4 (2D) or 5 (3D), got {features.ndim}")
        if logits.ndim != features.ndim:
            raise ValueError(f"`logits.ndim` ({logits.ndim}) must match `features.ndim` ({features.ndim})")
        v = self.probe_conv(features)
        margin_map, weight_map = self._margin_weight(logits)
        delta_logits = self.class_delta_from_v(v)
        base_probs = logits.softmax(dim=1)
        probe_patterns = self._deterministic_probe_patterns(v.device, v.dtype)
        delta_samples = torch.stack([
            self.class_delta_from_v(v, z=probe_patterns[i].unsqueeze(0).expand(v.shape[0], -1))
            for i in range(probe_patterns.shape[0])
        ], dim=0)
        pert_probs = (logits.unsqueeze(0) + delta_samples).softmax(dim=2)
        voxel_epi = pert_probs.var(dim=0, unbiased=False).sum(dim=1, keepdim=True)
        voxel_probe = v.square().mean(dim=1, keepdim=True)
        residual_map = delta_logits.square().mean(dim=1, keepdim=True)
        voxel_entropy = -(base_probs * torch.log(base_probs.clamp_min(1e-12))).sum(dim=1, keepdim=True)
        voxel_ale = None
        branch_in = [torch.log1p(voxel_epi + residual_map), margin_map]
        if self.use_aleatoric:
            ale_head = self.ale_head
            if ale_head is None:
                raise RuntimeError("Expected `ale_head` when `use_aleatoric=True`")
            voxel_ale = ale_head(features)
            voxel_ale = nn.functional.softplus(voxel_ale)
            branch_in.insert(1, torch.log1p(voxel_ale))
        branch_in = torch.cat(branch_in, dim=1)
        voxel_cal = self.calibration_head(branch_in)
        voxel_cal = nn.functional.softplus(voxel_cal)
        entropy_scale = torch.log(torch.tensor(float(self.num_classes), device=logits.device, dtype=logits.dtype))
        normalized_entropy = voxel_entropy / entropy_scale.clamp_min(1.0)
        anchor_map = (
            torch.log1p(voxel_epi)
            + .5 * torch.log1p(residual_map)
            + .25 * torch.log1p(voxel_cal)
            + .25 * normalized_entropy
            + weight_map
        )
        voxel_rnk = (
            (1 + .1 * torch.tanh(self.err_a)) * anchor_map
            + self.err_b
            + nn.functional.softplus(self.err_c) * weight_map
        )
        return UncertaintyOutputs(
            v=v, voxel_cal=voxel_cal, voxel_rnk=voxel_rnk, voxel_epi=voxel_epi, voxel_ale=voxel_ale,
            weight_map=weight_map, margin_map=margin_map, delta_logits=delta_logits,
            voxel_probe=voxel_probe, voxel_entropy=voxel_entropy, residual_map=residual_map, anchor_map=anchor_map
        )
