from typing import Literal, override

import torch
from mipcandy import DiceCELossWithLogits, Loss, SegmentationLoss, convert_logits_to_ids
from torch import nn
from torchmetrics.functional.classification import binary_average_precision, binary_auroc

from segwithu.net import UncertaintyOutputs


def apply_non_linearity(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    return x.sigmoid() if num_classes < 2 else x.softmax(dim=1)


class EverythingNeeded(object):
    def __init__(self, model: nn.Module, outputs: torch.Tensor, uncertainties: UncertaintyOutputs,
                 labels: torch.Tensor, *, eps: float = 1e-6) -> None:
        self.model: nn.Module = model
        self.outputs: torch.Tensor = outputs
        self.uncertainties: UncertaintyOutputs = uncertainties
        self.labels: torch.Tensor = labels
        self.eps: float = eps
        # temperature is used only for probabilistic scoring (NLL/Brier) – not for argmax predictions
        self.tempered_outputs: torch.Tensor = uncertainties.temper_logits(outputs)


class TrustLoss(SegmentationLoss):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, True)

    def forward(self, e: EverythingNeeded) -> torch.Tensor:
        delta_logits = e.uncertainties.delta_logits
        trust_logits = delta_logits.square().mean()
        base_probs = apply_non_linearity(e.outputs, self.num_classes)
        pert_probs = apply_non_linearity(e.outputs + delta_logits, self.num_classes)
        trust_probs = (pert_probs - base_probs).square().mean()
        return trust_logits + .25 * trust_probs


class NegativeLogLikelihood(Loss):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes: int = num_classes

    def forward(self, e: EverythingNeeded) -> torch.Tensor:
        probabilities = apply_non_linearity(e.tempered_outputs, self.num_classes)
        return -torch.log(probabilities.gather(dim=1, index=e.labels).clamp_min(1e-12)).mean()


class ErrorDetectionLoss(Loss):
    def __init__(self, *, mask: Literal["fg", "none"] = "none", background: int = 0) -> None:
        super().__init__()
        self.mask: Literal["fg", "none"] = mask
        self.background: int = background

    def get_uncertainty_scores_and_errors(self, e: EverythingNeeded, *,
                                          errors_dtype: torch.dtype = torch.bool) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        u = e.uncertainties.voxel_rnk
        # yo, be careful: define errors from the raw predictions (not tempered), otherwise sigma will fuck you
        err = (convert_logits_to_ids(e.outputs) != e.labels).to(errors_dtype)
        match self.mask:
            case "fg":
                m = (e.labels != self.background)
            case "none":
                m = torch.ones_like(err, dtype=torch.bool)
        return m, u[m], err[m]

    def forward(self, e: EverythingNeeded) -> tuple[torch.Tensor, dict[str, float]]:
        _, u, err = self.get_uncertainty_scores_and_errors(e)
        u = u.reshape(-1)
        err = err.reshape(-1)
        auprc = binary_average_precision(u, err)
        auroc = binary_auroc(u, err)
        return -auprc, {"auprc": auprc.item(), "auroc": auroc.item()}


class ErrorCorrelationLoss(ErrorDetectionLoss):
    def __init__(self, *, mask: Literal["fg", "none"] = "none", background: int = 0, tau: float = 1) -> None:
        super().__init__(mask=mask, background=background)
        self.tau: float = tau

    @override
    def forward(self, e: EverythingNeeded) -> torch.Tensor:
        _, u, err = self.get_uncertainty_scores_and_errors(e, errors_dtype=torch.float)
        # normalize per-batch for stable gradients and to prevent global sigma scaling from dominating.
        u = (u - u.mean()) / (u.std(unbiased=False) + e.eps)
        # logistic regression style: encourage s high when y=1 and low when y=0
        # equivalent to BCEWithLogits on score with target y.
        return nn.functional.binary_cross_entropy_with_logits(u / self.tau, err)


class PairwiseLoss(ErrorCorrelationLoss):
    def __init__(self, *, mask: Literal["fg", "none"] = "none", background: int = 0, num_pairs: int = 65536,
                 margin: float = 0, tau: float = 1) -> None:
        super().__init__(mask=mask, background=background, tau=tau)
        self.num_pairs: int = num_pairs
        self.margin: float = margin

    @override
    def forward(self, e: EverythingNeeded) -> torch.Tensor:
        _, u, err = self.get_uncertainty_scores_and_errors(e)
        err = err.reshape(-1)
        u = u.reshape(-1)
        # normalize so sigma scaling doesn't trivially dominate
        u = (u - u.mean()) / (u.std(unbiased=False) + e.eps)
        pos = torch.where(err)[0]
        neg = torch.where(~err)[0]
        if pos.numel() == 0 or neg.numel() == 0:
            return torch.zeros((), device=u.device, dtype=u.dtype)
        # sample pairs
        k = min(self.num_pairs, pos.numel() * neg.numel())
        pos_idx = pos[torch.randint(0, pos.numel(), (k,), device=u.device)]
        neg_idx = neg[torch.randint(0, neg.numel(), (k,), device=u.device)]
        up = u[pos_idx]
        un = u[neg_idx]
        # want up >= un + margin => minimize softplus((un - up + margin)/tau)
        return nn.functional.softplus((un - up + self.margin) / self.tau).mean()


class TailLoss(ErrorDetectionLoss):
    def __init__(self, *, temperature: float = .25, mask: Literal["fg", "none"] = "none", background: int = 0) -> None:
        super().__init__(mask=mask, background=background)
        self.temperature: float = temperature

    @override
    def forward(self, e: EverythingNeeded) -> torch.Tensor:
        _, u, err = self.get_uncertainty_scores_and_errors(e, errors_dtype=torch.float)
        u = u.reshape(-1)
        err = err.reshape(-1)
        w = torch.softmax(-u / self.temperature, dim=0)
        return (w * err).sum()


class AnchorLoss(Loss):
    def forward(self, e: EverythingNeeded) -> torch.Tensor:
        u = e.uncertainties.voxel_rnk
        a = e.uncertainties.anchor_map.detach()
        u = (u - u.mean()) / (u.std(unbiased=False) + e.eps)
        a = (a - a.mean()) / (a.std(unbiased=False) + e.eps)
        return nn.functional.smooth_l1_loss(u, a)


class ResidualLoss(Loss):
    def forward(self, e: EverythingNeeded) -> torch.Tensor:
        easy_weight = (1 - e.uncertainties.weight_map).detach()
        return (easy_weight * e.uncertainties.residual_map).mean()


class BrierLoss(SegmentationLoss):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, True)

    def forward(self, e: EverythingNeeded) -> torch.Tensor:
        probabilities = apply_non_linearity(e.tempered_outputs, self.num_classes)
        return (probabilities - self.logitfy_no_grad(e.labels)).pow(2).sum(dim=1).mean()


class SegWithUCriterion(SegmentationLoss):
    def __init__(self, num_classes: int, *, lambda_seg: float = 0, lambda_nll: float = .5,
                 lambda_ec: float = .25, lambda_pairwise: float = .25, lambda_tail: float = .25,
                 lambda_trust: float = .05, lambda_anchor: float = .05, lambda_residual: float = .05,
                 include_background: bool = False, background: int = 0) -> None:
        super().__init__(num_classes, include_background)
        # refine segmentation, which we don't since it's pre-trained
        # however, flat loss needs it so it can't be None
        self.seg_loss: SegmentationLoss = DiceCELossWithLogits(num_classes)
        # probabilistic quality, also used for validation
        self.nll_loss: Loss = NegativeLogLikelihood(num_classes)
        self.ec_loss: Loss | None = ErrorCorrelationLoss(background=background) if lambda_ec != 0 else None
        self.pairwise_loss: Loss | None = PairwiseLoss(background=background) if lambda_pairwise != 0 else None
        self.tail_loss: Loss | None = TailLoss() if lambda_tail != 0 else None
        self.trust_loss: Loss | None = TrustLoss(num_classes) if lambda_trust != 0 else None
        self.anchor_loss: Loss | None = AnchorLoss() if lambda_anchor != 0 else None
        self.residual_loss: Loss | None = ResidualLoss() if lambda_residual != 0 else None
        # validation-only metrics
        self.brier_loss: Loss = BrierLoss(num_classes)
        self.ed_loss: Loss = ErrorDetectionLoss(background=background)
        self.brier_loss.requires_grad_(False)
        self.ed_loss.requires_grad_(False)

        self.lambda_seg: float = lambda_seg
        self.lambda_nll: float = lambda_nll
        self.lambda_ec: float = lambda_ec
        self.lambda_pairwise: float = lambda_pairwise
        self.lambda_tail: float = lambda_tail
        self.lambda_trust: float = lambda_trust
        self.lambda_anchor: float = lambda_anchor
        self.lambda_residual: float = lambda_residual

    @staticmethod
    def _make_z(v: torch.Tensor) -> torch.Tensor:
        return torch.randn((*v.shape[:2], *([1] * (v.ndim - 2))), device=v.device, dtype=v.dtype)

    def forward(self, model: nn.Module, outputs: torch.Tensor, uncertainties: UncertaintyOutputs,
                labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        metrics = {
            "avg voxel epi": uncertainties.voxel_epi.mean().item(),
            "avg voxel probe": uncertainties.voxel_probe.mean().item(),
            "avg voxel entropy": uncertainties.voxel_entropy.mean().item(),
            "avg residual": uncertainties.residual_map.mean().item(),
        }
        e = EverythingNeeded(model, outputs, uncertainties, labels)
        loss = torch.zeros((), device=outputs.device, dtype=outputs.dtype)
        if self.lambda_seg != 0:
            seg_loss, seg_metrics = self.seg_loss(e)
            loss += self.lambda_seg * seg_loss
            metrics.update(seg_metrics)
            metrics["seg loss"] = seg_loss.item()
        if self.lambda_nll != 0:
            nll = self.nll_loss(e)
            loss += self.lambda_nll * nll
            metrics["nll"] = nll.item()
        if self.ec_loss:
            ec_loss = self.ec_loss(e)
            loss += self.lambda_ec * ec_loss
            metrics["ec loss"] = ec_loss.item()
        if self.pairwise_loss:
            pairwise_loss = self.pairwise_loss(e)
            loss += self.lambda_pairwise * pairwise_loss
            metrics["pairwise loss"] = pairwise_loss.item()
        if self.tail_loss:
            tail_loss = self.tail_loss(e)
            loss += self.lambda_tail * tail_loss
            metrics["tail loss"] = tail_loss.item()
        if self.trust_loss:
            trust_loss = self.trust_loss(e)
            loss += self.lambda_trust * trust_loss
            metrics["trust loss"] = trust_loss.item()
        if self.anchor_loss:
            anchor_loss = self.anchor_loss(e)
            loss += self.lambda_anchor * anchor_loss
            metrics["anchor loss"] = anchor_loss.item()
        if self.residual_loss:
            residual_loss = self.residual_loss(e)
            loss += self.lambda_residual * residual_loss
            metrics["residual loss"] = residual_loss.item()
        if not self.validation_mode:
            return loss, metrics
        with torch.no_grad():
            if self.lambda_nll == 0:
                metrics["nll"] = self.nll_loss(e).item()
            metrics["brier"] = self.brier_loss(e).item()
            metrics.update(self.ed_loss(e)[1])
        return loss, metrics
