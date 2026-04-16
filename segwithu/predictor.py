from json import dump
from os import PathLike, mkdir

import torch
from mipcandy import AmbiguousShape, Device, Shape, save_image, PathBasedUnsupervisedDataset, PathBasedSupervisedDataset
from monai.inferers import sliding_window_inference
from torch import nn

from segwithu.crit import SegWithUCriterion
from segwithu.net import UncertaintyOutputs
from segwithu.trainer import SegWithUNetwork


class SegWithUPredictor(SegWithUNetwork):
    tap_module_names = ("upsamples.1.conv_block", "upsamples.2.conv_block", "output_block")
    tap_channels = (256, 128, 32)

    def __init__(self, num_classes: int, experiment_folder: str | PathLike[str],
                 example_shape: AmbiguousShape, *, checkpoint: str = "checkpoint_best.pth", device: Device = "cpu") -> None:
        super().__init__(device)
        self.num_classes: int = num_classes
        if not (3 <= len(example_shape) <= 4):
            raise ValueError("`example_shape` must be [C, H, W] or [C, D, H, W]")
        self._roi_shape: Shape = example_shape[1:]
        self._experiment_folder: str = str(experiment_folder)
        self._example_shape: AmbiguousShape = example_shape
        self._checkpoint: str = checkpoint
        self._model: nn.Module | None = None
        self._criterion: SegWithUCriterion = SegWithUCriterion(self.num_classes)
        self._criterion.validation_mode = True

    def lazy_load_model(self) -> None:
        if self._model:
            return
        self._model = self.load_model(self._example_shape, False,
                                      path=f"{self._experiment_folder}/{self._checkpoint}")
        self._model.eval()

    def predictor(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.lazy_load_model()
        with torch.no_grad():
            outputs, uncertainties = self._model(x, return_uncertainty=True)
        return {
            "outputs": outputs, "v": uncertainties.v, "voxel_epi": uncertainties.voxel_epi,
            "voxel_ale": uncertainties.voxel_ale, "voxel_cal": uncertainties.voxel_cal,
            "voxel_rnk": uncertainties.voxel_rnk, "weight_map": uncertainties.weight_map,
            "margin_map": uncertainties.margin_map, "delta_logits": uncertainties.delta_logits,
            "voxel_probe": uncertainties.voxel_probe, "voxel_entropy": uncertainties.voxel_entropy,
            "residual_map": uncertainties.residual_map, "anchor_map": uncertainties.anchor_map
        }

    def predict_image(self, image: torch.Tensor, *, batch: bool = False) -> tuple[torch.Tensor, UncertaintyOutputs]:
        self.lazy_load_model()
        image = image.to(self._device)
        if not batch:
            image = image.unsqueeze(0)
        outputs = sliding_window_inference(image, self._roi_shape, 2, self.predictor, .5, "gaussian", device=self._device)
        return outputs.pop("outputs"), UncertaintyOutputs(**outputs)

    @staticmethod
    def _save_outputs_to_file(outputs: torch.Tensor, uncertainties: UncertaintyOutputs, folder: str | PathLike[str],
                              case_name: str) -> None:
        save_image(outputs, f"{folder}/seg_{case_name}.nii.gz")
        save_image(uncertainties.v, f"{folder}/vvv_{case_name}.nii.gz")
        save_image(uncertainties.voxel_cal, f"{folder}/cal_{case_name}.nii.gz")
        save_image(uncertainties.voxel_rnk, f"{folder}/rnk_{case_name}.nii.gz")

    def predict_to_files(self, x: PathBasedSupervisedDataset | PathBasedUnsupervisedDataset,
                         folder: str | PathLike[str]) -> None:
        case_folder = f"{folder}/cases"
        mkdir(case_folder)
        case_names = []
        if isinstance(x, PathBasedUnsupervisedDataset):
            for i, image in enumerate(x):
                image = image.to(self._device)
                outputs, uncertainties = self.predict_image(image)
                case_name = x.paths()[i]
                case_name = case_name[:case_name.find(".")]
                case_names.append(case_name)
                self._save_outputs_to_file(outputs, uncertainties, case_folder, case_name)
            with open(f"{folder}/case_names.txt", "w") as f:
                f.write("\n".join(case_names))
            return
        # if we are generously granted ground truths, we thank the user by producing additional metrics
        avg_metrics = {}
        for i, (image, label) in enumerate(x):
            image, label = image.to(self._device), label.to(self._device)
            outputs, uncertainties = self.predict_image(image)
            case_name = x.paths()[i][0]
            case_name = case_name[:case_name.index(".")]
            case_names.append(case_name)
            self._save_outputs_to_file(outputs, uncertainties, case_folder, case_name)
            with torch.no_grad():
                loss, metrics = self._criterion(self._model, outputs, uncertainties, label.unsqueeze(0))
                metrics["loss"] = loss.item()
                with open(f"{case_folder}/metrics_{case_name}.json", "w") as f:
                    dump(metrics, f)
                if len(avg_metrics) == 0:
                    avg_metrics = metrics
                else:
                    for k in avg_metrics:
                        avg_metrics[k] += metrics[k]
        with open(f"{folder}/case_names.txt", "w") as f:
            f.write("\n".join(case_names))
        with open(f"{folder}/avg_metrics.json", "w") as f:
            dump({k: v / len(x) for k, v in avg_metrics.items()}, f)
