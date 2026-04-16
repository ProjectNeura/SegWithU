"""
I know it's a very weird design, but the names "GTaR", "SegWithU", and "SegWithUv2" came from our iterations and have
become a historical artifact. You can simply interpret them as "ACDC", "BraTS2024", and "LiTS", respectively.
"""

from dataclasses import dataclass
from os.path import exists
from typing import Literal

from mipcandy import Device, auto_device, InspectionAnnotations, NNUNetDataset, inspect, load_inspection_annotations


@dataclass
class ExpConfig(object):
    project_name: str
    shared_datasets: str
    dataset_dir: str
    align_spacing: bool
    shared_weights: str
    project_weights: str
    spatial_dims: Literal[2, 3]
    num_classes: int
    example_shape: tuple[int, ...]
    fold: Literal[0, 1, 2, 3, 4, "all"] = "all"
    device: Device = auto_device()
    num_workers: int = 2
    prefetch_factor: int = 2
    batch_size: int = 2
    val_num_workers: int = 0
    val_prefetch_factor: int | None = None

    # trainer settings
    profiler: bool = False

    def annotations_path(self) -> str:
        return f"{self.dataset_dir}/annotations_align_spacing.json" if self.align_spacing else f"{
        self.dataset_dir}/annotations.json"

    def dataset(self, split: Literal["Tr","Ts"]) -> NNUNetDataset:
        return NNUNetDataset(self.dataset_dir, align_spacing=self.align_spacing, split=split)

    def annotations(self, dataset: NNUNetDataset, *, inspect_if_not_found: bool = True) -> InspectionAnnotations:
        annotations_path = self.annotations_path()
        if exists(annotations_path) or not inspect_if_not_found:
            return load_inspection_annotations(annotations_path, dataset)
        dataset.device(device=self.device)
        annotations = inspect(dataset)
        dataset.device(device="cpu")
        annotations.save(annotations_path)
        return annotations



SHARED_DATASETS: str = "S:/SharedDatasets"
SHARED_WEIGHTS: str = "S:/SharedWeights"
SERVER: bool = False


def acdc_config() -> ExpConfig:
    cfg = ExpConfig("GTaR", SHARED_DATASETS, f"{SHARED_DATASETS}/ACDC", True, SHARED_WEIGHTS, f"{SHARED_WEIGHTS}/GTaR",
                    3, 4, (1, 64, 128, 128))
    if SERVER:
        cfg.num_workers = 16
        cfg.prefetch_factor = 8
        cfg.batch_size = 16
        cfg.val_num_workers = 4
        cfg.val_prefetch_factor = 4
    return cfg


def brats_config() -> ExpConfig:
    cfg = ExpConfig("SegWithU", SHARED_DATASETS, f"{SHARED_DATASETS}/BraTS2024GLI", False, SHARED_WEIGHTS,
                    f"{SHARED_WEIGHTS}/SegWithU", 3, 5, (4, 128, 128, 128))
    if SERVER:
        cfg.num_workers = 16
        cfg.prefetch_factor = 8
        cfg.batch_size = 8
        cfg.val_num_workers = 4
        cfg.val_prefetch_factor = 4
    return cfg


def lits_config() -> ExpConfig:
    cfg = ExpConfig("SegWithUv2", SHARED_DATASETS, f"{SHARED_DATASETS}/LITS", False, SHARED_WEIGHTS,
                    f"{SHARED_WEIGHTS}/SegWithUv2", 3, 3, (1, 128, 128, 128))
    if SERVER:
        cfg.num_workers = 16
        cfg.prefetch_factor = 8
        cfg.batch_size = 8
        cfg.val_num_workers = 4
        cfg.val_prefetch_factor = 4
    return cfg
