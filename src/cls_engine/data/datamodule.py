from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from cls_engine.config.schema import DataConfig
from cls_engine.distributed.ddp import is_dist_avail_and_initialized

from .dataset import (
    DatasetLayout,
    ImageFolderWithPath,
    compute_class_counts,
    discover_dataset_layout,
    parse_data_roots,
    remap_dataset_to_class_order,
)
from .splits import stratified_split_indices
from .transforms import build_eval_transform, build_train_transform


def _emit_progress(progress_logger, message: str) -> None:
    if progress_logger is not None:
        progress_logger(message)


@dataclass
class PreparedData:
    class_names: list[str]
    num_classes: int
    train_set: object
    val_set: object
    test_set: object | None
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    train_counts: np.ndarray
    val_counts: np.ndarray
    test_counts: np.ndarray | None
    train_root_counts: dict[str, int]
    val_root_counts: dict[str, int]
    test_root_counts: dict[str, int]
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    layout: DatasetLayout


def build_split_indices_for_labels(
    labels: list[int],
    cfg: DataConfig,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    return stratified_split_indices(labels, cfg.val_ratio, cfg.test_ratio, seed)


def _compute_counts_from_labels(labels: list[int], num_classes: int, indices: list[int] | None = None) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    selected_labels = labels if indices is None else [labels[index] for index in indices]
    for label in selected_labels:
        counts[label] += 1
    return counts


def _build_split_dataset(
    split_dirs: list[Path],
    transform,
    class_names: list[str],
) -> tuple[object, list[int], np.ndarray, dict[str, int]]:
    datasets = []
    all_labels: list[int] = []
    total_counts = np.zeros(len(class_names), dtype=np.int64)
    per_root_counts: dict[str, int] = {}
    for split_dir in split_dirs:
        dataset = ImageFolderWithPath(root=str(split_dir), transform=transform)
        remap_dataset_to_class_order(dataset, class_names)
        datasets.append(dataset)
        all_labels.extend([sample[1] for sample in dataset.samples])
        total_counts += compute_class_counts(dataset)
        per_root_counts[str(split_dir.parent)] = len(dataset)
    if len(datasets) == 1:
        return datasets[0], all_labels, total_counts, per_root_counts
    return ConcatDataset(datasets), all_labels, total_counts, per_root_counts


def prepare_data(cfg: DataConfig, seed: int, batch_size: int, progress_logger=None) -> PreparedData:
    train_tf = build_train_transform(
        cfg.img_size,
        augment_backend=cfg.augment_backend,
        preprocess=cfg.preprocess,
        augment=cfg.augment,
    )
    val_tf = build_eval_transform(
        cfg.img_size,
        augment_backend=cfg.augment_backend,
        preprocess=cfg.preprocess,
        augment=cfg.augment,
    )
    _emit_progress(progress_logger, "[Data] discovering dataset layout...")
    layout = discover_dataset_layout(parse_data_roots(cfg.root))
    class_names = layout.class_names

    _emit_progress(progress_logger, "[Data] building train dataset...")
    train_dataset_full, all_labels, train_dataset_counts, train_root_counts = _build_split_dataset(
        layout.train_dirs, train_tf, class_names
    )

    val_dataset_explicit = None
    test_dataset_explicit = None
    split_eval_dataset = None
    val_root_counts: dict[str, int] = {}
    test_root_counts: dict[str, int] = {}
    _emit_progress(progress_logger, "[Data] building val dataset...")
    if layout.has_explicit_val:
        val_dataset_explicit, _, val_counts_explicit, val_root_counts = _build_split_dataset(
            layout.val_dirs, val_tf, class_names
        )
    else:
        split_eval_dataset, _, _, _ = _build_split_dataset(layout.train_dirs, val_tf, class_names)

    if layout.has_explicit_test:
        _emit_progress(progress_logger, "[Data] building test dataset...")
        test_dataset_explicit, _, test_counts_explicit, test_root_counts = _build_split_dataset(
            layout.test_dirs, val_tf, class_names
        )

    if layout.has_explicit_val:
        train_set = train_dataset_full
        val_set = val_dataset_explicit
        train_idx = list(range(len(train_dataset_full)))
        val_idx = list(range(len(val_dataset_explicit)))
        test_idx = list(range(len(test_dataset_explicit))) if test_dataset_explicit is not None else []
        test_set = test_dataset_explicit
        train_counts = train_dataset_counts
        val_counts = val_counts_explicit
        test_counts = test_counts_explicit if test_dataset_explicit is not None else None
    else:
        generated_test_ratio = 0.0 if test_dataset_explicit is not None else cfg.test_ratio
        train_idx, val_idx, test_idx = stratified_split_indices(
            all_labels,
            val_ratio=cfg.val_ratio,
            test_ratio=generated_test_ratio,
            seed=seed,
        )
        train_set = Subset(train_dataset_full, train_idx)
        val_set = Subset(split_eval_dataset, val_idx)
        if test_dataset_explicit is not None:
            test_set = test_dataset_explicit
            test_idx = list(range(len(test_dataset_explicit)))
            test_counts = test_counts_explicit
        elif test_idx:
            test_set = Subset(split_eval_dataset, test_idx)
            test_counts = _compute_counts_from_labels(all_labels, len(class_names), test_idx)
        else:
            test_set = None
            test_counts = None
        train_counts = _compute_counts_from_labels(all_labels, len(class_names), train_idx)
        val_counts = _compute_counts_from_labels(all_labels, len(class_names), val_idx)

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if is_dist_avail_and_initialized() else None
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False) if is_dist_avail_and_initialized() else None
    test_sampler = (
        DistributedSampler(test_set, shuffle=False, drop_last=False)
        if test_set is not None and is_dist_avail_and_initialized()
        else None
    )

    _emit_progress(progress_logger, "[Data] creating dataloaders...")
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(cfg.num_workers > 0),
        )

    return PreparedData(
        class_names=class_names,
        num_classes=len(class_names),
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts,
        train_root_counts=train_root_counts,
        val_root_counts=val_root_counts,
        test_root_counts=test_root_counts,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        layout=layout,
    )
