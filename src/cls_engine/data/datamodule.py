from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from cls_engine.config.schema import DataConfig
from cls_engine.distributed.ddp import is_dist_avail_and_initialized

from .dataset import (
    DatasetLayout,
    ImageFolderWithPath,
    compute_class_counts,
    discover_dataset_layout,
    remap_dataset_to_class_order,
)
from .splits import compute_class_counts_from_indices, stratified_split_indices
from .transforms import build_eval_transform, build_train_transform


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


def prepare_data(cfg: DataConfig, seed: int, batch_size: int) -> PreparedData:
    train_tf = build_train_transform(cfg.img_size, augment_backend=cfg.augment_backend, preprocess=cfg.preprocess)
    val_tf = build_eval_transform(cfg.img_size, augment_backend=cfg.augment_backend, preprocess=cfg.preprocess)
    layout = discover_dataset_layout(Path(cfg.root))
    class_names = layout.class_names

    train_dataset_full = ImageFolderWithPath(root=str(layout.train_dir), transform=train_tf)
    remap_dataset_to_class_order(train_dataset_full, class_names)

    val_dataset_explicit = None
    test_dataset_explicit = None
    split_eval_dataset = None
    if layout.has_explicit_val:
        val_dataset_explicit = ImageFolderWithPath(root=str(layout.val_dir), transform=val_tf)
        remap_dataset_to_class_order(val_dataset_explicit, class_names)
    else:
        split_eval_dataset = ImageFolderWithPath(root=str(layout.train_dir), transform=val_tf)
        remap_dataset_to_class_order(split_eval_dataset, class_names)

    if layout.has_explicit_test:
        test_dataset_explicit = ImageFolderWithPath(root=str(layout.test_dir), transform=val_tf)
        remap_dataset_to_class_order(test_dataset_explicit, class_names)

    if layout.has_explicit_val:
        train_set = train_dataset_full
        val_set = val_dataset_explicit
        train_idx = list(range(len(train_dataset_full)))
        val_idx = list(range(len(val_dataset_explicit)))
        test_idx = list(range(len(test_dataset_explicit))) if test_dataset_explicit is not None else []
        test_set = test_dataset_explicit
        train_counts = compute_class_counts(train_dataset_full)
        val_counts = compute_class_counts(val_dataset_explicit)
        test_counts = compute_class_counts(test_dataset_explicit) if test_dataset_explicit is not None else None
    else:
        all_labels = [sample[1] for sample in train_dataset_full.samples]
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
            test_counts = compute_class_counts(test_dataset_explicit)
        elif test_idx:
            test_set = Subset(split_eval_dataset, test_idx)
            test_counts = compute_class_counts_from_indices(split_eval_dataset, test_idx)
        else:
            test_set = None
            test_counts = None
        train_counts = compute_class_counts_from_indices(train_dataset_full, train_idx)
        val_counts = compute_class_counts_from_indices(split_eval_dataset, val_idx)

    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if is_dist_avail_and_initialized() else None
    val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False) if is_dist_avail_and_initialized() else None
    test_sampler = (
        DistributedSampler(test_set, shuffle=False, drop_last=False)
        if test_set is not None and is_dist_avail_and_initialized()
        else None
    )

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
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        layout=layout,
    )
