from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder


@dataclass(frozen=True)
class DatasetLayout:
    data_root: Path
    train_dir: Path
    val_dir: Path | None
    test_dir: Path | None
    class_names: list[str]
    has_explicit_val: bool
    has_explicit_test: bool
    mode: str


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, target, path


def collect_sorted_class_names(split_dir: Path) -> list[str]:
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise ValueError(f"Directory does not exist or is not a directory: {split_dir}")
    class_names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if not class_names:
        raise ValueError(f"No class subdirectories found in: {split_dir}")
    return class_names


def ensure_same_class_set(split_name: str, split_dir: Path, expected_class_names: list[str]) -> None:
    actual_class_names = collect_sorted_class_names(split_dir)
    expected = set(expected_class_names)
    actual = set(actual_class_names)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{split_name} class set differs from train. "
            f"missing(in {split_name}): {missing}; extra(in {split_name}): {extra}"
        )


def discover_dataset_layout(data_root: Path) -> DatasetLayout:
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    if not train_dir.is_dir():
        raise ValueError(f"Dataset root must contain train directory: {data_root}")

    class_names = collect_sorted_class_names(train_dir)
    has_explicit_val = val_dir.is_dir()
    has_explicit_test = test_dir.is_dir()
    if has_explicit_val:
        ensure_same_class_set("val", val_dir, class_names)
    if has_explicit_test:
        ensure_same_class_set("test", test_dir, class_names)

    mode = "explicit_splits" if (has_explicit_val or has_explicit_test) else "train_only_split"
    return DatasetLayout(
        data_root=data_root,
        train_dir=train_dir,
        val_dir=val_dir if has_explicit_val else None,
        test_dir=test_dir if has_explicit_test else None,
        class_names=class_names,
        has_explicit_val=has_explicit_val,
        has_explicit_test=has_explicit_test,
        mode=mode,
    )


def remap_dataset_to_class_order(ds, desired_order: list[str]) -> None:
    existing = set(ds.classes)
    desired = list(desired_order)
    desired_set = set(desired)
    missing = sorted(desired_set - existing)
    extra = sorted(existing - desired_set)
    if missing or extra:
        raise ValueError(
            f"Class set mismatch. missing(in dataset): {missing}; extra(in dataset): {extra}"
        )

    old_classes = list(ds.classes)
    new_class_to_idx = {name: i for i, name in enumerate(desired)}
    new_samples = []
    for path, old_y in ds.samples:
        class_name = old_classes[old_y]
        new_samples.append((path, new_class_to_idx[class_name]))

    ds.classes = desired
    ds.class_to_idx = new_class_to_idx
    ds.samples = new_samples
    ds.targets = [y for _, y in new_samples]


def compute_class_counts(dataset: ImageFolderWithPath) -> np.ndarray:
    counts = np.zeros(len(dataset.classes), dtype=np.int64)
    for _, y in dataset.samples:
        counts[y] += 1
    return counts
