from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder


@dataclass(frozen=True)
class DatasetLayout:
    data_roots: list[Path]
    train_dirs: list[Path]
    val_dirs: list[Path]
    test_dirs: list[Path]
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


def parse_data_roots(root_spec: str) -> list[Path]:
    roots = [Path(item.strip()) for item in root_spec.split(",") if item.strip()]
    if not roots:
        raise ValueError("data.root must not be empty")
    return roots


def _validate_split_presence(split_name: str, split_dirs: list[Path]) -> bool:
    presence = [split_dir.is_dir() for split_dir in split_dirs]
    if all(presence):
        return True
    if any(presence):
        missing_roots = [str(root.parent) for root, exists in zip(split_dirs, presence) if not exists]
        raise ValueError(
            f"All dataset roots must consistently provide {split_name}/. "
            f"Missing {split_name}/ under: {missing_roots}"
        )
    return False


def discover_dataset_layout(data_roots: Path | list[Path]) -> DatasetLayout:
    roots = [Path(data_roots)] if isinstance(data_roots, Path) else [Path(root) for root in data_roots]
    train_dirs = [root / "train" for root in roots]
    val_dirs = [root / "val" for root in roots]
    test_dirs = [root / "test" for root in roots]
    for root, train_dir in zip(roots, train_dirs):
        if not train_dir.is_dir():
            raise ValueError(f"Dataset root must contain train directory: {root}")

    class_names = collect_sorted_class_names(train_dirs[0])
    for train_dir in train_dirs[1:]:
        actual_class_names = collect_sorted_class_names(train_dir)
        if set(actual_class_names) != set(class_names):
            missing = sorted(set(class_names) - set(actual_class_names))
            extra = sorted(set(actual_class_names) - set(class_names))
            raise ValueError(
                f"Class set mismatch across dataset roots. "
                f"missing(in {train_dir.parent}): {missing}; extra(in {train_dir.parent}): {extra}"
            )

    has_explicit_val = _validate_split_presence("val", val_dirs)
    has_explicit_test = _validate_split_presence("test", test_dirs)
    if has_explicit_val:
        for val_dir in val_dirs:
            ensure_same_class_set("val", val_dir, class_names)
    if has_explicit_test:
        for test_dir in test_dirs:
            ensure_same_class_set("test", test_dir, class_names)

    mode = "explicit_splits" if (has_explicit_val or has_explicit_test) else "train_only_split"
    return DatasetLayout(
        data_roots=roots,
        train_dirs=train_dirs,
        val_dirs=val_dirs if has_explicit_val else [],
        test_dirs=test_dirs if has_explicit_test else [],
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
