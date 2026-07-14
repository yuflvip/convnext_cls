from collections import defaultdict

import numpy as np
import torch


def stratified_split_indices(
    labels: list[int],
    val_ratio: float,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.RandomState(seed)
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[int(label)].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for _, indices in sorted(class_to_indices.items()):
        arr = np.array(indices)
        rng.shuffle(arr)
        n = len(arr)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        val_idx.extend(arr[:n_val].tolist())
        test_idx.extend(arr[n_val:n_val + n_test].tolist())
        train_idx.extend(arr[n_val + n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def compute_class_counts_from_indices(dataset, indices: list[int]) -> np.ndarray:
    counts = np.zeros(len(dataset.classes), dtype=np.int64)
    for i in indices:
        _, y = dataset.samples[i]
        counts[y] += 1
    return counts


def build_class_weights(counts: np.ndarray, mode: str = "inv_sqrt") -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode not in {"inv", "inv_sqrt"}:
        raise ValueError("class weight mode must be 'none', 'inv', or 'inv_sqrt'")
    counts = np.maximum(counts, 1)
    if mode == "inv":
        weights = 1.0 / counts
    else:
        weights = 1.0 / np.sqrt(counts)
    weights = weights / (weights.mean() + 1e-12)
    return torch.tensor(weights, dtype=torch.float32)
