from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _read_results(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Results CSV contains no data rows: {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
        if key and all(row.get(key, "") != "" for row in rows)
    }


def _smooth(values: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    window = max(3, int(round(values.size * fraction)))
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window < 3:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def _best_epoch(data: dict[str, np.ndarray]) -> int | None:
    epochs = data.get("epoch")
    if epochs is None:
        return None
    flags = data.get("is_best")
    if flags is not None and np.any(flags > 0.5):
        return int(epochs[np.flatnonzero(flags > 0.5)[-1]])
    accuracy = data.get("val_acc_top1")
    return int(epochs[int(np.argmax(accuracy))]) if accuracy is not None else None


def _mark_best(axes, best_epoch: int | None) -> None:
    if best_epoch is None:
        return
    for axis in np.asarray(axes).flat:
        axis.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.2, alpha=0.8, label="best epoch")


def _plot_loss(axis, epochs: np.ndarray, values: np.ndarray, title: str) -> None:
    axis.plot(epochs, values, color="tab:blue", linewidth=1.0, alpha=0.35, label="raw")
    axis.plot(epochs, _smooth(values), color="tab:blue", linewidth=2.0, label="smooth")
    axis.set_title(title)
    axis.set_ylabel("Loss")
    axis.legend()


def _plot_accuracy(axis, epochs: np.ndarray, data: dict[str, np.ndarray], suffix: str, title: str) -> None:
    train_key = f"train_acc_{suffix}"
    val_key = f"val_acc_{suffix}"
    if train_key in data:
        axis.plot(epochs, data[train_key] * 100.0, linewidth=1.7, label="train")
    if val_key in data:
        axis.plot(epochs, data[val_key] * 100.0, linewidth=1.7, label="val")
    axis.set_title(title)
    axis.set_ylabel("Accuracy (%)")
    axis.legend()


def _finish_figure(fig, axes, output_path: Path, reserve_title: bool = False) -> None:
    for axis in np.asarray(axes).flat:
        axis.set_xlabel("Epoch")
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if reserve_title else None)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def plot_training_results(results_csv: str | Path, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_csv = Path(results_csv)
    output_dir = Path(output_dir) if output_dir is not None else results_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _read_results(results_csv)
    epochs = data["epoch"]
    best_epoch = _best_epoch(data)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _plot_loss(axes[0, 0], epochs, data["train_loss"], "Train Loss")
    _plot_loss(axes[0, 1], epochs, data["val_loss"], "Validation Loss")
    _plot_accuracy(axes[1, 0], epochs, data, "top1", "Top-1 Accuracy")
    _plot_accuracy(axes[1, 1], epochs, data, "top5", "Top-5 Accuracy")
    _mark_best(axes, best_epoch)
    if best_epoch is not None and "val_acc_top1" in data:
        best_index = int(np.where(epochs == best_epoch)[0][-1])
        fig.suptitle(f"Training Results — best epoch {best_epoch}, val Top-1 {data['val_acc_top1'][best_index] * 100:.2f}%")
    results_path = output_dir / "results.png"
    _finish_figure(fig, axes, results_path, reserve_title=(best_epoch is not None))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    lr_start = data.get("lr_start", data.get("lr"))
    lr_end = data.get("lr_end", data.get("lr"))
    if lr_start is not None and lr_end is not None:
        axes[0, 0].plot(epochs, lr_start, label="lr_start", linewidth=1.5)
        axes[0, 0].plot(epochs, lr_end, label="lr_end", linewidth=1.5)
        axes[0, 0].fill_between(epochs, lr_start, lr_end, alpha=0.18)
    axes[0, 0].set_title("Learning Rate")
    axes[0, 0].set_ylabel("LR")
    axes[0, 0].legend()

    for key, label in (
        ("val_macro_f1", "Macro-F1"),
        ("val_balanced_accuracy", "Balanced accuracy"),
        ("val_worst_class_recall", "Worst-class recall"),
    ):
        if key in data:
            axes[0, 1].plot(epochs, data[key] * 100.0, label=label, linewidth=1.5)
    axes[0, 1].set_title("Validation Class Metrics")
    axes[0, 1].set_ylabel("Metric (%)")
    axes[0, 1].legend()

    for key, label in (("val_ece", "ECE"), ("val_brier_score", "Brier score")):
        if key in data:
            axes[1, 0].plot(epochs, data[key], label=label, linewidth=1.5)
    axes[1, 0].set_title("Calibration")
    axes[1, 0].set_ylabel("Score (lower is better)")
    axes[1, 0].legend()

    skipped = data.get("amp_skipped_steps", np.zeros_like(epochs))
    axes[1, 1].bar(epochs, skipped, color="tab:orange", alpha=0.8)
    axes[1, 1].set_title("AMP Skipped Steps")
    axes[1, 1].set_ylabel("Steps")
    _mark_best(axes, best_epoch)
    diagnostics_path = output_dir / "results_diagnostics.png"
    _finish_figure(fig, axes, diagnostics_path)
    plt.close(fig)
    return results_path, diagnostics_path
