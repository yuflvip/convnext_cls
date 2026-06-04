from pathlib import Path
from typing import Any

import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_best_checkpoint(
    path: str | Path,
    model,
    epoch: int,
    model_name: str,
    best_val_acc: float,
    class_names: list[str],
    img_size: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "state_dict": unwrap_model(model).state_dict(),
            "best_val_acc": best_val_acc,
            "classes": class_names,
            "img_size": img_size,
            "val_loss": val_loss,
        },
        Path(path),
    )


def save_last_checkpoint(path: str | Path, model, epoch: int, class_names: list[str]) -> None:
    torch.save(
        {"epoch": epoch, "state_dict": unwrap_model(model).state_dict(), "classes": class_names},
        Path(path),
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
