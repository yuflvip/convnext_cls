from __future__ import annotations

import time
from pathlib import Path

from cls_engine.predict.predictor import IMAGE_EXTS


class EvalImageDataset:
    def __init__(self, samples: list[tuple[Path, int]], class_names: list[str], transform):
        self.samples = [(str(path), target) for path, target in samples]
        self.targets = [target for _, target in self.samples]
        self.classes = list(class_names)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path


def _scan_eval_samples(data_root: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    actual_class_names = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    expected_set = set(class_names)
    actual_set = set(actual_class_names)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise ValueError(
            f"Class set mismatch. missing(in dataset): {missing}; extra(in dataset): {extra}"
        )

    samples: list[tuple[Path, int]] = []
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = data_root / class_name
        for image_path in sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            samples.append((image_path, class_to_idx[class_name]))
    return samples


def build_eval_dataset(
    data_root: str | Path,
    class_names: list[str],
    input_size: int | tuple[int, int] | list[int],
    preprocess: str = "letterbox",
):
    from cls_engine.data.transforms import build_eval_transform

    data_root = Path(data_root)
    if not data_root.is_dir():
        raise ValueError(f"Directory does not exist or is not a directory: {data_root}")

    transform = build_eval_transform(input_size, augment_backend="cpu", preprocess=preprocess)
    samples = _scan_eval_samples(data_root, class_names)
    return EvalImageDataset(samples=samples, class_names=class_names, transform=transform)


def resolve_eval_output_dir(model_path: str | Path, output: str | Path | None) -> Path:
    if output:
        return Path(output)
    return Path("runs") / "eval" / f"{Path(model_path).stem}_eval"


def _resolve_device(device: str) -> torch.device:
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _print_eval_summary(eval_out: dict, class_names: list[str], sample_count: int, elapsed_sec: float) -> None:
    import numpy as np

    print(
        f"[Eval] done | acc_top1={eval_out['val_acc_top1']:.4f} | "
        f"acc_top5={eval_out['val_acc_top5']:.4f} | samples={sample_count} | time={elapsed_sec:.1f}s"
    )
    cm = eval_out.get("confusion_matrix")
    if cm is None:
        return
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.int64)
    for index, class_name in enumerate(class_names):
        denom = int(support[index])
        acc = (float(tp[index]) / float(denom)) if denom > 0 else 0.0
        print(f"[Eval][Class] {class_name} acc={acc:.4f} support={denom}")


def evaluate_checkpoint_directory(
    model_path: str | Path,
    data_root: str | Path,
    output: str | Path | None = None,
    device: str = "auto",
    imgsz: str = "224",
    preprocess: str = "letterbox",
    batch_size: int = 32,
    num_workers: int = 0,
    print_top_wrong: int = 20,
    log_interval: int = 1,
) -> Path:
    from torch.utils.data import DataLoader

    from cls_engine.engine.evaluator import write_eval_artifacts
    from cls_engine.engine.loops import evaluate_with_details
    from cls_engine.io.artifacts import ArtifactWriter
    from cls_engine.models.factory import build_model
    from cls_engine.predict.predictor import parse_predict_imgsz
    from cls_engine.predict.pth import load_predict_checkpoint_info

    info = load_predict_checkpoint_info(model_path)
    input_size = parse_predict_imgsz(imgsz)
    eval_dataset = build_eval_dataset(data_root, info["classes"], input_size, preprocess=preprocess)
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    torch_device = _resolve_device(device)
    model = build_model(info["model_name"], num_classes=info["num_classes"], pretrained=False).to(torch_device)
    model.load_state_dict(info["state_dict"], strict=True)
    model.eval()

    start = time.perf_counter()
    eval_out = evaluate_with_details(
        model,
        loader,
        torch_device,
        info["classes"],
        log_interval=log_interval,
        log_prefix="Eval",
    )
    elapsed_sec = time.perf_counter() - start
    output_dir = resolve_eval_output_dir(model_path, output)
    writer = ArtifactWriter(output_dir)
    write_eval_artifacts(writer, "eval", eval_out, info["classes"], print_top_wrong)
    writer.write_final_summary({
        "model_path": str(Path(model_path)),
        "model_name": info["model_name"],
        "data_root": str(Path(data_root)),
        "num_classes": info["num_classes"],
        "class_names": info["classes"],
        "samples": len(eval_dataset),
        "eval_loss": eval_out["val_loss"],
        "eval_acc_top1": eval_out["val_acc_top1"],
        "eval_acc_top5": eval_out["val_acc_top5"],
        "elapsed_sec": elapsed_sec,
    })
    _print_eval_summary(eval_out, info["classes"], len(eval_dataset), elapsed_sec)
    return output_dir
