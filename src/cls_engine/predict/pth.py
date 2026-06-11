from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from cls_engine.data.transforms import build_eval_transform
from cls_engine.models.checkpoint import load_checkpoint
from cls_engine.models.factory import build_model

from .predictor import (
    collect_input_images,
    parse_predict_imgsz,
    resolve_prediction_output_dir,
    write_prediction_outputs,
)


def load_predict_checkpoint_info(model_path: str | Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(model_path, map_location="cpu")
    classes = checkpoint.get("classes") or []
    model_name = checkpoint.get("model_name")
    if not model_name:
        raise ValueError("Checkpoint does not contain model_name.")
    if not classes:
        raise ValueError("Checkpoint does not contain classes.")
    return {
        "model_name": model_name,
        "classes": classes,
        "num_classes": len(classes),
        "state_dict": checkpoint["state_dict"],
    }


def predict_with_checkpoint(
    model_path: str | Path,
    input_path: str | Path,
    output: str | Path | None = None,
    device: str = "auto",
    imgsz: str = "224",
    preprocess: str = "letterbox",
    topk: int = 3,
) -> Path:
    info = load_predict_checkpoint_info(model_path)
    input_size = parse_predict_imgsz(imgsz)
    image_paths = collect_input_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {input_path}")

    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    model = build_model(info["model_name"], num_classes=info["num_classes"], pretrained=False).to(torch_device)
    model.load_state_dict(info["state_dict"], strict=True)
    model.eval()

    transform = build_eval_transform(input_size, augment_backend="cpu", preprocess=preprocess)
    rows: list[dict[str, Any]] = []
    effective_topk = max(1, min(topk, info["num_classes"]))

    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            x = transform(image).unsqueeze(0).to(torch_device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred = probs.max(dim=0)
            values, indices = probs.topk(effective_topk)
            topk_rows = [[info["classes"][int(idx)], float(val)] for val, idx in zip(values.cpu().tolist(), indices.cpu().tolist())]
            rows.append(
                {
                    "path": str(path),
                    "pred_idx": int(pred.item()),
                    "pred_name": info["classes"][int(pred.item())],
                    "conf": float(conf.item()),
                    "topk": topk_rows,
                }
            )

    output_dir = resolve_prediction_output_dir(model_path, output)
    write_prediction_outputs(output_dir, rows)
    return output_dir
