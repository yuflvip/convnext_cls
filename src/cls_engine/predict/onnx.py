from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cls_engine.data.transforms import build_eval_transform

from .predictor import (
    arrange_prediction_outputs,
    collect_input_images,
    load_onnx_classes,
    parse_predict_imgsz,
    resolve_prediction_output_dir,
    write_prediction_outputs,
)


def _resolve_onnx_providers(device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this onnxruntime build.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    raise ValueError("device must be auto, cpu, or cuda.")


def predict_with_onnx(
    model_path: str | Path,
    input_path: str | Path,
    output: str | Path | None = None,
    device: str = "auto",
    imgsz: str = "224",
    preprocess: str = "letterbox",
    topk: int = 3,
    classes_path: str | Path | None = None,
    arrange_mode: str | None = None,
) -> Path:
    import onnxruntime as ort

    classes = load_onnx_classes(model_path, classes_path)
    input_size = parse_predict_imgsz(imgsz)
    image_paths = collect_input_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {input_path}")

    providers = _resolve_onnx_providers(device)
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    transform = build_eval_transform(input_size, augment_backend="cpu", preprocess=preprocess)
    effective_topk = max(1, min(topk, len(classes)))

    rows = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        x = transform(image).unsqueeze(0).numpy().astype(np.float32)
        logits = session.run(None, {input_name: x})[0][0]
        logits = np.asarray(logits, dtype=np.float32)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        pred_idx = int(np.argmax(probs))
        topk_indices = np.argsort(probs)[::-1][:effective_topk]
        rows.append(
            {
                "path": str(path),
                "pred_idx": pred_idx,
                "pred_name": classes[pred_idx],
                "conf": float(probs[pred_idx]),
                "topk": [[classes[int(idx)], float(probs[int(idx)])] for idx in topk_indices],
            }
        )

    output_dir = resolve_prediction_output_dir(model_path, output)
    write_prediction_outputs(output_dir, rows)
    arrange_prediction_outputs(output_dir, rows, arrange_mode=arrange_mode)
    return output_dir
