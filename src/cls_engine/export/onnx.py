from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cls_engine.models.checkpoint import load_checkpoint
from cls_engine.models.factory import build_model


def parse_export_imgsz(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        if size <= 0:
            raise ValueError("imgsz must be positive.")
        return (size, size)
    if len(parts) == 2:
        height = int(parts[0])
        width = int(parts[1])
        if height <= 0 or width <= 0:
            raise ValueError("imgsz must be positive.")
        return (height, width)
    raise ValueError("imgsz must be a single integer or h,w.")


def resolve_export_output_path(model_path: str | Path, output: str | Path | None) -> Path:
    if output:
        return Path(output)
    model_path = Path(model_path)
    return model_path.with_suffix(".onnx")


def load_export_checkpoint_info(model_path: str | Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(model_path, map_location="cpu")
    classes = checkpoint.get("classes") or []
    model_name = checkpoint.get("model_name")
    if not model_name:
        raise ValueError("Checkpoint does not contain model_name.")
    if not classes:
        raise ValueError("Checkpoint does not contain classes.")
    img_size = checkpoint.get("img_size", (224, 224))
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    elif isinstance(img_size, (list, tuple)):
        if len(img_size) == 1:
            img_size = (int(img_size[0]), int(img_size[0]))
        elif len(img_size) == 2:
            img_size = (int(img_size[0]), int(img_size[1]))
        else:
            raise ValueError("Checkpoint img_size must contain one or two values.")
    else:
        raise ValueError("Checkpoint img_size is invalid.")
    return {
        "model_name": model_name,
        "classes": classes,
        "num_classes": len(classes),
        "img_size": img_size,
        "state_dict": checkpoint["state_dict"],
        "checkpoint": checkpoint,
    }


def export_checkpoint_to_onnx(
    model_path: str | Path,
    output: str | Path | None = None,
    imgsz: str | tuple[int, int] | None = None,
    opset: int = 13,
    device: str = "cpu",
    simplify: bool = False,
    dynamo: bool = False,
) -> Path:
    info = load_export_checkpoint_info(model_path)
    input_size = parse_export_imgsz(imgsz) if isinstance(imgsz, str) else (imgsz or info["img_size"])
    output_path = resolve_export_output_path(model_path, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    model = build_model(
        info["model_name"],
        num_classes=info["num_classes"],
        pretrained=False,
    ).to(torch_device)
    model.load_state_dict(info["state_dict"], strict=True)
    model.eval()

    height, width = input_size
    dummy = torch.randn(1, 3, height, width, device=torch_device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["images"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=dynamo,
            external_data=False,
        )

    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
        except ImportError as exc:
            raise ImportError("simplify=True requires onnxsim to be installed.") from exc
        import onnx

        model_onnx = onnx.load(output_path)
        simplified, ok = onnx_simplify(model_onnx)
        if not ok:
            raise RuntimeError("onnxsim failed to simplify exported model.")
        onnx.save(simplified, output_path)

    external_data_path = output_path.with_suffix(output_path.suffix + ".data")
    if external_data_path.exists():
        try:
            import onnx

            model_onnx = onnx.load(output_path)
            uses_external_data = any(
                getattr(initializer, "external_data", None)
                or getattr(initializer, "data_location", None) == onnx.TensorProto.EXTERNAL
                for initializer in model_onnx.graph.initializer
            )
        except Exception:
            uses_external_data = True
        if not uses_external_data:
            external_data_path.unlink()

    return output_path
