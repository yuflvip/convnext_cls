from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.config.schema import normalize_img_size
from cls_engine.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_gpu_eval_batch_augment,
    build_gpu_train_batch_augment,
    build_train_transform,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按当前训练预处理逻辑生成预处理结果预览图。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="输入图片路径或目录。必填。")
    parser.add_argument("--output", type=str, default="output/preprocess_preview", help="输出目录。默认: %(default)s")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval"], help="预处理阶段。默认: %(default)s")
    parser.add_argument(
        "--augment_backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="增强后端。默认: %(default)s",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[224],
        metavar=("HEIGHT", "WIDTH"),
        help="输入尺寸。支持: --imgsz 224 或 --imgsz 256 384。默认: 224",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="仅 augment_backend=gpu 时生效。支持: auto、cpu、cuda。默认: %(default)s",
    )
    return parser


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda but CUDA is not available.")
        return torch.device("cuda")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device must be auto, cpu, or cuda for this preview script.")


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_transforms(split: str, augment_backend: str, img_size):
    if split == "train":
        cpu_transform = build_train_transform(img_size, augment_backend=augment_backend)
        gpu_transform = build_gpu_train_batch_augment(img_size) if augment_backend == "gpu" else None
    else:
        cpu_transform = build_eval_transform(img_size, augment_backend=augment_backend)
        gpu_transform = build_gpu_eval_batch_augment(img_size) if augment_backend == "gpu" else None
    return cpu_transform, gpu_transform


def denormalize_image(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return torch.clamp(x * std + mean, 0.0, 1.0)


def process_one_image(
    image_path: Path,
    cpu_transform,
    gpu_transform,
    device: torch.device,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    x = cpu_transform(image)
    if gpu_transform is not None:
        gpu_transform = gpu_transform.to(device)
        gpu_transform.eval()
        with torch.no_grad():
            x = gpu_transform(x.unsqueeze(0).to(device)).squeeze(0).cpu()
    return denormalize_image(x)


def destination_path(src: Path, input_root: Path, output_root: Path) -> Path:
    relative = src.name if input_root.is_file() else str(src.relative_to(input_root))
    rel_path = Path(relative)
    return output_root / rel_path.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    img_size = normalize_img_size(args.imgsz)
    device = resolve_device(args.device)

    cpu_transform, gpu_transform = build_transforms(args.split, args.augment_backend, img_size)
    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")

    for image_path in images:
        processed = process_one_image(image_path, cpu_transform, gpu_transform, device)
        dst = destination_path(image_path, input_path, output_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_image(processed, dst)
        print(f"saved: {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
