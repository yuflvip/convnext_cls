from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按等比例缩放+居中padding(114)方式生成预处理结果预览图。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="输入图片路径或目录。必填。")
    parser.add_argument("--output", type=str, default="output/preprocess_letterbox", help="输出目录。默认: %(default)s")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[224],
        metavar=("HEIGHT", "WIDTH"),
        help="目标尺寸。支持: --imgsz 224 或 --imgsz 256 384。默认: 224",
    )
    parser.add_argument("--pad_value", type=int, default=114, help="padding 填充值。默认: %(default)s")
    return parser


def normalize_imgsz(value: list[int]) -> tuple[int, int]:
    if len(value) == 1:
        size = int(value[0])
        return (size, size)
    if len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("imgsz must be one or two integers.")


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def letterbox_image(image: Image.Image, target_size: tuple[int, int], pad_value: int) -> Image.Image:
    target_h, target_w = target_size
    src_w, src_h = image.size

    if src_h == target_h and src_w == target_w:
        return image.copy()

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), color=(pad_value, pad_value, pad_value))
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def destination_path(src: Path, input_root: Path, output_root: Path) -> Path:
    relative = src.name if input_root.is_file() else str(src.relative_to(input_root))
    return (output_root / Path(relative)).with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    target_size = normalize_imgsz(args.imgsz)
    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        processed = letterbox_image(image, target_size, args.pad_value)
        dst = destination_path(image_path, input_path, output_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        processed.save(dst)
        print(f"saved: {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
