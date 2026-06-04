import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.export.onnx import export_checkpoint_to_onnx, parse_export_imgsz


def parse_imgsz(value: str) -> tuple[int, int]:
    return parse_export_imgsz(value)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="导出图像分类模型。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="待导出的模型权重路径。必填。")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx"], help="导出格式。默认: %(default)s")
    parser.add_argument("--output", type=str, default=None, help="导出文件路径。可为空。默认: 根据模型路径自动推导")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="导出设备。默认: %(default)s")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset 版本。默认: %(default)s")
    parser.add_argument(
        "--imgsz",
        type=str,
        default="224",
        help="导出输入尺寸。支持: 224 或 256,384。默认: %(default)s",
    )
    parser.add_argument("--simplify", action="store_true", help="导出后使用 onnxsim 简化 ONNX 模型结构。默认: False")
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="启用新的 torch.export/dynamo ONNX 导出器；默认关闭，使用传统 TorchScript 导出器。",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    parse_imgsz(args.imgsz)
    if args.format != "onnx":
        raise ValueError(f"Unsupported export format: {args.format}")
    output_path = export_checkpoint_to_onnx(
        model_path=args.model,
        output=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        device=args.device,
        simplify=args.simplify,
        dynamo=args.dynamo,
    )
    print(f"Exported ONNX: {output_path.resolve()}")


if __name__ == "__main__":
    main()
