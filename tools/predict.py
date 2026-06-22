import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.predict.predictor import parse_predict_imgsz
from cls_engine.predict.pth import predict_with_checkpoint
from cls_engine.predict.onnx import predict_with_onnx


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="预测图像分类结果。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--format", type=str, default="pth", choices=["pth", "onnx"], help="预测后端格式。默认: %(default)s")
    parser.add_argument("--model", type=str, required=True, help="待预测的模型权重路径。必填。")
    parser.add_argument("--input", type=str, required=True, help="输入图片路径或目录。必填。")
    parser.add_argument("--output", type=str, default=None, help="输出目录。可为空。默认: runs/predict/predict_YYYYMMDDHHmmss")
    parser.add_argument("--classes", type=str, default=None, help="类别文件路径。ONNX 模式可为空，默认尝试读取模型同目录 classes.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="预测设备。默认: %(default)s")
    parser.add_argument("--imgsz", type=str, default="224", help="输入尺寸。支持: 224 或 256,384。默认: %(default)s")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="stretch",
        choices=["stretch", "letterbox", "crop"],
        help="预处理方式。默认: %(default)s",
    )
    parser.add_argument("--topk", type=int, default=3, help="输出 top-k 结果数。默认: %(default)s")
    parser.add_argument("--arrange_mode", type=str, default=None, choices=["copy", "move"], help="按预测类别整理图片。可选: copy, move。默认: 不整理")
    return parser


def parse_imgsz(value: str) -> tuple[int, int]:
    return parse_predict_imgsz(value)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    parse_imgsz(args.imgsz)
    if args.format == "pth":
        output_dir = predict_with_checkpoint(
            model_path=args.model,
            input_path=args.input,
            output=args.output,
            device=args.device,
            imgsz=args.imgsz,
            preprocess=args.preprocess,
            topk=args.topk,
            arrange_mode=args.arrange_mode,
        )
    else:
        output_dir = predict_with_onnx(
            model_path=args.model,
            input_path=args.input,
            output=args.output,
            device=args.device,
            imgsz=args.imgsz,
            preprocess=args.preprocess,
            topk=args.topk,
            classes_path=args.classes,
            arrange_mode=args.arrange_mode,
        )
    print(f"Predictions saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
