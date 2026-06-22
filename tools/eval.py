import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.eval.pth import evaluate_checkpoint_directory
from cls_engine.predict.predictor import parse_predict_imgsz


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="评估图像分类模型在外部验证集上的准确率。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="待评估的模型权重路径。必填。")
    parser.add_argument("--data", type=str, required=True, help="验证集根目录。目录下应按类别分子目录。必填。")
    parser.add_argument("--output", type=str, default=None, help="输出目录。可为空。默认: {model_stem}_eval")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="评估设备。默认: %(default)s")
    parser.add_argument("--imgsz", type=str, default="224", help="输入尺寸。支持: 224 或 256,384。默认: %(default)s")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="stretch",
        choices=["stretch", "letterbox", "crop"],
        help="预处理方式。默认: %(default)s",
    )
    parser.add_argument("--batch", type=int, default=32, help="评估 batch size。默认: %(default)s")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker 数。默认: %(default)s")
    parser.add_argument("--print_top_wrong", type=int, default=20, help="打印错误样本条数。默认: %(default)s")
    parser.add_argument("--log_interval", type=int, default=1, help="评估进度打印间隔（按 batch）。默认: %(default)s")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    parse_predict_imgsz(args.imgsz)
    output_dir = evaluate_checkpoint_directory(
        model_path=args.model,
        data_root=args.data,
        output=args.output,
        device=args.device,
        imgsz=args.imgsz,
        preprocess=args.preprocess,
        batch_size=args.batch,
        num_workers=args.workers,
        print_top_wrong=args.print_top_wrong,
        log_interval=args.log_interval,
    )
    print(f"Evaluation outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
