from __future__ import annotations

from pathlib import Path
import argparse
import os
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def clear_local_pycache(root: Path) -> int:
    removed = 0
    for pyc_path in Path(root).rglob("__pycache__/*.pyc"):
        pyc_path.unlink(missing_ok=True)
        removed += 1
    return removed


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="训练图像分类模型。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "cls_default.yaml"),
        help="配置文件路径。可为空。默认: %(default)s",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="训练数据集根目录。可为空；为空时使用配置文件中的 data.root，传入时覆盖配置值。",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="训练输出的项目目录名称。可为空；为空时使用配置文件中的 task.project，传入时覆盖配置值。",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="训练运行名称。可为空。默认: exp_YYYYMMDDHHmmss",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="显式指定最终输出目录。可为空。默认: runs/classify/{project}/{name}",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="若设置，则允许复用已存在的输出目录。默认: False",
    )
    parser.add_argument("--model", type=str, default=None, help="模型名称。可为空。默认: 取配置文件 model.name")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数。可为空。默认: 取配置文件 train.epochs")
    parser.add_argument("--patience", type=int, default=None, help="早停耐心轮数。可为空。默认: 取配置文件 train.patience")
    parser.add_argument("--batch", type=int, default=None, help="batch size。可为空。默认: 取配置文件 train.batch_size")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader worker 数。可为空。默认: 取配置文件 data.num_workers")
    parser.add_argument(
        "--preprocess",
        type=str,
        default=None,
        choices=["stretch", "letterbox", "crop"],
        help="预处理方式。可为空。可选: stretch拉伸, letterbox等比, crop裁剪。默认: 取配置文件 data.preprocess",
    )
    parser.add_argument(
        "--augment_backend",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="增强后端。可为空。可选: cpu, gpu。默认: 取配置文件 data.augment_backend",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="输入尺寸。可为空。支持: --imgsz 224 或 --imgsz 256 384。默认: 取配置文件 data.img_size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备选择。可为空。支持: auto、cpu、cuda、1、0,1,2。默认: 取配置文件 device",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="分布式训练主端口。可为空。默认: 取配置文件 distributed.master_port",
    )
    parser.add_argument(
        "--print_top_wrong",
        type=int,
        default=None,
        help="打印错误样本条数。可为空。默认: 取配置文件 eval.print_top_wrong",
    )
    return parser


def preconfigure_cuda_visible_devices(device: str | None) -> None:
    if device and re.fullmatch(r"\d+(,\d+)*", device):
        os.environ["CUDA_VISIBLE_DEVICES"] = device


def main(argv=None):
    clear_local_pycache(ROOT)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    preconfigure_cuda_visible_devices(args.device)
    from cls_engine.config.loader import load_train_config
    from cls_engine.engine.trainer import run_training

    cfg = load_train_config(args.config, args)
    run_training(cfg)


if __name__ == "__main__":
    main()
