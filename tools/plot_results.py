from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.metrics.plots import plot_training_results


def main() -> None:
    parser = argparse.ArgumentParser(description="从训练 results.csv 重新生成结果图。")
    parser.add_argument("--results", required=True, help="results.csv 路径。")
    parser.add_argument("--output", default=None, help="图片输出目录；默认与 results.csv 同目录。")
    args = parser.parse_args()
    paths = plot_training_results(args.results, args.output)
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
