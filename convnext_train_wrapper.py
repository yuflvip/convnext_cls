#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConvNextTrainConfig:
    code_dir: str
    conda_sh: str
    conda_env: str
    data_dir: str
    project: str
    epochs: int
    batch: int
    imgsz: int
    cuda_visible_devices: str
    nproc_per_node: int
    nccl_p2p_disable: int
    nccl_ib_disable: int
    name: str


def default_run_name(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("exp_%Y%m%d%H%M%S")


def build_expected_output_dir(code_dir: str, project: str, name: str) -> str:
    return str(Path(code_dir) / "runs" / "classify" / project / name)


def write_run_config(output_dir: str, data: dict) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_config_path = output_path / "run_config.json"
    run_config_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(run_config_path)


def collect_artifact_paths(output_dir: str) -> dict[str, str]:
    output_path = Path(output_dir)
    candidates = {
        "best_model": output_path / "best.pth",
        "last_model": output_path / "last.pth",
        "results_csv": output_path / "results.csv",
        "run_config_json": output_path / "run_config.json",
        "final_summary_json": output_path / "final_summary.json",
        "wrong_val_csv": output_path / "wrong_val.csv",
        "wrong_test_csv": output_path / "wrong_test.csv",
    }
    return {
        artifact_name: str(path)
        for artifact_name, path in candidates.items()
        if path.exists()
    }


def write_final_summary(output_dir: str, return_code: int, artifacts: dict[str, str]) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "final_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "return_code": return_code,
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(summary_path)


def build_run_summary(config: ConvNextTrainConfig, expected_output_dir: str) -> str:
    fields = [
        f"conda_env={config.conda_env}",
        f"code_dir={config.code_dir}",
        f"data_dir={config.data_dir}",
        f"project={config.project}",
        f"name={config.name}",
        f"epochs={config.epochs}",
        f"batch={config.batch}",
        f"imgsz={config.imgsz}",
        f"cuda_visible_devices={config.cuda_visible_devices}",
        f"nproc_per_node={config.nproc_per_node}",
        f"nccl_p2p_disable={config.nccl_p2p_disable}",
        f"nccl_ib_disable={config.nccl_ib_disable}",
        f"expected_output_dir={expected_output_dir}",
    ]
    return "Resolved args: " + ", ".join(fields)


def build_config_from_task_args(task_args: dict) -> ConvNextTrainConfig:
    return ConvNextTrainConfig(
        code_dir=task_args["code_dir"],
        conda_sh=task_args["conda_sh"],
        conda_env=task_args["conda_env"],
        data_dir=task_args["data_dir"],
        project=task_args["project"],
        epochs=int(task_args["epochs"]),
        batch=int(task_args["batch"]),
        imgsz=int(task_args["imgsz"]),
        cuda_visible_devices=str(task_args["cuda_visible_devices"]),
        nproc_per_node=int(task_args["nproc_per_node"]),
        nccl_p2p_disable=int(task_args["nccl_p2p_disable"]),
        nccl_ib_disable=int(task_args["nccl_ib_disable"]),
        name=task_args["name"],
    )


def build_shell_command(config: ConvNextTrainConfig) -> str:
    train_command = " ".join(
        [
            f"torchrun --nproc_per_node={config.nproc_per_node} tools/train.py",
            f"--data {shlex.quote(config.data_dir)}",
            f"--project {shlex.quote(config.project)}",
            f"--name {shlex.quote(config.name)}",
            f"--epochs {config.epochs}",
            f"--batch {config.batch}",
            f"--imgsz {config.imgsz}",
        ]
    )
    command_parts = [
        "unset PYTHONPATH",
        "unset PYTHONHOME",
        "unset VIRTUAL_ENV",
        "unset CONDA_PREFIX",
        "unset CONDA_DEFAULT_ENV",
        "unset CONDA_PROMPT_MODIFIER",
        f"source {shlex.quote(config.conda_sh)}",
        f"conda activate {shlex.quote(config.conda_env)}",
        f"cd {shlex.quote(config.code_dir)}",
        f"export NCCL_P2P_DISABLE={config.nccl_p2p_disable}",
        f"export NCCL_IB_DISABLE={config.nccl_ib_disable}",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(config.cuda_visible_devices)}",
        train_command,
    ]
    return " && ".join(command_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClearML ConvNeXt 训练包装脚本")
    parser.add_argument(
        "--code-dir",
        default="/mnt/sdb/yufl/cnn/ConvNext/image_classification",
        help="ConvNeXt 训练代码目录",
    )
    parser.add_argument(
        "--conda-sh",
        default="/root/miniconda3/etc/profile.d/conda.sh",
        help="conda.sh 的绝对路径，通常位于 miniconda3 或 anaconda3 的 etc/profile.d/conda.sh",
    )
    parser.add_argument(
        "--conda-env",
        default="convnext-env",
        help="ConvNeXt 训练使用的 conda 环境名",
    )
    parser.add_argument(
        "--data-dir",
        default="/mnt/imgs_coco_datasets_145/2026_imgs_datasets/imgs_cls_vehicle_10class/imgs_cls_vehicle_10class_baseline/",
        help="训练数据集目录",
    )
    parser.add_argument(
        "--project",
        default="classify_vehicle_10class",
        help="传给 tools/train.py 的 project 参数",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="训练输出目录名；为空时自动生成 exp_YYYYMMddHHmmss",
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="输入图像尺寸")
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1,2,3",
        help="设置 CUDA_VISIBLE_DEVICES，例如 0,1,2,3",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=4,
        help="torchrun 的 nproc_per_node 值",
    )
    parser.add_argument(
        "--nccl-p2p-disable",
        type=int,
        default=0,
        help="设置 NCCL_P2P_DISABLE 的值",
    )
    parser.add_argument(
        "--nccl-ib-disable",
        type=int,
        default=1,
        help="设置 NCCL_IB_DISABLE 的值",
    )
    parser.add_argument(
        "--clearml-project",
        default="convnext_classify/convnext_classify_vehicle_10class",
        help="ClearML 项目名",
    )
    parser.add_argument(
        "--clearml-task-name",
        default="convnext-train-wrapper",
        help="ClearML 任务名",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from clearml import Task

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name,
    )
    run_name = args.name or default_run_name()
    expected_output_dir = build_expected_output_dir(args.code_dir, args.project, run_name)

    task_args = vars(args).copy()
    task_args["name"] = run_name
    task_args["expected_output_dir"] = expected_output_dir
    task_args = dict(task.connect(task_args))

    if not task_args.get("name"):
        task_args["name"] = run_name
    task_args["expected_output_dir"] = build_expected_output_dir(
        task_args["code_dir"],
        task_args["project"],
        task_args["name"],
    )

    config = build_config_from_task_args(task_args)
    expected_output_dir = task_args["expected_output_dir"]
    run_config_path = write_run_config(expected_output_dir, task_args)
    command = build_shell_command(config)
    run_summary = build_run_summary(config, expected_output_dir)

    logger = task.get_logger()
    logger.report_text(run_summary)
    logger.report_text(f"Expected output dir: {expected_output_dir}")
    logger.report_text(f"Executing command: {command}")

    completed = subprocess.run(["bash", "-lc", command], check=False)
    artifacts = collect_artifact_paths(expected_output_dir)
    summary_path = write_final_summary(
        expected_output_dir,
        return_code=completed.returncode,
        artifacts=artifacts,
    )
    artifacts["final_summary_json"] = summary_path
    artifacts["run_config_json"] = run_config_path

    for artifact_name, artifact_path in artifacts.items():
        task.upload_artifact(name=artifact_name, artifact_object=artifact_path)

    if "best_model" in artifacts:
        from clearml import OutputModel

        output_model = OutputModel(task=task, framework="PyTorch")
        output_model.update_weights(artifacts["best_model"])

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
