import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import amp

from cls_engine.config.schema import TrainConfig
from cls_engine.data.datamodule import PreparedData, prepare_data
from cls_engine.data.splits import build_class_weights
from cls_engine.data.transforms import build_gpu_eval_batch_augment, build_gpu_train_batch_augment
from cls_engine.distributed.ddp import (
    barrier,
    cleanup_distributed,
    get_rank,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_distributed,
)
from cls_engine.distributed.device import (
    apply_device_spec,
    configure_torch_backend,
    parse_device_spec,
    resolve_device,
    validate_multi_gpu_launch,
)
from cls_engine.distributed.port import PortConfig, resolve_master_port
from cls_engine.engine.evaluator import write_eval_artifacts
from cls_engine.engine.loops import evaluate_with_details, train_one_epoch
from cls_engine.io.artifacts import ArtifactWriter
from cls_engine.models.checkpoint import load_checkpoint, save_best_checkpoint, save_last_checkpoint, unwrap_model
from cls_engine.models.factory import build_model
from cls_engine.utils.paths import ensure_dir, resolve_output_dir
from cls_engine.utils.seed import set_seed


RESULTS_CSV_HEADER = [
    "epoch",
    "train_loss",
    "train_acc_top1",
    "train_acc_top5",
    "val_loss",
    "val_acc_top1",
    "val_acc_top5",
    "lr",
    "epoch_time_sec",
]


def _run_id() -> str:
    return f"{int(time.time())}-{os.getpid()}"


def _build_scheduler(optimizer, epochs: int, steps_per_epoch: int):
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = max(100, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _dataset_layout_payload(data: PreparedData) -> dict:
    layout = data.layout
    return {
        "mode": layout.mode,
        "train_dir": str(layout.train_dir),
        "val_dir": str(layout.val_dir) if layout.val_dir else None,
        "test_dir": str(layout.test_dir) if layout.test_dir else None,
        "has_explicit_val": layout.has_explicit_val,
        "has_explicit_test": layout.has_explicit_test,
        "class_names": data.class_names,
    }


def _write_initial_artifacts(writer: ArtifactWriter, cfg: TrainConfig, data: PreparedData, port_info: dict) -> None:
    writer.write_classes(data.class_names)
    writer.write_run_config({
        "config": cfg.to_dict(),
        "dataset_layout": _dataset_layout_payload(data),
        "distributed": {
            "port": port_info,
            "rank": get_rank(),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "backend": cfg.distributed.backend,
        },
    })
    if not data.layout.has_explicit_val:
        writer.write_split_indices({
            "mode": data.layout.mode,
            "train_indices": data.train_indices,
            "val_indices": data.val_indices,
            "test_indices": data.test_indices if data.test_set is not None and not data.layout.has_explicit_test else [],
            "class_names": data.class_names,
            "seed": cfg.task.seed,
        })


def run_training(cfg: TrainConfig) -> None:
    resolved_output_dir, final_run_name = resolve_output_dir(
        project=cfg.task.project,
        name=cfg.task.name,
        output=cfg.task.output if cfg.task.output_explicit else None,
        exist_ok=cfg.task.exist_ok,
    )
    cfg.task.output = str(resolved_output_dir)
    cfg.task.name = final_run_name

    run_id = _run_id()
    port_cfg = PortConfig(
        master_port=cfg.distributed.master_port,
        range_start=cfg.distributed.port_range_start,
        range_end=cfg.distributed.port_range_end,
    )
    _, port_info = resolve_master_port(
        port_cfg,
        data_root=cfg.data.root,
        model_name=cfg.model.name,
        output_dir=cfg.task.output,
        run_id=run_id,
    )

    try:
        device_spec = parse_device_spec(cfg.device)
        apply_device_spec(device_spec)
        validate_multi_gpu_launch(device_spec)
        setup_distributed(cfg.distributed.backend)
        set_seed(cfg.task.seed + get_rank())
        device = resolve_device(cfg.device)
        configure_torch_backend(device)
        out_dir = ensure_dir(cfg.task.output) if is_main_process() else Path(cfg.task.output)

        data = prepare_data(cfg.data, seed=cfg.task.seed, batch_size=cfg.train.batch_size)
        writer = ArtifactWriter(out_dir) if is_main_process() else None
        if is_main_process():
            _write_initial_artifacts(writer, cfg, data, port_info)

        train_batch_transform = None
        eval_batch_transform = None
        if cfg.data.augment_backend == "gpu":
            train_batch_transform = build_gpu_train_batch_augment(cfg.data.img_size, preprocess=cfg.data.preprocess).to(device)
            eval_batch_transform = build_gpu_eval_batch_augment(cfg.data.img_size, preprocess=cfg.data.preprocess).to(device)

        class_weights = build_class_weights(data.train_counts, mode=cfg.train.class_weight_mode).to(device)
        model = build_model(
            cfg.model.name,
            num_classes=data.num_classes,
            pretrained=cfg.model.pretrained,
            cache_dir=cfg.model.cache_dir,
        ).to(device)
        if is_dist_avail_and_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = _build_scheduler(optimizer, cfg.train.epochs, len(data.train_loader))
        scaler = amp.GradScaler(enabled=(device.type == "cuda" and cfg.train.amp))
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        best_val_acc = -1.0
        test_acc = None
        test_loss = None

        for epoch in range(1, cfg.train.epochs + 1):
            sampler = getattr(data.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            train_loss, train_acc_top1, train_acc_top5, train_time = train_one_epoch(
                model,
                data.train_loader,
                optimizer,
                scaler,
                device,
                epoch,
                criterion,
                log_interval=50,
                batch_transform=train_batch_transform,
            )
            for _ in range(len(data.train_loader)):
                scheduler.step()

            eval_out = evaluate_with_details(
                model,
                data.val_loader,
                device,
                data.class_names,
                batch_transform=eval_batch_transform,
            )
            val_loss = eval_out["val_loss"]
            val_acc_top1 = eval_out["val_acc_top1"]
            val_acc_top5 = eval_out["val_acc_top5"]

            if is_main_process():
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"\n[Epoch {epoch}] train_loss={train_loss:.4f} "
                    f"train_acc_top1={train_acc_top1:.4f} train_acc_top5={train_acc_top5:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc_top1={val_acc_top1:.4f} "
                    f"val_acc_top5={val_acc_top5:.4f} | lr={lr_now:.6g} time={train_time:.1f}s"
                )
                if val_acc_top1 > best_val_acc:
                    best_val_acc = val_acc_top1
                    save_best_checkpoint(
                        out_dir / "best.pth",
                        model,
                        epoch,
                        cfg.model.name,
                        best_val_acc,
                        data.class_names,
                        cfg.data.img_size,
                        val_loss,
                    )
                    print(f"Saved best checkpoint: val_acc={best_val_acc:.4f}")
                save_last_checkpoint(out_dir / "last.pth", model, epoch, data.class_names)
                write_eval_artifacts(writer, "val", eval_out, data.class_names, cfg.eval.print_top_wrong)
                writer.append_epoch_result(
                    [
                        epoch,
                        f"{train_loss:.6f}",
                        f"{train_acc_top1:.6f}",
                        f"{train_acc_top5:.6f}",
                        f"{val_loss:.6f}",
                        f"{val_acc_top1:.6f}",
                        f"{val_acc_top5:.6f}",
                        f"{lr_now:.8f}",
                        f"{train_time:.4f}",
                    ],
                    RESULTS_CSV_HEADER,
                    first_epoch=(epoch == 1),
                )

        if data.test_loader is not None:
            if is_dist_avail_and_initialized():
                barrier()
            checkpoint = load_checkpoint(Path(cfg.task.output) / "best.pth", map_location="cpu")
            unwrap_model(model).load_state_dict(checkpoint["state_dict"])
            test_out = evaluate_with_details(
                model,
                data.test_loader,
                device,
                data.class_names,
                batch_transform=eval_batch_transform,
            )
            test_loss = test_out["val_loss"]
            test_acc = test_out["val_acc_top1"]
            if is_main_process():
                write_eval_artifacts(writer, "test", test_out, data.class_names, cfg.eval.print_top_wrong)

        if is_main_process():
            writer.write_final_summary({
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "num_classes": data.num_classes,
                "class_names": data.class_names,
                "train_samples": len(data.train_indices),
                "val_samples": len(data.val_indices),
                "test_samples": len(data.test_set) if data.test_set is not None else 0,
                "split_mode": data.layout.mode,
            })
            print(f"Training done. Best val acc = {best_val_acc:.4f}")
            print(f"Outputs in: {Path(cfg.task.output).resolve()}")
    finally:
        cleanup_distributed()
