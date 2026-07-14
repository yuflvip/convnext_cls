import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp

from cls_engine.distributed.ddp import (
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_sum,
    safe_all_gather_object,
)
from cls_engine.metrics import classification as classification_metrics


compute_topk_correct = classification_metrics.compute_topk_correct
metrics_from_confusion_matrix = getattr(
    classification_metrics,
    "metrics_from_confusion_matrix",
    lambda _cm: {"macro_f1": 0.0, "balanced_accuracy": 0.0, "worst_class_recall": 0.0},
)


def now() -> float:
    return time.perf_counter()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    criterion,
    scheduler=None,
    amp_enabled: bool = True,
    log_interval=50,
    batch_transform=None,
    announce_first_batch_wait: bool = False,
    announce_prefix: str = "Train",
):
    model.train()
    if batch_transform is not None:
        batch_transform.train()
    start = now()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_num = 0
    amp_skipped_steps = 0

    if announce_first_batch_wait and is_main_process():
        print(f"[{announce_prefix}] waiting for first batch...")

    for it, batch in enumerate(loader):
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if batch_transform is not None:
            x = batch_transform(x)
        optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda" and amp_enabled
        with amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale() if scaler.is_enabled() else None
        scaler.step(optimizer)
        scaler.update()
        optimizer_updated = not scaler.is_enabled() or scaler.get_scale() >= scale_before
        if scheduler is not None and optimizer_updated:
            scheduler.step()
        if not optimizer_updated:
            amp_skipped_steps += 1

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct_top1 += (logits.argmax(1) == y).sum().item()
        total_correct_top5 += compute_topk_correct(logits, y, k=5)
        total_num += bs
        if (it + 1) % log_interval == 0 and is_main_process():
            print(
                f"[Epoch {epoch}] iter {it + 1}/{len(loader)} "
                f"loss={(total_loss / max(total_num, 1)):.4f} "
                f"acc_top1={(total_correct_top1 / max(total_num, 1)):.4f} "
                f"acc_top5={(total_correct_top5 / max(total_num, 1)):.4f} "
                f"time={now() - start:.1f}s"
            )

    t_loss = reduce_sum(torch.tensor(total_loss, device=device, dtype=torch.float64))
    t_cor_top1 = reduce_sum(torch.tensor(total_correct_top1, device=device, dtype=torch.float64))
    t_cor_top5 = reduce_sum(torch.tensor(total_correct_top5, device=device, dtype=torch.float64))
    t_num = reduce_sum(torch.tensor(total_num, device=device, dtype=torch.float64))
    t_amp_skipped = torch.tensor(amp_skipped_steps, device=device, dtype=torch.int64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(t_amp_skipped, op=dist.ReduceOp.MAX)

    avg_loss = (t_loss / torch.clamp(t_num, min=1)).item()
    avg_acc_top1 = (t_cor_top1 / torch.clamp(t_num, min=1)).item()
    avg_acc_top5 = (t_cor_top5 / torch.clamp(t_num, min=1)).item()
    return avg_loss, avg_acc_top1, avg_acc_top5, now() - start, int(t_amp_skipped.item())


@torch.no_grad()
def evaluate_with_details(
    model,
    loader,
    device,
    class_names: list[str],
    batch_transform=None,
    log_interval: int = 0,
    log_prefix: str = "Eval",
    calibration_bins: int = 15,
) -> dict[str, Any]:
    model.eval()
    eval_model = model.module if hasattr(model, "module") else model
    if batch_transform is not None:
        batch_transform.eval()
    class_count = len(class_names)
    start = now()
    total_loss_sum = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_num = 0
    cm_local = torch.zeros((class_count, class_count), device=device, dtype=torch.int64)
    calibration_local = torch.zeros((calibration_bins, 3), device=device, dtype=torch.float64)
    brier_sum_local = torch.zeros((), device=device, dtype=torch.float64)
    preds_local = []

    for it, (x, y, paths) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if batch_transform is not None:
            x = batch_transform(x)
        logits = eval_model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        loss_sum = F.cross_entropy(logits, y, reduction="sum")
        one_hot = F.one_hot(y, num_classes=class_count).to(probs.dtype)
        brier_sum_local += ((probs - one_hot) ** 2).sum(dim=1).sum().double()
        total_loss_sum += loss_sum.item()
        total_correct_top1 += (pred == y).sum().item()
        total_correct_top5 += compute_topk_correct(logits, y, k=5)
        total_num += x.size(0)

        for gt_i, pr_i in zip(y.view(-1), pred.view(-1)):
            cm_local[gt_i.long(), pr_i.long()] += 1
        bin_indices = torch.clamp((conf * calibration_bins).long(), max=calibration_bins - 1)
        calibration_local[:, 0].scatter_add_(0, bin_indices, torch.ones_like(conf, dtype=torch.float64))
        calibration_local[:, 1].scatter_add_(0, bin_indices, conf.double())
        calibration_local[:, 2].scatter_add_(0, bin_indices, (pred == y).double())
        for path, gt_i, pr_i, cf in zip(
            paths,
            y.detach().cpu().tolist(),
            pred.detach().cpu().tolist(),
            conf.detach().cpu().tolist(),
        ):
            preds_local.append([path, int(gt_i), int(pr_i), float(cf)])

        if log_interval > 0 and is_main_process() and ((it + 1) % log_interval == 0):
            elapsed = now() - start
            avg_batch_time = elapsed / max(it + 1, 1)
            remaining_batches = max(len(loader) - (it + 1), 0)
            eta_sec = avg_batch_time * remaining_batches
            print(
                f"[{log_prefix}] {it + 1}/{len(loader)} batches | "
                f"{total_num} images | elapsed={elapsed:.1f}s | eta={eta_sec:.1f}s"
            )

    t_loss = reduce_sum(torch.tensor(total_loss_sum, device=device, dtype=torch.float64))
    t_cor_top1 = reduce_sum(torch.tensor(total_correct_top1, device=device, dtype=torch.float64))
    t_cor_top5 = reduce_sum(torch.tensor(total_correct_top5, device=device, dtype=torch.float64))
    t_num = reduce_sum(torch.tensor(total_num, device=device, dtype=torch.float64))
    val_loss = (t_loss / torch.clamp(t_num, min=1)).item()
    val_acc_top1 = (t_cor_top1 / torch.clamp(t_num, min=1)).item()
    val_acc_top5 = (t_cor_top5 / torch.clamp(t_num, min=1)).item()

    cm_total = cm_local.clone()
    calibration_total = calibration_local.clone()
    brier_sum_total = brier_sum_local.clone()
    if is_dist_avail_and_initialized():
        dist.all_reduce(cm_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(calibration_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(brier_sum_total, op=dist.ReduceOp.SUM)
    bin_counts = calibration_total[:, 0]
    nonempty = bin_counts > 0
    bin_confidence = torch.zeros_like(bin_counts)
    bin_accuracy = torch.zeros_like(bin_counts)
    bin_confidence[nonempty] = calibration_total[nonempty, 1] / bin_counts[nonempty]
    bin_accuracy[nonempty] = calibration_total[nonempty, 2] / bin_counts[nonempty]
    ece = ((bin_counts / torch.clamp(t_num, min=1)) * (bin_accuracy - bin_confidence).abs()).sum().item()
    brier_score = (brier_sum_total / torch.clamp(t_num, min=1)).item()
    class_metrics = metrics_from_confusion_matrix(cm_total.detach().cpu().numpy())
    gathered = safe_all_gather_object(preds_local)
    out: dict[str, Any] = {
        "val_loss": val_loss,
        "val_acc_top1": val_acc_top1,
        "val_acc_top5": val_acc_top5,
        "macro_f1": class_metrics["macro_f1"],
        "balanced_accuracy": class_metrics["balanced_accuracy"],
        "worst_class_recall": class_metrics["worst_class_recall"],
        "ece": ece,
        "brier_score": brier_score,
        "nll": val_loss,
    }
    if is_main_process():
        all_rows = []
        for part in gathered:
            all_rows.extend(part)
        all_rows_named = []
        wrong_rows_named = []
        for path, gt_i, pr_i, cf in all_rows:
            gt_name = class_names[gt_i]
            pr_name = class_names[pr_i]
            row = [path, gt_i, gt_name, pr_i, pr_name, f"{cf:.6f}"]
            all_rows_named.append(row)
            if gt_i != pr_i:
                wrong_rows_named.append(row)
        out["confusion_matrix"] = cm_total.detach().cpu().numpy().astype(np.int64)
        out["all_rows"] = all_rows_named
        out["wrong_rows"] = wrong_rows_named
    return out
