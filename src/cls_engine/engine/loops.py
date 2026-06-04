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
from cls_engine.metrics.classification import compute_topk_correct


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
    log_interval=50,
    batch_transform=None,
):
    model.train()
    if batch_transform is not None:
        batch_transform.train()
    start = now()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_num = 0

    for it, batch in enumerate(loader):
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if batch_transform is not None:
            x = batch_transform(x)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

    avg_loss = (t_loss / torch.clamp(t_num, min=1)).item()
    avg_acc_top1 = (t_cor_top1 / torch.clamp(t_num, min=1)).item()
    avg_acc_top5 = (t_cor_top5 / torch.clamp(t_num, min=1)).item()
    return avg_loss, avg_acc_top1, avg_acc_top5, now() - start


@torch.no_grad()
def evaluate_with_details(model, loader, device, class_names: list[str], batch_transform=None) -> dict[str, Any]:
    model.eval()
    if batch_transform is not None:
        batch_transform.eval()
    class_count = len(class_names)
    total_loss_sum = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_num = 0
    cm_local = torch.zeros((class_count, class_count), device=device, dtype=torch.int64)
    preds_local = []

    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if batch_transform is not None:
            x = batch_transform(x)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        loss_sum = F.cross_entropy(logits, y, reduction="sum")
        total_loss_sum += loss_sum.item()
        total_correct_top1 += (pred == y).sum().item()
        total_correct_top5 += compute_topk_correct(logits, y, k=5)
        total_num += x.size(0)

        for gt_i, pr_i in zip(y.view(-1), pred.view(-1)):
            cm_local[gt_i.long(), pr_i.long()] += 1
        for path, gt_i, pr_i, cf in zip(
            paths,
            y.detach().cpu().tolist(),
            pred.detach().cpu().tolist(),
            conf.detach().cpu().tolist(),
        ):
            preds_local.append([path, int(gt_i), int(pr_i), float(cf)])

    t_loss = reduce_sum(torch.tensor(total_loss_sum, device=device, dtype=torch.float64))
    t_cor_top1 = reduce_sum(torch.tensor(total_correct_top1, device=device, dtype=torch.float64))
    t_cor_top5 = reduce_sum(torch.tensor(total_correct_top5, device=device, dtype=torch.float64))
    t_num = reduce_sum(torch.tensor(total_num, device=device, dtype=torch.float64))
    val_loss = (t_loss / torch.clamp(t_num, min=1)).item()
    val_acc_top1 = (t_cor_top1 / torch.clamp(t_num, min=1)).item()
    val_acc_top5 = (t_cor_top5 / torch.clamp(t_num, min=1)).item()

    cm_total = cm_local.clone()
    if is_dist_avail_and_initialized():
        dist.all_reduce(cm_total, op=dist.ReduceOp.SUM)
    gathered = safe_all_gather_object(preds_local)
    out: dict[str, Any] = {"val_loss": val_loss, "val_acc_top1": val_acc_top1, "val_acc_top5": val_acc_top5}
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
