import torch
import numpy as np


def compute_topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, logits.size(1))
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1))
    return correct.any(dim=1).sum().item()


def metrics_from_confusion_matrix(confusion_matrix: np.ndarray) -> dict[str, float]:
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    true_positive = np.diag(cm)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(recall), where=(precision + recall) > 0)
    present = support > 0
    if not np.any(present):
        return {"macro_f1": 0.0, "balanced_accuracy": 0.0, "worst_class_recall": 0.0}
    return {
        "macro_f1": float(f1[present].mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "worst_class_recall": float(recall[present].min()),
    }
