import numpy as np


def confusion_to_report(cm: np.ndarray, class_names: list[str]) -> str:
    class_count = cm.shape[0]
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_cnt = cm.sum(axis=0).astype(np.float64)
    recall = tp / (support + eps)
    precision = tp / (pred_cnt + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = tp.sum() / (cm.sum() + eps)

    lines = ["Per-class metrics:", "class,precision,recall,f1,support"]
    for i in range(class_count):
        lines.append(f"{class_names[i]},{precision[i]:.6f},{recall[i]:.6f},{f1[i]:.6f},{int(support[i])}")
    lines.extend([
        "",
        f"Overall accuracy: {acc:.6f}",
        f"Macro precision: {precision.mean():.6f}",
        f"Macro recall   : {recall.mean():.6f}",
        f"Macro F1       : {f1.mean():.6f}",
    ])
    return "\n".join(lines)
