import torch


def compute_topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, logits.size(1))
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1))
    return correct.any(dim=1).sum().item()
