import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler[int]):
    """Split evaluation indices across ranks without padding or dropping samples."""

    def __init__(self, dataset, num_replicas: int | None = None, rank: int | None = None):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("DistributedEvalSampler requires an initialized process group")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("DistributedEvalSampler requires an initialized process group")
            rank = dist.get_rank()
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas
