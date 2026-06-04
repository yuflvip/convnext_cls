import os
from typing import Any

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        kwargs = {"backend": backend, "init_method": "env://"}
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            if backend == "nccl":
                kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**kwargs)


def barrier() -> None:
    if not is_dist_avail_and_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
    else:
        dist.barrier()


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        barrier()
        dist.destroy_process_group()


def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return tensor
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def safe_all_gather_object(obj: Any) -> list[Any]:
    if not is_dist_avail_and_initialized():
        return [obj]
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered
