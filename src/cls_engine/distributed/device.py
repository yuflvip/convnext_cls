from dataclasses import dataclass
import os
import re

import torch


@dataclass(frozen=True)
class DeviceSpec:
    raw: str
    mode: str
    cuda_visible_devices: str | None
    requested_gpu_count: int
    requires_torchrun: bool


def parse_device_spec(requested: str = "auto") -> DeviceSpec:
    requested = requested or "auto"
    if requested == "cpu":
        return DeviceSpec(requested, "cpu", None, 0, False)
    if requested == "auto":
        return DeviceSpec(requested, "auto", None, 0, False)
    if requested == "cuda":
        return DeviceSpec(requested, "cuda", None, 0, False)
    if re.fullmatch(r"\d+(,\d+)*", requested):
        gpu_ids = requested.split(",")
        return DeviceSpec(
            raw=requested,
            mode="cuda",
            cuda_visible_devices=requested,
            requested_gpu_count=len(gpu_ids),
            requires_torchrun=(len(gpu_ids) > 1),
        )
    raise ValueError("device must be auto, cpu, cuda, a GPU id like 1, or a GPU list like 0,1,2")


def apply_device_spec(spec: DeviceSpec) -> None:
    if spec.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices


def validate_multi_gpu_launch(spec: DeviceSpec) -> None:
    if not spec.requires_torchrun:
        return
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            f"--device {spec.raw} selects {spec.requested_gpu_count} GPUs, "
            "but this command started only one Python process. Use torchrun, for example: "
            f"torchrun --nproc_per_node={spec.requested_gpu_count} tools/train.py --device={spec.raw} ..."
        )
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size and int(local_world_size) != spec.requested_gpu_count:
        raise RuntimeError(
            f"--device {spec.raw} selects {spec.requested_gpu_count} GPUs, "
            f"but torchrun LOCAL_WORLD_SIZE={local_world_size}. "
            f"Use --nproc_per_node={spec.requested_gpu_count} or adjust --device."
        )


def resolve_device(requested: str = "auto") -> torch.device:
    spec = parse_device_spec(requested)
    if spec.mode == "cpu":
        return torch.device("cpu")
    if spec.mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_torch_backend(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
