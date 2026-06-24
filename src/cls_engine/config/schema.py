from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass
class TaskConfig:
    project: str = "default"
    name: str = ""
    seed: int = 42
    output: str = ""
    output_explicit: bool = False
    exist_ok: bool = False


@dataclass
class DataConfig:
    root: str = ""
    img_size: int | tuple[int, int] | list[int] = 224
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 8
    preprocess: str = "letterbox"
    augment_backend: str = "cpu"


@dataclass
class ModelConfig:
    name: str = "convnext_nano.in12k_ft_in1k"
    pretrained: bool = True
    cache_dir: str = "/mnt/sdb/cls_dataset_clean/pretrained_cache/huggingface/hub"


@dataclass
class TrainSettings:
    epochs: int = 30
    patience: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.05
    class_weight_mode: str = "inv_sqrt"
    amp: bool = True


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    master_port: int | str = "auto"
    port_range_start: int = 20000
    port_range_end: int = 65000


@dataclass
class EvalConfig:
    print_top_wrong: int = 20
    save_predictions: bool = True


@dataclass
class TrainConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainSettings = field(default_factory=TrainSettings)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "auto"

    def __post_init__(self) -> None:
        self.data.img_size = normalize_img_size(self.data.img_size)

    def validate(self) -> None:
        if not self.task.project:
            raise ValueError("task.project must not be empty")
        if not self.task.name:
            raise ValueError("task.name must not be empty")
        if not (0.0 < self.data.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {self.data.val_ratio}")
        if not (0.0 <= self.data.test_ratio < 1.0):
            raise ValueError(f"test_ratio must be in [0, 1), got {self.data.test_ratio}")
        if self.data.val_ratio + self.data.test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio must be less than 1")
        if self.train.class_weight_mode not in {"inv", "inv_sqrt"}:
            raise ValueError("class_weight_mode must be 'inv' or 'inv_sqrt'")
        if self.train.patience < 0:
            raise ValueError("train.patience must be >= 0")
        if self.data.preprocess not in {"crop", "letterbox", "stretch"}:
            raise ValueError("preprocess must be 'crop', 'letterbox', or 'stretch'")
        if self.data.augment_backend not in {"cpu", "gpu"}:
            raise ValueError("augment_backend must be 'cpu' or 'gpu'")
        if self.distributed.backend not in {"nccl", "gloo"}:
            raise ValueError("distributed backend must be 'nccl' or 'gloo'")
        if not is_valid_device_spec(self.device):
            raise ValueError("device must be auto, cpu, cuda, a GPU id like 1, or a GPU list like 0,1,2")
        if self.distributed.port_range_start >= self.distributed.port_range_end:
            raise ValueError("port_range_start must be smaller than port_range_end")
        height, width = self.data.img_size
        if height <= 0 or width <= 0:
            raise ValueError("img_size values must be positive integers")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_img_size(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            size = int(value[0])
            return (size, size)
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
    raise ValueError("img_size must be an int or one/two integers: HEIGHT [WIDTH]")


def is_valid_device_spec(value: str) -> bool:
    if value in {"auto", "cpu", "cuda"}:
        return True
    return re.fullmatch(r"\d+(,\d+)*", value or "") is not None
