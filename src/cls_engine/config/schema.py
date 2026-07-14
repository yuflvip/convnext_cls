from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass
class FlipAugmentConfig:
    enabled: bool = True
    p: float = 0.5


@dataclass
class ColorJitterAugmentConfig:
    enabled: bool = True
    p: float = 1.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.02


@dataclass
class BlurAugmentConfig:
    enabled: bool = False
    p: float = 0.1
    kernel_size: int = 3
    sigma_min: float = 0.1
    sigma_max: float = 1.0


@dataclass
class ErasingAugmentConfig:
    enabled: bool = False
    p: float = 0.1
    scale_min: float = 0.02
    scale_max: float = 0.12
    ratio_min: float = 0.3
    ratio_max: float = 3.3
    value: float = 0.0


@dataclass
class AffineAugmentConfig:
    enabled: bool = False
    p: float = 0.1
    degrees: float = 5.0
    translate: float = 0.05
    scale_min: float = 0.95
    scale_max: float = 1.05


@dataclass
class ResizedCropAugmentConfig:
    enabled: bool = False
    p: float = 1.0
    scale_min: float = 0.8
    scale_max: float = 1.0


@dataclass
class JpegAugmentConfig:
    enabled: bool = False
    p: float = 0.1
    quality_min: int = 60
    quality_max: int = 95


@dataclass
class AugmentConfig:
    enabled: bool = True
    flip: FlipAugmentConfig = field(default_factory=FlipAugmentConfig)
    color_jitter: ColorJitterAugmentConfig = field(default_factory=ColorJitterAugmentConfig)
    blur: BlurAugmentConfig = field(default_factory=BlurAugmentConfig)
    erasing: ErasingAugmentConfig = field(default_factory=ErasingAugmentConfig)
    affine: AffineAugmentConfig = field(default_factory=AffineAugmentConfig)
    resized_crop: ResizedCropAugmentConfig = field(default_factory=ResizedCropAugmentConfig)
    jpeg: JpegAugmentConfig = field(default_factory=JpegAugmentConfig)


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
    augment: AugmentConfig = field(default_factory=AugmentConfig)


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
    class_weight_mode: str = "none"
    label_smoothing: float = 0.05
    amp: bool = True
    warmup_epochs: float = 5.0
    warmup_steps: int | str = "auto"
    min_lr_ratio: float = 0.01


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
    calibration_bins: int = 15


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
        if not self.data.root or not self.data.root.strip():
            raise ValueError("data.root must be set in the config file or provided with --data")
        if not (0.0 < self.data.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {self.data.val_ratio}")
        if not (0.0 <= self.data.test_ratio < 1.0):
            raise ValueError(f"test_ratio must be in [0, 1), got {self.data.test_ratio}")
        if self.data.val_ratio + self.data.test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio must be less than 1")
        if self.train.class_weight_mode not in {"none", "inv", "inv_sqrt"}:
            raise ValueError("class_weight_mode must be 'none', 'inv', or 'inv_sqrt'")
        if not (0.0 <= self.train.label_smoothing < 1.0):
            raise ValueError("train.label_smoothing must be in [0, 1)")
        if self.train.patience < 0:
            raise ValueError("train.patience must be >= 0")
        if self.train.warmup_epochs < 0:
            raise ValueError("train.warmup_epochs must be >= 0")
        if not (
            self.train.warmup_steps == "auto"
            or (
                isinstance(self.train.warmup_steps, int)
                and not isinstance(self.train.warmup_steps, bool)
                and self.train.warmup_steps >= 0
            )
        ):
            raise ValueError("train.warmup_steps must be 'auto' or a non-negative integer")
        if not (0.0 <= self.train.min_lr_ratio <= 1.0):
            raise ValueError("train.min_lr_ratio must be in [0, 1]")
        if self.data.preprocess not in {"crop", "letterbox", "stretch"}:
            raise ValueError("preprocess must be 'crop', 'letterbox', or 'stretch'")
        if self.data.augment_backend not in {"cpu", "gpu"}:
            raise ValueError("augment_backend must be 'cpu' or 'gpu'")
        _validate_augment_config(self.data.augment)
        if self.distributed.backend not in {"nccl", "gloo"}:
            raise ValueError("distributed backend must be 'nccl' or 'gloo'")
        if self.eval.calibration_bins <= 0:
            raise ValueError("eval.calibration_bins must be a positive integer")
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


def _validate_probability(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_augment_config(cfg: AugmentConfig) -> None:
    _validate_probability("augment.flip.p", cfg.flip.p)
    _validate_probability("augment.color_jitter.p", cfg.color_jitter.p)
    _validate_probability("augment.blur.p", cfg.blur.p)
    _validate_probability("augment.erasing.p", cfg.erasing.p)
    _validate_probability("augment.affine.p", cfg.affine.p)
    _validate_probability("augment.resized_crop.p", cfg.resized_crop.p)
    _validate_probability("augment.jpeg.p", cfg.jpeg.p)

    if cfg.blur.kernel_size <= 0 or cfg.blur.kernel_size % 2 == 0:
        raise ValueError("augment.blur.kernel_size must be a positive odd integer")
    if cfg.blur.sigma_min > cfg.blur.sigma_max:
        raise ValueError("augment.blur.sigma_min must be <= sigma_max")

    if cfg.erasing.scale_min > cfg.erasing.scale_max:
        raise ValueError("augment.erasing.scale_min must be <= scale_max")
    if cfg.erasing.ratio_min > cfg.erasing.ratio_max:
        raise ValueError("augment.erasing.ratio_min must be <= ratio_max")

    if cfg.affine.degrees < 0:
        raise ValueError("augment.affine.degrees must be >= 0")
    if not (0.0 <= cfg.affine.translate < 1.0):
        raise ValueError("augment.affine.translate must be in [0, 1)")
    if cfg.affine.scale_min > cfg.affine.scale_max:
        raise ValueError("augment.affine.scale_min must be <= scale_max")

    if cfg.resized_crop.scale_min > cfg.resized_crop.scale_max:
        raise ValueError("augment.resized_crop.scale_min must be <= scale_max")

    if cfg.jpeg.quality_min > cfg.jpeg.quality_max:
        raise ValueError("augment.jpeg.quality_min must be <= quality_max")
