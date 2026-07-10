import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .schema import (
    AffineAugmentConfig,
    AugmentConfig,
    BlurAugmentConfig,
    ColorJitterAugmentConfig,
    DataConfig,
    DistributedConfig,
    ErasingAugmentConfig,
    EvalConfig,
    FlipAugmentConfig,
    JpegAugmentConfig,
    ModelConfig,
    ResizedCropAugmentConfig,
    TaskConfig,
    TrainConfig,
    TrainSettings,
    normalize_img_size,
)
from cls_engine.utils.paths import build_default_run_name


def _read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML config requires pyyaml. Use JSON config or install pyyaml.") from exc
    loaded = yaml.safe_load(text)
    return loaded or {}


def _build_dataclass(cls, payload: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in allowed})


def _set_if_present(obj: object, attr: str, value: Any) -> None:
    if value is not None:
        setattr(obj, attr, value)


def _has_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _build_augment_config(payload: dict[str, Any] | None) -> AugmentConfig:
    payload = dict(payload or {})
    return AugmentConfig(
        enabled=payload.get("enabled", True),
        flip=_build_dataclass(FlipAugmentConfig, payload.get("flip", {})),
        color_jitter=_build_dataclass(ColorJitterAugmentConfig, payload.get("color_jitter", {})),
        blur=_build_dataclass(BlurAugmentConfig, payload.get("blur", {})),
        erasing=_build_dataclass(ErasingAugmentConfig, payload.get("erasing", {})),
        affine=_build_dataclass(AffineAugmentConfig, payload.get("affine", {})),
        resized_crop=_build_dataclass(ResizedCropAugmentConfig, payload.get("resized_crop", {})),
        jpeg=_build_dataclass(JpegAugmentConfig, payload.get("jpeg", {})),
    )


def _build_data_config(payload: dict[str, Any]) -> DataConfig:
    payload = dict(payload)
    augment_raw = payload.pop("augment", {})
    cfg = _build_dataclass(DataConfig, payload)
    cfg.augment = _build_augment_config(augment_raw)
    return cfg


def load_train_config(path: str | Path, args: object | None = None) -> TrainConfig:
    raw = _read_config(Path(path))
    task_raw = dict(raw.get("task", {}))
    if _has_nonempty_value(task_raw.get("output")):
        task_raw["output_explicit"] = True
    if "output" not in task_raw and "output_dir" in task_raw:
        task_raw["output"] = task_raw["output_dir"]
        task_raw["output_explicit"] = _has_nonempty_value(task_raw["output"])
    cfg = TrainConfig(
        task=_build_dataclass(TaskConfig, task_raw),
        data=_build_data_config(raw.get("data", {})),
        model=_build_dataclass(ModelConfig, raw.get("model", {})),
        train=_build_dataclass(TrainSettings, raw.get("train", {})),
        distributed=_build_dataclass(DistributedConfig, raw.get("distributed", {})),
        eval=_build_dataclass(EvalConfig, raw.get("eval", {})),
        device=raw.get("device", "auto"),
    )

    if args is not None:
        _set_if_present(cfg.data, "root", getattr(args, "data", None))
        _set_if_present(cfg.task, "project", getattr(args, "project", None))
        _set_if_present(cfg.task, "name", getattr(args, "name", None))
        _set_if_present(cfg.task, "output", getattr(args, "output", None))
        if _has_nonempty_value(getattr(args, "output", None)):
            cfg.task.output_explicit = True
        if getattr(args, "exist_ok", None) is not None:
            cfg.task.exist_ok = bool(getattr(args, "exist_ok"))
        _set_if_present(cfg.model, "name", getattr(args, "model", None))
        _set_if_present(cfg.train, "epochs", getattr(args, "epochs", None))
        _set_if_present(cfg.train, "patience", getattr(args, "patience", None))
        _set_if_present(cfg.train, "batch_size", getattr(args, "batch", None))
        _set_if_present(cfg.data, "num_workers", getattr(args, "workers", None))
        _set_if_present(cfg.data, "preprocess", getattr(args, "preprocess", None))
        _set_if_present(cfg.data, "augment_backend", getattr(args, "augment_backend", None))
        _set_if_present(cfg.data, "img_size", getattr(args, "imgsz", None))
        _set_if_present(cfg, "device", getattr(args, "device", None))
        _set_if_present(cfg.distributed, "master_port", getattr(args, "master_port", None))
        _set_if_present(cfg.eval, "print_top_wrong", getattr(args, "print_top_wrong", None))

    if not cfg.task.name:
        cfg.task.name = build_default_run_name()
    cfg.data.img_size = normalize_img_size(cfg.data.img_size)
    cfg.validate()
    return cfg
