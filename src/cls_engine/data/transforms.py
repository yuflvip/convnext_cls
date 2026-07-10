import io
import random

import torch
from PIL import Image
from torchvision import transforms

from cls_engine.config.schema import AugmentConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_size(img_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(img_size, int):
        return (img_size, img_size)
    if len(img_size) == 1:
        return (int(img_size[0]), int(img_size[0]))
    return (int(img_size[0]), int(img_size[1]))


def _resize_size(img_size: tuple[int, int]) -> tuple[int, int]:
    return (int(img_size[0] * 1.15), int(img_size[1] * 1.15))


class LetterboxTransform:
    def __init__(self, size: tuple[int, int], pad_value: int = 114):
        self.size = (int(size[0]), int(size[1]))
        self.pad_value = int(pad_value)

    def __call__(self, image: Image.Image) -> Image.Image:
        target_h, target_w = self.size
        src_w, src_h = image.size
        if src_h == target_h and src_w == target_w:
            return image.copy()

        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (target_w, target_h), color=(self.pad_value, self.pad_value, self.pad_value))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas


class RandomProbabilityTransform:
    def __init__(self, transform, p: float):
        self.transform = transform
        self.p = float(p)

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        return self.transform(image)


class RandomJpegCompression:
    def __init__(self, quality_min: int, quality_max: int, p: float):
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)
        self.p = float(p)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def _resolve_augment(augment: AugmentConfig | None) -> AugmentConfig:
    return augment if augment is not None else AugmentConfig()


def _build_cpu_preprocess_ops(
    crop_size: tuple[int, int],
    preprocess: str,
    train: bool,
) -> list:
    if preprocess == "letterbox":
        return [LetterboxTransform(crop_size)]
    if preprocess == "stretch":
        return [transforms.Resize(crop_size)]

    ops = [transforms.Resize(_resize_size(crop_size))]
    if not train:
        ops.append(transforms.CenterCrop(crop_size))
    return ops


def _build_cpu_common_augments(
    crop_size: tuple[int, int],
    preprocess: str,
    augment: AugmentConfig,
) -> tuple[list, list]:
    pil_ops: list = []
    tensor_ops: list = []

    if preprocess == "crop":
        if augment.enabled and augment.resized_crop.enabled:
            pil_ops.append(
                RandomProbabilityTransform(
                    transforms.RandomResizedCrop(
                        crop_size,
                        scale=(augment.resized_crop.scale_min, augment.resized_crop.scale_max),
                    ),
                    p=augment.resized_crop.p,
                )
            )
        else:
            pil_ops.append(transforms.CenterCrop(crop_size))

    if augment.enabled:
        if augment.affine.enabled:
            pil_ops.append(
                RandomProbabilityTransform(
                    transforms.RandomAffine(
                        degrees=augment.affine.degrees,
                        translate=(augment.affine.translate, augment.affine.translate),
                        scale=(augment.affine.scale_min, augment.affine.scale_max),
                    ),
                    p=augment.affine.p,
                )
            )
        if augment.flip.enabled:
            pil_ops.append(transforms.RandomHorizontalFlip(p=augment.flip.p))
        if augment.color_jitter.enabled:
            color_jitter = transforms.ColorJitter(
                brightness=augment.color_jitter.brightness,
                contrast=augment.color_jitter.contrast,
                saturation=augment.color_jitter.saturation,
                hue=augment.color_jitter.hue,
            )
            if augment.color_jitter.p < 1.0:
                pil_ops.append(RandomProbabilityTransform(color_jitter, p=augment.color_jitter.p))
            else:
                pil_ops.append(color_jitter)
        if augment.blur.enabled:
            pil_ops.append(
                RandomProbabilityTransform(
                    transforms.GaussianBlur(
                        kernel_size=augment.blur.kernel_size,
                        sigma=(augment.blur.sigma_min, augment.blur.sigma_max),
                    ),
                    p=augment.blur.p,
                )
            )
        if augment.jpeg.enabled:
            pil_ops.append(
                RandomJpegCompression(
                    quality_min=augment.jpeg.quality_min,
                    quality_max=augment.jpeg.quality_max,
                    p=augment.jpeg.p,
                )
            )
        if augment.erasing.enabled:
            tensor_ops.append(
                transforms.RandomErasing(
                    p=augment.erasing.p,
                    scale=(augment.erasing.scale_min, augment.erasing.scale_max),
                    ratio=(augment.erasing.ratio_min, augment.erasing.ratio_max),
                    value=augment.erasing.value,
                )
            )

    return pil_ops, tensor_ops


def build_train_transform(
    img_size: int | tuple[int, int] | list[int],
    augment_backend: str = "cpu",
    preprocess: str = "letterbox",
    augment: AugmentConfig | None = None,
):
    crop_size = _normalize_size(img_size)
    augment = _resolve_augment(augment)
    if augment_backend == "gpu":
        return transforms.Compose(_build_cpu_preprocess_ops(crop_size, preprocess, train=True) + [transforms.ToTensor()])

    preprocess_ops = _build_cpu_preprocess_ops(crop_size, preprocess, train=True)
    pil_augments, tensor_augments = _build_cpu_common_augments(crop_size, preprocess, augment)
    return transforms.Compose(
        preprocess_ops
        + pil_augments
        + [
            transforms.ToTensor(),
            *tensor_augments,
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(
    img_size: int | tuple[int, int] | list[int],
    augment_backend: str = "cpu",
    preprocess: str = "letterbox",
    augment: AugmentConfig | None = None,
):
    crop_size = _normalize_size(img_size)
    if augment_backend == "gpu":
        return transforms.Compose(_build_cpu_preprocess_ops(crop_size, preprocess, train=False) + [transforms.ToTensor()])
    return transforms.Compose(
        _build_cpu_preprocess_ops(crop_size, preprocess, train=False)
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _validate_gpu_supported_augments(augment: AugmentConfig) -> None:
    if not augment.enabled:
        return
    if augment.jpeg.enabled:
        raise ValueError("augment.jpeg is not supported with augment_backend=gpu")


def build_gpu_train_batch_augment(
    img_size: int | tuple[int, int] | list[int],
    preprocess: str = "letterbox",
    augment: AugmentConfig | None = None,
):
    crop_size = _normalize_size(img_size)
    augment = _resolve_augment(augment)
    _validate_gpu_supported_augments(augment)
    try:
        import kornia.augmentation as K
    except ImportError as exc:
        raise ImportError("augment_backend=gpu requires kornia to be installed.") from exc

    class GpuTrainBatchAugment(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_crop = K.CenterCrop(size=crop_size, p=1.0) if preprocess == "crop" else None
            self.random_crop = None
            if augment.enabled and preprocess == "crop" and augment.resized_crop.enabled:
                self.random_crop = K.RandomResizedCrop(
                    size=crop_size,
                    scale=(augment.resized_crop.scale_min, augment.resized_crop.scale_max),
                    same_on_batch=False,
                    p=augment.resized_crop.p,
                )
            self.random_affine = None
            if augment.enabled and augment.affine.enabled:
                self.random_affine = K.RandomAffine(
                    degrees=augment.affine.degrees,
                    translate=(augment.affine.translate, augment.affine.translate),
                    scale=(augment.affine.scale_min, augment.affine.scale_max),
                    p=augment.affine.p,
                )
            self.random_flip = K.RandomHorizontalFlip(p=augment.flip.p) if augment.enabled and augment.flip.enabled else None
            self.color_jitter = None
            if augment.enabled and augment.color_jitter.enabled:
                self.color_jitter = K.ColorJiggle(
                    brightness=augment.color_jitter.brightness,
                    contrast=augment.color_jitter.contrast,
                    saturation=augment.color_jitter.saturation,
                    hue=augment.color_jitter.hue,
                    p=augment.color_jitter.p,
                )
            self.random_blur = None
            if augment.enabled and augment.blur.enabled:
                self.random_blur = K.RandomGaussianBlur(
                    kernel_size=(augment.blur.kernel_size, augment.blur.kernel_size),
                    sigma=(augment.blur.sigma_min, augment.blur.sigma_max),
                    p=augment.blur.p,
                )
            self.random_erasing = None
            if augment.enabled and augment.erasing.enabled:
                self.random_erasing = K.RandomErasing(
                    scale=(augment.erasing.scale_min, augment.erasing.scale_max),
                    ratio=(augment.erasing.ratio_min, augment.erasing.ratio_max),
                    value=augment.erasing.value,
                    p=augment.erasing.p,
                )
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.random_crop is not None:
                x = self.random_crop(x)
            elif self.base_crop is not None:
                x = self.base_crop(x)
            if self.random_affine is not None:
                x = self.random_affine(x)
            if self.random_flip is not None:
                x = self.random_flip(x)
            if self.color_jitter is not None:
                x = self.color_jitter(x)
            if self.random_blur is not None:
                x = self.random_blur(x)
            if self.random_erasing is not None:
                x = self.random_erasing(x)
            return (x - self.mean) / self.std

    return GpuTrainBatchAugment()


def build_gpu_eval_batch_augment(
    img_size: int | tuple[int, int] | list[int],
    preprocess: str = "letterbox",
    augment: AugmentConfig | None = None,
):
    crop_size = _normalize_size(img_size)
    _ = _resolve_augment(augment)
    try:
        import kornia.augmentation as K
    except ImportError as exc:
        raise ImportError("augment_backend=gpu requires kornia to be installed.") from exc

    class GpuEvalBatchAugment(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.center_crop = K.CenterCrop(size=crop_size, p=1.0) if preprocess == "crop" else None
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.center_crop is not None:
                x = self.center_crop(x)
            return (x - self.mean) / self.std

    return GpuEvalBatchAugment()
