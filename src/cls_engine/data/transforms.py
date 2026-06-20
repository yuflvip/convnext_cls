import torch
from torchvision import transforms
from PIL import Image


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


def build_train_transform(
    img_size: int | tuple[int, int] | list[int],
    augment_backend: str = "cpu",
    preprocess: str = "letterbox",
):
    crop_size = _normalize_size(img_size)
    if preprocess == "letterbox":
        if augment_backend == "gpu":
            return transforms.Compose([
                LetterboxTransform(crop_size),
                transforms.ToTensor(),
            ])
        return transforms.Compose([
            LetterboxTransform(crop_size),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    if preprocess == "stretch":
        return transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    if augment_backend == "gpu":
        return transforms.Compose([
            transforms.Resize(_resize_size(crop_size)),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(_resize_size(crop_size)),
        transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(
    img_size: int | tuple[int, int] | list[int],
    augment_backend: str = "cpu",
    preprocess: str = "letterbox",
):
    crop_size = _normalize_size(img_size)
    if preprocess == "letterbox":
        if augment_backend == "gpu":
            return transforms.Compose([
                LetterboxTransform(crop_size),
                transforms.ToTensor(),
            ])
        return transforms.Compose([
            LetterboxTransform(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    if preprocess == "stretch":
        return transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    if augment_backend == "gpu":
        return transforms.Compose([
            transforms.Resize(_resize_size(crop_size)),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(_resize_size(crop_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_gpu_train_batch_augment(
    img_size: int | tuple[int, int] | list[int],
    preprocess: str = "letterbox",
):
    crop_size = _normalize_size(img_size)
    try:
        import kornia.augmentation as K
    except ImportError as exc:
        raise ImportError("augment_backend=gpu requires kornia to be installed.") from exc

    class GpuTrainBatchAugment(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.random_crop = None
            if preprocess == "crop":
                self.random_crop = K.RandomResizedCrop(
                    size=crop_size,
                    scale=(0.7, 1.0),
                    same_on_batch=False,
                    p=1.0,
                )
            self.random_flip = K.RandomHorizontalFlip(p=0.6)
            self.color_jitter = K.ColorJiggle(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
                p=1.0,
            )
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.random_crop is not None:
                x = self.random_crop(x)
            x = self.random_flip(x)
            x = self.color_jitter(x)
            return (x - self.mean) / self.std

    return GpuTrainBatchAugment()


def build_gpu_eval_batch_augment(
    img_size: int | tuple[int, int] | list[int],
    preprocess: str = "letterbox",
):
    crop_size = _normalize_size(img_size)
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
