import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.config.schema import DataConfig, TaskConfig, TrainConfig


TRANSFORMS_PATH = SRC / "cls_engine" / "data" / "transforms.py"


def _load_transforms_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.Tensor = object

    class FakeModule:
        def register_buffer(self, name, value, persistent=False):
            setattr(self, name, value)

    class FakeNN(types.SimpleNamespace):
        Module = FakeModule

    fake_torch.nn = FakeNN()
    fake_torch.tensor = lambda *args, **kwargs: types.SimpleNamespace(view=lambda *a, **k: ("tensor", args, kwargs))

    fake_transforms_module = types.ModuleType("torchvision.transforms")

    class Compose:
        def __init__(self, transforms):
            self.transforms = list(transforms)

    class Resize:
        def __init__(self, size):
            self.size = size

    class RandomResizedCrop:
        def __init__(self, size, scale):
            self.size = size
            self.scale = scale

    class CenterCrop:
        def __init__(self, size):
            self.size = size

    class RandomHorizontalFlip:
        def __init__(self, p):
            self.p = p

    class ColorJitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ToTensor:
        pass

    class Normalize:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

    fake_transforms_module.Compose = Compose
    fake_transforms_module.Resize = Resize
    fake_transforms_module.RandomResizedCrop = RandomResizedCrop
    fake_transforms_module.CenterCrop = CenterCrop
    fake_transforms_module.RandomHorizontalFlip = RandomHorizontalFlip
    fake_transforms_module.ColorJitter = ColorJitter
    fake_transforms_module.ToTensor = ToTensor
    fake_transforms_module.Normalize = Normalize

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.transforms = fake_transforms_module

    fake_pil = types.ModuleType("PIL")
    fake_image = types.SimpleNamespace(Image=object)
    fake_pil.Image = fake_image

    fake_kornia_aug = types.ModuleType("kornia.augmentation")

    class RandomResizedCropK:
        def __init__(self, size, scale, same_on_batch, p):
            self.size = size
            self.scale = scale
            self.same_on_batch = same_on_batch
            self.p = p

    class CenterCropK:
        def __init__(self, size, p):
            self.size = size
            self.p = p

    class RandomHorizontalFlipK:
        def __init__(self, p):
            self.p = p

    class ColorJiggle:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_kornia_aug.RandomResizedCrop = RandomResizedCropK
    fake_kornia_aug.CenterCrop = CenterCropK
    fake_kornia_aug.RandomHorizontalFlip = RandomHorizontalFlipK
    fake_kornia_aug.ColorJiggle = ColorJiggle

    fake_kornia = types.ModuleType("kornia")
    fake_kornia.augmentation = fake_kornia_aug

    module_name = "test_transforms_module"
    spec = importlib.util.spec_from_file_location(module_name, TRANSFORMS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.update(
        {
            "torch": fake_torch,
            "torchvision": fake_torchvision,
            "torchvision.transforms": fake_transforms_module,
            "PIL": fake_pil,
            "PIL.Image": fake_image,
            "kornia": fake_kornia,
            "kornia.augmentation": fake_kornia_aug,
        }
    )
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StretchPreprocessTests(unittest.TestCase):
    def test_train_config_accepts_stretch_preprocess(self):
        cfg = TrainConfig(
            task=TaskConfig(project="demo", name="exp"),
            data=DataConfig(preprocess="stretch"),
        )

        cfg.validate()

    def test_build_train_transform_stretch_uses_direct_resize(self):
        module = _load_transforms_module()

        tf = module.build_train_transform((224, 224), augment_backend="cpu", preprocess="stretch")

        self.assertEqual([type(item).__name__ for item in tf.transforms], ["Resize", "RandomHorizontalFlip", "ColorJitter", "ToTensor", "Normalize"])
        self.assertEqual(tf.transforms[0].size, (224, 224))

    def test_build_eval_transform_stretch_uses_direct_resize(self):
        module = _load_transforms_module()

        tf = module.build_eval_transform((224, 224), augment_backend="cpu", preprocess="stretch")

        self.assertEqual([type(item).__name__ for item in tf.transforms], ["Resize", "ToTensor", "Normalize"])
        self.assertEqual(tf.transforms[0].size, (224, 224))

    def test_build_gpu_train_batch_augment_stretch_skips_random_crop(self):
        module = _load_transforms_module()

        augment = module.build_gpu_train_batch_augment((224, 224), preprocess="stretch")

        self.assertIsNone(augment.random_crop)

    def test_build_gpu_eval_batch_augment_stretch_skips_center_crop(self):
        module = _load_transforms_module()

        augment = module.build_gpu_eval_batch_augment((224, 224), preprocess="stretch")

        self.assertIsNone(augment.center_crop)


if __name__ == "__main__":
    unittest.main()
