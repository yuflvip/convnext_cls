import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_module(module_name: str, path: Path, stub_modules: dict[str, object]):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    package_parts = module_name.split(".")
    for index in range(1, len(package_parts)):
        package_name = ".".join(package_parts[:index])
        if package_name not in sys.modules:
            package_module = types.ModuleType(package_name)
            package_module.__path__ = []
            sys.modules[package_name] = package_module
    with mock.patch.dict(sys.modules, stub_modules, clear=False):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class TrainingProgressLogTests(unittest.TestCase):
    def test_prepare_data_emits_stage_logs(self):
        fake_torch_utils_data = types.ModuleType("torch.utils.data")

        class FakeLoader:
            def __init__(self, *args, **kwargs):
                pass

        class FakeSubset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

        fake_torch_utils_data.ConcatDataset = lambda datasets: datasets
        fake_torch_utils_data.DataLoader = FakeLoader
        fake_torch_utils_data.Subset = FakeSubset
        fake_torch_utils_distributed = types.ModuleType("torch.utils.data.distributed")
        fake_torch_utils_distributed.DistributedSampler = lambda *args, **kwargs: None

        fake_dataset_module = types.ModuleType("cls_engine.data.dataset")
        fake_dataset_module.DatasetLayout = object
        fake_dataset_module.ImageFolderWithPath = object
        fake_dataset_module.compute_class_counts = lambda dataset: [len(getattr(dataset, "samples", []))]
        fake_dataset_module.parse_data_roots = lambda root: [Path(root)]
        fake_dataset_module.remap_dataset_to_class_order = lambda dataset, class_names: None

        class FakeDataset:
            def __init__(self, name):
                self.name = name
                self.samples = [("a.jpg", 0), ("b.jpg", 0)]

            def __len__(self):
                return len(self.samples)

        layout = types.SimpleNamespace(
            class_names=["car"],
            train_dirs=[Path("/data/train")],
            val_dirs=[],
            test_dirs=[],
            has_explicit_val=False,
            has_explicit_test=False,
            mode="train_only_split",
        )
        fake_dataset_module.discover_dataset_layout = lambda roots: layout

        fake_splits_module = types.ModuleType("cls_engine.data.splits")
        fake_splits_module.stratified_split_indices = lambda labels, val_ratio, test_ratio, seed: ([0], [1], [])

        fake_transforms_module = types.ModuleType("cls_engine.data.transforms")
        fake_transforms_module.build_train_transform = lambda *args, **kwargs: "train_tf"
        fake_transforms_module.build_eval_transform = lambda *args, **kwargs: "eval_tf"

        fake_dist_module = types.ModuleType("cls_engine.distributed.ddp")
        fake_dist_module.is_dist_avail_and_initialized = lambda: False

        fake_config_module = types.ModuleType("cls_engine.config.schema")
        fake_config_module.DataConfig = object

        module = _load_module(
            "cls_engine.data.datamodule_test",
            SRC / "cls_engine" / "data" / "datamodule.py",
            {
                "torch.utils.data": fake_torch_utils_data,
                "torch.utils.data.distributed": fake_torch_utils_distributed,
                "cls_engine.data.dataset": fake_dataset_module,
                "cls_engine.data.splits": fake_splits_module,
                "cls_engine.data.transforms": fake_transforms_module,
                "cls_engine.distributed.ddp": fake_dist_module,
                "cls_engine.config.schema": fake_config_module,
            },
        )

        module._build_split_dataset = lambda split_dirs, transform, class_names: (FakeDataset("ds"), [0, 0], [2], {"/data": 2})
        cfg = types.SimpleNamespace(img_size=(224, 224), augment_backend="cpu", preprocess="letterbox", root="/data", val_ratio=0.5, test_ratio=0.0, num_workers=0)

        logs = []
        module.prepare_data(cfg, seed=42, batch_size=2, progress_logger=logs.append)

        self.assertEqual(
            logs,
            [
                "[Data] discovering dataset layout...",
                "[Data] building train dataset...",
                "[Data] building val dataset...",
                "[Data] creating dataloaders...",
            ],
        )

    def test_train_one_epoch_can_announce_first_batch_wait(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.tensor = lambda *args, **kwargs: types.SimpleNamespace(item=lambda: 0.0)
        fake_torch.float64 = "float64"
        fake_torch.clamp = lambda x, min=1: x
        fake_torch.no_grad = lambda: (lambda fn: fn)
        fake_torch.nn = types.ModuleType("torch.nn")

        fake_amp = types.SimpleNamespace(autocast=lambda **kwargs: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: False))
        fake_torch.amp = fake_amp
        fake_dist = types.ModuleType("torch.distributed")
        fake_f = types.ModuleType("torch.nn.functional")
        fake_f.cross_entropy = lambda *args, **kwargs: 0.0

        fake_ddp = types.ModuleType("cls_engine.distributed.ddp")
        fake_ddp.is_dist_avail_and_initialized = lambda: False
        fake_ddp.is_main_process = lambda: True
        fake_ddp.reduce_sum = lambda x: x
        fake_ddp.safe_all_gather_object = lambda x: [x]

        fake_metrics = types.ModuleType("cls_engine.metrics.classification")
        fake_metrics.compute_topk_correct = lambda logits, y, k=5: 0

        module = _load_module(
            "cls_engine.engine.loops_test",
            SRC / "cls_engine" / "engine" / "loops.py",
            {
                "torch": fake_torch,
                "torch.nn": fake_torch.nn,
                "torch.distributed": fake_dist,
                "torch.nn.functional": fake_f,
                "cls_engine.distributed.ddp": fake_ddp,
                "cls_engine.metrics.classification": fake_metrics,
            },
        )

        output = io.StringIO()
        with redirect_stdout(output):
            try:
                module.train_one_epoch(
                    model=types.SimpleNamespace(train=lambda: None),
                    loader=[],
                    optimizer=types.SimpleNamespace(zero_grad=lambda **kwargs: None),
                    scaler=types.SimpleNamespace(scale=lambda loss: types.SimpleNamespace(backward=lambda: None), step=lambda optimizer: None, update=lambda: None),
                    device=types.SimpleNamespace(type="cpu"),
                    epoch=1,
                    criterion=lambda logits, y: 0.0,
                    announce_first_batch_wait=True,
                    announce_prefix="Train",
                )
            except Exception:
                pass

        self.assertIn("[Train] waiting for first batch...", output.getvalue())

    def test_print_startup_stage_uses_expected_prefix(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.amp = types.SimpleNamespace()
        fake_torch_nn = types.ModuleType("torch.nn")
        fake_module = _load_module(
            "cls_engine.engine.trainer_test",
            SRC / "cls_engine" / "engine" / "trainer.py",
            {
                "torch": fake_torch,
                "torch.nn": fake_torch_nn,
                "cls_engine.config.schema": types.SimpleNamespace(TrainConfig=object),
                "cls_engine.data.datamodule": types.SimpleNamespace(PreparedData=object, prepare_data=lambda *args, **kwargs: None),
                "cls_engine.data.dataset": types.SimpleNamespace(parse_data_roots=lambda root: [root]),
                "cls_engine.data.splits": types.SimpleNamespace(build_class_weights=lambda *args, **kwargs: None),
                "cls_engine.data.transforms": types.SimpleNamespace(build_gpu_eval_batch_augment=lambda *args, **kwargs: None, build_gpu_train_batch_augment=lambda *args, **kwargs: None),
                "cls_engine.distributed.ddp": types.SimpleNamespace(barrier=lambda: None, cleanup_distributed=lambda: None, get_rank=lambda: 0, is_dist_avail_and_initialized=lambda: False, is_main_process=lambda: True, setup_distributed=lambda backend: None),
                "cls_engine.distributed.device": types.SimpleNamespace(apply_device_spec=lambda spec: None, configure_torch_backend=lambda device: None, parse_device_spec=lambda device: None, resolve_device=lambda device: None, validate_multi_gpu_launch=lambda spec: None),
                "cls_engine.distributed.port": types.SimpleNamespace(PortConfig=object, resolve_master_port=lambda *args, **kwargs: (None, {})),
                "cls_engine.engine.evaluator": types.SimpleNamespace(write_eval_artifacts=lambda *args, **kwargs: None),
                "cls_engine.engine.loops": types.SimpleNamespace(evaluate_with_details=lambda *args, **kwargs: {}, train_one_epoch=lambda *args, **kwargs: (0, 0, 0, 0)),
                "cls_engine.io.artifacts": types.SimpleNamespace(ArtifactWriter=object),
                "cls_engine.models.checkpoint": types.SimpleNamespace(load_checkpoint=lambda *args, **kwargs: {}, save_best_checkpoint=lambda *args, **kwargs: None, save_last_checkpoint=lambda *args, **kwargs: None, unwrap_model=lambda model: model),
                "cls_engine.models.factory": types.SimpleNamespace(build_model=lambda *args, **kwargs: None),
                "cls_engine.utils.paths": types.SimpleNamespace(ensure_dir=lambda path: path, resolve_output_dir=lambda **kwargs: (Path("runs/x"), "exp")),
                "cls_engine.utils.seed": types.SimpleNamespace(set_seed=lambda seed: None),
            },
        )

        output = io.StringIO()
        with redirect_stdout(output):
            fake_module._print_startup_stage("preparing data...")

        self.assertEqual(output.getvalue().strip(), "[Startup] preparing data...")


if __name__ == "__main__":
    unittest.main()
