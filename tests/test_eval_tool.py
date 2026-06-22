import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_eval_test_stubs() -> None:
    dataset_module = types.ModuleType("cls_engine.data.dataset")
    transforms_module = types.ModuleType("cls_engine.data.transforms")

    class FakeImageFolderWithPath:
        def __init__(self, root: str, transform=None):
            self.root = str(root)
            self.transform = transform
            root_path = Path(root)
            self.classes = sorted(path.name for path in root_path.iterdir() if path.is_dir())
            self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
            self.samples = []
            for class_name in self.classes:
                class_dir = root_path / class_name
                for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                    self.samples.append((str(image_path), self.class_to_idx[class_name]))
            self.targets = [target for _, target in self.samples]

        def __len__(self):
            return len(self.samples)

    def fake_remap_dataset_to_class_order(ds, desired_order):
        existing = set(ds.classes)
        desired = list(desired_order)
        desired_set = set(desired)
        missing = sorted(desired_set - existing)
        extra = sorted(existing - desired_set)
        if missing or extra:
            raise ValueError(
                f"Class set mismatch. missing(in dataset): {missing}; extra(in dataset): {extra}"
            )

        old_classes = list(ds.classes)
        new_class_to_idx = {name: i for i, name in enumerate(desired)}
        new_samples = []
        for path, old_y in ds.samples:
            class_name = old_classes[old_y]
            new_samples.append((path, new_class_to_idx[class_name]))

        ds.classes = desired
        ds.class_to_idx = new_class_to_idx
        ds.samples = new_samples
        ds.targets = [target for _, target in new_samples]

    def fake_build_eval_transform(*args, **kwargs):
        return ("fake-transform", args, kwargs)

    dataset_module.ImageFolderWithPath = FakeImageFolderWithPath
    dataset_module.remap_dataset_to_class_order = fake_remap_dataset_to_class_order
    transforms_module.build_eval_transform = fake_build_eval_transform
    sys.modules["cls_engine.data.dataset"] = dataset_module
    sys.modules["cls_engine.data.transforms"] = transforms_module


_install_eval_test_stubs()


def _install_eval_runtime_stubs() -> None:
    torch_module = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    def fake_device(name: str):
        return name

    torch_module.cuda = FakeCuda()
    torch_module.device = fake_device
    sys.modules["torch"] = torch_module

    data_module = types.ModuleType("torch.utils.data")

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def __len__(self):
            return len(self.dataset)

    data_module.DataLoader = FakeDataLoader
    sys.modules["torch.utils.data"] = data_module

    evaluator_module = types.ModuleType("cls_engine.engine.evaluator")

    def fake_write_eval_artifacts(writer, prefix, eval_out, class_names, print_top_wrong):
        writer.write_eval_outputs(prefix, class_names, eval_out)

    evaluator_module.write_eval_artifacts = fake_write_eval_artifacts
    sys.modules["cls_engine.engine.evaluator"] = evaluator_module

    loops_module = types.ModuleType("cls_engine.engine.loops")
    loops_module.last_call = None

    def fake_evaluate_with_details(model, loader, device, class_names, batch_transform=None, log_interval=0, log_prefix="Eval"):
        loops_module.last_call = {
            "log_interval": log_interval,
            "log_prefix": log_prefix,
            "batch_transform": batch_transform,
        }
        return {
            "val_loss": 0.25,
            "val_acc_top1": 0.75,
            "val_acc_top5": 1.0,
            "confusion_matrix": [[1, 0], [0, 1]],
            "all_rows": [["/tmp/a.jpg", 0, class_names[0], 0, class_names[0], "0.900000"]],
            "wrong_rows": [],
        }

    loops_module.evaluate_with_details = fake_evaluate_with_details
    sys.modules["cls_engine.engine.loops"] = loops_module

    artifacts_module = types.ModuleType("cls_engine.io.artifacts")

    class FakeArtifactWriter:
        def __init__(self, out_dir):
            self.out_dir = Path(out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)

        def write_eval_outputs(self, prefix, class_names, eval_out):
            (self.out_dir / f"classification_report_{prefix}.txt").write_text(
                f"classes={','.join(class_names)} acc={eval_out['val_acc_top1']}",
                encoding="utf-8",
            )

        def write_final_summary(self, payload):
            (self.out_dir / "final_summary.json").write_text(str(payload), encoding="utf-8")

    artifacts_module.ArtifactWriter = FakeArtifactWriter
    sys.modules["cls_engine.io.artifacts"] = artifacts_module

    factory_module = types.ModuleType("cls_engine.models.factory")

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict, strict=True):
            self.state_dict = state_dict

        def eval(self):
            return self

    def fake_build_model(model_name, num_classes, pretrained=False):
        return FakeModel()

    factory_module.build_model = fake_build_model
    sys.modules["cls_engine.models.factory"] = factory_module

    predictor_module = types.ModuleType("cls_engine.predict.predictor")

    def fake_parse_predict_imgsz(value: str):
        return (224, 224) if "," not in value else tuple(int(part) for part in value.split(","))

    predictor_module.parse_predict_imgsz = fake_parse_predict_imgsz
    sys.modules["cls_engine.predict.predictor"] = predictor_module

    predict_pth_module = types.ModuleType("cls_engine.predict.pth")

    def fake_load_predict_checkpoint_info(model_path):
        return {
            "model_name": "demo_model",
            "classes": ["dog", "cat"],
            "num_classes": 2,
            "state_dict": {"weight": 1},
        }

    predict_pth_module.load_predict_checkpoint_info = fake_load_predict_checkpoint_info
    sys.modules["cls_engine.predict.pth"] = predict_pth_module

from cls_engine.eval.pth import build_eval_dataset, evaluate_checkpoint_directory
from cls_engine.eval.pth import resolve_eval_output_dir
from tools.eval import build_arg_parser


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-image")


class EvalToolTests(unittest.TestCase):
    def test_build_eval_dataset_remaps_targets_to_checkpoint_class_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_image(base / "cat" / "1.jpg")
            _write_image(base / "dog" / "2.jpg")

            dataset = build_eval_dataset(
                data_root=base,
                class_names=["dog", "cat"],
                input_size=(32, 32),
                preprocess="letterbox",
            )

        self.assertEqual(dataset.classes, ["dog", "cat"])
        self.assertEqual(dataset.class_to_idx, {"dog": 0, "cat": 1})
        self.assertEqual(sorted(dataset.targets), [0, 1])

    def test_build_eval_dataset_rejects_class_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_image(base / "cat" / "1.jpg")

            with self.assertRaisesRegex(ValueError, "Class set mismatch"):
                build_eval_dataset(
                    data_root=base,
                    class_names=["dog", "cat"],
                    input_size=(32, 32),
                    preprocess="letterbox",
                )

    def test_build_eval_dataset_allows_empty_class_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_image(base / "dog" / "2.jpg")
            (base / "cat").mkdir(parents=True, exist_ok=True)

            dataset = build_eval_dataset(
                data_root=base,
                class_names=["dog", "cat"],
                input_size=(32, 32),
                preprocess="letterbox",
            )

        self.assertEqual(dataset.classes, ["dog", "cat"])
        self.assertEqual(dataset.class_to_idx, {"dog": 0, "cat": 1})
        self.assertEqual(dataset.targets, [0])

    def test_eval_cli_accepts_model_and_data_arguments(self):
        parser = build_arg_parser()

        args = parser.parse_args(["--model", "runs/demo/best.pth", "--data", "/tmp/val"])

        self.assertEqual(args.model, "runs/demo/best.pth")
        self.assertEqual(args.data, "/tmp/val")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.imgsz, "224")
        self.assertEqual(args.preprocess, "stretch")
        self.assertEqual(args.log_interval, 1)

    def test_eval_cli_accepts_custom_log_interval(self):
        parser = build_arg_parser()

        args = parser.parse_args(["--model", "runs/demo/best.pth", "--data", "/tmp/val", "--log_interval", "5"])

        self.assertEqual(args.log_interval, 5)

    def test_resolve_eval_output_dir_defaults_under_runs_eval(self):
        resolved = resolve_eval_output_dir("/tmp/demo/best.pth", None, now=datetime(2026, 6, 22, 9, 8, 7))

        self.assertEqual(resolved, Path("runs") / "eval" / "eval_20260622090807")

    def test_evaluate_checkpoint_directory_writes_summary_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_image(base / "cat" / "1.jpg")
            _write_image(base / "dog" / "2.jpg")
            output_dir = base / "eval_outputs"
            _install_eval_runtime_stubs()

            with mock.patch("builtins.print") as mock_print:
                result = evaluate_checkpoint_directory(
                    model_path=base / "best.pth",
                    data_root=base,
                    output=output_dir,
                )

            self.assertEqual(result, output_dir)
            self.assertTrue((output_dir / "classification_report_eval.txt").exists())
            self.assertTrue((output_dir / "final_summary.json").exists())
            self.assertEqual(sys.modules["cls_engine.engine.loops"].last_call["log_interval"], 1)
            self.assertEqual(sys.modules["cls_engine.engine.loops"].last_call["log_prefix"], "Eval")
            printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
            self.assertTrue(any("[Eval] done" in line and "acc_top1=0.7500" in line for line in printed_lines))
            self.assertTrue(any("[Eval][Class] dog acc=1.0000 support=1" in line for line in printed_lines))
            self.assertTrue(any("[Eval][Class] cat acc=1.0000 support=1" in line for line in printed_lines))


if __name__ == "__main__":
    unittest.main()
