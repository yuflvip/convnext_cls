import importlib.util
import json
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

from cls_engine.predict.predictor import (
    arrange_prediction_outputs,
    cleanup_temporary_inputs,
    prepare_prediction_inputs,
    resolve_prediction_output_dir,
)


def _load_predict_parser():
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_tools_predict", ROOT / "tools" / "predict.py")
    module = importlib.util.module_from_spec(spec)
    stub_pth = types.ModuleType("cls_engine.predict.pth")
    stub_pth.predict_with_checkpoint = lambda **kwargs: Path("runs/predict/demo")
    stub_onnx = types.ModuleType("cls_engine.predict.onnx")
    stub_onnx.predict_with_onnx = lambda **kwargs: Path("runs/predict/demo")
    with mock.patch.dict(
        sys.modules,
        {
            "cls_engine.predict.pth": stub_pth,
            "cls_engine.predict.onnx": stub_onnx,
        },
        clear=False,
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module.build_arg_parser


def _load_predict_runtime_modules():
    stub_torch = types.ModuleType("torch")

    class FakeNoGrad:
        def __call__(self, fn=None):
            if fn is None:
                return self
            return fn

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    stub_torch.no_grad = FakeNoGrad
    stub_torch.device = lambda value: value
    stub_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    class FakeProbVector:
        def __getitem__(self, index):
            return self

        def max(self, dim=0):
            return types.SimpleNamespace(item=lambda: 0.95), types.SimpleNamespace(item=lambda: 1)

        def topk(self, k):
            return (
                types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(tolist=lambda: [0.95])),
                types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(tolist=lambda: [1])),
            )

    stub_torch.softmax = lambda logits, dim=1: FakeProbVector()

    class FakeTensor:
        def unsqueeze(self, dim):
            return self

        def to(self, device):
            return self

    class FakeImage:
        def convert(self, mode):
            return self

    stub_pil_image = types.SimpleNamespace(open=lambda path: FakeImage())
    stub_pil = types.ModuleType("PIL")
    stub_pil.Image = stub_pil_image

    stub_transforms = types.ModuleType("cls_engine.data.transforms")
    stub_transforms.build_eval_transform = lambda *args, **kwargs: (lambda image: FakeTensor())

    stub_checkpoint = types.ModuleType("cls_engine.models.checkpoint")
    stub_checkpoint.load_checkpoint = lambda *args, **kwargs: {
        "classes": ["cat", "dog"],
        "model_name": "demo_model",
        "state_dict": {"weight": 1},
    }

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict, strict=True):
            self.state_dict = state_dict

        def eval(self):
            return self

        def __call__(self, x):
            return "fake_logits"

    stub_factory = types.ModuleType("cls_engine.models.factory")
    stub_factory.build_model = lambda *args, **kwargs: FakeModel()

    predictor_spec = importlib.util.spec_from_file_location("test_predictor_runtime", SRC / "cls_engine" / "predict" / "predictor.py")
    predictor_module = importlib.util.module_from_spec(predictor_spec)
    sys.modules["test_predictor_runtime"] = predictor_module
    assert predictor_spec.loader is not None
    predictor_spec.loader.exec_module(predictor_module)

    with mock.patch.dict(
        sys.modules,
        {
            "torch": stub_torch,
            "PIL": stub_pil,
            "PIL.Image": stub_pil_image,
            "cls_engine.data.transforms": stub_transforms,
            "cls_engine.models.checkpoint": stub_checkpoint,
            "cls_engine.models.factory": stub_factory,
            "test_predict_runtime.predictor": predictor_module,
        },
        clear=False,
    ):
        pth_spec = importlib.util.spec_from_file_location("test_predict_runtime.pth", SRC / "cls_engine" / "predict" / "pth.py")
        pth_module = importlib.util.module_from_spec(pth_spec)
        sys.modules["test_predict_runtime.pth"] = pth_module
        assert pth_spec.loader is not None
        pth_spec.loader.exec_module(pth_module)

    return predictor_module, pth_module


class PredictArrangeTests(unittest.TestCase):
    def test_resolve_prediction_output_dir_defaults_under_runs_predict(self):
        resolved = resolve_prediction_output_dir("/tmp/demo/best.pth", None, now=datetime(2026, 6, 21, 15, 30, 10))

        self.assertEqual(resolved, Path("runs") / "predict" / "predict_20260621153010")

    def test_predict_cli_accepts_arrange_mode_with_data(self):
        parser = _load_predict_parser()()

        args = parser.parse_args(["--model", "demo.pth", "--data", "/tmp/images", "--arrange_mode", "copy"])

        self.assertEqual(args.arrange_mode, "copy")
        self.assertEqual(args.data, "/tmp/images")
        self.assertEqual(args.temp_dir, "/tmp/predict_cls_url/")

    def test_predict_cli_rejects_removed_input_argument(self):
        parser = _load_predict_parser()()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--model", "demo.pth", "--input", "/tmp/images"])

    def test_arrange_prediction_outputs_copy_copies_into_class_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_dir = base / "input"
            output_dir = base / "output"
            source = input_dir / "a.jpg"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"image-a")
            rows = [{"path": str(source), "pred_name": "cat", "pred_idx": 0, "conf": 0.9, "topk": [["cat", 0.9]]}]

            arrange_prediction_outputs(output_dir, rows, arrange_mode="copy")

            arranged = output_dir / "cat" / "a.jpg"
            self.assertTrue(source.exists())
            self.assertTrue(arranged.exists())
            self.assertEqual(arranged.read_bytes(), b"image-a")

    def test_arrange_prediction_outputs_move_moves_into_class_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_dir = base / "input"
            output_dir = base / "output"
            source = input_dir / "b.jpg"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"image-b")
            rows = [{"path": str(source), "pred_name": "dog", "pred_idx": 1, "conf": 0.8, "topk": [["dog", 0.8]]}]

            arrange_prediction_outputs(output_dir, rows, arrange_mode="move")

            arranged = output_dir / "dog" / "b.jpg"
            self.assertFalse(source.exists())
            self.assertTrue(arranged.exists())
            self.assertEqual(arranged.read_bytes(), b"image-b")

    def test_arrange_prediction_outputs_overwrites_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_dir = base / "input"
            output_dir = base / "output"
            source = input_dir / "c.jpg"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"new-content")
            target = output_dir / "truck" / "c.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"old-content")
            rows = [{"path": str(source), "pred_name": "truck", "pred_idx": 2, "conf": 0.7, "topk": [["truck", 0.7]]}]

            arrange_prediction_outputs(output_dir, rows, arrange_mode="copy")

            self.assertEqual(target.read_bytes(), b"new-content")

    def test_predict_with_checkpoint_prints_path_class_id_and_conf(self):
        predictor_module, pth_module = _load_predict_runtime_modules()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "demo.jpg"
            image_path.write_bytes(b"img")

            with mock.patch.object(predictor_module, "collect_input_images", return_value=[image_path]):
                with mock.patch("builtins.print") as mock_print:
                    pth_module.predict_with_checkpoint(model_path=base / "best.pth", input_path=base)

        printed_lines = [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]
        self.assertTrue(
            any(
                f"[Predict] 1/1 {image_path} -> class=dog id=1 conf=0.9500" in line
                for line in printed_lines
            )
        )

    def test_prepare_prediction_inputs_downloads_single_url_to_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp) / "predict_cls_url"

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self):
                    return b"remote-image"

            with mock.patch("urllib.request.urlopen", return_value=FakeResponse()):
                with mock.patch("cls_engine.predict.predictor.random.randint", return_value=123456):
                    with mock.patch("cls_engine.predict.predictor.datetime") as mock_datetime:
                        mock_datetime.now.return_value = datetime(2026, 6, 24, 13, 30, 10, 123000)
                        input_paths, temp_paths = prepare_prediction_inputs(
                            "https://example.com/a.jpg",
                            temp_dir=temp_dir,
                        )

            self.assertEqual(len(input_paths), 1)
            self.assertEqual(temp_paths, input_paths)
            self.assertTrue(input_paths[0].exists())
            self.assertEqual(input_paths[0].parent, temp_dir)
            self.assertEqual(input_paths[0].name, "20260624133010123_123456.jpg")

    def test_cleanup_temporary_inputs_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_file = Path(tmp) / "tmp.jpg"
            temp_file.write_bytes(b"temp")

            cleanup_temporary_inputs([temp_file], temp_dir=Path(tmp))

            self.assertFalse(temp_file.exists())


if __name__ == "__main__":
    unittest.main()
