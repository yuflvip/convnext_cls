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

from cls_engine.predict.predictor import arrange_prediction_outputs, resolve_prediction_output_dir


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


class PredictArrangeTests(unittest.TestCase):
    def test_resolve_prediction_output_dir_defaults_under_runs_predict(self):
        resolved = resolve_prediction_output_dir("/tmp/demo/best.pth", None, now=datetime(2026, 6, 21, 15, 30, 10))

        self.assertEqual(resolved, Path("runs") / "predict" / "predict_20260621153010")

    def test_predict_cli_accepts_arrange_mode(self):
        parser = _load_predict_parser()()

        args = parser.parse_args(["--model", "demo.pth", "--input", "/tmp/images", "--arrange_mode", "copy"])

        self.assertEqual(args.arrange_mode, "copy")

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


if __name__ == "__main__":
    unittest.main()
