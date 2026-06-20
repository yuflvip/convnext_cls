import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.config.schema import DataConfig
from cls_engine.data.datamodule import prepare_data
from cls_engine.data.dataset import discover_dataset_layout, parse_data_roots


def _touch_dataset_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-image")


class MultiDatasetRootTests(unittest.TestCase):
    def test_parse_data_roots_supports_single_and_comma_separated_values(self):
        self.assertEqual(parse_data_roots("/data/a"), [Path("/data/a")])
        self.assertEqual(parse_data_roots("/data/a,"), [Path("/data/a")])
        self.assertEqual(
            parse_data_roots(" /data/a , /data/b ,, /data/c "),
            [Path("/data/a"), Path("/data/b"), Path("/data/c")],
        )

    def test_discover_dataset_layout_accepts_multiple_roots_with_same_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for dataset_name in ("side", "front"):
                _touch_dataset_file(base / dataset_name / "train" / "cat" / "1.jpg")
                _touch_dataset_file(base / dataset_name / "train" / "dog" / "1.jpg")

            layout = discover_dataset_layout(parse_data_roots(f"{base / 'side'},{base / 'front'}"))

        self.assertEqual(layout.class_names, ["cat", "dog"])
        self.assertEqual(len(layout.data_roots), 2)
        self.assertEqual(layout.mode, "train_only_split")
        self.assertFalse(layout.has_explicit_val)
        self.assertFalse(layout.has_explicit_test)

    def test_discover_dataset_layout_rejects_mismatched_class_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _touch_dataset_file(base / "side" / "train" / "cat" / "1.jpg")
            _touch_dataset_file(base / "side" / "train" / "dog" / "1.jpg")
            _touch_dataset_file(base / "front" / "train" / "cat" / "1.jpg")
            _touch_dataset_file(base / "front" / "train" / "bird" / "1.jpg")

            with self.assertRaisesRegex(ValueError, "Class set mismatch"):
                discover_dataset_layout(parse_data_roots(f"{base / 'side'},{base / 'front'}"))

    def test_prepare_data_concatenates_multiple_train_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _touch_dataset_file(base / "side" / "train" / "cat" / "1.jpg")
            _touch_dataset_file(base / "side" / "train" / "dog" / "1.jpg")
            _touch_dataset_file(base / "front" / "train" / "cat" / "2.jpg")
            _touch_dataset_file(base / "front" / "train" / "dog" / "2.jpg")

            cfg = DataConfig(
                root=f"{base / 'side'},{base / 'front'}",
                img_size=32,
                val_ratio=0.5,
                test_ratio=0.0,
                num_workers=0,
            )
            prepared = prepare_data(cfg, seed=42, batch_size=2)

        self.assertEqual(prepared.class_names, ["cat", "dog"])
        self.assertEqual(len(prepared.layout.data_roots), 2)
        self.assertEqual(prepared.train_root_counts, {str(base / "side"): 2, str(base / "front"): 2})
        self.assertEqual(len(prepared.train_indices) + len(prepared.val_indices), 4)
        self.assertEqual(int(prepared.train_counts.sum() + prepared.val_counts.sum()), 4)


if __name__ == "__main__":
    unittest.main()
