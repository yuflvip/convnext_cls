import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "preview_preprocess_letterbox.py"


class PreviewPreprocessLetterboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="preview-letterbox-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--input", result.stdout)
        self.assertIn("--pad_value", result.stdout)

    def test_process_single_image(self):
        image_path = self.tmp / "sample.jpg"
        output_dir = self.tmp / "out"
        Image.new("RGB", (120, 60), color=(10, 20, 30)).save(image_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(image_path),
                "--output",
                str(output_dir),
                "--imgsz",
                "64",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        saved_path = output_dir / "sample.png"
        self.assertTrue(saved_path.is_file())
        with Image.open(saved_path) as saved:
            self.assertEqual(saved.size, (64, 64))

    def test_same_size_image_keeps_geometry(self):
        image_path = self.tmp / "sample.jpg"
        output_dir = self.tmp / "out"
        Image.new("RGB", (64, 64), color=(100, 110, 120)).save(image_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(image_path),
                "--output",
                str(output_dir),
                "--imgsz",
                "64",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        with Image.open(output_dir / "sample.png") as saved:
            self.assertEqual(saved.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
