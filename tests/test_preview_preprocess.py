import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "preview_preprocess.py"


class PreviewPreprocessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="preview-preprocess-"))

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
        self.assertIn("--augment_backend", result.stdout)

    def test_process_single_image_cpu_eval(self):
        image_path = self.tmp / "sample.jpg"
        output_dir = self.tmp / "out"
        Image.new("RGB", (80, 60), color=(120, 160, 200)).save(image_path)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input",
                str(image_path),
                "--output",
                str(output_dir),
                "--split",
                "eval",
                "--augment_backend",
                "cpu",
                "--imgsz",
                "64",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue((output_dir / "sample.png").is_file())


if __name__ == "__main__":
    unittest.main()
