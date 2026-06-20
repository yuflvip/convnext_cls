import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train import clear_local_pycache


class TrainStartupTests(unittest.TestCase):
    def test_clear_local_pycache_removes_pyc_files_under_pycache_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pycache_dir = base / "pkg" / "__pycache__"
            pycache_dir.mkdir(parents=True, exist_ok=True)
            pyc_path = pycache_dir / "module.cpython-312.pyc"
            pyc_path.write_bytes(b"broken")

            removed = clear_local_pycache(base)

        self.assertEqual(removed, 1)
        self.assertFalse(pyc_path.exists())


if __name__ == "__main__":
    unittest.main()
