import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.distributed.device import parse_device_spec, validate_multi_gpu_launch


class MultiGpuLaunchTests(unittest.TestCase):
    def test_multi_gpu_device_requires_torchrun(self):
        spec = parse_device_spec("1,2")

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Use torchrun"):
                validate_multi_gpu_launch(spec)

    def test_torchrun_local_world_size_must_match_device_count(self):
        spec = parse_device_spec("1,2")
        env = {"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}

        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "--nproc_per_node=2"):
                validate_multi_gpu_launch(spec)

    def test_torchrun_matching_local_world_size_is_allowed(self):
        spec = parse_device_spec("1,2")
        env = {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"}

        with mock.patch.dict(os.environ, env, clear=True):
            validate_multi_gpu_launch(spec)


if __name__ == "__main__":
    unittest.main()
