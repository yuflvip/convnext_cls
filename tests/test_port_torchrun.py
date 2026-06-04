import os
import unittest
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cls_engine.distributed.port import PortConfig, resolve_master_port


class TorchrunPortTests(unittest.TestCase):
    def test_torchrun_default_master_port_is_respected(self):
        env = {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "MASTER_PORT": "29500",
        }

        with mock.patch.dict(os.environ, env, clear=False):
            port, info = resolve_master_port(
                PortConfig(master_port="auto"),
                data_root="/data",
                model_name="model",
                output_dir="out",
                run_id="rank-specific-run-id",
            )
            current_master_port = os.environ["MASTER_PORT"]

        self.assertEqual(port, 29500)
        self.assertEqual(info["mode"], "torchrun_env")
        self.assertEqual(current_master_port, "29500")


if __name__ == "__main__":
    unittest.main()
