import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "check_nightly_board", Path(__file__).with_name("check_nightly_board.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeCuda:
    def __init__(self, names):
        self.names = names

    def device_count(self):
        return len(self.names)

    def get_device_name(self, index):
        return self.names[index]


class NightlyBoardTest(unittest.TestCase):
    def run_board(self, board, devices, names):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "device_fingerprint.json"
            with patch.dict(
                "sys.modules", {"torch": type("Torch", (), {"cuda": FakeCuda(names)})}
            ):
                MODULE.check_board(board, devices, output)
            return json.loads(output.read_text())

    def test_accepts_both_board_families(self):
        self.assertEqual(
            self.run_board("zw-810e", 1, ["PPU-ZW810E"])["board"], "zw-810e"
        )
        self.assertEqual(
            self.run_board("zw-m890p", 2, ["ZW-M890P", "ZW-M890P"])["device_count"],
            2,
        )

    def test_rejects_wrong_board_and_records_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "Expected board zw-m890p"):
            self.run_board("zw-m890p", 1, ["PPU-ZW810E"])

    def test_rejects_wrong_device_count(self):
        with self.assertRaisesRegex(RuntimeError, "Expected 2 assigned devices"):
            self.run_board("zw-810e", 2, ["PPU-ZW810E"])


if __name__ == "__main__":
    unittest.main()
