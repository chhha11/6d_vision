import tempfile
import unittest
from pathlib import Path

from core import orbbec_capture


class OrbbecCaptureTest(unittest.TestCase):
    def test_read_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "summary.txt").write_text(
                "paired_frames=12\ncallback_errors=0\n", encoding="utf-8")
            self.assertEqual(orbbec_capture.read_summary(output), {
                "paired_frames": "12",
                "callback_errors": "0",
            })

    def test_missing_binary_has_build_hint(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "cmake --build"):
                orbbec_capture.run_capture(Path(folder) / "missing", Path(folder), 1, 1)


if __name__ == "__main__":
    unittest.main()

