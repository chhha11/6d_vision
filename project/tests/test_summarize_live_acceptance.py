import json
import tempfile
import unittest
from pathlib import Path

from core.summarize_live_acceptance import summarize_run


class SummarizeLiveAcceptanceTest(unittest.TestCase):
    def test_aggregates_attempts_events_and_frame_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            attempt = run / "attempt_001" / "tracking"
            attempt.mkdir(parents=True)
            (attempt.parent / "initialization_timing.json").write_text(
                json.dumps({"status": "ready", "source": "rgb_reference_pnp"}),
                encoding="utf-8")
            (attempt / "frames.csv").write_text(
                "elapsed_s\n0.01\n0.07\n0.24\n", encoding="utf-8")
            (attempt / "summary.json").write_text(json.dumps({
                "frames": 3,
                "tracking_frames": 2,
                "duration_s": 0.24,
                "average_fps": 12.5,
                "status_counts": {"TRACKING": 2, "TARGET LOST": 1},
                "display": {"status_counts": {"TRACKING": 2, "SEARCHING": 1}},
                "reacquire_requested": True,
                "final_pose": {"tracking_valid": False},
                "max_inliers": 24,
            }), encoding="utf-8")
            (run / "session_events.jsonl").write_text(
                '{"stage":"depth_capture_failed"}\n{"stage":"depth_capture_failed"}\n',
                encoding="utf-8")

            result = summarize_run(run)

        self.assertEqual(result["initialization"]["ready"], 1)
        self.assertEqual(result["tracking"]["frames"], 3)
        self.assertEqual(result["tracking"]["tracking_frames"], 2)
        self.assertEqual(result["tracking"]["max_frame_interval_ms"], 170.0)
        self.assertEqual(result["tracking"]["hard_loss_counts"], {"TARGET LOST": 1})
        self.assertEqual(result["events"]["by_stage"], {"depth_capture_failed": 2})
        self.assertTrue(result["review_flags"])

    def test_missing_tracking_is_reported_without_failing(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            attempt = run / "attempt_001"
            attempt.mkdir()
            (attempt / "initialization_timing.json").write_text(
                '{"status":"rejected","error":"no target"}', encoding="utf-8")
            result = summarize_run(run)

        self.assertEqual(result["initialization"]["rejected"], 1)
        self.assertEqual(result["tracking"]["attempts"], 0)
        self.assertIn("没有生成 tracking/summary.json；可能尚未识别到目标或只进行了搜索。",
                      result["review_flags"])


if __name__ == "__main__":
    unittest.main()
