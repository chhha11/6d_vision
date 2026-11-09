"""Exercise the isolated YOLO pipe without importing Torch into the test runner."""
import sys
import unittest

import numpy as np

from core.detector_process import DetectorProcess


FIXTURE = '''import json,struct,sys,time
def send(value):
 data=json.dumps(value).encode()
 sys.stdout.buffer.write(struct.pack('<I',len(data))+data)
 sys.stdout.buffer.flush()
send(dict(ready=True,timings=dict(detector_import_s=.1)))
'''


class DetectorProcessTest(unittest.TestCase):
    def worker(self, source, timeout_s=1.):
        timings = {}
        worker = DetectorProcess('unused', timings, timeout_s,
            command=[sys.executable, '-u', '-c', FIXTURE + source])
        self.addCleanup(worker.close)
        self.assertEqual(timings['detector_import_s'], .1)
        return worker

    def test_frame_and_rotation_roundtrip_and_owned_process_cleanup(self):
        worker = self.worker('''h,w,turn=struct.unpack('<III',sys.stdin.buffer.read(12))
frame=sys.stdin.buffer.read(h*w*3)
assert (h,w,turn)==(12,16,3) and frame==bytes([23])*(h*w*3)
send(dict(box=[1.,2.,13.,11.]))
time.sleep(5)
''')
        process = worker.process
        frame = np.full((12,16,3),23,np.uint8)
        np.testing.assert_array_equal(worker.get_box(frame,3), [1,2,13,11])
        worker.close()
        self.assertIsNotNone(process.poll())
        worker.close()

    def test_timeout_kills_worker_and_does_not_wait_for_a_detector_forever(self):
        worker = self.worker('time.sleep(5)')
        process = worker.process
        worker.timeout_s = .05
        with self.assertRaises(TimeoutError):
            worker.get_box(np.zeros((12,16,3),np.uint8))
        self.assertIsNotNone(process.poll())

    def test_crashed_worker_is_reported_as_optional_assistance_failure(self):
        worker = self.worker("sys.stdin.buffer.read(12)\nsys.exit(1)")
        with self.assertRaisesRegex(RuntimeError, 'stopped'):
            worker.get_box(np.zeros((12,16,3),np.uint8))
        self.assertIsNone(worker.process)

    def test_invalid_box_is_rejected(self):
        worker = self.worker("sys.stdin.buffer.read(12)\nsend(dict(box=[1,2,3]))")
        with self.assertRaisesRegex(RuntimeError, 'Invalid YOLO worker box'):
            worker.get_box(np.zeros((12,16,3),np.uint8))

    def test_multiple_boxes_roundtrip(self):
        worker = self.worker("""h,w,turn=struct.unpack('<III',sys.stdin.buffer.read(12))
frame=sys.stdin.buffer.read(h*w*3)
assert turn==5
send(dict(boxes=[[1,2,6,10],[8,1,15,11]]))
""")
        np.testing.assert_array_equal(
            worker.get_boxes(np.zeros((12,16,3),np.uint8), 1),
            [[1,2,6,10],[8,1,15,11]])

    def test_progressive_multi_mode_roundtrip(self):
        worker = self.worker("""h,w,turn=struct.unpack('<III',sys.stdin.buffer.read(12))
frame=sys.stdin.buffer.read(h*w*3)
assert turn==11
send(dict(boxes=[[1,2,6,10]]))
""")
        np.testing.assert_array_equal(
            worker.get_boxes(np.zeros((12,16,3),np.uint8), 3, progressive=True),
            [[1,2,6,10]])

    def test_empty_multi_result(self):
        worker = self.worker("sys.stdin.buffer.read(12)\nsend(dict(boxes=[]))")
        self.assertEqual(worker.get_boxes(np.zeros((12,16,3),np.uint8)), [])

    def test_invalid_multi_box_is_rejected(self):
        for boxes in ([1,2,3,4], [[1,2,3]], [[8,1,2,10]], [[1,2,30,10]], None):
            with self.subTest(boxes=boxes):
                worker = self.worker(f"sys.stdin.buffer.read(12)\nsend(dict(boxes={boxes!r}))")
                with self.assertRaisesRegex(RuntimeError, 'Invalid YOLO worker boxes'):
                    worker.get_boxes(np.zeros((12,16,3),np.uint8))
                self.assertIsNone(worker.process)

    def test_invalid_input_is_rejected_before_sending(self):
        worker = self.worker('time.sleep(5)')
        with self.assertRaises(ValueError):
            worker.get_box(np.zeros((12,16,3),np.float32))
