"""Check the native boundary and independent rejection of region proposals."""
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core.cad_video_tracker import project, make_xyz_record
from core.m3t_live import NativeRegionTracker, RegionAssist, RegionPose, validate_region_pose


class NativeBoundaryTest(unittest.TestCase):
    def stream(self, response):
        native = NativeRegionTracker.__new__(NativeRegionTracker)
        native.size, native.timeout_s, native.sequence = (16, 12), .5, 0
        native.log = tempfile.TemporaryFile()
        script = '''import sys,struct,time
h=sys.stdin.buffer.read(56)
command,sequence,*pose=struct.unpack('<II6d',h)
frame=sys.stdin.buffer.read(16*12*3)
''' + response
        native.process = subprocess.Popen([sys.executable, '-u', '-c', script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=native.log, bufsize=0)
        os.set_blocking(native.process.stdin.fileno(), False)
        os.set_blocking(native.process.stdout.fileno(), False)
        self.addCleanup(native.close)
        return native

    def test_reset_keeps_radians_millimeters_sequence_and_frame_bytes(self):
        native = self.stream('''assert command==1 and sequence==1
assert pose==[.1,.2,.3,10.,20.,500.] and frame==bytes([23])*(16*12*3)
sys.stdout.buffer.write(struct.pack('<II7d',sequence,1,*pose,2.5))
''')
        result = native.step(np.full((12,16,3),23,np.uint8),
                             (np.array([.1,.2,.3]), np.array([10.,20.,500.])))
        np.testing.assert_array_equal(result.tvec, [10.,20.,500.])
        np.testing.assert_array_equal(result.rvec, [.1,.2,.3])
        self.assertEqual(result.processing_ms, 2.5)

    def test_rejects_unsynchronized_native_reply(self):
        native = self.stream("sys.stdout.buffer.write(struct.pack('<II7d',sequence+1,1,0,0,0,0,0,500,2))")
        with self.assertRaisesRegex(RuntimeError, 'unsynchronized'):
            native.step(np.zeros((12,16,3),np.uint8))

    def test_hung_native_times_out_and_can_be_reaped(self):
        native = self.stream('time.sleep(5)')
        native.timeout_s = .05
        process = native.process
        with self.assertRaises(TimeoutError):
            native.step(np.zeros((12,16,3),np.uint8))
        native.close()
        self.assertIsNotNone(process.poll())
        native.close()

    def test_wrong_calibrated_frame_is_rejected_before_write(self):
        native = NativeRegionTracker.__new__(NativeRegionTracker)
        native.size = (16,12)
        for frame in [np.zeros((12,15,3),np.uint8), np.zeros((12,16,3),np.float32)]:
            with self.assertRaises(ValueError):
                native.step(frame)


class RegionQualityTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[530.,0,320.],[0,530.,240.],[0,0,1.]])
        self.dist = np.zeros(5)
        self.vertices = np.array([[x,y,z] for x in (-45.,45.) for y in (-45.,45.)
                                  for z in (-75.,75.)])
        self.points = np.array([[x,y,-75.] for x in (-40.,-20.,0.,20.,40.)
                                for y in (-40.,0.,40.)])
        self.pose = RegionPose(np.zeros(3),np.array([0.,0.,500.]),2.)
        self.pixels = project(self.points,self.pose.rvec,self.pose.tvec,self.K,self.dist)
        def mask(shape,vertices,rvec,tvec,**_kwargs):
            result = np.zeros(shape[:2],np.uint8)
            uv = project(vertices,rvec,tvec,self.K,self.dist)
            cv2.fillConvexPoly(result,cv2.convexHull(np.rint(uv).astype(np.int32)),255)
            return result
        self.seeder = SimpleNamespace(mask=mask)
        silhouette = mask((480,640,3),self.vertices,self.pose.rvec,self.pose.tvec)
        self.frame = cv2.cvtColor(silhouette,cv2.COLOR_GRAY2BGR)

    def validate(self, frame=None, pose=None, points=None, pixels=None, box=None):
        return validate_region_pose(self.frame if frame is None else frame,
            self.pose if pose is None else pose,(self.pose.rvec,self.pose.tvec),
            self.vertices,self.K,self.dist,self.seeder,
            self.points if points is None else points,
            self.pixels if pixels is None else pixels,box)

    def test_distributed_flow_and_matching_silhouette_accept_slow_motion(self):
        moved = RegionPose(np.zeros(3),np.array([2.,0.,500.]),2.)
        pixels = project(self.points,moved.rvec,moved.tvec,self.K,self.dist)
        frame = cv2.cvtColor(self.seeder.mask(self.frame.shape,self.vertices,
                            moved.rvec,moved.tvec),cv2.COLOR_GRAY2BGR)
        quality = self.validate(frame,moved,pixels=pixels)
        self.assertTrue(quality['accepted'],quality)
        self.assertEqual(quality['flow_support'],15)
        self.assertTrue(make_xyz_record(0,0,'REGION TRACKED',moved.rvec,moved.tvec)['valid'])

    def test_matching_edges_alone_cannot_certify_region_pose(self):
        quality = self.validate(points=np.empty((0,3)),pixels=np.empty((0,2)))
        self.assertFalse(quality['accepted'])
        self.assertEqual(quality['reason'],'no_independent_image_support')

    def test_low_pnp_residual_cannot_certify_blank_or_occluded_scene(self):
        quality = self.validate(frame=np.zeros_like(self.frame))
        self.assertFalse(quality['accepted'])
        self.assertEqual(quality['reason'],'silhouette_mismatch')

    def test_fresh_detector_conflict_cannot_be_erased_by_good_flow(self):
        # Deliberately short/offset detection conflicts despite perfect PnP.
        box=np.array([250.,180.,300.,240.])
        quality=self.validate(box=box)
        self.assertFalse(quality['accepted'])
        self.assertEqual(quality['reason'],'detector_conflict')
        from core.cad_video_tracker import current_detection_agrees
        self.assertFalse(current_detection_agrees(self.frame,box,self.vertices,
            self.pose.rvec,self.pose.tvec,self.K,self.dist))
        self.assertTrue(current_detection_agrees(self.frame,None,self.vertices,
            self.pose.rvec,self.pose.tvec,self.K,self.dist))

    def test_geometric_jump_and_bad_depth_rejected(self):
        for t in ([0,0,-500],[200,0,500],[0,0,float('nan')]):
            quality = self.validate(pose=RegionPose(np.zeros(3),np.array(t),2.))
            self.assertFalse(quality['accepted'])
            self.assertEqual(quality['reason'],'geometry')

    def test_narrow_line_features_cannot_validate_background_edge(self):
        points = self.points.copy();points[:,1]=0
        pixels = project(points,self.pose.rvec,self.pose.tvec,self.K,self.dist)
        self.assertFalse(self.validate(points=points,pixels=pixels)['accepted'])

    def test_optimizer_reset_is_not_independent_tracking_evidence(self):
        native = Mock(); native.step.return_value = self.pose
        assist = RegionAssist(native)
        def update(certified):
            return assist.update(self.frame,self.pose.rvec,self.pose.tvec,certified,
                self.vertices,self.K,self.dist,self.seeder,self.points,self.pixels,None,150,2400)
        self.assertIsNone(update(False));native.step.assert_not_called()
        self.assertIsNone(update(True));self.assertEqual(assist.stats['resets'],1)
        with patch('core.m3t_live.validate_region_pose',return_value=dict(accepted=False,reason='fixture')):
            for _ in range(3):self.assertIsNone(update(False))
        self.assertFalse(assist.initialized)
        self.assertIsNone(update(False));self.assertEqual(native.step.call_count,4)
        self.assertIsNone(update(True));self.assertEqual(assist.stats['resets'],2)

    def test_native_failure_disables_only_assistance(self):
        native=Mock();native.step.side_effect=TimeoutError('fixture timeout')
        assist=RegionAssist(native)
        self.assertIsNone(assist.update(self.frame,self.pose.rvec,self.pose.tvec,True,
            self.vertices,self.K,self.dist,self.seeder,self.points,self.pixels,None,150,2400))
        self.assertTrue(assist.disabled)
        native.close.assert_called_once()
        self.assertEqual(assist.stats['errors'],1)


if __name__ == '__main__':
    unittest.main()
