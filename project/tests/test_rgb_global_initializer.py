import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.rgb_global_initializer import (ReferenceView, RgbGlobalInitializer, advance_pose,
                                          default_reference_paths, prepare_initializer,
                                          solve_correspondences, verified_reference_paths)


class RgbGlobalInitializerTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        self.dist = np.zeros(5)
        self.points = np.array([[x, y, z] for x, y, z in (
            (-70, -35, 0), (-40, -30, 12), (0, -28, 20), (40, -30, 10), (70, -35, 0),
            (-65, 0, 18), (-20, 5, 30), (25, 4, 12), (65, 0, 20),
            (-60, 35, 2), (-15, 30, 20), (30, 32, 5), (65, 35, 18))], dtype=np.float64)
        self.rvec = np.array([.2, -.15, .1])
        self.tvec = np.array([20., -15., 700.])
        pixels, _ = cv2.projectPoints(self.points, self.rvec, self.tvec, self.K, self.dist)
        self.uv = pixels.reshape(-1, 2)
        self.box = np.r_[self.uv.min(axis=0), self.uv.max(axis=0)]

    def solve(self, pixels=None, box=None):
        return solve_correspondences(self.points, self.uv if pixels is None else pixels,
                                     self.K, self.dist, self.points,
                                     self.box if box is None else box, 'fixture')

    def test_solves_current_pose_from_metric_surface_points(self):
        pose = self.solve()
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.tvec, self.tvec, atol=.1)
        self.assertLess(pose.rms_px, .1)

    def test_pnp_discards_out_of_range_inlier_indices(self):
        malformed = np.arange(12, dtype=np.int32).reshape(-1, 1)
        malformed[-1, 0] = len(self.points) + 10
        with patch('core.rgb_global_initializer.cv2.solvePnPRansac',
                   return_value=(True, self.rvec.copy(), self.tvec.copy(), malformed)):
            self.assertIsNone(self.solve())

    def test_malformed_reference_view_is_ignored_before_matching(self):
        # A stale cache can contain more descriptors than CAD points. Such a
        # view must be discarded at the initializer boundary because queryIdx
        # names a row in the descriptor/point correspondence table.
        malformed = ReferenceView('stale', self.points[:-1],
                                  np.ones((len(self.points), 8), np.uint8),
                                  pixels=self.uv[:-1])
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, [malformed])
        self.assertEqual(initializer.views, [])

    def test_pose_on_another_object_or_small_patch_is_rejected(self):
        self.assertIsNone(self.solve(box=[20., 20., 150., 150.]))
        self.assertIsNone(self.solve(pixels=np.random.default_rng(17).uniform(0, 400, (13, 2))))
        tiny = self.points * .001
        uv, _ = cv2.projectPoints(tiny, self.rvec, self.tvec, self.K, self.dist)
        self.assertIsNone(solve_correspondences(tiny, uv, self.K, self.dist,
                                               tiny, self.box))

    def test_pnp_accepts_a_partially_visible_edge_box(self):
        points = np.array([[x, y, (x + y) % 20] for x in (-70, -40, -10, 20, 50, 70)
                           for y in (-35, 0, 35)], np.float64)
        tvec = np.array([-350., -15., 700.])
        projected, _ = cv2.projectPoints(points, self.rvec, tvec, self.K, self.dist)
        uv = projected.reshape(-1, 2)
        visible = uv[:, 0] >= 0
        box = np.r_[np.maximum(uv.min(0), 0), uv.max(0)]
        pose = solve_correspondences(points[visible], uv[visible], self.K, self.dist,
                                     points, box, 'fixture',
                                     image_shape=(480, 640, 3))
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.tvec, tvec, atol=.5)

    def test_delayed_pose_is_updated_to_the_current_image(self):
        pose = self.solve()
        random = np.random.default_rng(4)
        source = random.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        affine = np.array([[1., 0., 12.], [0., 1., -5.]])
        current = cv2.warpAffine(source, affine, (640, 480))
        updated = advance_pose(pose, source, current, self.K, self.dist, self.points, self.box)
        self.assertIsNotNone(updated)
        expected = self.uv + [12., -5.]
        actual, _ = cv2.projectPoints(self.points, updated.rvec, updated.tvec, self.K, self.dist)
        self.assertLess(np.linalg.norm(actual.reshape(-1, 2) - expected, axis=1).mean(), .5)
        self.assertGreater(np.linalg.norm(updated.tvec - pose.tvec), 10.)

    def test_delayed_pose_can_be_updated_without_a_detector_box(self):
        pose = self.solve()
        random = np.random.default_rng(4)
        source = random.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        current = cv2.warpAffine(source, np.array([[1., 0., 12.], [0., 1., -5.]]),
                                 (640, 480))
        updated = advance_pose(pose, source, current, self.K, self.dist,
                               self.points, None)
        self.assertIsNotNone(updated)

    def test_session_reference_excludes_surface_misses_and_keeps_shared_bank(self):
        original = ReferenceView('disk_reference', self.points.copy(),
                                 np.ones((13, 8), np.uint8))
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, [original])
        descriptors = np.arange(13 * 8, dtype=np.uint8).reshape(13, 8)
        keys = [cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv]
        initializer.detectors = {
            kind: SimpleNamespace(detectAndCompute=lambda *_: (keys, descriptors))
            for kind in ('akaze', 'sift')}
        valid = np.ones(13, bool)
        valid[0] = False
        points = self.points.copy()
        points[0] = np.nan
        mask = np.ones((480, 640), np.uint8)
        seeder = SimpleNamespace(mask=lambda *a: mask,
                                 intersect=lambda *a: (points, valid))
        frame = np.zeros((480, 640, 3), np.uint8)
        augmented = initializer.with_reference(frame, self.rvec, self.tvec, seeder)
        self.assertIsNot(augmented, initializer)
        self.assertEqual(len(initializer.views), 1)
        self.assertEqual(len(augmented.views), 3)
        for view in augmented.views[1:]:
            self.assertEqual(view.name, 'session_initial')
            np.testing.assert_array_equal(view.points, self.points[1:])
            np.testing.assert_array_equal(view.descriptors, descriptors[1:])
        self.assertEqual({v.descriptor_kind for v in augmented.views[1:]}, {'akaze', 'sift'})
        descriptors[:] = 0
        self.assertTrue(augmented.views[1].descriptors.any())
        # Reusing a session bank replaces its one view instead of growing it.
        augmented.detectors = initializer.detectors
        replaced = augmented.with_reference(frame, self.rvec, self.tvec, seeder)
        self.assertEqual(len(replaced.views), 3)

    def test_initial_image_already_in_disk_bank_counts_as_one_reference(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        image_id = hashlib.sha256(frame).hexdigest()
        descriptors = np.ones((13, 8), np.uint8)
        originals = [ReferenceView('copied_capture', self.points, descriptors, kind, image_id)
                     for kind in ('akaze', 'sift')]
        other = ReferenceView('different_image', self.points, descriptors, 'akaze', 'other')
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, originals + [other])
        keys = [cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv]
        initializer.detectors = {
            kind: SimpleNamespace(detectAndCompute=lambda *_: (keys, descriptors))
            for kind in ('akaze', 'sift')}
        seeder = SimpleNamespace(mask=lambda *a: np.ones(frame.shape[:2], np.uint8),
                                 intersect=lambda *a: (self.points, np.ones(13, bool)))
        local = initializer.with_reference(frame, self.rvec, self.tvec, seeder)
        self.assertEqual([v.name for v in local.views],
                         ['different_image', 'session_initial', 'session_initial'])
        self.assertEqual(len(initializer.views), 3)
        self.assertEqual({v.image_id for v in local.views[1:]}, {image_id})

    def test_invalid_or_empty_initial_reference_cannot_enter_search_bank(self):
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, [])
        frame = np.zeros((480, 640, 3), np.uint8)
        seeder = SimpleNamespace(mask=lambda *a: np.zeros((480, 640), np.uint8))
        for tvec in (self.tvec, np.array([0., 0., -100.]), np.array([0., 0., np.nan])):
            self.assertIs(initializer.with_reference(frame, self.rvec, tvec, seeder),
                          initializer)
        seeder.mask = lambda *a: np.ones((480, 640), np.uint8)
        self.assertIs(initializer.with_reference(frame, self.rvec, self.tvec, seeder),
                      initializer)

    def test_empty_scene_does_not_produce_a_pose(self):
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, [])
        self.assertIsNone(initializer.estimate(np.zeros((480, 640, 3), np.uint8), self.box))

    def test_full_frame_search_does_not_use_full_frame_as_detector_box(self):
        descriptors = np.random.default_rng(42).integers(0, 256, (13, 61), dtype=np.uint8)
        views = [ReferenceView('view-a', self.points, descriptors),
                 ReferenceView('view-b', self.points, descriptors)]
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, views)
        source = np.random.default_rng(5).integers(0, 255, (480, 640, 3), dtype=np.uint8)

        def features(_image, _mask):
            return ([cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv],
                    descriptors)

        initializer.detector = SimpleNamespace(detectAndCompute=features)
        with patch('core.rgb_global_initializer.observe_long_axis',
                   side_effect=AssertionError('full-frame search must skip axis prior')):
            pose = initializer.estimate(source, [0., 0., 640., 480.],
                                        validation_box=None)
        self.assertIsNotNone(pose)

    def test_sift_runs_after_akaze_fails_reference_support(self):
        random = np.random.default_rng(43)
        akaze = random.integers(0, 256, (13, 61), dtype=np.uint8)
        sift = random.random((13, 128)).astype(np.float32)
        keys = [cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv]
        frame = np.zeros((480, 640, 3), np.uint8)
        for enough_sift_support in (False, True):
            with self.subTest(enough_sift_support=enough_sift_support):
                views = [ReferenceView('a', self.points, akaze),
                         ReferenceView('a', self.points, sift, 'sift')]
                if enough_sift_support:
                    views.append(ReferenceView('b', self.points, sift, 'sift'))
                initializer = RgbGlobalInitializer(self.K, self.dist, self.points, views)
                initializer.detector = SimpleNamespace(detectAndCompute=lambda *_: (keys, akaze))
                with patch.dict(initializer.detectors, sift=SimpleNamespace(
                        detectAndCompute=unittest.mock.Mock(return_value=(keys, sift)))):
                    pose = initializer.estimate(frame, [0, 0, 640, 480])
                    initializer.detectors['sift'].detectAndCompute.assert_called_once()
                if enough_sift_support:
                    self.assertIsNotNone(pose)
                    self.assertEqual(pose.reference_support, 2)
                else:
                    self.assertIsNone(pose)

    def test_complementary_families_solve_one_reference_without_duplicate_pixels(self):
        random = np.random.default_rng(48)
        akaze = random.integers(0, 256, (13, 61), dtype=np.uint8)
        sift = random.random((13, 128)).astype(np.float32)
        frame = np.zeros((480, 640, 3), np.uint8)
        for support, sift_indices, accepted in ((2, np.arange(5, 13), True),
                                               (1, np.arange(5, 13), False),
                                               (2, np.arange(8), False)):
            with self.subTest(support=support, sift_indices=sift_indices):
                views = [ReferenceView(str(i), self.points, descriptors, kind)
                         for i in range(support)
                         for kind, descriptors in (('akaze', akaze), ('sift', sift))]
                initializer = RgbGlobalInitializer(self.K, self.dist, self.points, views)
                def features(indices, descriptors):
                    return ([cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv[indices]],
                            descriptors[indices])
                initializer.detector = SimpleNamespace(detectAndCompute=lambda *_:
                                                        features(np.arange(8), akaze))
                initializer.detectors['sift'] = SimpleNamespace(detectAndCompute=lambda *_:
                                                               features(sift_indices, sift))
                pose = initializer.estimate(frame, [0, 0, 640, 480])
                self.assertEqual(pose is not None, accepted)
                if accepted:
                    self.assertEqual(len(pose.points), 13)
                    self.assertEqual(pose.reference_support, 2)
                    np.testing.assert_allclose(pose.tvec, self.tvec, atol=.1)

    def test_supported_akaze_pose_skips_sift_cost(self):
        descriptors = np.random.default_rng(44).integers(0, 256, (13, 61), dtype=np.uint8)
        views = [ReferenceView(name, self.points, descriptors) for name in ('a', 'b')]
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, views)
        keys = [cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv]
        initializer.detector = SimpleNamespace(detectAndCompute=lambda *_: (keys, descriptors))
        initializer.detectors['sift'] = SimpleNamespace(
            detectAndCompute=lambda *_: self.fail('valid AKAZE must skip SIFT'))
        self.assertIsNotNone(initializer.estimate(np.zeros((480, 640, 3), np.uint8),
                                                 [0, 0, 640, 480]))

    def test_moving_search_combines_two_incomplete_sets_of_matches(self):
        self.check_moving_feature_evidence('akaze')

    def test_sift_combines_incomplete_matches_without_mixing_akaze_indices(self):
        self.check_moving_feature_evidence('sift')

    def check_moving_feature_evidence(self, kind):
        random = np.random.default_rng(41)
        descriptors = (random.integers(0, 256, (13, 61), dtype=np.uint8) if kind == 'akaze'
                       else random.random((13, 128)).astype(np.float32))
        views = [ReferenceView(name, self.points, descriptors, kind)
                 for name in ('view-a', 'view-b')]
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, views)
        search = initializer.new_search()
        source = np.random.default_rng(4).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        affine = np.array([[1., 0., 8.], [0., 1., -3.]])
        current = cv2.warpAffine(source, affine, (640, 480))

        def features(indices, pixels, box):
            # Feature detectors return crop-local coordinates; PnP and optical
            # flow must continue receiving full camera-image coordinates.
            local = pixels - np.maximum(np.rint(box[:2]) - 48, 0)
            return ([cv2.KeyPoint(float(local[i, 0]), float(local[i, 1]), 4.) for i in indices],
                    descriptors[indices])

        observations = iter([features(np.arange(8), self.uv, self.box),
                             features(np.arange(5, 13), self.uv + [8., -3.],
                                      self.box + [8., -3., 8., -3.])])
        detector = SimpleNamespace(detectAndCompute=lambda *_: next(observations))
        if kind == 'akaze':
            initializer.detector = detector
        else:
            initializer.detector = SimpleNamespace(detectAndCompute=lambda *_: ([], None))
            initializer.detectors['sift'] = detector
            # Same view names and indices belong to different 3D features in
            # each descriptor family. An AKAZE bank must not contaminate SIFT.
            search.remember(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), self.box,
                            {'view-a': {i: self.uv[i] + 80 for i in range(8)}})
        with patch('core.rgb_global_initializer.observe_long_axis', return_value=None):
            self.assertIsNone(search.estimate(source, self.box))
            pose = search.estimate(current, self.box + [8., -3., 8., -3.])
        self.assertIsNotNone(pose)
        self.assertGreaterEqual(len(pose.points), 12)
        actual, _ = cv2.projectPoints(self.points, pose.rvec, pose.tvec, self.K, self.dist)
        self.assertLess(np.linalg.norm(actual.reshape(-1, 2) - self.uv - [8., -3.], axis=1).mean(), .5)
        self.assertIsNone(search.estimate(current, None))
        self.assertFalse(search.evidence)
        self.assertFalse(search.sift_evidence.evidence)

    def test_roi_features_use_crop_and_restore_camera_coordinates(self):
        descriptors = np.random.default_rng(45).integers(0, 256, (13, 61), dtype=np.uint8)
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points,
            [ReferenceView(name, self.points, descriptors) for name in ('a', 'b')])
        frame = np.random.default_rng(46).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        origin = np.maximum(np.rint(self.box[:2]).astype(int) - 48, 0)
        end = np.minimum(np.rint(self.box[2:]).astype(int) + 48, [640, 480])
        keys = [cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv - origin]
        local_before = np.array([key.pt for key in keys])

        def features(image, mask):
            self.assertEqual(image.shape, (end[1] - origin[1], end[0] - origin[0]))
            np.testing.assert_array_equal(image, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[
                origin[1]:end[1], origin[0]:end[0]])
            self.assertEqual(mask.shape, image.shape)
            return keys, descriptors

        initializer.detector = SimpleNamespace(detectAndCompute=features)
        pose = initializer.estimate(frame, self.box)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.tvec, self.tvec, atol=.1)
        np.testing.assert_array_equal([key.pt for key in keys], local_before)

    def test_outside_and_empty_regions_clear_both_evidence_banks(self):
        descriptors = np.zeros((13, 61), np.uint8)
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points,
                                          [ReferenceView('a', self.points, descriptors)])
        initializer.detector = SimpleNamespace(detectAndCompute=lambda *_:
                                               self.fail('invalid ROI must skip extraction'))
        frame = np.zeros((480, 640, 3), np.uint8)
        for box in ([700, 0, 800, 100], [-200, 0, -100, 100], [100, 100, 50, 50], [0, 0, 0, 0]):
            search = initializer.new_search()
            for bank in (search, search.sift_evidence):
                bank.remember(frame[:, :, 0], self.box, {'a': {0: self.uv[0]}})
            self.assertIsNone(search.estimate(frame, box))
            self.assertFalse(search.evidence)
            self.assertFalse(search.sift_evidence.evidence)

    def test_temporal_matches_are_discarded_when_target_jumps(self):
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, [])
        search = initializer.new_search()
        gray = np.zeros((480, 640), np.uint8)
        search.remember(gray, self.box, {'view': {0: self.uv[0]}})
        self.assertEqual(search.track(gray, self.box + [200., 0., 200., 0.]), {})
        self.assertFalse(search.evidence)
        self.assertFalse(search.sift_evidence.evidence)

    def test_roi_features_use_crop_and_restore_camera_coordinates(self):
        descriptors = np.random.default_rng(45).integers(0, 256, (13, 61), dtype=np.uint8)
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points,
            [ReferenceView(name, self.points, descriptors) for name in ('a', 'b')])
        frame = np.random.default_rng(46).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        origin = np.maximum(np.rint(self.box[:2]).astype(int) - 48, 0)
        end = np.minimum(np.rint(self.box[2:]).astype(int) + 48, [640, 480])
        keys = [cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv - origin]
        local_before = np.array([key.pt for key in keys])

        def features(image, mask):
            self.assertEqual(image.shape, (end[1] - origin[1], end[0] - origin[0]))
            np.testing.assert_array_equal(image, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[
                origin[1]:end[1], origin[0]:end[0]])
            self.assertEqual(mask.shape, image.shape)
            return keys, descriptors

        initializer.detector = SimpleNamespace(detectAndCompute=features)
        pose = initializer.estimate(frame, self.box)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.tvec, self.tvec, atol=.1)
        np.testing.assert_array_equal([key.pt for key in keys], local_before)

    def test_outside_and_empty_regions_clear_both_evidence_banks(self):
        descriptors = np.zeros((13, 61), np.uint8)
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points,
                                          [ReferenceView('a', self.points, descriptors)])
        initializer.detector = SimpleNamespace(detectAndCompute=lambda *_:
                                               self.fail('invalid ROI must skip extraction'))
        frame = np.zeros((480, 640, 3), np.uint8)
        for box in ([700, 0, 800, 100], [-200, 0, -100, 100], [100, 100, 50, 50], [0, 0, 0, 0]):
            search = initializer.new_search()
            for bank in (search, search.sift_evidence):
                bank.remember(frame[:, :, 0], self.box, {'a': {0: self.uv[0]}})
            self.assertIsNone(search.estimate(frame, box))
            self.assertFalse(search.evidence)
            self.assertFalse(search.sift_evidence.evidence)

    def test_temporal_indices_outside_reference_view_are_discarded(self):
        descriptors = np.random.default_rng(51).integers(0, 256, (13, 61), dtype=np.uint8)
        view = ReferenceView('view', self.points, descriptors)
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, [view])
        keys = [cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv[:11]]
        initializer.detector = SimpleNamespace(
            detectAndCompute=lambda *_: (keys, descriptors[:11]))
        search = initializer.new_search()
        # Simulate evidence from a stale reference bank after a view rebuild.
        search.track = lambda *_: {'view': {index: self.uv[0]
                                            for index in range(13, 25)}}
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertIsNone(search.estimate(frame, self.box))

    def test_descriptor_family_temporal_banks_are_independent_and_clear_together(self):
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, [])
        search = initializer.new_search()
        self.assertIsNot(search.sift_evidence, search.orb_evidence)
        search.sift_evidence.evidence = {'sift': {0: self.uv[0]}}
        search.orb_evidence.evidence = {'orb': {0: self.uv[1]}}
        search.clear()
        self.assertFalse(search.sift_evidence.evidence)
        self.assertFalse(search.orb_evidence.evidence)

    def test_temporal_matches_expire_when_search_is_interrupted(self):
        initializer = RgbGlobalInitializer(self.K, self.dist, self.points, [])
        search = initializer.new_search()
        gray = np.zeros((480, 640), np.uint8)
        with patch('core.rgb_global_initializer.time.monotonic', return_value=10.):
            search.remember(gray, self.box, {'view': {0: self.uv[0]}})
        with patch('core.rgb_global_initializer.time.monotonic', return_value=10.6):
            self.assertEqual(search.track(gray, self.box), {})
        self.assertFalse(search.evidence)



class ReferenceDiscoveryTest(unittest.TestCase):
    @staticmethod
    def capture(root, name, *, collection=False, **changes):
        path = root / name / ('live/attempt_001' if collection else 'attempt_001') / 'rgb_initial_pose.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(depth_pose_quality=dict(accepted=True, partial_depth=False,
                                           requires_rgb_verification=False),
                    rgb_stillness=dict(global_response=.8, global_shift_px=[0, 0],
                                       target_shift_px=[0, 0]))
        for field, values in changes.items():
            data[field].update(values)
        path.write_text(json.dumps(data))
        return path

    def test_default_bank_includes_only_fully_verified_collection_captures(self):
        with TemporaryDirectory() as directory:
            data = Path(directory)
            legacy = data / 'tracking/live_depth_6dof'
            collection = data / 'tracking/targeted_collection'
            old = self.capture(legacy, 'old')
            new = self.capture(collection, 'new', collection=True)
            self.capture(collection, 'partial', collection=True,
                         depth_pose_quality=dict(partial_depth=True, requires_rgb_verification=True))
            self.capture(collection, 'needs_rgb', collection=True,
                         depth_pose_quality=dict(requires_rgb_verification=True))
            self.capture(collection, 'missing_depth_status', collection=True)
            missing = collection / 'missing_depth_status/live/attempt_001/rgb_initial_pose.json'
            payload = json.loads(missing.read_text())
            del payload['depth_pose_quality']['partial_depth']
            missing.write_text(json.dumps(payload))
            self.capture(collection, 'moving', collection=True,
                         rgb_stillness=dict(motion_tolerated=True))
            self.capture(collection, 'failed', collection=True,
                         depth_pose_quality=dict(accepted=False))
            with patch('core.rgb_global_initializer.DATA_ROOT', data):
                self.assertEqual(set(default_reference_paths()), {old, new})
            # An explicit root remains restricted to that caller's library.
            self.assertEqual(verified_reference_paths(legacy), [old])

    def test_stillness_gate_rejects_weak_or_shifted_collection_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            good = self.capture(root, 'good', collection=True)
            for name, field in [('response', dict(global_response=.2)),
                                ('global_shift', dict(global_shift_px=[4, 0])),
                                ('target_shift', dict(target_shift_px=[0, 3]))]:
                self.capture(root, name, collection=True, rgb_stillness=field)
            broken = self.capture(root, 'broken', collection=True)
            broken.write_text('{invalid')
            paths = verified_reference_paths(root, pattern='*/live/attempt_*/rgb_initial_pose.json',
                                              require_full_depth=True)
            self.assertEqual(paths, [good])

    def test_copied_images_do_not_become_independent_reference_support(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.ply'
            model.write_text('fixture')
            image = np.random.default_rng(18).integers(0, 255, (48, 64, 3), dtype=np.uint8)
            paths = []
            for name in ('a', 'b'):
                path = self.capture(root, name)
                rgb_path = path.parent / 'rgb.png'
                cv2.imwrite(str(rgb_path), image)
                payload = json.loads(path.read_text())
                payload.update(rgb_image=str(rgb_path), intrinsics=str(root/'camera.npz'),
                               rvec_rgb_model_rad=[0, 0, 0], tvec_rgb_model_mm=[0, 0, 500])
                path.write_text(json.dumps(payload))
                depth = path.parent / 'depth_fit/diagnostics.json'
                depth.parent.mkdir()
                depth.write_text(json.dumps(dict(input=dict(model=str(model)))))
                paths.append(path)
            keys = [cv2.KeyPoint(float(i+2), float(i+2), 4.) for i in range(13)]
            detector = SimpleNamespace(detectAndCompute=lambda *_: (keys, np.ones((13, 8), np.uint8)))
            seeder = SimpleNamespace(mask=lambda shape, *a: np.ones(shape[:2], np.uint8),
                                     intersect=lambda *a: (np.arange(39).reshape(13, 3), np.ones(13, bool)))
            with patch('core.rgb_global_initializer.DATA_ROOT', root), \
                 patch('core.rgb_global_initializer.default_reference_paths', return_value=paths), \
                 patch('core.rgb_global_initializer.load_calibration', return_value=(np.eye(3), np.zeros(5), (64,48))), \
                 patch('core.rgb_global_initializer.make_edge_points', return_value=(np.ones((13,3)), None, None)), \
                 patch('core.rgb_global_initializer.VisibleSurfaceSeeder', return_value=seeder), \
                 patch('core.rgb_global_initializer.cv2.AKAZE_create', return_value=detector), \
                 patch('core.rgb_global_initializer.cv2.SIFT_create', return_value=detector):
                initial = prepare_initializer(model, root/'camera.npz')
                self.assertEqual(len(initial.views), 2)
                self.assertEqual({v.name for v in initial.views}, {str(paths[0].resolve())})
                self.assertEqual({v.image_id for v in initial.views}, {hashlib.sha256(image).hexdigest()})
                # The same rule applies when descriptors come from the disk cache.
                with patch('core.rgb_global_initializer.VisibleSurfaceSeeder', side_effect=AssertionError('cache miss')):
                    cached = prepare_initializer(model, root/'camera.npz')
                self.assertEqual(len(cached.views), 2)
                self.assertEqual({v.name for v in cached.views}, {str(paths[0].resolve())})
                with patch('core.rgb_global_initializer.default_reference_paths') as defaults, \
                     patch('core.rgb_global_initializer.verified_reference_paths', return_value=paths) as explicit:
                    restricted = prepare_initializer(model, root/'camera.npz', root=root/'private')
                defaults.assert_not_called()
                explicit.assert_called_once_with(root/'private')
                self.assertEqual(len(restricted.views), 2)


if __name__ == '__main__':
    unittest.main()
