"""Initialize the current RGB pose from depth-verified views and CAD features.

Each descriptor is attached to a visible CAD surface in a verified reference.
PnP solves a fresh pose in the query frame; reference poses are never copied to
the query. Different reference views are solved separately because repeated
rails and equivalent roll poses can give inconsistent cross-view identities.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cad_pose_overlay import load_calibration
from .cad_video_tracker import detection_matches_pose, make_edge_points, project
from .paths import DATA_ROOT
from .reference_pose_corrector import CONTROL_POINTS, project_points
from .rgb_pose_recovery import axis_matches_pose, observe_long_axis
from .surface_seeding import VisibleSurfaceSeeder


def _reference_view_arrays_are_aligned(points, descriptors, pixels=None):
    """Return whether one reference keeps 3-D/descriptor/pixel rows aligned.

    A descriptor match's ``queryIdx`` is an index into the reference descriptor
    array.  It is therefore only safe to use it for CAD lookup when all three
    arrays have the same row count.  Older feature caches were written during
    an intermediate pipeline revision and can violate this invariant.
    """
    points = np.asarray(points)
    descriptors = np.asarray(descriptors)
    if points.ndim != 2 or points.shape[1] != 3:
        return False
    if descriptors.ndim != 2 or len(points) != len(descriptors):
        return False
    if pixels is not None:
        pixels = np.asarray(pixels)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or len(points) != len(pixels):
            return False
    return bool(np.isfinite(points).all()
                and (pixels is None or np.isfinite(pixels).all()))


@dataclass(frozen=True)
class ReferenceView:
    name: str
    points: np.ndarray
    descriptors: np.ndarray
    descriptor_kind: str = 'akaze'
    image_id: str = ''
    # Pixel coordinates are needed by the oblique-view homography fallback.
    # Older in-memory/test views may omit them and simply use direct PnP.
    pixels: np.ndarray | None = None


@dataclass(frozen=True)
class RgbPose:
    rvec: np.ndarray
    tvec: np.ndarray
    points: np.ndarray
    pixels: np.ndarray
    reference: str
    rms_px: float
    inlier_ratio: float
    elapsed_ms: float
    reference_support: int = 1
    # Median image displacement measured by LK for this temporal update.
    # This is association metadata only; it must never be folded into tvec.
    motion_shift_px: np.ndarray | None = None


def solve_correspondences(points, pixels, K, dist, vertices, box, reference='',
                          min_depth=150., max_depth=2000., image_shape=None,
                          pnp_method=cv2.SOLVEPNP_EPNP, min_inlier_ratio=.5,
                          refine_inliers=True, prior_pose=None) -> RgbPose | None:
    """Solve metric pose with distributed, independently checked 2D/3D evidence."""
    started = time.perf_counter()
    points = np.ascontiguousarray(points, np.float64).reshape(-1, 3)
    pixels = np.ascontiguousarray(pixels, np.float64).reshape(-1, 2)
    if len(points) < 12 or len(points) != len(pixels):
        return None
    if not np.isfinite(points).all() or not np.isfinite(pixels).all():
        return None
    # A temporal pose is a useful *initial guess*, not a measurement.  When it
    # exists, use ITERATIVE PnP from that guess before falling back to the
    # normal RANSAC branch.  This prevents EPNP from selecting the equally
    # valid 180-degree solution for a symmetric elongated object on an
    # otherwise continuous track.
    pnp_kwargs = dict(iterationsCount=160, reprojectionError=3.,
                      confidence=.999, flags=pnp_method)
    if prior_pose is not None and pnp_method == cv2.SOLVEPNP_ITERATIVE:
        pnp_kwargs.update(useExtrinsicGuess=True,
                          rvec=np.asarray(prior_pose.rvec, np.float64).reshape(3, 1).copy(),
                          tvec=np.asarray(prior_pose.tvec, np.float64).reshape(3, 1).copy())
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(points, pixels, K, dist,
                                                  **pnp_kwargs)
    if (not ok or inliers is None or len(inliers) < 12
            or len(inliers) / len(points) < min_inlier_ratio):
        return None
    # solvePnPRansac returns row indices, not a boolean mask. Building the
    # mask explicitly is important when the correspondence array is larger
    # than the returned consensus set.
    # OpenCV should return row indices into the input arrays.  In long-running
    # live sessions a stale descriptor bank or an occasional native-side
    # malformed result can still produce an index outside that array.  Treat
    # those entries as invalid evidence instead of letting one background
    # search terminate the live pose service.
    indices = np.unique(inliers.ravel().astype(np.intp))
    indices = indices[(indices >= 0) & (indices < len(points))]
    if len(indices) < 12 or len(indices) / len(points) < min_inlier_ratio:
        return None
    keep = np.zeros(len(points), dtype=bool)
    keep[indices] = True
    # Rail-only matches have ambiguous translation/orientation. Require a
    # two-dimensional patch spanning a useful fraction of the physical object.
    xyz, uv = points[keep], pixels[keep]
    singular = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    if singular[1] / np.sqrt(len(xyz)) < 6.:
        return None

    if refine_inliers:
        # RANSAC gives a good branch, but its inlier set is based on the
        # pre-refine pose. Reproject all correspondences after each light LM
        # step and retain only points that agree with the refined pose. This is
        # bounded to two passes, so relocalization gets a cleaner metric pose
        # without changing the descriptor search budget.
        # Keep the same consensus radius as RANSAC while reclassifying. The
        # final RMS gate remains 2.5 px; using that tighter value here would
        # discard otherwise valid handheld optical-flow tracks before LM settles.
        refine_threshold = 3.0
        for _ in range(2):
            xyz, uv = points[keep], pixels[keep]
            # OpenCV may return a duplicated RANSAC index on a partially
            # visible edge set. The reclassification below can recover the
            # omitted unique point, so only require a usable LM set here.
            if len(xyz) < 6:
                return None
            rvec, tvec = cv2.solvePnPRefineLM(xyz, uv, K, dist, rvec, tvec)
            rvec, tvec = rvec.reshape(3), tvec.reshape(3)
            reprojection = project(points, rvec, tvec, K, dist)
            errors = np.linalg.norm(reprojection - pixels, axis=1)
            candidate_keep = np.isfinite(errors) & (errors <= refine_threshold)
            # Preserve the caller's confidence gate after reclassification. If
            # the stricter set is too small, keep the previous verified RANSAC
            # set and finish with that pose rather than weakening acceptance.
            if (np.count_nonzero(candidate_keep) < 12
                    or np.count_nonzero(candidate_keep) / len(points) < min_inlier_ratio):
                break
            if np.array_equal(candidate_keep, keep):
                break
            keep = candidate_keep

        xyz, uv = points[keep], pixels[keep]
        # One final refinement on the reclassified inliers removes the small
        # bias introduced by the first RANSAC consensus.
        rvec, tvec = cv2.solvePnPRefineLM(xyz, uv, K, dist, rvec, tvec)
    else:
        # Optical-flow tracks are already temporally filtered. Keep their
        # original single-pass behavior so a fast, mildly noisy handheld frame
        # is not rejected by a relocalization-only consensus split.
        xyz, uv = points[keep], pixels[keep]
        if len(xyz) < 6:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(xyz, uv, K, dist, rvec, tvec)
    rvec, tvec = rvec.reshape(3), tvec.reshape(3)
    rotation, _ = cv2.Rodrigues(rvec)
    if (not np.isfinite([*rvec, *tvec]).all() or not min_depth <= tvec[2] <= max_depth
            or np.min((vertices @ rotation.T + tvec)[:, 2]) <= 0):
        return None
    projected = project(vertices, rvec, tvec, K, dist)
    hull_area = cv2.contourArea(cv2.convexHull(projected.astype(np.float32)))
    spread = cv2.contourArea(cv2.convexHull(uv.astype(np.float32)))
    final_errors = np.linalg.norm(project(xyz, rvec, tvec, K, dist) - uv, axis=1)
    rms = float(np.sqrt(np.mean(final_errors ** 2)))
    visible_fraction = 1.
    if image_shape is not None:
        height, width = image_shape[:2]
        full_box = np.r_[projected.min(axis=0), projected.max(axis=0)]
        clipped = np.clip(full_box, [0, 0, 0, 0], [width-1, height-1, width-1, height-1])
        full_area = float(np.prod(np.maximum(full_box[2:] - full_box[:2], 0)))
        visible_fraction = float(np.prod(np.maximum(clipped[2:] - clipped[:2], 0))) / max(full_area, 1.)
    if (rms > 2.5 or visible_fraction < .35 or spread < max(200., hull_area * visible_fraction * .12)
            or (box is not None and not detection_matches_pose(
                box, vertices, rvec, tvec, K, dist, image_shape=image_shape))):
        return None
    return RgbPose(rvec, tvec, xyz.copy(), uv.astype(np.float32), reference,
                   rms, len(keep) / len(points), (time.perf_counter() - started) * 1000.)


def feature_region(gray, mask, bounds, padding=48):
    """Preserve descriptor context around an ROI and return its camera offset."""
    x1, y1, x2, y2 = np.rint(bounds).astype(int)
    left, top = max(0, x1 - padding), max(0, y1 - padding)
    right, bottom = min(gray.shape[1], x2 + padding), min(gray.shape[0], y2 + padding)
    return (np.ascontiguousarray(gray[top:bottom, left:right]),
            np.ascontiguousarray(mask[top:bottom, left:right]),
            np.array([left, top], np.float32))


class RgbGlobalInitializer:
    def __init__(self, K, dist, vertices, views):
        self.K, self.dist, self.vertices = K, dist, vertices
        # A malformed cached view cannot produce a valid correspondence: its
        # descriptor query indices no longer name the same CAD points. Drop it
        # at the boundary so one stale cache entry cannot kill the background
        # 6DoF worker. prepare_initializer also rebuilds such cache entries.
        self.views = [view for view in views
                      if len(view.points) >= 12
                      and _reference_view_arrays_are_aligned(
                          view.points, view.descriptors, view.pixels)]
        # AKAZE remains the cheap first pass. SIFT is an adaptive fallback for
        # oblique/side views where low-texture or scale-changed AKAZE evidence
        # is insufficient. Both paths still require the same CAD/PnP gates.
        self.detectors = {
            'akaze': cv2.AKAZE_create(threshold=.0001),
            'sift': cv2.SIFT_create(nfeatures=1600),
            # ORB gives the homography fallback many more repeatable binary
            # edge/corner matches on the low-texture side surface.  It is not
            # accepted by itself; its homography still has to produce a fresh
            # CAD/PnP pose and pass every projection gate.
            'orb': cv2.ORB_create(nfeatures=2500, fastThreshold=5),
        }
        self.detector = self.detectors['akaze']

    def with_reference(self, frame, rvec, tvec, seeder):
        """Build a session-local bank from the image of a confirmed initial pose.

        Raycast only visible CAD surfaces, excluding background and silhouette
        edges. Never learn from a held/lost pose or mutate the shared disk bank.
        Both descriptor families name the same view, so they cannot count as
        independent reference support.
        """
        rvec = np.asarray(rvec, np.float64).reshape(3)
        tvec = np.asarray(tvec, np.float64).reshape(3)
        if not np.isfinite(np.r_[rvec, tvec]).all():
            return self
        rotation, _ = cv2.Rodrigues(rvec)
        if np.min((self.vertices @ rotation.T + tvec)[:, 2]) <= 1.:
            return self
        mask = seeder.mask(frame.shape, self.vertices, rvec, tvec)
        if not np.any(mask):
            return self
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        views = []
        image_id = hashlib.sha256(np.ascontiguousarray(frame)).hexdigest()
        for kind, detector in self.detectors.items():
            keys, descriptors = detector.detectAndCompute(gray, mask)
            if descriptors is None or len(keys) < 12:
                continue
            pixels = np.array([key.pt for key in keys], np.float32)
            points, valid = seeder.intersect(pixels, rvec, tvec)
            valid = np.asarray(valid, bool) & np.isfinite(points).all(axis=1)
            if np.count_nonzero(valid) >= 12:
                views.append(ReferenceView('session_initial', points[valid].copy(),
                                           descriptors[valid].copy(), kind, image_id,
                                           pixels[valid].copy()))
        if not views:
            return self
        # Keep exactly one initial view per live session, even if a caller
        # provides an already augmented initializer.
        base = [view for view in self.views
                if view.name != 'session_initial' and view.image_id != image_id]
        return RgbGlobalInitializer(self.K, self.dist, self.vertices, base + views)

    def new_search(self):
        return RgbPoseSearch(self)

    def estimate(self, frame, box, temporal=None, validation_box=None, *,
                 cross_family_support=False, prior_pose=None) -> RgbPose | None:
        """Find a metric pose with adaptive current-frame feature evidence.

        AKAZE is tried first for speed. If it cannot produce a geometrically
        valid pose, SIFT is run on the same current frame to improve side and
        scale-changing views. Complementary matches may then be combined
        within each reference, without counting coincident features twice.
        Temporal banks augment measured correspondences, never a prior pose.
        """
        started = time.perf_counter()
        if box is None or not self.views:
            if temporal is not None:
                temporal.clear()
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(frame.shape[:2], np.uint8)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, mask.shape[1]), min(y2, mask.shape[0])
        if x2 <= x1 or y2 <= y1:
            if temporal is not None:
                temporal.clear()
            return None
        mask[y1:y2, x1:x2] = 255

        # A mask only filters returned keypoints: OpenCV still constructs the
        # feature pyramid over the entire image. Crop with context so a small
        # target gets both a cheaper pyramid and its own SIFT feature budget,
        # instead of competing with textured background outside the ROI.
        feature_gray, feature_mask, offset = feature_region(gray, mask, (x1, y1, x2, y2))

        family_matches = {}
        homography_matches = {}

        def search_kind(kind):
            # Do not touch a temporal bank when this descriptor family has no
            # reference views. In particular, an empty ORB family must not
            # erase SIFT evidence accumulated for the next frame.
            views = [view for view in self.views
                     if getattr(view, 'descriptor_kind', 'akaze') == kind]
            if not views:
                return []
            detector = self.detector if kind == 'akaze' else self.detectors[kind]
            keypoints, descriptors = detector.detectAndCompute(feature_gray, feature_mask)
            pixels_by_key = np.array([key.pt for key in keypoints], np.float32).reshape(-1, 2)
            pixels_by_key += offset
            bank = None
            if temporal is not None:
                bank = {'akaze': temporal, 'sift': temporal.sift_evidence,
                        'orb': temporal.orb_evidence}[kind]
            previous = bank.track(gray, box) if bank is not None else {}
            if descriptors is None or len(descriptors) < 2:
                descriptors = None
            norm = cv2.NORM_HAMMING if kind in ('akaze', 'orb') else cv2.NORM_L2
            ratio = .72 if kind == 'akaze' else .75
            if kind == 'orb':
                ratio = .80
            matcher = cv2.BFMatcher(norm)
            candidates = []
            evidence = {}
            # The full bank contains many verified viewpoints. Use one cheap
            # merged-bank pass to retain only the most likely reference views,
            # then run the stricter per-view ratio test on that shortlist. This
            # keeps oblique-view coverage without paying 22 independent matcher
            # calls for every detector hit.
            search_views = views
            if len(views) > 8 and descriptors is not None:
                bank_descriptors = np.concatenate([view.descriptors for view in views], axis=0)
                owners = np.concatenate([
                    np.full(len(view.descriptors), index, np.int32)
                    for index, view in enumerate(views)])
                shortlist_matches = matcher.knnMatch(bank_descriptors, descriptors, k=2)
                scores = np.zeros(len(views), np.int32)
                seen_train = [set() for _ in views]
                for pair in shortlist_matches:
                    if len(pair) != 2 or pair[0].distance >= ratio * pair[1].distance:
                        continue
                    match = pair[0]
                    owner = int(owners[match.queryIdx])
                    seen_train[owner].add(int(match.trainIdx))
                scores = np.array([len(items) for items in seen_train], np.int32)
                keep = np.argsort(-scores, kind='stable')[:8]
                search_views = [views[index] for index in keep if scores[index] >= 4]
                if not search_views:
                    search_views = views
            for view in search_views:
                pairs = (matcher.knnMatch(view.descriptors, descriptors, k=2)
                         if descriptors is not None else [])
                unique = {}
                for pair in pairs:
                    if len(pair) != 2 or pair[0].distance >= ratio * pair[1].distance:
                        continue
                    match = pair[0]
                    old = unique.get(match.trainIdx)
                    if old is None or match.distance < old.distance:
                        unique[match.trainIdx] = match
                current = {int(match.queryIdx): pixels_by_key[match.trainIdx].copy()
                           for match in unique.values()
                           if 0 <= int(match.queryIdx) < len(view.points)}
                # Temporal evidence can outlive a reference-view rebuild
                # during a live rotation. Drop stale descriptor indices before
                # they reach PnP correspondence assembly; one stale index must
                # not terminate the background pose worker.
                tracked = {
                    int(index): np.asarray(pixel, np.float32).copy()
                    for index, pixel in previous.get(view.name, {}).items()
                    if 0 <= int(index) < len(view.points)
                    and np.asarray(pixel).shape == (2,)
                    and np.isfinite(pixel).all()
                }
                merged = dict(current)
                for index, pixel in tracked.items():
                    # A tracked feature and a newly matched feature can describe
                    # the same pixel with different rail identities. Do not count
                    # that image evidence twice towards PnP support.
                    if index not in merged and all(np.linalg.norm(pixel - other) >= 3.
                                                   for other in merged.values()):
                        merged[index] = pixel
                if len(merged) >= 4:
                    evidence[view.name] = merged
                    family_matches.setdefault(view.name, {})[kind] = (view, merged)
                    if kind == 'orb' and view.pixels is not None:
                        homography_matches[view.name] = (view, merged)
                if len(merged) >= 12:
                    candidates.append((view, merged))
            candidates.sort(key=lambda entry: len(entry[1]), reverse=True)
            if bank is not None:
                bank.remember(gray, box, evidence)
            valid_candidates = []
            for view, matches in candidates[:6]:
                if not _reference_view_arrays_are_aligned(
                        view.points, view.descriptors, view.pixels):
                    continue
                indices = [int(index) for index in matches
                           if 0 <= int(index) < len(view.points)]
                if len(indices) < 12:
                    continue
                valid_candidates.append((
                    view.name,
                    np.asarray([view.points[index] for index in indices]),
                    np.asarray([matches[index] for index in indices])))
            return solve_candidates(valid_candidates)

        def solve_candidates(candidates, *, min_inlier_ratio=.5):
            poses = []
            observed_axis = None
            for name, points, pixels in candidates:
                method = (cv2.SOLVEPNP_ITERATIVE if prior_pose is not None
                          else cv2.SOLVEPNP_EPNP)
                pose = solve_correspondences(
                    points, pixels, self.K, self.dist, self.vertices, validation_box,
                    name, image_shape=frame.shape,
                    min_inlier_ratio=min_inlier_ratio, pnp_method=method,
                    prior_pose=prior_pose)
                if pose is None:
                    continue
                if validation_box is None:
                    poses.append(pose)
                    continue
                if observed_axis is None:
                    observed_axis = observe_long_axis(frame, validation_box)
                if axis_matches_pose(observed_axis, pose.rvec, pose.tvec, self.K, self.dist):
                    poses.append(pose)
            return poses

        def combined_candidates():
            candidates = []
            for name, families in family_matches.items():
                if len(families) < 2:
                    continue
                points, pixels = [], []
                # Families have separate index spaces. Combine their measured
                # 3D/2D pairs only within the same depth-verified reference.
                # Coincident image features cannot count twice as PnP support.
                for view, matches in families.values():
                    for index, pixel in matches.items():
                        point = view.points[index]
                        if pixels and (np.min(np.linalg.norm(np.asarray(pixels) - pixel, axis=1)) < 3.
                                       or np.min(np.linalg.norm(np.asarray(points) - point, axis=1)) < 1.):
                            continue
                        points.append(point)
                        pixels.append(pixel)
                if len(points) >= 12:
                    candidates.append((name, np.array(points), np.array(pixels)))
            candidates.sort(key=lambda entry: len(entry[1]), reverse=True)
            return solve_candidates(candidates[:6])

        def homography_candidates():
            """Recover a fresh pose from a planar/near-planar side surface.

            Oblique and horizontal views often provide only a handful of
            descriptor matches per reference.  A RANSAC homography expands
            those measured matches to the other CAD-attached features in the
            same reference image, after which PnP is solved from the expanded
            3D/2D evidence.  The homography is only a proposal: the normal
            CAD reprojection, visible-area, spread and detector gates still
            decide whether the pose is accepted.
            """
            candidates = []
            for name, (view, matches) in homography_matches.items():
                if view.pixels is None or len(matches) < 8:
                    continue
                indices = np.fromiter(matches.keys(), dtype=np.int32)
                src = np.asarray(view.pixels[indices], np.float32)
                dst = np.asarray(list(matches.values()), np.float32)
                if len(src) < 8 or cv2.contourArea(cv2.convexHull(src)) < 250.:
                    continue
                homography, inlier_mask = cv2.findHomography(
                    src, dst, cv2.RANSAC, 6.0, maxIters=2000, confidence=.999)
                if homography is None or inlier_mask is None:
                    continue
                inliers = inlier_mask.ravel().astype(bool)
                # Side and horizontal views can expose only a small planar
                # patch. Eight geometrically consistent ORB matches are the
                # minimum proposal; the expanded CAD correspondences still
                # have to pass the full metric PnP gates below.
                if np.count_nonzero(inliers) < 8:
                    continue
                mapped = cv2.perspectiveTransform(
                    np.asarray(view.pixels, np.float32).reshape(-1, 1, 2),
                    homography).reshape(-1, 2)
                height, width = frame.shape[:2]
                valid = (np.isfinite(mapped).all(axis=1)
                         & (mapped[:, 0] >= -8.) & (mapped[:, 0] < width + 8.)
                         & (mapped[:, 1] >= -8.) & (mapped[:, 1] < height + 8.))
                # Keep the proposal local to this detector hit. This prevents
                # a numerically valid H from turning distant reference points
                # into a second object in a multi-target frame.
                target_box = np.asarray(validation_box if validation_box is not None else box)
                valid &= (mapped[:, 0] <= target_box[2] + 16.)
                valid &= (mapped[:, 1] <= target_box[3] + 16.)
                valid &= (mapped[:, 0] >= target_box[0] - 16.)
                valid &= (mapped[:, 1] >= target_box[1] - 16.)

                # Homography-expanded pixels are only a planar approximation
                # for a 3-D CAD surface. Put the real ORB observations first
                # and use the expansion only to resolve depth/pose ambiguity.
                # This keeps the final LM fit anchored to measured image points
                # instead of allowing a large set of extrapolated pixels to
                # shift the silhouette.
                direct_indices = indices[inliers]
                direct_pixels = dst[inliers]
                direct_valid = (np.isfinite(direct_pixels).all(axis=1)
                                & (direct_pixels[:, 0] >= -8.)
                                & (direct_pixels[:, 0] < width + 8.)
                                & (direct_pixels[:, 1] >= -8.)
                                & (direct_pixels[:, 1] < height + 8.)
                                & (direct_pixels[:, 0] >= target_box[0] - 16.)
                                & (direct_pixels[:, 0] <= target_box[2] + 16.)
                                & (direct_pixels[:, 1] >= target_box[1] - 16.)
                                & (direct_pixels[:, 1] <= target_box[3] + 16.))
                if np.count_nonzero(direct_valid) < 8:
                    continue
                direct_indices = direct_indices[direct_valid]
                direct_pixels = direct_pixels[direct_valid]
                direct_mask = np.zeros(len(view.points), bool)
                direct_mask[direct_indices] = True
                expanded_valid = valid & ~direct_mask
                if (len(direct_indices) + np.count_nonzero(expanded_valid)) < 12:
                    continue
                candidate_points = np.concatenate((view.points[direct_indices],
                                                   view.points[expanded_valid]), axis=0)
                candidate_pixels = np.concatenate((direct_pixels, mapped[expanded_valid]), axis=0)
                candidates.append((name, candidate_points, candidate_pixels))
            candidates.sort(key=lambda entry: len(entry[1]), reverse=True)
            return solve_candidates(candidates[:6], min_inlier_ratio=.35)

        # Fall back only after a feature family fails every acceptance gate,
        # including reference support. A weak AKAZE PnP candidate must not
        # suppress a potentially valid SIFT or combined result for this image.
        verified_by_reference = {}
        for kind in ('akaze', 'sift', 'orb', 'combined'):
            poses = combined_candidates() if kind == 'combined' else search_kind(kind)
            if cross_family_support:
                # Descriptor families are alternative measurements, not separate
                # references. Retain independent views that passed ALL gates in
                # earlier families; count each reference at most once.
                for candidate in poses:
                    previous = verified_by_reference.get(candidate.reference)
                    if previous is None or (candidate.inlier_ratio, -candidate.rms_px,
                                              len(candidate.points)) > (
                            previous.inlier_ratio, -previous.rms_px, len(previous.points)):
                        verified_by_reference[candidate.reference] = candidate
                poses = list(verified_by_reference.values())
            def pose_key(item):
                # Once a track exists, continuity is stronger evidence than a
                # tiny RMS advantage.  The old key preferred whichever PnP
                # branch happened to have the lower residual, which is exactly
                # how the 180-degree branch won on frame 821 in the recorded
                # session.
                if prior_pose is None:
                    return (0., -item.inlier_ratio, item.rms_px, -len(item.points))
                previous_axis = project_points(CONTROL_POINTS, prior_pose.rvec,
                                               prior_pose.tvec, self.K, self.dist)
                current_axis = project_points(CONTROL_POINTS, item.rvec,
                                              item.tvec, self.K, self.dist)
                old_vector = previous_axis[1] - previous_axis[0]
                new_vector = current_axis[1] - current_axis[0]
                old_length = np.linalg.norm(old_vector)
                new_length = np.linalg.norm(new_vector)
                if old_length < 20. or new_length < 20.:
                    directed_jump = 0.
                else:
                    cosine = np.clip(np.dot(old_vector, new_vector) /
                                     (old_length * new_length), -1., 1.)
                    directed_jump = float(np.degrees(np.arccos(cosine)))
                old_center = previous_axis[0]
                new_center = current_axis[0]
                center_jump = float(np.linalg.norm(new_center - old_center))
                return (directed_jump, center_jump, -item.inlier_ratio,
                        item.rms_px, -len(item.points))

            for pose in sorted(poses, key=pose_key):
                # Repeated rails sometimes give a well-fitting but shifted PnP.
                # Small match sets need a second reference agreeing on metric
                # center and undirected long axis; equivalent roll is ignored.
                rotation, _ = cv2.Rodrigues(pose.rvec)
                support = 1
                for other in poses:
                    if other is pose or other.reference == pose.reference:
                        continue
                    other_rotation, _ = cv2.Rodrigues(other.rvec)
                    axis_dot = abs(float(np.dot(rotation[:, 2], other_rotation[:, 2])))
                    if (np.linalg.norm(pose.tvec - other.tvec) <= 12.
                            and axis_dot >= np.cos(np.deg2rad(12.))):
                        support += 1
                if len(pose.points) < 20 and support < 2:
                    continue
                return RgbPose(pose.rvec, pose.tvec, pose.points, pose.pixels,
                               pose.reference, pose.rms_px, pose.inlier_ratio,
                               (time.perf_counter() - started) * 1000., support)
        for pose in sorted(homography_candidates(),
                           key=lambda item: (-item.inlier_ratio, item.rms_px,
                                             -len(item.points))):
            return RgbPose(pose.rvec, pose.tvec, pose.points, pose.pixels,
                           pose.reference, pose.rms_px, pose.inlier_ratio,
                           (time.perf_counter() - started) * 1000.,
                           pose.reference_support)
        return None


class RgbFeatureEvidence:
    """Short-lived measured correspondences for one descriptor family."""

    def __init__(self):
        self.gray = None
        self.box = None
        self.evidence = {}
        self.updated_at = 0.
        self.frames = 0

    def clear(self):
        self.gray = self.box = None
        self.evidence = {}
        self.frames = 0

    def track(self, gray, box):
        now = time.monotonic()
        box = np.asarray(box)
        if (self.gray is None or self.box is None or self.gray.shape != gray.shape
                or now - self.updated_at > .5 or self.frames >= 6):
            RgbFeatureEvidence.clear(self)
            return {}
        size = np.maximum(np.asarray(box)[2:] - box[:2], 1.)
        old_size = np.maximum(self.box[2:] - self.box[:2], 1.)
        shift = np.linalg.norm((np.asarray(box)[:2] + box[2:] - self.box[:2] - self.box[2:]) * .5)
        if shift > max(40., np.linalg.norm(size) * .4) or np.max(np.abs(np.log(size / old_size))) > .4:
            RgbFeatureEvidence.clear(self)
            return {}
        keys = [(name, index) for name, values in self.evidence.items() for index in values]
        if not keys:
            return {}
        pixels = np.array([self.evidence[name][index] for name, index in keys],
                           np.float32).reshape(-1, 1, 2)
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, pixels, None, winSize=(21, 21), maxLevel=3)
        if forward is None or status is None:
            return {}
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, forward, None, winSize=(21, 21), maxLevel=3)
        if backward is None or back_status is None:
            return {}
        good = (status.ravel().astype(bool) & back_status.ravel().astype(bool) &
                (np.linalg.norm(backward - pixels, axis=2).ravel() <= 1.))
        current = forward.reshape(-1, 2)
        good &= (np.all(current >= np.asarray(box)[:2] - 3, axis=1) &
                 np.all(current <= np.asarray(box)[2:] + 3, axis=1))
        tracked = {}
        for (name, index), point, keep in zip(keys, current, good):
            if keep:
                tracked.setdefault(name, {})[index] = point
        return tracked

    def remember(self, gray, box, evidence):
        # Limit work and lifetime: these are measured image features, never a
        # predicted pose. Each accepted pose still passes the full PnP gates.
        best = sorted(evidence, key=lambda name: len(evidence[name]), reverse=True)[:6]
        self.evidence = {name: evidence[name] for name in best}
        self.gray, self.box = gray.copy(), np.asarray(box).copy()
        self.updated_at = time.monotonic()
        self.frames += 1


class RgbPoseSearch(RgbFeatureEvidence):
    """Keep AKAZE and SIFT evidence separate while accumulating moving features."""

    def __init__(self, initializer):
        super().__init__()
        self.initializer = initializer
        self.cross_family_support = False
        self.K, self.dist, self.vertices = initializer.K, initializer.dist, initializer.vertices
        self.sift_evidence = RgbFeatureEvidence()
        self.orb_evidence = RgbFeatureEvidence()

    def clear(self):
        super().clear()
        self.sift_evidence.clear()
        self.orb_evidence.clear()

    def estimate(self, frame, box, validation_box=None, prior_pose=None):
        options = dict(cross_family_support=True) if self.cross_family_support else {}
        return self.initializer.estimate(frame, box, temporal=self,
                                         validation_box=validation_box,
                                         prior_pose=prior_pose, **options)


def verified_reference_paths(root: Path, *, pattern='*/attempt_*/rgb_initial_pose.json',
                             require_full_depth=False):
    """Find captures with accepted depth and a still RGB/depth handoff.

    Newly included collection captures must have a full depth solution; a
    partial solution requiring RGB verification is not an independent reference.
    """
    result = []
    for path in root.glob(pattern):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            stillness = data.get('rgb_stillness')
            quality = data.get('depth_pose_quality', {})
            if require_full_depth and (quality.get('partial_depth') is not False
                                       or quality.get('requires_rgb_verification') is not False):
                continue
            if (quality.get('accepted') is True
                    and isinstance(stillness, dict)
                    and not stillness.get('motion_tolerated', False)
                    and stillness.get('global_response', 0) >= .35
                    and np.linalg.norm(stillness.get('global_shift_px', [1e6, 1e6])) <= 3.
                    and np.linalg.norm(stillness.get('target_shift_px', [1e6, 1e6])) <= 2.):
                result.append(path)
        except (OSError, ValueError, TypeError):
            continue
    return sorted(result, key=lambda path: path.stat().st_mtime_ns, reverse=True)[:48]


def default_reference_paths():
    """Include verified captures from both recording entry points."""
    paths = verified_reference_paths(DATA_ROOT / 'tracking/live_depth_6dof')
    paths += verified_reference_paths(
        DATA_ROOT / 'tracking/targeted_collection',
        pattern='*/live/attempt_*/rgb_initial_pose.json', require_full_depth=True)
    return sorted(set(paths), key=lambda path: path.stat().st_mtime_ns, reverse=True)[:48]


def prepare_initializer(model: Path, intrinsics: Path,
                        root: Path | None = None) -> RgbGlobalInitializer:
    K, dist, size = load_calibration(intrinsics)
    vertices, _, _ = make_edge_points(model, 700, 3)
    paths = default_reference_paths() if root is None else verified_reference_paths(root)
    detectors = {
        'akaze': cv2.AKAZE_create(threshold=.0001),
        'sift': cv2.SIFT_create(nfeatures=1600),
        'orb': cv2.ORB_create(nfeatures=2500, fastThreshold=5),
    }
    seeder = None
    views = []
    seen_images = set()
    cache = DATA_ROOT / 'cache/rgb_pose_features'
    cache.mkdir(parents=True, exist_ok=True)
    calibration_id = hashlib.sha256(K.tobytes() + dist.tobytes()).hexdigest()
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            image_path = Path(data['rgb_image'])
            reference_K, reference_dist, reference_size = load_calibration(Path(data['intrinsics']))
            if (reference_size != size or not np.array_equal(reference_K, K)
                    or not np.array_equal(reference_dist, dist)):
                continue
            diagnostics_path = path.parent / 'depth_fit/diagnostics.json'
            diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8'))
            if Path(diagnostics['input']['model']).resolve() != model.resolve():
                continue
            frame = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[1::-1] != size:
                continue
            image_id = hashlib.sha256(np.ascontiguousarray(frame)).hexdigest()
            if image_id in seen_images:
                continue
            seen_images.add(image_id)
            rvec = tvec = mask = None
            for kind, detector in detectors.items():
                # Cache each descriptor family independently. Version 4 also
                # stores reference pixels for the ORB homography fallback.
                source = str((path.resolve(), path.stat().st_mtime_ns,
                              image_path.stat().st_mtime_ns, image_path.stat().st_size,
                              model.resolve(), model.stat().st_mtime_ns, calibration_id,
                              4, kind))
                cache_path = cache / (hashlib.sha256(source.encode()).hexdigest() + '.npz')
                cached = False
                if cache_path.is_file():
                    try:
                        with np.load(cache_path, allow_pickle=False) as stored:
                            if 'pixels' in stored:
                                points = stored['points'].copy()
                                descriptors = stored['descriptors'].copy()
                                pixels = stored['pixels'].copy()
                                cached = _reference_view_arrays_are_aligned(
                                    points, descriptors, pixels)
                    except (OSError, ValueError, KeyError):
                        cached = False
                if not cached:
                    if mask is None:
                        rvec = np.array(data['rvec_rgb_model_rad'], np.float64)
                        tvec = np.array(data['tvec_rgb_model_mm'], np.float64)
                        if seeder is None:
                            seeder = VisibleSurfaceSeeder(model, K, dist)
                        mask = seeder.mask(frame.shape, vertices, rvec, tvec)
                    keys, descriptors = detector.detectAndCompute(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
                    if descriptors is None or len(keys) < 12:
                        # Persist a valid empty result. This prevents every
                        # initializer rebuild from rerunning the expensive
                        # surface intersection for a descriptor family that
                        # has no usable support in this reference image.
                        descriptor_width = (0 if descriptors is None else descriptors.shape[1])
                        descriptor_dtype = (np.uint8 if kind in ('akaze', 'orb')
                                            else np.float32)
                        empty_descriptors = np.empty((0, descriptor_width), descriptor_dtype)
                        np.savez_compressed(cache_path,
                                            points=np.empty((0, 3), np.float64),
                                            descriptors=empty_descriptors,
                                            pixels=np.empty((0, 2), np.float32))
                        continue
                    pixels = np.array([key.pt for key in keys])
                    points, valid = seeder.intersect(pixels, rvec, tvec)
                    points, descriptors, pixels = (points[valid], descriptors[valid],
                                                   pixels[valid])
                    np.savez_compressed(cache_path, points=points,
                                        descriptors=descriptors, pixels=pixels)
                if (len(points) >= 12
                        and _reference_view_arrays_are_aligned(points, descriptors, pixels)):
                    views.append(ReferenceView(str(path.resolve()), points, descriptors,
                                               kind, image_id, pixels))
        except (OSError, ValueError, KeyError, RuntimeError, cv2.error):
            continue
    return RgbGlobalInitializer(K, dist, vertices, views)


def advance_pose(pose: RgbPose, source, current, K, dist, vertices, box, *,
                 retry_pnp=False, prior_pose=None):
    """Track matched pixels to the newest frame with a temporal PnP prior."""
    started = time.perf_counter()
    first = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    pixels = pose.pixels.reshape(-1, 1, 2)
    height, width = current.shape[:2]
    forward = good = None
    for win_size, max_level, fb_limit in (
            ((31, 31), 3, 1.),
            ((41, 41), 4, 1.5)):
        candidate_forward, valid, _ = cv2.calcOpticalFlowPyrLK(
            first, second, pixels, None, winSize=win_size, maxLevel=max_level)
        if candidate_forward is None or valid is None:
            continue
        backward, back_valid, _ = cv2.calcOpticalFlowPyrLK(
            second, first, candidate_forward, None,
            winSize=win_size, maxLevel=max_level)
        if backward is None or back_valid is None:
            continue
        candidate_good = (valid.ravel().astype(bool) & back_valid.ravel().astype(bool)
                          & (np.linalg.norm(backward - pixels, axis=2).ravel()
                             <= fb_limit))
        candidate_good &= (np.all(candidate_forward[:, 0] >= [0, 0], axis=1) &
                           np.all(candidate_forward[:, 0] < [width, height], axis=1))
        if np.count_nonzero(candidate_good) < 12:
            continue
        forward, good = candidate_forward, candidate_good
        break
    if forward is None or good is None:
        return None
    # The detector box belongs to source; move it with the measured features
    # for geometric validation. With detector-free global search there is no
    # box to advance, and the independent PnP/CAD gates remain authoritative.
    motion_shift = np.median(forward[good, 0] - pixels[good, 0], axis=0).astype(np.float32)
    current_box = None
    if box is not None:
        current_box = np.asarray(box) + np.tile(motion_shift, 2)
    arguments = (pose.points[good], forward[good, 0], K, dist, vertices,
                 current_box, pose.reference)
    # Temporal updates are the high-rate tracking path. They must stay in the
    # basin of the previous measured pose; otherwise EPNP/RANSAC can select the
    # geometrically equivalent 180-degree branch even with sub-pixel residuals.
    # ITERATIVE is useful here only when it receives the previous pose as its
    # extrinsic guess.
    prior_pose = pose if prior_pose is None else prior_pose
    method = cv2.SOLVEPNP_ITERATIVE if prior_pose is not None else cv2.SOLVEPNP_EPNP
    current_pose = solve_correspondences(
        *arguments, image_shape=current.shape, pnp_method=method,
        prior_pose=prior_pose, refine_inliers=False)
    if current_pose is None and prior_pose is not None:
        # Keep a measured fallback for difficult blur/occlusion frames. The
        # caller still applies long-axis and box-motion gates before accepting
        # this branch.
        current_pose = solve_correspondences(
            *arguments, image_shape=current.shape, pnp_method=cv2.SOLVEPNP_EPNP,
            refine_inliers=False)
    if current_pose is None:
        return None
    return RgbPose(current_pose.rvec, current_pose.tvec, current_pose.points,
                   current_pose.pixels, current_pose.reference, current_pose.rms_px,
                   current_pose.inlier_ratio,
                   pose.elapsed_ms + (time.perf_counter() - started) * 1000.,
                   pose.reference_support, motion_shift)
