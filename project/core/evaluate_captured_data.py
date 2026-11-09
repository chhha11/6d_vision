"""Evaluate captured RGB-D samples without using their manual ROI for detection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .depth_pose_handoff import compose_pose

from .paths import CALIBRATION_ROOT, MODEL_ROOT, PROJECT_ROOT, SAMPLE_ROOT, TRACKING_ROOT

DEFAULT_MODEL = MODEL_ROOT / "energy_unit_centered_mm.stl"
DEFAULT_EXTRINSICS = CALIBRATION_ROOT / "rgb_ir" / "rgb_ir_extrinsics.npz"
SAMPLE_NAME = re.compile(r"(?:P\d{3}|N\d{3}|sample_\d{4})\Z")
BOARD_NAME = re.compile(r"sample_\d{4}\Z")
SESSION_NAME = re.compile(r"session_\d{2}\Z")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    group: str
    folder: Path
    manifest: dict
    session_id: str | None = None

    @property
    def kind(self) -> str:
        return self.manifest["sample"]["kind"]

    @property
    def target_present(self) -> bool:
        return bool(self.manifest["target_present"])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Return a file hash on Python versions before hashlib.file_digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_samples(root: Path, groups: set[str]) -> list[Sample]:
    samples: list[Sample] = []
    for group, folder_name, pattern in (
        ("global", "global_dataset", SAMPLE_NAME),
        ("board", "board_reference", BOARD_NAME),
    ):
        if group not in groups:
            continue
        dataset = root / folder_name
        for folder in sorted(dataset.iterdir()):
            if not folder.is_dir() or not pattern.fullmatch(folder.name):
                continue
            manifest = read_json(folder / "manifest.json")
            if manifest.get("accepted"):
                samples.append(Sample(f"{group}/{folder.name}", group, folder, manifest))

    if "integration" in groups:
        dataset = root / "integration_sessions"
        for session in sorted(dataset.iterdir()):
            if not session.is_dir() or not SESSION_NAME.fullmatch(session.name):
                continue
            pairing = read_json(session / "session_manifest.json")
            for anchor, label in (("start_rgb_d", "start"), ("end_rgb_d", "end")):
                entry = pairing[anchor]
                manifest_path = session / entry["manifest"]
                manifest = read_json(manifest_path)
                if entry.get("accepted") and manifest.get("accepted"):
                    samples.append(Sample(
                        f"integration/{session.name}/{label}", "integration",
                        manifest_path.parent, manifest, session.name,
                    ))
    return samples


def calibration_path(sample: Sample, fallback: Path) -> tuple[Path, str]:
    capture_calibration = sample.manifest.get("rgb_depth_calibration")
    if not capture_calibration:
        return fallback, "shared_calibration"
    path = sample.folder / capture_calibration["file"]
    expected_hash = capture_calibration.get("sha256")
    if expected_hash:
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Calibration snapshot SHA-256 mismatch: {path}")
    return path, "capture_snapshot"


def bbox_iou(roi_xywh: list[float], bbox_xyxy: list[float]) -> float:
    x, y, width, height = (float(v) for v in roi_xywh)
    left, top, right, bottom = (float(v) for v in bbox_xyxy)
    intersection = max(0.0, min(x + width, right) - max(x, left)) * max(
        0.0, min(y + height, bottom) - max(y, top))
    union = width * height + max(0.0, right - left) * max(0.0, bottom - top) - intersection
    return intersection / union if union > 0 else 0.0


def projection_check(T_rgb_model: np.ndarray, K: np.ndarray, dist: np.ndarray,
                     vertices: np.ndarray, roi: list[int]) -> dict[str, object]:
    camera_vertices = vertices @ T_rgb_model[:3, :3].T + T_rgb_model[:3, 3]
    visible = camera_vertices[:, 2] > 1.0
    if not np.any(visible):
        raise ValueError("CAD model lies entirely behind the RGB camera.")
    rvec, _ = cv2.Rodrigues(T_rgb_model[:3, :3])
    projected, _ = cv2.projectPoints(
        vertices[visible], rvec, T_rgb_model[:3, 3], K, dist)
    projected = projected.reshape(-1, 2)
    bbox = [float(projected[:, 0].min()), float(projected[:, 1].min()),
            float(projected[:, 0].max()), float(projected[:, 1].max())]
    center, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64), rvec,
        T_rgb_model[:3, 3], K, dist)
    center_xy = center.reshape(2)
    x, y, width, height = (float(v) for v in roi)
    roi_center = np.array([x + width / 2, y + height / 2])
    return {
        "projected_center_px": center_xy.tolist(),
        "projected_bbox_xyxy": bbox,
        "roi_center_error_px": float(np.linalg.norm(center_xy - roi_center)),
        "roi_center_inside": bool(x <= center_xy[0] <= x + width
                                  and y <= center_xy[1] <= y + height),
        "roi_bbox_iou": bbox_iou(roi, bbox),
    }


def evaluate_sample(sample: Sample, output: Path, model_path: Path,
                    fallback_extrinsics: Path, vertices: np.ndarray,
                    settings: object) -> dict[str, object]:
    from .depth_global_initializer import run

    row: dict[str, object] = {
        "sample_id": sample.sample_id,
        "group": sample.group,
        "kind": sample.kind,
        "target_present": sample.target_present,
        "session_id": sample.session_id,
        "rgb_roi_xywh": sample.manifest.get("rgb_roi_xywh"),
        "rgb_projection_status": "not_attempted",
    }
    result_dir = output / "samples" / sample.sample_id
    diagnostics_path = result_dir / "diagnostics.json"
    for name in ("diagnostics.json", "selected_candidate.ply", "aligned_model.ply",
                 "segmentation_preview.png", "registration_preview.png",
                 "rgb_tracker_pose.json"):
        (result_dir / name).unlink(missing_ok=True)
    try:
        diagnostics = run(
            sample.folder / "depth" / "depth_burst.npz",
            sample.folder / "depth" / "metadata.json",
            model_path, result_dir, settings,
        )
    except RuntimeError as exc:
        new_diagnostics = diagnostics_path.exists()
        if new_diagnostics:
            details = read_json(diagnostics_path)
            row["diagnostics"] = str(diagnostics_path.resolve())
            row["rejection_reason"] = details.get("rejection_reason")
        if (new_diagnostics and details.get("rejection_reason")
                and details.get("pose_quality", {}).get("accepted") is False):
            row["status"] = "no_candidate"
            return row
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return row
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return row

    quality = diagnostics["pose_quality"]
    pose = diagnostics["pose"]
    row.update(
        status="accepted" if quality["accepted"] else "rejected_quality",
        candidate_points=diagnostics["selected_cluster"]["point_count"],
        cluster_eps_used_mm=diagnostics["cluster_eps_used_mm"],
        icp_fitness=pose["fitness"],
        icp_rmse_mm=pose["inlier_rmse_mm"],
        depth_center_mm=pose["translation_mm"],
        diagnostics=str(diagnostics_path.resolve()),
    )
    if not sample.target_present:
        return row
    try:
        extrinsics_path, source = calibration_path(sample, fallback_extrinsics)
        with np.load(extrinsics_path) as extrinsics:
            T_rgb_depth = extrinsics["T_rgb_depth"].copy()
            K_rgb = extrinsics["K_rgb"].copy()
            dist_rgb = extrinsics["dist_rgb"].copy()
            image_size = tuple(int(value) for value in extrinsics["image_size"])
        image = cv2.imdecode(np.fromfile(sample.folder / "rgb.png", dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None or (image.shape[1], image.shape[0]) != image_size:
            raise ValueError(f"RGB image and calibration size differ: {sample.folder}")
        T_rgb_model = compose_pose(
            T_rgb_depth, np.asarray(pose["model_to_camera"], dtype=np.float64))
        row.update(
            calibration_source=source,
            rgb_center_mm=T_rgb_model[:3, 3].tolist(),
            T_rgb_model=T_rgb_model.tolist(),
        )
        roi = sample.manifest.get("rgb_roi_xywh")
        if roi is not None:
            row.update(projection_check(T_rgb_model, K_rgb, dist_rgb, vertices, roi))
            row["rgb_projection_status"] = "checked"
        else:
            row["rgb_projection_status"] = "not_annotated"
        if (quality["accepted"] and sample.kind == "positive"
                and row.get("roi_center_inside") is True):
            rvec, _ = cv2.Rodrigues(T_rgb_model[:3, :3])
            tracker_pose = result_dir / "rgb_tracker_pose.json"
            tracker_pose.write_text(json.dumps({
                "source_sample_id": sample.sample_id,
                "definition": "T_rgb_model = T_rgb_depth @ T_depth_model",
                "T_rgb_model": T_rgb_model.tolist(),
                "rvec_rgb_model_rad": rvec.reshape(3).tolist(),
                "tvec_rgb_model_mm": T_rgb_model[:3, 3].tolist(),
                "depth_pose_quality": quality,
                "roi_center_inside": True,
                "calibration_source": source,
                "use": "tracking_initialization_only",
                "limitation": "Long-axis roll is ambiguous; this is not 6DoF ground truth.",
            }, indent=2), encoding="utf-8")
            row["tracker_pose"] = str(tracker_pose.resolve())
    except Exception as exc:
        row.update(rgb_projection_status="error",
                   rgb_projection_error=f"{type(exc).__name__}: {exc}")
    return row


def summarize(rows: list[dict], board_report: dict | None = None) -> dict[str, object]:
    def count(items: list[dict], status: str) -> int:
        return sum(row["status"] == status for row in items)

    positives = [row for row in rows if row["group"] == "global"
                 and row["kind"] == "positive"]
    absent = [row for row in rows if row["group"] == "global"
              and row["kind"] == "negative" and not row["target_present"]]
    present_negatives = [row for row in rows if row["group"] == "global"
                         and row["kind"] == "negative" and row["target_present"]]
    board = [row for row in rows if row["group"] == "board"]
    integration = [row for row in rows if row["group"] == "integration"]
    summary: dict[str, object] = {
        "total_samples": len(rows),
        "errors": sum(row["status"] == "error" or row.get("rgb_projection_status") == "error"
                      for row in rows),
        "depth_errors": count(rows, "error"),
        "rgb_projection_errors": sum(row.get("rgb_projection_status") == "error"
                                     for row in rows),
        "global_positive": {
            "samples": len(positives),
            "depth_pose_accepted": count(positives, "accepted"),
            "accepted_center_inside_manual_roi": sum(
                row["status"] == "accepted" and row.get("roi_center_inside") is True
                for row in positives),
        },
        "global_target_absent_negative": {
            "samples": len(absent),
            "false_acceptances": count(absent, "accepted"),
        },
        "global_target_present_negative": {
            "samples": len(present_negatives),
            "accepted_poses": count(present_negatives, "accepted"),
            "note": "These contain the target; accepted poses are not false positives.",
        },
        "board": {"samples": len(board), "depth_pose_accepted": count(board, "accepted")},
        "integration": {
            "anchors": len(integration),
            "depth_pose_accepted": count(integration, "accepted"),
            "sessions_with_both_anchors_accepted": sum(
                all(row["status"] == "accepted" for row in integration
                    if row["session_id"] == session_id)
                for session_id in sorted({row["session_id"] for row in integration})
                if sum(row["session_id"] == session_id for row in integration) == 2),
        },
    }
    if board_report is not None:
        board_transforms = {
            sample["sample_id"]: np.asarray(
                sample["T_first_camera_current_camera"], dtype=np.float64)
            for sample in board_report["samples"]
            if sample.get("board_detected")
            and sample.get("rgb_depth_pair_stable") is True
            and sample.get("corner_order_consistent_with_first") is True
            and "T_first_camera_current_camera" in sample
        }
        centers = []
        axes = []
        for row in board:
            name = row["sample_id"].split("/")[-1]
            if (row["status"] != "accepted" or name not in board_transforms
                    or "T_rgb_model" not in row):
                continue
            T_first_model = board_transforms[name] @ np.asarray(
                row["T_rgb_model"], dtype=np.float64)
            centers.append(T_first_model[:3, 3])
            axes.append(T_first_model[:3, 2])
        if len(centers) >= 2:
            center_array = np.stack(centers)
            center_reference = np.median(center_array, axis=0)
            errors = np.linalg.norm(center_array - center_reference, axis=1)
            axis_array = np.stack(axes)
            axis_reference = axis_array[0]
            axis_angles = np.rad2deg(np.arccos(np.clip(
                np.abs(axis_array @ axis_reference), 0.0, 1.0)))
            summary["board"].update(
                comparable_poses=len(centers),
                center_scatter_median_mm=float(np.median(errors)),
                center_scatter_p95_mm=float(np.percentile(errors, 95)),
                axis_line_scatter_median_deg=float(np.median(axis_angles)),
                note="Relative consistency for a fixed target, not absolute pose error.",
            )
    return summary


def endpoint_center_check(tracker_center_mm: list[float], anchor_center_mm: list[float],
                          tolerance_mm: float) -> dict[str, object]:
    tracker = np.asarray(tracker_center_mm, dtype=np.float64)
    anchor = np.asarray(anchor_center_mm, dtype=np.float64)
    if (tracker.shape != (3,) or anchor.shape != (3,)
            or not np.isfinite(tracker).all() or not np.isfinite(anchor).all()):
        raise ValueError("Tracker and endpoint centers must be finite XYZ vectors.")
    delta = tracker - anchor
    error = float(np.linalg.norm(delta))
    return {
        "end_anchor_center_delta_xyz_mm": delta.tolist(),
        "end_anchor_center_error_mm": error,
        "end_anchor_center_check_passed": error <= tolerance_mm,
    }


def endpoint_axis_line_check(tracker_rvec: list[float], anchor_transform: list,
                             tolerance_deg: float) -> dict[str, object]:
    rvec = np.asarray(tracker_rvec, dtype=np.float64)
    anchor = np.asarray(anchor_transform, dtype=np.float64)
    if (rvec.shape != (3,) or anchor.shape != (4, 4)
            or not np.isfinite(rvec).all() or not np.isfinite(anchor).all()):
        raise ValueError("Tracker rotation and endpoint transform must be finite.")
    rotation, _ = cv2.Rodrigues(rvec)
    cosine = float(np.clip(abs(rotation[:, 2] @ anchor[:3, 2]), 0.0, 1.0))
    error = float(np.rad2deg(np.arccos(cosine)))
    return {
        "end_anchor_axis_line_error_deg": error,
        "end_anchor_axis_line_check_passed": error <= tolerance_deg,
    }


def replay_integration_sessions(data_root: Path, output: Path, rows: list[dict],
                                model: Path, intrinsics: Path,
                                endpoint_tolerance_mm: float,
                                axis_tolerance_deg: float) -> list[dict[str, object]]:
    starts = {row["session_id"]: row for row in rows
              if row["group"] == "integration" and row["sample_id"].endswith("/start")}
    ends = {row["session_id"]: row for row in rows
            if row["group"] == "integration" and row["sample_id"].endswith("/end")}
    results: list[dict[str, object]] = []
    for session_id, start in sorted(starts.items()):
        result: dict[str, object] = {"session_id": session_id}
        replay_dir = output / "tracking" / session_id
        summary_path = replay_dir / "summary.json"
        for name in ("summary.json", "frames.csv", "final_pose.json"):
            (replay_dir / name).unlink(missing_ok=True)
        if not start.get("tracker_pose"):
            result["status"] = "skipped_no_verified_start_pose"
            results.append(result)
            continue
        session = data_root / "integration_sessions" / session_id
        pairing = read_json(session / "session_manifest.json")
        video = session / pairing["motion_rgb"]["video"]
        command = [
            sys.executable, "-m", "core.cad_video_tracker",
            "--video", str(video.resolve()), "--headless",
            "--pose", start["tracker_pose"],
            "--model", str(model.resolve()),
            "--intrinsics", str(intrinsics.resolve()),
            "--no-static-correction",
            "--output", str(replay_dir / "final_pose.json"),
            "--log", str(replay_dir / "frames.csv"),
            "--summary", str(summary_path),
        ]
        replay_dir.mkdir(parents=True, exist_ok=True)
        print(f"Replaying {session_id}", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True,
                                   check=False)
        fresh_summary = summary_path.exists()
        if completed.returncode != 0 or not fresh_summary:
            result.update(status="error", error=(completed.stderr or completed.stdout).strip(),
                          returncode=completed.returncode)
        else:
            tracker_summary = read_json(summary_path)
            result.update(
                status="complete",
                video=str(video.resolve()),
                summary=str(summary_path.resolve()),
                frames=tracker_summary["frames"],
                tracking_frames=tracker_summary["tracking_frames"],
                video_reached_eof=tracker_summary["video_reached_eof"],
                final_tracking_valid=tracker_summary["final_pose"]["tracking_valid"],
                end_nominal_depth_error_mm=tracker_summary["end_nominal_depth_error_mm"],
                end_nominal_depth_check_raw=tracker_summary[
                    "end_nominal_depth_check_passed"],
                end_nominal_depth_check_passed=(
                    tracker_summary["end_nominal_depth_check_passed"]
                    if (tracker_summary["video_reached_eof"]
                        and tracker_summary["final_pose"]["tracking_valid"]) else None),
            )
            end = ends.get(session_id)
            if not (result["video_reached_eof"] and result["final_tracking_valid"]):
                result["endpoint_check_status"] = "skipped_untrusted_final_pose"
            elif end and end.get("status") == "accepted" and "rgb_center_mm" in end:
                result.update(endpoint_center_check(
                    tracker_summary["final_pose"]["tvec_mm"],
                    end["rgb_center_mm"], endpoint_tolerance_mm))
                result.update(endpoint_axis_line_check(
                    tracker_summary["final_pose"]["rvec"],
                    end["T_rgb_model"], axis_tolerance_deg))
                result["end_anchor_center_and_axis_consistent"] = bool(
                    result["end_anchor_center_check_passed"]
                    and result["end_anchor_axis_line_check_passed"])
                result["end_anchor_pose"] = end.get("tracker_pose")
                result["endpoint_pairing_review"] = pairing.get("pairing_review")
                result["endpoint_check_status"] = "checked"
            else:
                result["endpoint_check_status"] = "skipped_no_end_anchor_pose"
        results.append(result)
    return results


def write_report(output: Path, rows: list[dict], summary: dict,
                 selection: list[str], tracking: list[dict] | None = None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "selection": selection,
        "method": (
            "Blind depth initialization. RGB manual ROI is used for post-hoc projection "
            "checks and gates which accepted start poses are passed to video replay."
        ),
        "limitations": (
            "ICP fit and ROI agreement are not absolute 6DoF ground truth. "
            "Target-present negative captures are reported separately."
        ),
        "summary": summary,
        "samples": rows,
    }
    if tracking is not None:
        report["tracking"] = tracking
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = (
        "sample_id", "group", "kind", "target_present", "session_id", "status",
        "candidate_points", "cluster_eps_used_mm", "icp_fitness", "icp_rmse_mm",
        "rgb_projection_status", "roi_center_error_px", "roi_center_inside",
        "roi_bbox_iou", "diagnostics", "rejection_reason", "error",
        "rgb_projection_error", "tracker_pose",
    )
    with (output / "samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--groups", nargs="+", choices=("global", "board", "integration"),
                        default=("global", "board", "integration"))
    parser.add_argument("--sample", action="append", help="Run one sample ID, e.g. global/P001")
    parser.add_argument("--track-videos", action="store_true",
                        help="Replay paired integration videos from accepted start poses")
    parser.add_argument("--endpoint-center-tolerance-mm", type=float, default=50.0,
                        help="Tolerance for video final center vs independent end RGB-D anchor")
    parser.add_argument("--endpoint-axis-tolerance-deg", type=float, default=15.0,
                        help="Tolerance for the sign-invariant CAD long-axis line at video end")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--intrinsics", type=Path,
                        default=CALIBRATION_ROOT / "camera_intrinsics.npz")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.track_videos and "integration" not in args.groups:
        parser.error("--track-videos requires the integration group.")
    if args.endpoint_center_tolerance_mm <= 0:
        parser.error("--endpoint-center-tolerance-mm must be positive.")
    if not 0 < args.endpoint_axis_tolerance_deg <= 90:
        parser.error("--endpoint-axis-tolerance-deg must be in (0, 90].")

    from .cad_pose_overlay import load_edges
    from .depth_global_initializer import Settings

    output = args.output or TRACKING_ROOT / "evaluations" / datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    samples = discover_samples(args.data_root, set(args.groups))
    if args.sample:
        unknown = set(args.sample) - {sample.sample_id for sample in samples}
        if unknown:
            parser.error(f"Unknown accepted sample IDs: {', '.join(sorted(unknown))}")
        samples = [sample for sample in samples if sample.sample_id in args.sample]
    if not samples:
        parser.error("No accepted samples selected.")
    if args.track_videos and not any(
            sample.group == "integration" and sample.sample_id.endswith("/start")
            for sample in samples):
        parser.error("--track-videos requires at least one selected integration start anchor.")
    vertices, _ = load_edges(args.model, 1)
    vertices = np.unique(vertices, axis=0)
    rows = []
    for index, sample in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] {sample.sample_id}", flush=True)
        row = evaluate_sample(sample, output, args.model, args.extrinsics,
                              vertices, Settings())
        rows.append(row)
        print(f"  {row['status']}" +
              (f"; {row['error']}" if row.get("error") else ""), flush=True)
    if "board" in args.groups:
        from .analyze_board_reference import analyze
        board_report = analyze(args.data_root / "board_reference", args.intrinsics)
    else:
        board_report = None
    summary = summarize(rows, board_report)
    tracking = None
    if args.track_videos:
        tracking = replay_integration_sessions(
            args.data_root, output, rows, args.model, args.intrinsics,
            args.endpoint_center_tolerance_mm, args.endpoint_axis_tolerance_deg)
        summary["integration"].update(
            video_replays=sum(item["status"] != "skipped_no_verified_start_pose"
                              for item in tracking),
            video_replay_skipped=sum(item["status"] == "skipped_no_verified_start_pose"
                                     for item in tracking),
            video_replay_complete=sum(item["status"] == "complete" for item in tracking),
            videos_reached_eof=sum(item.get("video_reached_eof") is True for item in tracking),
            final_tracking_valid=sum(item.get("final_tracking_valid") is True for item in tracking),
            end_nominal_depth_check_passed=sum(
                item.get("end_nominal_depth_check_passed") is True for item in tracking),
            end_anchor_center_check_passed=sum(
                item.get("end_anchor_center_check_passed") is True for item in tracking),
            end_anchor_axis_line_check_passed=sum(
                item.get("end_anchor_axis_line_check_passed") is True for item in tracking),
            end_anchor_center_and_axis_consistent=sum(
                item.get("end_anchor_center_and_axis_consistent") is True for item in tracking),
            endpoint_center_tolerance_mm=args.endpoint_center_tolerance_mm,
            endpoint_axis_tolerance_deg=args.endpoint_axis_tolerance_deg,
            video_replay_errors=sum(item["status"] == "error" for item in tracking),
            video_replay_note=(
                "End-anchor center agreement tests drift against a separate static RGB-D "
                "estimate near the video endpoint. The nominal depth check is constrained "
                "by the tracker's session depth gate and is not independent. Endpoint "
                "pairing is approximate; this is not absolute 6DoF accuracy."
            ),
        )
        summary["errors"] += (summary["integration"]["video_replay_errors"]
                              + summary["integration"]["video_replay_skipped"])
    report_path = write_report(output, rows, summary,
                               args.sample or list(args.groups), tracking)
    print(f"Saved {report_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
