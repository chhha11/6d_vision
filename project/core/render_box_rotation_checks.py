"""Compare the roll-neutral display box on accepted static rotation captures."""
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cad_video_tracker import (BOX_DISPLAY_SETTINGS_PATH, BOX_DISPLAY_Y_OFFSET_PX,
                               load_box_display_y_offset, load_model_box,
                               render_box_layer)

from .paths import DATA_ROOT

DEFAULT_REPORT = DATA_ROOT / "tracking" / "evaluations" / "captured_20261031" / "report.json"
DEFAULT_OUTPUT = DATA_ROOT / "tracking" / "evaluations" / "captured_20261031" / "videos" / "box_rotation_checks.png"
DEFAULT_SAMPLES = ("P001", "P002", "P003", "P025")
CROP_WIDTH, CROP_HEIGHT = 340, 230
SCALE = 2


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    center_x: int
    center_y: int
    measurements: dict


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read RGB image: {path}")
    return image


def load_sample(sample_id: str, row: dict, data_root: Path) -> Sample:
    folder = data_root / "global_dataset" / sample_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if row["status"] != "accepted" or row.get("rgb_projection_status") != "checked":
        raise ValueError(f"Sample {sample_id} has no accepted RGB depth pose")
    image = read_rgb(folder / "rgb.png")
    calibration_path = folder / manifest["rgb_depth_calibration"]["file"]
    with np.load(calibration_path) as calibration:
        K = calibration["K_rgb"].copy()
        dist = calibration["dist_rgb"].copy()
        image_size = tuple(int(value) for value in calibration["image_size"])
    if image.shape[1::-1] != image_size:
        raise ValueError(f"Image size and calibration differ for {sample_id}")
    T = np.asarray(row["T_rgb_model"], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    x, y, w, h = manifest["rgb_roi_xywh"]
    center_x, center_y = round(x + w / 2), round(y + h / 2)
    axis = T[:3, 2]
    axis_ends = np.array([T[:3, 3] - 75 * axis, T[:3, 3] + 75 * axis])
    projected, _ = cv2.projectPoints(axis_ends, np.zeros(3), np.zeros(3), K, dist)
    direction = np.diff(projected.reshape(2, 2), axis=0).reshape(2)
    result = {
        "sample_id": sample_id,
        "nominal_pose_label": manifest["sample"]["pose"],
        "long_axis_camera": axis.tolist(),
        "projected_axis_length_px": float(np.linalg.norm(direction)),
        "projected_axis_angle_deg": float(np.degrees(np.arctan2(direction[1], direction[0]))),
        "projected_center_minus_roi_center_y_px": float(
            row["projected_center_px"][1] - (y + h / 2)),
    }
    return Sample(sample_id, image, K, dist, rvec.reshape(3), T[:3, 3],
                  center_x, center_y, result)


def render_sample(sample: Sample, box_corners: np.ndarray,
                  display_y_offset_px: float) -> np.ndarray:
    layer = render_box_layer(sample.image.shape, box_corners, sample.K, sample.dist,
                             sample.rvec, sample.tvec, display_y_offset_px)
    overlay = cv2.addWeighted(sample.image, 1.0, layer, 0.72, 0)
    pad_x, pad_y = CROP_WIDTH, CROP_HEIGHT
    padded = cv2.copyMakeBorder(overlay, pad_y, pad_y, pad_x, pad_x,
                                cv2.BORDER_CONSTANT, value=(30, 30, 30))
    left = sample.center_x + pad_x - CROP_WIDTH // 2
    top = sample.center_y + pad_y - CROP_HEIGHT // 2
    crop = padded[top:top + CROP_HEIGHT, left:left + CROP_WIDTH]
    return cv2.resize(crop, (CROP_WIDTH * SCALE, CROP_HEIGHT * SCALE),
                      interpolation=cv2.INTER_LINEAR)


def render_sheet(samples: list[Sample], box_corners: np.ndarray,
                 display_y_offset_px: float) -> np.ndarray:
    panels = []
    for sample in samples:
        crop = render_sample(sample, box_corners, display_y_offset_px)
        title = f"{sample.sample_id}  {sample.measurements['nominal_pose_label']}"
        panel = np.full((crop.shape[0] + 36, crop.shape[1], 3), 28, np.uint8)
        cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (240, 240, 240), 2, cv2.LINE_AA)
        panel[36:] = crop
        panels.append(panel)
    return np.concatenate((np.concatenate(panels[:2], axis=1),
                           np.concatenate(panels[2:], axis=1)), axis=0)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(path)


def show_adjustment_window(samples: list[Sample], box_corners: np.ndarray,
                           output: Path, initial_offset: float) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("Box alignment")
    root.configure(bg="#202629")
    toolbar = tk.Frame(root, bg="#202629", padx=14, pady=10)
    toolbar.pack(fill="x")
    tk.Label(toolbar, text="Vertical offset", fg="#e8eded", bg="#202629").pack(side="left")
    value_label = tk.Label(toolbar, width=8, fg="#50d4c7", bg="#202629")
    value_label.pack(side="left", padx=(12, 8))
    image_label = tk.Label(root, bg="#202629", bd=0)
    status_label = tk.Label(root, text="", anchor="w", fg="#aebabc", bg="#202629",
                            padx=14, pady=7)
    pending = None
    current_sheet = None

    def update_preview() -> None:
        nonlocal pending, current_sheet
        pending = None
        offset = int(slider.get())
        current_sheet = render_sheet(samples, box_corners, offset)
        max_width = max(640, root.winfo_screenwidth() - 100)
        max_height = max(480, root.winfo_screenheight() - 180)
        scale = min(1.0, max_width / current_sheet.shape[1],
                    max_height / current_sheet.shape[0])
        if scale < 1.0:
            preview = cv2.resize(current_sheet, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
        else:
            preview = current_sheet
        ok, encoded = cv2.imencode(".png", preview)
        if not ok:
            raise RuntimeError("Cannot encode box preview")
        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
        image_label.configure(image=photo)
        image_label.image = photo

    def on_slide(_value: str) -> None:
        nonlocal pending
        value_label.configure(text=f"{int(slider.get()):+d} px")
        status_label.configure(text="")
        if pending is not None:
            root.after_cancel(pending)
        pending = root.after(25, update_preview)

    slider = tk.Scale(toolbar, from_=-100, to=100, resolution=1,
                      orient="horizontal", showvalue=False, length=520,
                      bg="#202629", fg="#e8eded", troughcolor="#58666a",
                      highlightthickness=0, bd=0, command=on_slide)
    slider.pack(side="left", fill="x", expand=True, padx=(0, 14))

    def save_selection(_event=None) -> None:
        nonlocal current_sheet
        offset = int(slider.get())
        if pending is not None or current_sheet is None:
            current_sheet = render_sheet(samples, box_corners, offset)
        try:
            save_png(output, current_sheet)
            BOX_DISPLAY_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            BOX_DISPLAY_SETTINGS_PATH.write_text(
                json.dumps({"display_y_offset_px": offset}, indent=2) + "\n",
                encoding="utf-8")
        except (OSError, RuntimeError) as exc:
            status_label.configure(text=str(exc), fg="#ff9a90")
            return
        status_label.configure(text=f"Saved {offset:+d} px", fg="#83dec7")
        print(f"Saved {output.resolve()} and {BOX_DISPLAY_SETTINGS_PATH.resolve()}")

    tk.Button(toolbar, text="Reset", command=lambda: slider.set(BOX_DISPLAY_Y_OFFSET_PX),
              bg="#384448", fg="#f0f2f2", relief="flat", padx=12).pack(side="left", padx=4)
    tk.Button(toolbar, text="Save", command=save_selection,
              bg="#238c84", fg="#ffffff", relief="flat", padx=18).pack(side="left", padx=4)
    image_label.pack(padx=12)
    status_label.pack(fill="x")
    root.bind("<Control-s>", save_selection)
    root.bind("<Escape>", lambda _event: root.destroy())
    slider.set(round(initial_offset))
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT / "samples")
    parser.add_argument("--model", type=Path,
                        default=DATA_ROOT / "models" / "energy_unit_centered_mm.stl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", action="append", dest="samples")
    parser.add_argument("--box-display-y-offset-px", type=float,
                        help="Display-only vertical offset; defaults to saved setting")
    parser.add_argument("--interactive", action="store_true",
                        help="Open a four-view alignment window with a vertical offset slider")
    args = parser.parse_args()
    selected = args.samples or list(DEFAULT_SAMPLES)
    if len(selected) != 4:
        parser.error("Select exactly four samples for the comparison sheet")
    try:
        offset = (args.box_display_y_offset_px if args.box_display_y_offset_px is not None
                  else load_box_display_y_offset())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not np.isfinite(offset) or abs(offset) > 100:
        parser.error("Box display Y offset must be finite and within +/-100 pixels")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = {item["sample_id"].split("/")[-1]: item
            for item in report["samples"] if item["group"] == "global"}
    box_corners = load_model_box(args.model)
    samples = []
    for sample_id in selected:
        if sample_id not in rows:
            parser.error(f"Sample {sample_id} is absent from the report")
        samples.append(load_sample(sample_id, rows[sample_id], args.data_root))
    if args.interactive:
        show_adjustment_window(samples, box_corners, args.output, offset)
    else:
        save_png(args.output, render_sheet(samples, box_corners, offset))
        print(json.dumps([sample.measurements for sample in samples], indent=2))
        print(f"Saved {args.output.resolve()} with {offset:+g} px display offset")


if __name__ == "__main__":
    main()
