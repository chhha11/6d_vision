"""Canonical project paths shared by command-line tools and library modules.

The constants in this module describe storage boundaries only.  Runtime code
should keep its existing data flow and choose an output subdirectory explicitly;
centralizing the roots avoids slightly different defaults being introduced by
individual tools.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Reference data and model inputs.
CALIBRATION_ROOT = DATA_ROOT / "calibration"
MODEL_ROOT = DATA_ROOT / "models"
SAMPLE_ROOT = DATA_ROOT / "samples"
CACHE_ROOT = DATA_ROOT / "cache"
TRAINING_ROOT = DATA_ROOT / "runs"

# Runtime outputs.  These roots are intentionally separate from reference data
# so they can be cleaned without touching inputs used by tests or startup.
TRACKING_ROOT = DATA_ROOT / "tracking"
RGBD_CAPTURE_ROOT = DATA_ROOT / "rgbd_capture"
TMP_ROOT = DATA_ROOT / "tmp"

SDK_ROOT = PROJECT_ROOT / "sdk"

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "CALIBRATION_ROOT",
    "MODEL_ROOT",
    "SAMPLE_ROOT",
    "CACHE_ROOT",
    "TRAINING_ROOT",
    "TRACKING_ROOT",
    "RGBD_CAPTURE_ROOT",
    "TMP_ROOT",
    "SDK_ROOT",
]
