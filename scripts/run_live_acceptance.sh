#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CAMERA_INDEX="${CAMERA_INDEX:-}"
CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"
CAMERA_FPS="${CAMERA_FPS:-30}"
DETECTOR_INTERVAL="${DETECTOR_INTERVAL:-2}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the project virtualenv Python." >&2
  exit 2
fi

cd "${ROOT_DIR}/project"
CAMERA_ARGS=()
if [[ -n "${CAMERA_INDEX}" ]]; then
  CAMERA_ARGS+=(--camera "${CAMERA_INDEX}")
fi
CAMERA_ARGS+=(--camera-fourcc "${CAMERA_FOURCC}")
CAMERA_ARGS+=(--camera-fps "${CAMERA_FPS}")
CAMERA_ARGS+=(--detector-interval "${DETECTOR_INTERVAL}")
exec "${PYTHON_BIN}" -m core.live_depth_6dof "${CAMERA_ARGS[@]}" "$@"
