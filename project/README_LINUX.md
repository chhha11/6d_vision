# Linux migration notes

This directory is the Linux working copy of the original 6D Vision project.
The Windows-only virtual environment, DLLs, OpenNI configuration, and wheel
were intentionally left out.

## Current camera backend

The supported Linux target is Ubuntu 22.04. During development, connect the
Astra Pro Plus directly to the laptop. The current 640x480 RGB-D profile runs
at USB 2.0 bandwidth on this machine; a USB 3 hub is not required and can
make camera recovery less predictable.

The Astra Pro Plus is opened through the native OrbbecSDK backend in the
parent project:

```bash
cd /home/chhha/6d_vision
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/orbbec_rgbd_capture \
  --output project/data/rgbd_capture/smoke \
  --seconds 10 \
  --pairs 30
```

The same backend can be launched from the migrated Python project:

```bash
cd /home/chhha/6d_vision/project
python3 -m core.orbbec_capture \
  --output data/rgbd_capture/smoke \
  --seconds 10 \
  --pairs 30 \
  --summary
```

This opens color and depth in one pipeline and saves only `FrameSet` objects
that contain both streams and a non-empty depth image. The old `core/astra_depth_probe.py` still describes
the Windows/OpenNI2 depth-only workflow and is not the live Linux capture
entry point.

The camera does not support hardware frame sync. Pairing is therefore defined
by the SDK `FrameSet`; use `system_timestamp_delta_us` in `pairs.csv` for
diagnostics. `depth_value_scale` records the conversion from stored 16-bit
depth values to millimeters.

## Python environment

Install the native build tools and create the project environment:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev python3-venv

cd /home/chhha/6d_vision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r project/requirements-linux.txt
```

`opencv-contrib-python` is intentional: the project uses feature extractors
that are not included in the base `opencv-python` wheel. For a CPU-only
machine, install PyTorch from the CPU index after the requirements file:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision
```

The native Orbbec SDK remains outside this Python environment. Build it from
the project root and run the Python tests from `project`:

```bash
cd /home/chhha/6d_vision
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build -j2

cd /home/chhha/6d_vision/project
python3 -m unittest discover -s tests -p 'test_*.py'
```

The live depth worker now runs the native Orbbec RGB-D binary for five paired
frames, then materializes `depth_burst.npz` and `metadata.json` for the existing
Python depth fitter. The color stream is opened in the same native pipeline as
the depth stream; the current tracker still reopens the OpenCV RGB stream for
the live preview and tracking stage.

The Astra Pro Plus reports that hardware frame synchronization is unsupported.
The capture backend therefore treats color and depth as paired only when both
frames are present in the same SDK `FrameSet`; `pairs.csv` records timestamp
deltas for diagnostics. The current depth initialization remains compatible
with the original project's RGB-depth extrinsics. A future Linux calibration
run can replace that legacy estimate with newly measured extrinsics.

## Linux migration status

The Windows-only OpenNI DLLs, wheel, and virtual environment are not used by
the Linux path. The migrated code is split into a native C++ camera boundary
and the existing Python fitting/tracking code, so a future C++ replacement of
the Python stages can reuse the same RGB-D files and metadata contract.

