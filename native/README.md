# Linux RGB-D capture backend

`orbbec_rgbd_capture` uses OrbbecSDK v1.10.37 to open the Astra Pro Plus color
and depth streams in one pipeline. It saves only `FrameSet` objects that
contain both streams, so the Python tracking code can consume a known paired
sample instead of reopening the legacy OpenNI depth driver.

Build from the project root:

```bash
cmake -S . -B build
cmake --build build -j2
```

Capture 30 paired samples:

```bash
./build/orbbec_rgbd_capture \
  --output data/rgbd_capture/current \
  --seconds 20 \
  --pairs 30
```

The default color profile is 640x480 MJPG at the device-advertised 30 FPS.
The Astra currently delivers color at about 16.7 FPS in this USB/SDK setup;
`--color-format rgb` and `--color-format bgra` are available for diagnostics,
but do not improve the measured rate.

Output files:

- `color/000000.jpg`, ...: decoded BGR color images
- `depth/000000.png`, ...: 16-bit depth images in millimeters
- `pairs.csv`: device and host timestamps for each saved pair. On this Astra,
  `color_timestamp_us` and `depth_timestamp_us` use different sensor clocks,
  so use `system_timestamp_delta_us` for diagnostics. The pair itself is
  defined by both frames being present in the same `FrameSet` and by the depth
  image containing at least one non-zero pixel. Astra units can emit an
  all-zero depth buffer while the stream is warming up; those frames are
  counted as `empty_depth_frames` and are not saved as valid pairs.
- `summary.txt`: stream and capture counts
- `camera_params.json`: depth and color intrinsics returned by the active
  OrbbecSDK profiles

The device does not provide hardware frame sync. `pairs.csv` is therefore
the source of truth for measuring system timestamp differences before adding
application-level matching or RGB-D registration. `depth_value_scale` gives
the scale needed to convert stored 16-bit depth values to millimeters.
