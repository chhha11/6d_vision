# 6D Vision

基于 Orbbec Astra Pro Plus RGB-D 相机的 6DoF 姿态估计与目标跟踪项目，包含 Python 视觉算法、C++ 相机采集后端以及 M3T 跟踪集成。

## 目录

| 路径 | 内容 |
| --- | --- |
| `project/core/` | 采集、标定、姿态估计、跟踪与评测 |
| `project/tests/` | Python 单元测试 |
| `project/docs/` | 使用指南、实验记录与故障排查 |
| `project/data/` | CAD 模型、标定参数及本地运行数据 |
| `native/` | C++ RGB-D 采集和 M3T 后端 |
| `scripts/` | 依赖准备、数据采集与验收脚本 |
| `tools/` | 相机诊断工具源码 |

## Linux 环境

当前 Linux 工作环境为 Ubuntu 22.04。安装系统依赖并建立 Python 环境：

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r project/requirements-linux.txt
```

### Orbbec 相机后端

Orbbec SDK 需要单独安装。当前工程使用 v1.10.37；默认目录由根目录的 `CMakeLists.txt` 指定，也可通过 `ORBBEC_SDK_ROOT` 指向其他安装路径：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DORBBEC_SDK_ROOT=/absolute/path/to/OrbbecSDK_v1.10.37
cmake --build build -j2
./build/orbbec_rgbd_capture --output project/data/rgbd_capture/smoke --seconds 10 --pairs 30
```

### M3T 后端

准备固定版本的上游源码和本地依赖，然后编译：

```bash
python3 scripts/setup_m3t_offline.py
```

该脚本需要网络、Git 和 Debian/Ubuntu 的包管理工具，会将依赖放在 `third_party/`，将编译产物放在 `build/`。

### 运行与测试

```bash
source .venv/bin/activate
cd project
python -m core.live_depth_6dof
```

运行单元测试：

```bash
cd project
python -m unittest discover -s tests -p 'test_*.py'
```

详细说明见 [项目说明](project/README.md)、[Linux 环境说明](project/README_LINUX.md)、[原生采集后端](native/README.md) 和 [项目交接说明](project/docs/PROJECT_HANDOFF.md)。

## 仓库中的数据

仓库保留现有 CAD/STL 模型、标定参数与标定报告。原始采集样本、标定采集图像、跟踪录像、训练权重、SDK、虚拟环境和下载的第三方依赖由 `.gitignore` 排除，需要在其他机器上另行准备。依赖这些本地数据或相机硬件的功能与测试需要相应资源。

第三方 M3T 源码由准备脚本从 `DLR-RM/3DObjectTracking` 下载并固定到提交 `f02106186a1e756450226b3b793e309506817c58`，使用时应遵守上游许可。
