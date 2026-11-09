# 6D Vision

项目目录按用途划分：

| 目录 | 内容 |
| --- | --- |
| `core/` | 相机采集、标定、姿态估计、跟踪和评测代码 |
| `tests/` | `unittest` 测试 |
| `data/` | 原始 CAD、采集样本、标定结果、模型和运行输出 |
| `docs/` | 采集、标定和状态说明 |
| `sdk/` | Orbbec/OpenNI 运行依赖 |

从项目根目录使用模块方式运行功能脚本，例如：

```powershell
python -m core.live_depth_6dof
python -m core.evaluate_captured_data --groups global
```

运行测试：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

新采集的文件和默认评测结果会写入 `data/`。已有数据和操作说明见 [数据采集状态](docs/CAPTURE_STATUS.md)、[采集指南](docs/GLOBAL_CAPTURE_GUIDE.md) 和 [相机标定](docs/calibration.md)。

深度采集卡死、相机重开失败及异常记录见 [采集故障恢复说明](docs/DEPTH_RECOVERY.md)。

姿态切换卡顿和误切换电脑摄像头的处理、实测耗时及驱动限制见 [相机身份与重定位说明](docs/CAMERA_IDENTITY_AND_REACQUISITION.md)。

目标从画面外缓慢进入并在运动中首次框选，见 [缓慢入场 6DoF 说明](docs/SLOW_TARGET_ENTRY.md)。

跟踪状态来回切换、框闪烁及误判丢失的修复和验证，见 [跟踪稳定性说明](docs/TRACKING_STABILITY.md)。

目标水平转动、端面朝向镜头时的初始化改进，见 [端面姿态识别说明](docs/END_FACE_INITIALIZATION.md)。

SEARCHING 搜索耗时、漏检改进及离线回放结果，见 [搜索阶段改进说明](docs/SEARCHING_IMPROVEMENTS.md)。

同一画面内多个能量单元的识别、独立编号、RGB 位姿验证和 6DoF CAD 边缘轮廓显示，见 [多目标使用说明](docs/MULTI_TARGET.md)。运行 `python -m core.live_depth_6dof --multi-target` 开启。
