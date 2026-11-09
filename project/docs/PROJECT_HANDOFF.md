# 6D Vision 项目交接摘要

更新时间：2026-11-02

## 1. 项目目标

对带有 CAD 模型的目标实现持续、稳定的 6DoF 位姿跟踪：从目标识别开始直接进入 3D 跟踪，运动、旋转、短时遮挡和重新进入画面时尽量保持连续，避免 2D 检测框与 3D 位姿之间来回切换。

最终部署目标是 Linux。后续可以采用 C++ 或 Python/C++ 混合架构，但算法稳定前不做整套 C++ 重写。

## 2. 当前硬件和限制

- 深度相机：旧款 Orbbec Astra，使用结构光深度。
- 当前电脑：没有 NVIDIA GPU，方案必须以 CPU 可运行作为基线。
- 当前 Windows RGB 接口：UVC/DirectShow。
- 当前深度接口：旧版 Orbbec OpenNI2 驱动。
- 当前 RGB 与深度不能可靠同时开流。现有程序在需要深度重定位时会释放 RGB、采集深度、再重开 RGB，因此会出现明显停顿和 2D/3D 交接。
- 项目内的 `sdk/` 包含 Windows OpenNI2 DLL 和驱动，不应直接复制到 Linux 使用。

## 3. 当前代码结构

- `core/live_depth_6dof.py`：直播入口、2D 搜索、深度初始化、跟踪交接和重定位。
- `core/cad_video_tracker.py`：CAD 边线/表面播种、光流、PnP、状态维护、RGB 恢复。
- `core/depth_global_initializer.py`：深度点云与 CAD 的全局初始化、ICP 和质量门限。
- `core/astra_depth_probe.py`：通过 OpenNI2 采集深度帧。
- `core/rgb_pose_recovery.py`、`core/rgb_pose_relocalizer.py`：RGB 独立位姿复核和恢复。
- `core/live_detector.py`：CPU 路径上的 Ultralytics YOLO 检测。
- `data/models/`：CAD 模型和检测模型。
- `data/calibration/`：RGB、IR、深度内参和外参。
- `data/samples/`：全局初始化、运动和集成会话样本。
- `docs/`：采集、标定、初始化和跟踪验证记录。

## 4. 已有算法和实测结论

当前跟踪主要使用：YOLO 初始检测、CAD 可见表面播种、光流点跟踪、PnP 位姿更新、检测框/尺度/长轴复核，以及必要时的 RGB 参考重定位和深度重新初始化。

已完成的改进包括：运动中首次初始化、端面姿态分支、可见表面绑定、检测冲突确认、短时位姿保持、RGB 姿态恢复、相机物理身份检查和深度采集进程隔离。

实测边界：

- 普通跟踪计算约 1.5 ms，重新选点约 7 ms，RGB 读帧约 44 ms；性能瓶颈主要不是 Python 算术，而是相机切换、驱动恢复和深度初始化。
- 直播常态约 20 FPS；成功的跟踪段可以连续运行，但新表面、强遮挡、弱纹理和较大三维翻转仍可能导致对应关系失效。
- 最近一次长记录中 44 次初始化有 18 次 ready、26 次 rejected，说明深度初始化稳定性仍不足。
- 已有标定的 RGB-深度投影误差为亚像素量级，但这不等于动态跟踪的绝对 6DoF 误差。

详细记录：`CAMERA_IDENTITY_AND_REACQUISITION.md`、`DEPTH_RECOVERY.md`、`TRACKING_STABILITY.md`、`OBJECT_MOTION_TRACKING.md`、`RGB_POSE_RECOVERY.md`、`CAPTURE_STATUS.md`。

## 5. 方案决策

### 5.1 FoundationPose

FoundationPose 可作为后续算法原型参考，但当前电脑没有 NVIDIA GPU，不能把原版 CUDA 路线作为主方案。即使使用它，也仍需要可靠 RGB-D、目标掩码和 CAD/网格输入，不能解决相机流中断本身。

### 5.2 第二台单目相机

这是优先级较高的硬件方向：让新增相机持续提供 RGB，Astra 专门提供深度，避免争用 Astra 的 RGB 接口。必须先验证两路设备的时间戳、外参、同步误差、掉帧率和长期稳定性；单纯增加相机并不会自动完成 RGB-D 对齐。

### 5.3 C++ 或混合架构

先迁移到 Linux 并用 Python 验证采流和算法，再根据性能数据决定。最终可采用：C++ 负责相机采集、同步、时间戳和实时跟踪；Python 保留模型实验、数据分析或通过 ONNX Runtime/C++ 推理接入。算法尚未定型前不要维护两套完整实现。

## 6. Linux 迁移计划

1. 在原生 Linux 上安装 Ubuntu 22.04 或 24.04，并确认 Astra 精确型号和固件版本。
2. 优先测试 Orbbec 页面为旧 Astra 提供的 OpenNI2 SDK/Driver；不要把 `OrbbecSDK_v2` 当作旧 Astra 的默认方案，页面要求的 `v2.xx` 固件未确认适用。
3. 编写最小采流测试：同一设备创建 color stream 和 depth stream，同时启动并连续读取 10 至 30 分钟，记录 FPS、时间戳、掉帧、深度有效率、注册状态和进程异常。
4. 测试失败时，再验证新增单目 RGB 相机 + Astra 深度的双设备方案，并重新做双相机内外参标定。
5. 在 Linux 上先安装 Python 依赖，运行已有离线录像和单元测试，确保算法基线成立。
6. 用真实运动录像比较：初始化耗时、稳定跟踪帧比例、重定位次数、丢失恢复时间和 CPU 占用。
7. 算法达到验收标准后，再逐模块迁移 C++，每个模块与 Python 输出逐帧对比。

## 7. Linux 环境注意事项

可以复制：`core/`、`tests/`、`docs/`、`data/models/`、`data/calibration/`、`data/samples/`、`README.md`。

不要直接复制：`.venv-cad/`、`__pycache__/`、Windows `.dll`、Windows OpenNI 驱动。Linux 上需重新安装 Python 环境、OpenCV、NumPy、Open3D、trimesh、Ultralytics，以及 Linux 版 OpenNI2/Orbbec 驱动。

Windows 路径如 `D:\\new_6d_vision` 不能直接用于 Linux；标定数值、CAD 文件和样本本身可以复用。

## 8. 暂定验收标准

- 识别成功后不因正常运动在 2D 和 3D 状态间闪回。
- RGB-D 输入连续运行至少 30 分钟，无相机崩溃或需要人工重启。
- 目标正常运动时跟踪有效率、丢失恢复时间和端到端延迟达到现场要求。
- 发生遮挡或出画后，系统能区分“暂时不可见”和“位姿仍然有效”，不输出未经验证的错误 3D 位姿。
- 用独立测量或夹具提供 6DoF 真值后，再报告毫米和角度误差；现有棋盘回放不能替代绝对真值。

## 9. 当前第一步

Linux 安装完成后，不要先改跟踪算法。先完成 Astra 的 OpenNI2 并发 color/depth 采流测试，并保存测试日志、设备信息、固件版本和 USB 拓扑。这个结果决定后续是继续单设备 RGB-D，还是采用第二台 RGB 相机。
