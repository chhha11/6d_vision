# Astra RGB 相机标定

项目根目录运行，系统 Python 已有 OpenCV 4.11.0、NumPy 和 trimesh。
本步骤使用 USB 彩色相机，不需要读取 OpenNI 深度流。sdk 文件夹保留供后续使用。
实时跟踪器还使用 `pygrabber` 按 DirectShow 设备名选择相机；当前环境已安装，可用 `python -m pip install pygrabber` 补装。

## 采集

```powershell
python -m core.calibrate_camera capture --camera 1
```

务必确认窗口显示外接 Astra 的彩色画面。用手在 Astra 镜头前移动可以辨别。
如画面来自笔记本，按 Q 退出，改用 `--camera 0`。摄像头索引可能随设备连接变化。
程序检测到 9x6 个内角点时显示彩色标记；S 保存当前无标记原图，Q / Esc 退出。
如果所有角点未检出，当前帧不保存。拍摄 25-35 张；每次改变位置、距离或倾角。
覆盖画面中心、四角及边缘，保证棋盘格完整、清晰。留出另外几张照片用于独立检查。
标定时使用完整的 640x480 图像，不缩放、不裁剪、不镜像。尽量保持后续使用的镜头焦点不变。
默认棋盘内角点 (9,6)，方格 25 mm；打印尺寸有差异时用 `--square-mm 实测边长`，采集与计算命令保持一致。

图片存于 `data/calibration/images`，设备和图像设置存于同目录的 `capture_settings.json`。
不同相机或分辨率请使用不同 `--images` 文件夹；不要混在一起。
2026-10-29 探测：索引 0 和 1 可读取 640x480，索引 2 无法打开；索引本身不能确认设备身份。

## 计算

```powershell
python -m core.calibrate_camera calibrate
```

需要至少 12 张有效图，建议 25-35 张。输出：

- `camera_intrinsics.npz`：K、dist（兼有 camera_matrix、dist_coeffs 别名）、图像尺寸、棋盘参数、误差和各图外参。
- `calibration_report.json`：整体及逐图 RMS 重投影误差、图片记录、采集设置。
- `undistorted_example.png`：保持原 K、不裁剪的去畸变示例；边缘可能有黑区。

方格用毫米建模，所以外参中的 tvecs 单位为毫米。相机内参焦距和主点单位为像素。
以整体 RMS <0.5 像素作为目标；还需检查视角分布、清晰度及独立照片去畸变效果。
不要仅为降低 RMS 删除所有边缘或倾斜视角。
后续处理原始图像时使用 K 和 dist；处理本程序去畸变图像时使用 K 和零畸变，不要重复去畸变。

## 已完成的程序验证

28 张已知相机参数的合成透视棋盘格均检出角点。端到端标定 RMS 约 0.096 px，恢复焦距误差约 0.27%。
该结果仅验证程序流程，不能代替真实 Astra 相机标定。验证数据位于 data/tmp/calibration_check。

## 真实相机独立验证

运行 `python -m core.calibrate_camera validate`：固定已保存 K/dist，仅求新照片中的棋盘位姿。
另外用棋盘交错的一半角点拟合位姿、另一半评估误差，减少角点参与自身拟合的影响。
2026-10-29：35 张标定图 RMS=0.3957 px；5 张独立图均检出，整体 RMS=0.5167 px，留出角点 RMS=0.5481 px。
单图 RMS 为 0.3371、0.3429、0.5434、0.7116、0.5494 px。结果保存在 validation 子目录。
内参未重新拟合，并已核对与原始报告一致。独立误差略超过 0.5 px 目标，可先用于 CAD 投影原型；不代表毫米级定位精度已经验证。
部分图片中的标定板边缘有弯曲迹象，后续精度优化需确保板面平整。

## 采集能量单元照片

```powershell
python -m core.calibrate_camera capture --camera 0 --no-checkerboard --images data/samples/energy_unit
```

该模式不检测棋盘，S 保存无标记原图、Q 退出。继续使用同一相机、640x480 和相同镜头焦点。
单个物体放在背景简单、颜色有区别的平面上，完整露出两端；采集正侧面、略俯视和不同斜视共 3-5 张。
这批照片用于下一步静态模型投影验证，不是 YOLO 训练集，也不要求放置棋盘格。

## CAD 静态姿态验证

先用第一张横向照片打开姿态编辑器：

```powershell
python -m core.cad_pose_overlay edit --image data/samples/energy_unit/frame_20261029_153630_557391.png --rx 0 --ry 90 --rz 0 --tx 0 --ty -10 --tz 260
```

窗口里的 `rx/ry/rz` 是绕物体坐标轴的角度（度），`tx/ty/tz` 是相机坐标系中的毫米位置。
现在滑块的最小间隔为角度 0.1 度、平移 0.5 mm；窗口左上角会实时显示当前数值。
拖动滑块，让黄色 CAD 线框同时贴合两个端盖外轮廓、主体两侧纵向棱条；红点应落在物体中心，红线应沿物体长轴。
按 S 保存到照片同目录的 `_cad_overlay.png` 和 `_cad_overlay.json`，按 Q 退出。

先调 `tz` 改整体大小，再调 `tx/ty` 改中心位置，最后调三个角度。绕长轴的角度暂时不作为验收条件。
首张照片能稳定贴合后，再对另外四张分别编辑。每张照片保存的 JSON 是后续跟踪器的初始化姿态。

`cad_pose_fit.py` 是边缘拟合实验，只能从接近的初始姿态开始。当前照片中木纹和反光会产生强边缘，自动拟合结果必须人工检查，不能直接用于抓取。

## 实时 RGB 跟踪验证

保存第一张照片的姿态后运行：

```powershell
python -m core.cad_video_tracker
```

程序默认按 DirectShow 精确设备名 `USB Camera` 锁定 Astra RGB；找不到时直接报错，不会回退到电脑摄像头。需要手工指定索引时使用 `--camera N`，该参数会覆盖设备名选择。

启动时让物体处于与第一张初始化照片相近的位置。窗口中的黄色线框是 CAD，红点是中心，红线是长轴。`TRACKING` 只表示当前特征与 PnP 内部一致，不能独立证明已经找到目标；物体未对齐初始化位置时，背景特征也可能产生假跟踪。

程序在投影 ROI 内播种，使用双向光流跟踪特征，再用 `solvePnP` 更新物体位姿；状态栏显示内点数、重投影 RMS 和帧率。默认使用 700 条 CAD 边、最多 100 个点，并每两帧更新一次 CAD 显示层。

物体连续静止约 20 帧后，程序会在后台与首张人工确认图做 AKAZE 外观匹配和相似变换估计。只有连续两次高置信度结果一致时，才以每次最多 5 px、5 mm、1 度的小步修正中心、深度和投影长轴角；随后进入 `STATIC LOCKED`。检测到运动后显示 `MOTION RESEED`，重新播种并恢复 PnP 跟踪。三个单步上限可分别通过 `--static-max-center-step`、`--static-max-depth-step`、`--static-max-angle-step` 调整。周期重播种默认关闭，避免静止时反复重绑特征造成漂移。

静态校正仅对接近首张确认图的外观有效；不同视角匹配不足时会拒绝更新，而不是猜测姿态。需要扩大覆盖范围时，应先为其他视角人工确认准确 CAD 姿态，再加入参考库。可用 `--no-static-correction` 关闭该功能。

按 `R` 只会在最后可信姿态附近重新播种，不是全局重新寻找目标；按 `S` 保存当前 `R、t` 到 `data/tracking/last_pose.json`，按 `Q` 退出。出现 `REJECTED QUALITY`、`REJECTED JUMP`、`REJECTED DEPTH`、`FLOW LOST` 或 `PNP FAILED` 时，应先把物体放回初始化位置再重新播种。

每次运行自动生成 `data/tracking/track_log_时间戳.csv` 和 `track_summary_时间戳.json`，记录状态、位姿、FPS 和各阶段耗时。先缓慢平移物体，再缓慢转动，并短暂遮挡一次；记录线框是否持续贴合。

这一步验证的是连续 RGB 跟踪，尚未验证任意姿态自动初始化、可靠的遮挡恢复或机械臂坐标转换。

## Astra 深度全局初始化

OpenNI2 使用项目内 `sdk/OpenNI2.dll` 和 `sdk/OpenNI2/Drivers/orbbec.dll`。正式深度流必须关闭驱动默认的水平镜像；`core.astra_depth_probe` 已强制执行并将状态写入 metadata。旧版 Orbbec OpenNI 2.3 驱动在 Python 解释器退出时会破坏堆，因此深度和 IR 采集都在短生命周期进程中完成，写完文件后直接退出，不应导入实时 RGB 跟踪主进程。

```powershell
python -m core.astra_depth_probe --output data/tracking/depth_probe_current
python -m core.depth_global_initializer --depth data/tracking/depth_probe_current/depth_burst.npz --metadata data/tracking/depth_probe_current/metadata.json
```

深度初始化器执行 5 帧去零中值、工作区裁剪、RANSAC 去平面、DBSCAN 聚类、尺寸/PCA 筛选及 24 个 CAD ICP 假设。输出 `diagnostics.json`、候选点云、对齐 CAD 点云和预览图。当前静态回归数据得到 850 个目标体素点，ICP fitness=1.000、RMSE=1.794 mm；中心和长轴线可信，但端向置信度不足，绕长轴仍有约 120 度对称歧义。实际部署必须用多姿态数据继续验证。

2026-10-30 的另一组水平面旋转测试中，深度缺口使 10 mm DBSCAN 把目标分成两个短簇。只有首次没有合格候选时，初始化器才用 20 mm 半径重试，并仍经过尺寸与 CAD ICP 验收。该组长轴相对上一组偏转约 28.4 度，重试后有 1210 个目标体素点，ICP RMSE=2.403 mm，RGB 叠加基本贴合。诊断中的 `cluster_eps_used_mm` 记录实际使用的半径；复杂背景下的误合并风险尚需更多姿态和干扰物回归。

## RGB 与深度外参

RGB 是独立 DirectShow UVC 设备，OpenNI 只暴露深度和 IR，因此不能使用硬件同步或直接注册。外参采集按 `RGB 前帧 -> 释放 RGB -> IR -> 重开 RGB -> RGB 后帧` 执行，并按前后角点漂移拒绝移动样本：

```powershell
python -m core.capture_rgb_ir_calibration --target-count 20
python -m core.rgb_ir_extrinsic_calibrator
python -m core.validate_rgb_depth_extrinsics
```

IR 与深度使用不同投影模型，以下两套变换不得混用：

- `T_rgb_ir`：配合自由标定的 `K_ir/dist_ir`，用于 IR 图像点。
- `T_rgb_depth`：固定为深度点云使用的 OpenNI FOV 内参和零畸变，负责把深度点或 `T_depth_model` 转到 RGB 坐标。

位姿交接公式为 `T_rgb_model = T_rgb_depth @ T_depth_model`，可用以下命令生成 RGB 叠加检查：

```powershell
python -m core.depth_pose_handoff
```

2026-10-30 实测 17 组有效配对：IR RMS=0.402 px，IR-RGB stereo RMS=0.568 px，depth-RGB stereo RMS=0.421 px，深度到 RGB 基线=28.96 mm。独立 RGB/IR/深度留出组的棋盘平面残差中位数=0.783 mm，RGB 投影误差中位数=0.326 px、RMS=0.472 px、p95=0.801 px，正确矩阵方向相对逆矩阵有明显优势。结果位于 `data/calibration/rgb_ir` 和 `data/calibration/rgb_ir_validation/held_out_001`。
