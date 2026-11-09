# 明日实机验收

今天不启动相机。明天把目标带到电脑前后，直接在项目根目录执行：

```bash
cd /home/chhha/6d_vision
./scripts/run_live_acceptance.sh
```

默认按设备名 `USB Camera` 自动查找 Astra 的可读 RGB 视频节点，输出目录自动按时间创建在
`project/data/tracking/live_depth_6dof/YYYYMMDD_HHMMSS/`，不会覆盖历史结果。
如果需要手工指定当前节点，可用 `CAMERA_INDEX=4 ./scripts/run_live_acceptance.sh`；
不要把 `1` 当作 Astra 的索引，当前机器的 `video1` 属于笔记本摄像头且不可读。
Linux 下启动会使用 OpenCV 的默认视频后端打开 UVC 彩色流，不要求安装
Windows 专用的 `comtypes/pygrabber`；Windows 下仍保留 DirectShow 设备身份锁定。

当前 Linux 工作副本没有随项目提供 YOLO `best.pt`，因此启动时会自动进入无
YOLO 人工启动模式，不会再因模型缺失而退出。窗口会持续显示摄像头画面，
把目标放入画面并保持稳定后按 `Space` 开始深度初始化；如果背景干扰较多，按
`R` 用鼠标框选目标，再按 `Space`。初始化成功后窗口才会出现 CAD 框、XYZ 和
`TRACKING` 状态；没有这些叠加说明仍未完成初始化，不是跟踪已经在后台运行。
当前没有 `best.pt` 时暂时不具备“目标出画后自动检测并重捕获”的能力；以后把模型
放回默认路径，或用 `--reacquire-model PATH` 启动，即可恢复自动重捕获。

## 动作顺序

1. 把目标放在画面中央、完整可见，距离先保持约 0.5 到 0.8 米并停稳。
2. 窗口显示 `No YOLO model` 时按 `Space`；背景复杂或深度簇混在一起时先按 `R` 框住目标，再按 `Space`。
3. 等待窗口出现 CAD 框和 `TRACKING` 后，缓慢左右平移，再上下平移。
4. 缓慢绕竖直轴、水平轴旋转，并改变目标距离。
5. 用手短暂遮挡目标或让目标出画，再移回画面；无 YOLO 时丢失后需要重新按 `Space`，这是当前模型缺失的预期限制。
6. 按 `q` 退出。若窗口没有响应，按 `Esc`。

不要刻意追求 30 FPS；本次主要观察 6DoF 是否连续、姿态是否跳变，以及短暂丢失后是否能恢复。

## 结束后汇总

先找出刚才生成的时间目录，再运行：

```bash
./scripts/summarize_live_acceptance.sh \
  data/tracking/live_depth_6dof/YYYYMMDD_HHMMSS \
  --output data/tracking/live_depth_6dof/YYYYMMDD_HHMMSS/acceptance_summary.json
```

不带目录时会自动选择最新一次运行：

```bash
./scripts/summarize_live_acceptance.sh
```

汇总重点看：

- `tracking.tracking_ratio`：有效跟踪帧比例；
- `tracking.max_frame_interval_ms`：跟踪循环最大间隔，不等同于相机硬件 FPS；
- `tracking.hard_loss_counts`：`TARGET LOST`、`FLOW LOST`、`PNP FAILED` 等；
- `tracking.reacquire_requested`：是否触发重新初始化；
- `initialization.ready/rejected`：初始化成功和拒绝次数；
- `events.by_stage`：是否出现深度采集或相机恢复异常。

今天没有目标，因此不会提前生成或伪造验收结论。明天第一次真实动作验收后，再决定是否需要调整跟踪阈值或继续排查驱动。
