# SEARCHING 搜索提速和漏检改进（2026-11-04）

已修改 `core/rgb_global_initializer.py`，初次搜索和跟踪丢失后的 RGB 重定位共用这条路径。重新启动现有直播/采集命令即可加载。

## 问题证据

录像 `targeted_collection/20261104_223848_790626_motion` 共记录 6884 个跟踪帧，其中 3079 帧为 `TARGET LOST`。从 TRACKING 和 TARGET LOST 各均匀抽取 16 帧，原版独立搜索只有 4/32 帧输出位姿。YOLO 提供框后，原实现仍对全图建立特征金字塔；32 次检查的性能剖析中，`detectAndCompute` 占 3.929/5.330 秒，约 74%。

代码还有 0.5 秒结果时效限制，但本次抽样原版搜索中位数约 183 ms，不能据此把这批漏检主要归因于超时。

## 修改

- 有目标区域时，在四周保留 48 px 上下文后裁剪，再提取特征。匹配像素还原到相机坐标，保持原内参与几何校验；全图搜索继续保留。
- SIFT 增加独立短时特征库。每类特征分别使用前后向光流检查、0.5 秒更新时效和最多六次累积，不混用同名参考视角的特征索引。无目标区域和非法区域会清空两类库。
- AKAZE、SIFT 各自无法通过验收时，按同一个经过深度验证的参考视角合并互补的 3D/2D 对应关系。像素距离小于 3 px 或模型点距离小于 1 mm 的重复证据不重复计数，再执行原有 PnP、空间分布、重投影、参考视角支持、目标框和长轴检查。
- 保留原有位姿结果时效及当前帧推进/确认流程，未修改相机配置、模型权重和几何门槛。

## 离线验证

评测目录：`data/tracking/evaluations/searching_20261104/`。

单帧抽查：相同 32 帧、相同 19 个参考视角（每视角 AKAZE/SIFT 两组描述子），位姿候选从 4/32 增至 6/32，搜索中位数从 183.2 ms 降至 144.2 ms。候选不等于实时跟踪恢复。

连续帧回放：在同一批起始帧后，每隔四个录像帧进行一次搜索，最多七次；短时特征时效使用录像时间戳。候选需推进到后续当前帧，并通过真实 `RgbPoseRelocalizer.confirm`，才能计为恢复。录像末尾起始帧缺少后续帧，排除后共 31 段。

| 指标 | 修改前 | 修改后 |
| --- | ---: | ---: |
| 成功恢复片段 | 8/31 | 18/31 |
| 原 TARGET LOST 片段恢复 | 0/15 | 5/15 |
| 搜索耗时中位数 | 176.6 ms | 143.8 ms |
| 搜索耗时 P90 | 193.7 ms | 214.3 ms |

联合求解使部分困难检查耗时增加，因此 P90 没有改善；不能把中位数提速解释为每次搜索都更快。

21 张已保存 N 类负样本图，分别做全图与中央 ROI、各连续三次搜索，共 126 次检查，没有输出位姿。新增单元测试覆盖 ROI 坐标还原、非法框、SIFT 运动累积、重复特征去重和联合求解的参考支持门槛。初始化、重定位、直播和跟踪器四组回归共 **123 项通过**。

已查看单帧接受结果及五段 TARGET LOST 恢复结果的 CAD 投影图：`accepted_poses.jpg`、`recovered_poses.jpg`。这属于可视化检查，不是独立 6DoF 精度真值。

## 适用边界

这是短片段离线测试，不是实时相机全流程复测。每段使用起始帧的固定 YOLO 提示，保留全图回退；没有模拟背景推理竞争、真实提交队列和整个跟踪器后续行为。成功后提前停止会使两版耗时样本组成不同，因此搜索耗时也须结合固定 32 帧结果阅读。剩余 10/15 个丢失片段仍未恢复，不能宣称漏检已解决。

目前只有 19 个通过深度/静止校验的参考视角。对仍失败的姿态，需要进一步检查遮挡、有效特征和参考覆盖；本次结果不能区分这些因素各自的贡献。

## 复现

在工作区根目录使用：

```bash
PYTHONPATH=project .venv/bin/python project/data/tracking/evaluations/searching_20261104/sequences.py before --baseline
PYTHONPATH=project .venv/bin/python project/data/tracking/evaluations/searching_20261104/sequences.py final
PYTHONPATH=project .venv/bin/python project/data/tracking/evaluations/searching_20261104/benchmark.py combined
PYTHONPATH=project .venv/bin/python project/data/tracking/evaluations/searching_20261104/validate.py
PYTHONPATH=project .venv/bin/python -m unittest project/tests/test_rgb_global_initializer.py project/tests/test_rgb_pose_relocalizer.py project/tests/test_live_depth_6dof.py project/tests/test_cad_video_tracker.py
```

`baseline_initializer.py` 保存修改前版本；`before.json` 是固定抽样和起始 YOLO 框；`sequences_before.json`、`sequences_final.json`、`negative_results.json`、`summary.json`、`tests.log` 保存最终对比证据。其他 crop/reference_crop/after/combined 中间文件为试验记录；最终实现仍使用原有第 2 版参考特征缓存。
