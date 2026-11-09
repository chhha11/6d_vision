# 能量单元模型检查

来源：`data/能量单元.stp`。原始文件未修改。

## 检查结果

- STEP 长度单位为毫米，Open Cascade 几何有效性检查通过。
- 1 个实体、2 个壳、1424 个面。导出网格为封闭网格。
- 原始包围盒约为 X=[-47.5,47.5]、Y=[0,150]、Z=[-47.5,47.43418] mm。
- 总长 150 mm；最大 X 向宽度 95 mm。模型有端盖、棱条、槽口，两端外形不同。
- 长轴沿原始 Y 轴。约定轴线两端中点 (0,75,0) mm 为新的物体原点；这不是质量中心，也不是略微偏心的包围盒中心。
- 尚未验证实物尺寸、精确旋转对称阶数及夹爪对绕轴旋转的容差。

## 文件与坐标约定

- `data/models/energy_unit_original_mm.stl`：原始坐标，毫米。
- `data/models/energy_unit_centered_mm.stl`：中心坐标，毫米。
- `data/models/energy_unit_centered_m.stl`：中心坐标，米；后续视觉程序建议使用此文件。
- `data/models/cad_inspection.json`：几何检查结果、曲面统计和原始坐标到中心坐标的变换。
- `data/models/cad_preview.png`：原始坐标下的三视图；仅展示几何，不代表实际颜色。

中心坐标采用右手系：x'=x，y'=-z，z'=y-75（输入输出均为毫米）。
新的 +Z 指向原始 Y=150 的端部；两端轴线中心坐标为 (0,0,-75) 和 (0,0,75) mm。
换算为米时，三个坐标均乘以 0.001。STL 不自带单位，加载时必须遵循文件名约定。

网格线性离散误差参数 0.1 mm，角度参数 0.15 rad。
两个中心版本均通过封闭性、尺度比例和轴向范围检查。

## 重现

在项目根目录使用 `.venv-cad\Scripts\python.exe -m core.inspect_cad`。
CAD 环境使用 cadquery-ocp 8.0.1.1.0、cadquery-ocp-proxy 8.0.1.1.0、vtk 9.6.2。
三视图使用系统 Python 的 trimesh、numpy、matplotlib：`python -m core.preview_cad`。

## 用户下一步

测量实物总长、最大端盖直径，并与 150 mm、约 95 mm 核对。
同时确认两端是否允许互换抓取。尺寸核对后，进入 Astra RGB 相机标定。
