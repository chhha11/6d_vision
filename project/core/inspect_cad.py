"""Inspect the STEP with Open Cascade; run using .venv-cad/Scripts/python.exe."""
import json
from collections import Counter
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.gp import gp_Trsf, gp_Pnt
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

from .paths import DATA_ROOT

reader = STEPControl_Reader()
assert reader.ReadFile(str(DATA_ROOT / '能量单元.stp')) == IFSelect_RetDone
assert reader.TransferRoots() > 0
shape = reader.OneShape()
box = Bnd_Box()
BRepBndLib.AddOptimal_s(shape, box, False, False)
pmin, pmax = box.CornerMin(), box.CornerMax()
bounds = [pmin.X(), pmin.Y(), pmin.Z(), pmax.X(), pmax.Y(), pmax.Z()]
counts = {}
for label, kind in [('solids', TopAbs_SOLID), ('shells', TopAbs_SHELL), ('faces', TopAbs_FACE)]:
    exp = TopExp_Explorer(shape, kind)
    count = 0
    while exp.More():
        count += 1
        exp.Next()
    counts[label] = count
cylinders = Counter()
exp = TopExp_Explorer(shape, TopAbs_FACE)
while exp.More():
    surf = BRepAdaptor_Surface(TopoDS.Face(exp.Current()))
    if surf.GetType() == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        direction = cyl.Axis().Direction()
        key = (round(cyl.Radius(), 4), tuple(round(abs(v), 5) for v in (direction.X(), direction.Y(), direction.Z())))
        cylinders[key] += 1
    exp.Next()
out = DATA_ROOT / 'models'
out.mkdir(exist_ok=True)
BRepMesh_IncrementalMesh(shape, 0.1, False, 0.15, True).Perform()
writer = StlAPI_Writer()
writer.ASCIIMode = False
assert writer.Write(shape, str(out / 'energy_unit_original_mm.stl'))
transform = gp_Trsf()
# Right-handed coordinates: x'=x, y'=-z, z'=y-75 (millimetres).
transform.SetValues(1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, -75)
centered = BRepBuilderAPI_Transform(shape, transform, True).Shape()
BRepMesh_IncrementalMesh(centered, 0.1, False, 0.15, True).Perform()
assert writer.Write(centered, str(out / 'energy_unit_centered_mm.stl'))
scale = gp_Trsf()
scale.SetScale(gp_Pnt(0, 0, 0), 0.001)
metric = BRepBuilderAPI_Transform(centered, scale, True).Shape()
BRepMesh_IncrementalMesh(metric, 0.0001, False, 0.15, True).Perform()
assert writer.Write(metric, str(out / 'energy_unit_centered_m.stl'))
report = dict(unit='mm', bounds_min=bounds[:3], bounds_max=bounds[3:],
              extents=[bounds[i+3]-bounds[i] for i in range(3)],
              bbox_center=[(bounds[i+3]+bounds[i])/2 for i in range(3)],
              topology=counts, valid=BRepCheck_Analyzer(shape).IsValid(),
              original_long_axis='+Y', chosen_center_original_mm=[0, 75, 0],
              original_mm_to_centered_mm=[[1,0,0,0],[0,0,-1,0],[0,1,0,-75],[0,0,0,1]],
              centered_axis='+Z',
              cylindrical_faces=[dict(radius_mm=k[0], abs_axis=k[1], count=v) for k,v in cylinders.most_common()])
(out / 'cad_inspection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
