"""Create three orthographic previews of the exported original CAD mesh."""
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .paths import DATA_ROOT

mesh = trimesh.load(DATA_ROOT / 'models/energy_unit_original_mm.stl', force='mesh')
fig, axes = plt.subplots(1, 3, figsize=(13, 7), constrained_layout=True)
for ax, (i, j), title in zip(axes, [(0, 1), (2, 1), (0, 2)], ['Front (X,Y)', 'Side (Z,Y)', 'End (X,Z)']):
    triangles = mesh.triangles[:, :, [i, j]]
    # Sort faces from far to near for a readable orthographic projection.
    depth_axis = ({0, 1, 2} - {i, j}).pop()
    order = np.argsort(mesh.triangles_center[:, depth_axis])
    light = np.array([0.3, 0.6, 0.74])
    shade = 0.35 + 0.6 * np.abs(mesh.face_normals @ light)
    colors = plt.cm.Greys(1 - shade)
    ax.add_collection(PolyCollection(triangles[order], facecolors=colors[order], edgecolors='none'))
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('XYZ'[i] + ' (mm)')
    ax.set_ylabel('XYZ'[j] + ' (mm)')
    ax.grid(alpha=0.2)
fig.suptitle('Energy unit - CAD geometry (original coordinates)')
fig.savefig(DATA_ROOT / 'models/cad_preview.png', dpi=150)
print('mesh bounds:', mesh.bounds.tolist())
print('mesh watertight:', mesh.is_watertight, 'triangles:', len(mesh.faces))
