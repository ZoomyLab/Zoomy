"""Gate 19: the vectorised mesh builders (face topology, cell neighbours,
FVMMesh geometry, LSQ stencils and matrices, face neighbours, subvolumes)
equal the per-cell Python loops they replaced (zoomy_core 6b20967) on quads,
jittered quads, triangles, 1-D, 3-D and gmsh meshes: integers exactly,
floats to 1e-14.
    python gate19_mesh_builders.py [REV]"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import numpy as np
import gate16_parallel as G

REV = sys.argv[1] if len(sys.argv) > 1 else "6b20967"
CORE = os.path.join(os.path.dirname(__file__), "../../../../library/zoomy_core")
MSH = os.path.join(os.path.dirname(__file__), "../../../../library/zoomy_dmplex")
old_dir = tempfile.mkdtemp()
for name in ("base_mesh", "fvm_mesh", "lsq_mesh", "lsq_reconstruction"):
    src = subprocess.run(["git", "-C", CORE, "show", f"{REV}:zoomy_core/mesh/{name}.py"],
                         capture_output=True, text=True, check=True).stdout
    open(os.path.join(old_dir, f"{name}.py"), "w").write(src)


def load(name):
    spec = importlib.util.spec_from_file_location(f"old_{name}", f"{old_dir}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ob, of, ol = load("base_mesh"), load("fvm_mesh"), load("lsq_mesh")
ol.least_squares_reconstruction_local = load("lsq_reconstruction").least_squares_reconstruction_local
from zoomy_core.mesh import base_mesh as nb, LSQMesh, FVMMesh  # noqa: E402

ATTR = ["vertex_coordinates", "cell_vertices", "cell_faces", "face_cells", "cell_neighbors",
        "boundary_face_cells", "boundary_face_ghosts", "boundary_face_function_numbers",
        "boundary_face_physical_tags", "boundary_face_face_indices",
        "_cell_centers", "_cell_volumes", "_cell_inradius", "_face_normals", "_face_volumes",
        "_face_centers", "_lsq_gradQ", "_lsq_neighbors", "_lsq_boundary_face_neighbors",
        "_lsq_scale_factors", "_face_neighbors"]


def compare(label, a, b, extra=()):
    bad = []
    for k, x, y in [(k, getattr(a, k, None), getattr(b, k, None)) for k in ATTR] + list(extra):
        if x is None and y is None:
            continue
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            bad.append(f"{k} shape {x.shape}/{y.shape}")
            continue
        d = np.abs(x.astype(float) - y.astype(float)).max() if x.size else 0.0
        if d > (0 if x.dtype.kind in "iu" else 1e-14):
            bad.append(f"{k} {d:.1e}")
    print(f"{label}: {'equal' if not bad else bad}", flush=True)


rng = np.random.default_rng(1)
q = nb.BaseMesh.create_2d((0, 20, 0, 20), 12, 9)
X = q.vertex_coordinates.copy()
X[:2] += 0.15 * rng.uniform(-1, 1, X[:2].shape)
tri = G.tri_bump(1, nx=9)[0]
cases = {
    "create_2d": lambda m: m.BaseMesh.create_2d((0, 3, 0, 2), 7, 5),
    "jittered quads": lambda m: m.BaseMesh.from_arrays(
        X[:2], q.cell_vertices, "quad", default_tag=0, tag_names={0: "wall"}),
    "triangles": lambda m: m.BaseMesh.from_arrays(
        tri.vertex_coordinates[:2], tri.cell_vertices[:, :tri.n_inner_cells], "triangle",
        default_tag=0, tag_names={0: "wall"}),
    "create_1d": lambda m: m.BaseMesh.create_1d((0, 1), 9),
    "from_msh quad": lambda m: m.BaseMesh.from_msh(os.path.join(MSH, "bump_2d.msh")),
    "from_msh line": lambda m: m.BaseMesh.from_msh(os.path.join(MSH, "mesh_n240.msh")),
    "create_3d": lambda m: m.BaseMesh.create_3d((0, 1, 0, 1, 0, 1), 3, 4, 2),
}
for label, make in cases.items():
    try:
        bo = make(ob)
    except AttributeError:          # from_arrays is new: same input either way
        bo = make(nb)
    bn = make(nb)
    compare(f"BaseMesh {label}", bn, bo)
    fn, fo = FVMMesh.from_base(bn), of.FVMMesh.from_base(bo)
    compare(f"FVMMesh {label}", fn, fo, [("face_subvolumes",
            nb.BaseMesh.face_subvolumes_computed(fn), ob.BaseMesh.face_subvolumes_computed(fo))])
    ln, lo = LSQMesh.from_fvm(fn), ol.LSQMesh.from_fvm(fo)
    note = (" (intended change: 1-D from_msh stencils fixed, see below)"
            if label == "from_msh line" else "")
    compare(f"LSQMesh deg 1 {label}{note}", ln, lo)
    ln._build_lsq_stencil(2)
    lo._build_lsq_stencil(2)
    compare(f"LSQMesh deg 2 {label}{note}", ln, lo)

# the 1-D fix: a gmsh 1-D mesh gets create_1d's stencils (uniform, same cells)
m = LSQMesh.from_msh(os.path.join(MSH, "mesh_n240.msh"))
c = LSQMesh.create_1d((-1.5, 1.5), 240)
n = 240                     # inner cells; ghost rows and tag numbers differ by design
compare("LSQMesh 1-D from_msh vs create_1d, inner cells", None, None, [
    (k, getattr(m, k)[:n], getattr(c, k)[:n]) for k in ("_lsq_neighbors", "_lsq_gradQ")])
