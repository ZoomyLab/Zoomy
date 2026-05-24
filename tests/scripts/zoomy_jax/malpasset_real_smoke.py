"""Step 1: real Malpasset mesh, JAX solver, single-device.  Uses the
existing HDF5 writer (``solver.run_simulation`` → ``simulation.h5``)
and the existing VTK converter (``zoomy_core.misc.io.generate_vtk``).

Target wall-clock under ~120 s (Malpasset has 26k triangles; JIT
compile of the flux op alone is ~45 s)."""
from __future__ import annotations

import os
import sys
import time

# Force output directory to a user-writable location by overriding
# the project's main-directory autodetection (which would write to
# the root-owned /home/ingo/git/Zoomy/outputs/).
_OUT_ROOT = (
    os.path.expanduser("~/sciebo")
    if os.path.isdir(os.path.expanduser("~/sciebo"))
    else "/tmp"
)
os.environ["ZOOMY_DIR"] = _OUT_ROOT
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from malpasset_swe_model import MalpassetSWE  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.misc.misc import Settings, Zstruct  # noqa: E402
from zoomy_core.misc import io  # noqa: E402
from zoomy_core.model.models.system_model import SystemModel  # noqa: E402
from zoomy_core.numerics import NumericalSystemModel  # noqa: E402
from zoomy_core.numerics.numerical_system_model import (
    ReconstructionSpec,
)  # noqa: E402
from zoomy_jax.fvm.solver_jax import HyperbolicSolver  # noqa: E402


MESH_PATH = "/home/ingo/git/Zoomy/data/malpasset/geo_malpasset-small.msh"
OUT_DIR_RELATIVE = "outputs/malpasset_jax_serial"   # rel. to ZOOMY_DIR
OUT_FILENAME = "simulation"
TIME_END = 0.2     # very short run for the smoke test
N_SNAPSHOTS = 3


def _load_ic_from_msh(solver, Q):
    """Set Q from the .msh point data (B, H, U, V → b, h, hu, hv).
    Cell-average over the 3 vertices of each triangle."""
    import meshio
    m = meshio.read(MESH_PATH)
    cell_vertices = np.asarray(solver._rt_mesh.cell_vertices)
    nc = int(solver._rt_mesh.n_inner_cells)
    cv = cell_vertices[:, :nc]
    B = m.point_data["B"]
    H = m.point_data["H"]
    U = m.point_data["U"]
    V = m.point_data["V"]
    b = (B[cv[0]] + B[cv[1]] + B[cv[2]]) / 3.0
    h = (H[cv[0]] + H[cv[1]] + H[cv[2]]) / 3.0
    u = (U[cv[0]] + U[cv[1]] + U[cv[2]]) / 3.0
    v = (V[cv[0]] + V[cv[1]] + V[cv[2]]) / 3.0
    hu = h * u
    hv = h * v
    Q = Q.at[0, :nc].set(jnp.asarray(b, dtype=Q.dtype))
    Q = Q.at[1, :nc].set(jnp.asarray(h, dtype=Q.dtype))
    Q = Q.at[2, :nc].set(jnp.asarray(hu, dtype=Q.dtype))
    Q = Q.at[3, :nc].set(jnp.asarray(hv, dtype=Q.dtype))
    print(f"  IC: b∈[{b.min():.2f}, {b.max():.2f}]m, "
          f"h∈[{h.min():.2f}, {h.max():.2f}]m, "
          f"|u|max={np.sqrt(u**2 + v**2).max():.2f}m/s")
    return Q


def main():
    t0 = time.perf_counter()

    print(f"[{time.perf_counter() - t0:6.2f}s] Loading Malpasset mesh "
          f"({MESH_PATH})...")
    mesh_np = LSQMesh.from_msh(MESH_PATH)
    print(f"[{time.perf_counter() - t0:6.2f}s] n_inner_cells="
          f"{mesh_np.n_inner_cells}, n_faces={mesh_np.n_faces}, "
          f"n_bf={mesh_np.n_boundary_faces}")

    settings = Settings(
        output=Zstruct(
            directory=OUT_DIR_RELATIVE,
            filename=OUT_FILENAME,
            snapshots=N_SNAPSHOTS,
            clean_directory=True,
        )
    )

    model = MalpassetSWE()
    nsm = NumericalSystemModel.from_system_model(
        SystemModel.from_model(model),
        reconstruction=ReconstructionSpec(order=1),
    )
    solver = HyperbolicSolver(settings=settings, time_end=TIME_END)
    print(f"[{time.perf_counter() - t0:6.2f}s] setup_simulation...")
    Q, Qaux = solver.setup_simulation(mesh_np, nsm)
    print(f"[{time.perf_counter() - t0:6.2f}s] setup done")

    Q = _load_ic_from_msh(solver, Q)
    Qaux = solver.update_qaux(
        Q, Qaux, Q, Qaux, solver._rt_mesh, solver._rt_model,
        solver._rt_parameters, 0.0, 1.0,
    )
    print(f"[{time.perf_counter() - t0:6.2f}s] IC loaded.")

    print(f"[{time.perf_counter() - t0:6.2f}s] run_simulation "
          f"(time_end={TIME_END}, snapshots={N_SNAPSHOTS})...")
    t_run0 = time.perf_counter()
    Q_final, Qaux_final = solver.run_simulation(Q, Qaux, write_output=True)
    Q_final.block_until_ready()
    t_run = time.perf_counter() - t_run0
    print(f"[{time.perf_counter() - t0:6.2f}s] run_simulation done "
          f"(wall {t_run:.2f}s, including JIT compile)")

    nc = int(solver._rt_mesh.n_inner_cells)
    Q_np = np.asarray(Q_final)
    print(f"  final h range: [{Q_np[1, :nc].min():.3f}, "
          f"{Q_np[1, :nc].max():.3f}] m,  |hu|max={np.abs(Q_np[2, :nc]).max():.3f}")
    print(f"  finite: {bool(np.isfinite(Q_np).all())}")

    h5_path = os.path.join(OUT_DIR_RELATIVE, f"{OUT_FILENAME}.h5")
    abs_h5 = os.path.join(_OUT_ROOT, h5_path)
    print(f"\nHDF5 written to: {abs_h5}")

    # Convert to VTK time series.
    print(f"\n[{time.perf_counter() - t0:6.2f}s] Converting HDF5 → VTK...")
    io.generate_vtk(
        h5_path,
        field_names=["b", "h", "hu", "hv"],
        aux_field_names=["hinv"],
        filename="malpasset_jax",
    )
    out_dir_abs = os.path.dirname(abs_h5)
    vtks = sorted(f for f in os.listdir(out_dir_abs)
                  if f.startswith("malpasset_jax") and f.endswith(".vtk"))
    print(f"VTK series written: {out_dir_abs}/malpasset_jax.*.vtk "
          f"({len(vtks)} snapshots + .vtk.series manifest)")
    print(f"  → open in ParaView: {out_dir_abs}/malpasset_jax.vtk.series")

    print(f"\nTOTAL: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
