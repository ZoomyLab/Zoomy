"""C1 bump on the NumPy backend of the symbolic solvers, for CPython and Pyodide.

Matrix case C1 (as `library/zoomy_core/tests/symsolver/test_numpy_backend.py`): SME level 0
(= SWE), walled 10 x 10 box, Gaussian surface bump at rest, O2 minmod, 20 x 20 cells,
`SymbolicHyperbolicSolverNumpy` to t = 0.5. `main()` returns a JSON-able report: the march,
the mass error, which optional packages got imported, and every `zoomy_core` module whose
import fails here.
"""
import importlib
import importlib.util
import pkgutil
import sys
import time

import numpy as np

T_END = 0.5
#: Optional add-ons: none may be needed by the NumPy path.
OPTIONAL = ("jax", "jaxlib", "pyamg", "mpi4py", "petsc4py", "h5py", "meshio", "matplotlib")


def bump(nx=20):
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
    from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel

    tags = ("left", "right", "bottom", "top")
    model = SME(level=0, dimension=3, closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                        for t in tags]))
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=lambda x: np.array(
        [0.0, 0.5 + 0.05 * np.exp(-((x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2) / 2.0), 0.0, 0.0]))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
    return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx), nsm


def march(t_end=T_END):
    from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy

    t0 = time.perf_counter()
    mesh, nsm = bump()
    s = SymbolicHyperbolicSolverNumpy(time_end=t_end, cfl=0.9, mood=False)
    Q0, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q0, A)
    vol = np.asarray(s.mesh_np.cell_volumes)[:Q.shape[1]]
    h = [str(v) for v in nsm.state].index("h")
    mass0, mass = float(Q0[h] @ vol), float(Q[h] @ vol)
    return {"t": float(t), "finite": bool(np.isfinite(Q).all()),
            "moved": float(np.abs(Q[h] - Q0[h]).max()),
            "mass_rel_err": abs(mass - mass0) / mass0,
            "seconds": round(time.perf_counter() - t0, 1)}


def import_audit(prefix="zoomy_core"):
    """`{module: error}` for every module under `prefix` that does not import here."""
    failures = {}
    pkg = importlib.import_module(prefix)
    for m in pkgutil.walk_packages(pkg.__path__, prefix + ".", onerror=lambda n: None):
        if ".legacy" in m.name or ".tests" in m.name:
            continue
        try:
            importlib.import_module(m.name)
        except BaseException as e:   # noqa: BLE001 -- the audit reports, it does not stop
            failures[m.name] = f"{type(e).__name__}: {e}"[:300]
    return failures


def main():
    report = {"platform": sys.platform, "python": sys.version.split()[0]}
    try:
        report["march"] = march()
    except BaseException as e:   # noqa: BLE001
        import traceback
        report["march"] = {"error": traceback.format_exc()[-3000:]}
    report["imported_optional"] = sorted(m for m in OPTIONAL if m in sys.modules)
    report["absent_optional"] = sorted(m for m in OPTIONAL if importlib.util.find_spec(m) is None)
    report["import_failures"] = import_audit()
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=1))
