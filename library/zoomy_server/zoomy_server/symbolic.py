"""The symbolic solvers (`zoomy_core.symsolver` and its backends) behind the adapters' settings
runner: a case folder without a `run.py` (model.py, mesh.py, settings.json) is translated into
one symbolic solve.

Settings (the effective ones, `SolverAdapter.load_settings`): `time_end` (0.1), `cfl` (0.9),
`output_snapshots` (10), `reconstruction_order` (1), `limiter`, `imex` (`lie` | `ssp2`; legacy
`solver_type: imex` means `lie` for a model with a source), `mesh` (`mesh.h5`; or `domain` +
`n_cells` for a uniform grid).  Strings from the JSON become option members here.
"""
import os

import numpy as np


def load_mesh(case_dir, settings):
    """The case mesh: `domain` + `n_cells` (uniform 1-D / 2-D grid), else the `mesh` file."""
    from zoomy_core.mesh import LSQMesh
    if "domain" in settings and "n_cells" in settings:
        d, n = settings["domain"], settings["n_cells"]
        if len(d) == 2:
            return LSQMesh.create_1d(tuple(map(float, d)), int(n if np.isscalar(n) else n[0]))
        return LSQMesh.create_2d(tuple(map(float, d)), int(n[0]), int(n[1]))
    path = os.path.join(case_dir, settings.get("mesh", "mesh.h5"))
    if path.endswith(".h5"):
        return LSQMesh.from_hdf5(path)
    from zoomy_core.mesh import BaseMesh
    from zoomy_core.mesh.fvm_mesh import FVMMesh
    return LSQMesh.from_fvm(FVMMesh.from_base(BaseMesh.from_msh(path)))


def _has_source(nsm):
    """Whether the model carries a nonzero source."""
    import sympy as sp
    S = getattr(nsm, "source", None)
    return S is not None and any(sp.sympify(e) != 0 for e in sp.flatten(S))


def systems(adapter, case_dir, settings):
    """`(nsm, extra)`: the case model as an NSM with the settings' reconstruction, and the
    pressure / corrector systems of a Chorin split (`split` in model.py), else `()`."""
    from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
    from zoomy_core.symsolver.strategies.chorin import predictor_nsm
    rec = ReconstructionSpec(
        order=int(settings.get("reconstruction_order", settings.get("order", 1))),
        limiter=Limiter(settings["limiter"]) if "limiter" in settings
        else Limiter.venkatakrishnan)
    mod = adapter.import_from_case(case_dir, "model")
    split = getattr(mod, "split", None)
    if split is not None:
        nsm = predictor_nsm(split.SM_pred, split.SM_press, split.SM_corr, rec)
        return nsm, (split.SM_press, split.SM_corr)
    model = getattr(mod, "model", None)
    if model is None:
        model = adapter.resolve_model(case_dir)
    return NumericalSystemModel.from_system_model(model, reconstruction=rec), ()


def options(settings, nsm):
    """The solver keywords: time window, CFL, output interval, IMEX."""
    from zoomy_core.symsolver.discretization.time import Imex
    time_end = float(settings.get("time_end", 0.1))
    if settings.get("imex"):
        imex = Imex(settings["imex"])
    elif settings.get("solver_type") == "imex" and _has_source(nsm):
        imex = Imex.lie
    else:
        imex = False
    return dict(time_end=time_end, cfl=float(settings.get("cfl", 0.9)),
                dt_write=time_end / int(settings.get("output_snapshots", 10)), imex=imex)


def solve_in_process(adapter, case_dir, output_dir, on_progress, hyperbolic, chorin):
    """March the case on an in-process symbolic solver (NumPy, JAX) into
    `output_dir/simulation.h5`; `hyperbolic` / `chorin` are the solver classes."""
    settings = adapter.load_settings(case_dir)
    mesh = load_mesh(case_dir, settings)
    nsm, extra = systems(adapter, case_dir, settings)
    kw = options(settings, nsm)
    solver = (chorin if extra else hyperbolic)(
        **kw, output=os.path.join(os.path.abspath(output_dir), "simulation.h5"))
    Q, Qaux = solver.setup(mesh, nsm, *extra)
    time, _, _ = solver.run(Q, Qaux)
    on_progress(-1, float(time), 0.0)


def solve_cxx(adapter, case_dir, output_dir, on_progress, hyperbolic, chorin, **solver_kw):
    """March the case on a C++ symbolic solver (AMReX, foam): print, build, run in
    `output_dir/run`, write its snapshots to `output_dir/simulation.h5`."""
    from zoomy_core.mesh import ensure_lsq_mesh
    from zoomy_core.symsolver.backend_cxx import initial_state, write_snapshots
    settings = adapter.load_settings(case_dir)
    nsm, extra = systems(adapter, case_dir, settings)
    mesh = ensure_lsq_mesh(load_mesh(case_dir, settings), nsm)
    solver = (chorin if extra else hyperbolic)(**options(settings, nsm), **solver_kw)
    model = (nsm, *extra) if extra else nsm
    time, _ = solver.run(mesh, model, initial_state(mesh, nsm),
                         os.path.join(os.path.abspath(output_dir), "run"))
    write_snapshots(os.path.join(output_dir, "simulation.h5"), mesh, nsm, solver.snapshots)
    on_progress(-1, float(time), 0.0)
