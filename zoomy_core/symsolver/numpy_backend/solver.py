"""Driver of the NumPy backend: prints the solver template (hyperbolic, Chorin) of a
NumericalSystemModel with `PyPrinter` and runs it against `numpy_backend.runtime`, serially;
the default backend (no JAX, no compiler).  Mirrors `zoomy_jax.symsolver.solver` without AD,
GPU, MPI and the hyperbolic preCICE coupling (the Chorin march couples by overlapping Schwarz).
"""
from __future__ import annotations

import os

import numpy as np
import param
import sympy as sp

from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.mesh.geometry_xp import build_topology
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.misc.options import OptionParam
from zoomy_core.solver.constants import elliptic_constants
from zoomy_core.symsolver import PyPrinter, default_wave_speed
from zoomy_core.symsolver.numpy_backend import runtime as R
from zoomy_core.symsolver.numpy_backend.kernels import NumpyKernels
from zoomy_core.symsolver.settings import ChorinSettings, HyperbolicSettings, LinearSolver
from zoomy_core.symsolver.strategies.chorin import ChorinTemplate, check_overlap
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_core.systemmodel.padding import pad_to_square
from zoomy_core.systemmodel.system_model import _DT_SYMBOL


class SymbolicHyperbolicSolverNumpy(HyperbolicSettings):
    """Finite-volume solver in NumPy: prints `HyperbolicTemplate` for a NumericalSystemModel and
    runs it serially.  Use `setup(mesh, model)`, then `run(Q, Qaux)`.
    """
    on_write = param.Callable(default=None, allow_None=True, doc="on_write(time, Q, Qaux).")
    output = param.String(default=None, allow_None=True,
                          doc="HDF5 file for the io.write snapshots.")
    comm = param.Parameter(default=None, doc="Not supported (serial only).")

    # ── setup ──────────────────────────────────────────────────────────────
    def setup(self, mesh, model):
        """Lift `model` to an NSM if needed, then build the solver on `mesh`.

        Args:
            mesh: a mesh, or the path of a mesh file
            model: SystemModel or NumericalSystemModel
        Returns:
            initial `(Q, Qaux)`
        """
        nsm = (model if isinstance(model, NumericalSystemModel)
               else NumericalSystemModel.from_system_model(model))
        return self._setup(mesh, nsm)

    def _setup(self, mesh, nsm, systems=None, linsolve=None):
        """Read the mesh, build topology, geometry, kernels and the printed program; returns the
        initial `(Q, Qaux)`.
        """
        if self.comm is not None:
            raise NotImplementedError("numpy backend: serial only (comm), use zoomy_jax")
        from zoomy_core.mesh import distributed
        bcs = getattr(nsm.sm, "_bc_source", None)
        self.mesh_file = mesh if isinstance(mesh, (str, os.PathLike)) else None
        if self.mesh_file is not None:
            mesh = distributed.read_mesh(mesh)
        mesh = ensure_lsq_mesh(mesh, nsm)
        if hasattr(mesh, "resolve_periodic_bcs") and bcs is not None:
            mesh.resolve_periodic_bcs(bcs)
        self.partition, self.global_mesh = None, mesh          # serial (the JAX API)
        sm, nc = nsm.sm, int(mesh.n_inner_cells)
        Q = np.zeros((len(sm.state), nc))
        Qaux = np.zeros((len(sm.aux_state), nc))
        if sm.initial_conditions is not None:
            Q = sm.initial_conditions.apply(mesh.cell_centers[:, :nc], Q)
        if sm.aux_initial_conditions is not None:
            Qaux = sm.aux_initial_conditions.apply(mesh.cell_centers[:, :nc], Qaux)
        self.nsm, self.mesh_np = nsm, mesh
        self.topo = build_topology(mesh, nsm=nsm,
                                   systems=[sub for sub, _ in (systems or {}).values()])
        X = np.asarray(mesh.vertex_coordinates)[:mesh.dimension]
        self.mesh, geom = R.geometry_of(self.topo, X)
        default_wave_speed(nsm)
        self.rt = NumpyKernels(nsm)
        order = int(nsm.reconstruction.order)
        self.mood_active = bool(self.mood and order >= 2 and nsm.call.admissible is not None)
        on_save, on_load = self._restart()
        self.module = R.runtime(nsm, self.rt, mesh, self.topo, geom, on_write=self._writer(),
                                on_save=on_save, on_load=on_load, systems=systems,
                                couple=getattr(self, "couple", None),
                                implicit=dict(tol=self.implicit_tol,
                                              maxiter=self.implicit_maxiter,
                                              gmres_tol=self.gmres_tol,
                                              gmres_maxiter=self.gmres_maxiter,
                                              linsolve=linsolve))
        if self.imex and nsm.source_derivative_specs:
            raise NotImplementedError(
                "IMEX with derivative-dependent sources needs the global (Newton-GMRES) "
                "source solve; only the cell-local one exists")
        name, self.source = self._print()
        self._fn = R.build(self.source, name, self.module)
        Q = np.asarray(Q, dtype=float)
        return Q, self.initial_aux(Q, np.asarray(Qaux, dtype=float))

    def _config(self):
        """The configuration of the solver template."""
        return dict(cfl=self.cfl, dimension=int(self.mesh_np.dimension),
                    mood=self.mood_active, dt_write=self.dt_write, imex=self.imex,
                    cfl_check_every=self.cfl_check_every, wave_speed=self.cfl_wave_speed,
                    dt_constant=self.dt, time_order=self.time_order,
                    dt_checkpoint=self.dt_checkpoint)

    def _print(self):
        """Print the template to Python source; returns `(name, source)`."""
        Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
        template = HyperbolicTemplate(self.nsm, **self._config())
        return "hyperbolic", PyPrinter().procedure(
            "hyperbolic", (Q, Qaux, p, time, time_end, mesh), template.march)

    def _writer(self):
        """`on_write`, plus the HDF5 snapshots when `output` is set."""
        if not self.output:
            return self.on_write
        from zoomy_core.misc import misc
        from zoomy_core.misc.io import _save_fields_to_hdf5, write_mesh_to_hdf5
        names = [str(v) for v in self.nsm.state]
        path = os.path.join(misc.get_main_directory(), self.output)
        if self.mesh_file is not None:
            import h5py
            with h5py.File(self.mesh_file, "r") as src, h5py.File(path, "w") as f:
                src.copy(src["mesh"], f, "mesh")
                f.create_group("fields").attrs["names"] = np.array(names, dtype="S")
        else:
            write_mesh_to_hdf5(self.output, self.mesh_np, names=names)
        count, user = [0], self.on_write

        def write(time, Q, Qaux):
            """Write snapshot number `count` and call the user's `on_write`."""
            _save_fields_to_hdf5(self.output, count[0], float(time), np.asarray(Q),
                                 np.asarray(Qaux))
            count[0] += 1
            if user is not None:
                user(time, Q, Qaux)
        return write

    def _restart(self):
        """The restart pair of `io.save_state` / `io.load_state`: one HDF5 file
        (`zoomy_core.symsolver.restart`); checks `restart_from` first.
        """
        from zoomy_core.symsolver import restart as rs
        names = [str(v) for v in self.nsm.state]
        path = rs.restart_path(self.checkpoint, self.output)
        on_save = None if path is None else (
            lambda time, Q, Qaux: rs.write_state(path, time, Q, Qaux, names))
        if self.restart_from is None:
            return on_save, None
        rs.check_state(self.restart_from, names, int(self.mesh_np.n_inner_cells))
        return on_save, lambda time, Q, Qaux: rs.read_state(self.restart_from)

    def initial_aux(self, Q, Qaux, parameters=None):
        """The aux of the initial state: the pointwise formula plus the derivative aux with the
        BC ghosts and the boundary closure.
        """
        from zoomy_core.symsolver.discretization.residual import derivative_rows
        p = self.parameters if parameters is None else parameters
        m, rt = self.module, self.rt
        if rt.update_aux_variables is not None:
            Qaux = m["cells"](rt.update_aux_variables, Q, Qaux, p, 0.0, 1.0, self.mesh)
        if not self.nsm.aux_derivatives:
            return Qaux
        rows, functions = derivative_rows(self.nsm)
        ghost = rt.boundary_ghost_state if self.nsm.call.boundary_ghost_state is not None \
            else None
        if functions or ghost is None:
            Qaux = m["derivative_aux"](rt.aux_derivatives, Q, Qaux, self.mesh)
        if rows and ghost is not None:
            Qg = m["boundary_ghosts"](ghost, Q, Qaux, p, 0.0, self.mesh)
            Qaux = m["derivative_aux"](rt.aux_derivatives, Q, Qaux, self.mesh, Qg, rows)
        return Qaux

    @property
    def parameters(self):
        """The kernels' parameter vector."""
        return self.rt.parameters

    # ── run ────────────────────────────────────────────────────────────────
    def run(self, Q, Qaux, parameters=None, time0=None):
        """March from `time0` (default `time_start`; a restart: the saved state) to `time_end`;
        returns `(time, Q, Qaux)`."""
        p = self.parameters if parameters is None else np.asarray(parameters, dtype=float)
        time0 = self.time_start if time0 is None else time0
        end = () if getattr(self, "couple", None) else (float(self.time_end),)
        out = self._fn(np.asarray(Q, dtype=float), np.asarray(Qaux, dtype=float), p,
                       float(time0), *end, self.mesh, 0)
        return float(out[0]), out[1], out[2]


class SymbolicChorinSolverNumpy(SymbolicHyperbolicSolverNumpy, ChorinSettings):
    """The Chorin projection solver in NumPy (`ChorinTemplate`): `setup(mesh, pred, press,
    corr)` takes the predictor NSM on the full state and the pressure and corrector systems.

    References:
        Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
    """
    pressure_linear_solver = OptionParam(
        LinearSolver, default=LinearSolver.auto,
        members=[LinearSolver.auto, LinearSolver.gmres, LinearSolver.pyamg],
        doc="pyamg (optional PyAMG) or gmres (matrix-free); auto: pyamg when installed, else "
            "gmres.")
    pressure_amg_rebuild = param.Integer(
        default=10, bounds=(1, None),
        doc="'pyamg': rebuild the AMG hierarchy at least every this many pressure solves.")
    pressure_amg_growth = param.Number(
        default=2.0, bounds=(1, None),
        doc="'pyamg': rebuild early once the GMRES iterations exceed this factor times those "
            "right after the last rebuild.")
    couple = param.Dict(default=None, allow_None=True,
                        doc="preCICE: dict(adapter=, tag=, overlap=layers) "
                            "(`ChorinTemplate.precice_march`: the mesh reaches `layers` cell "
                            "rings across the interface, its outer boundary tagged `tag`; "
                            "the adapter a `zoomy_core.symsolver.precice_adapter."
                            "PreciceAdapter`).")

    def setup(self, mesh, pred, press, corr):
        """Check the split, build the pressure and corrector kernels and the linear-solver
        settings, then `_setup`; returns the initial `(Q, Qaux)`.
        """
        if self.couple and self.couple.get("overlap") is None:
            raise ValueError("Chorin with preCICE coupling: couple=dict(adapter=, tag=, "
                             "overlap=layers) (`ChorinTemplate.precice_march`)")
        if not isinstance(pred, NumericalSystemModel):
            pred = NumericalSystemModel.from_system_model(
                pad_to_square(pred), additional_systems=[press, corr])
        if pred.n_equations != pred.n_state:
            raise ValueError("pred must act on the full state (pad it)")
        systems = {}
        for name, sm in (("press", press), ("corr", corr)):
            used = set().union(*(e.free_symbols for e in
                                 (sm.source, sm.update_variables) if e is not None))
            stray = {s for s in used if str(s) == "dt" and s != _DT_SYMBOL}
            if stray:
                raise ValueError(f"{name}: split with zoomy's dt symbol (positive=True), "
                                 f"found {stray}")
            if list(sm.parameters.keys()) != list(pred.parameters.keys()):
                raise ValueError(f"{name}: parameters differ from pred's")
            nsm = (sm if isinstance(sm, NumericalSystemModel)
                   else NumericalSystemModel.from_system_model(sm))
            nsm._symbolic_name = name
            systems[name] = (nsm, NumpyKernels(default_wave_speed(nsm)))
        pred._symbolic_name = "pred"
        self.press, self.corr = systems["press"][0], systems["corr"][0]
        from zoomy_core.mesh import distributed
        n_cells = (distributed.source(mesh).n_cells
                   if isinstance(mesh, (str, os.PathLike)) else int(mesh.n_inner_cells))
        n = len(press.equation_to_state_index) * n_cells
        ell = elliptic_constants(n_unknowns=n, tol=self.pressure_tol,
                                 restart=self.pressure_restart).values
        tol = float(ell["c_elliptic_tol"])
        return self._setup(mesh, pred, systems, linsolve=dict(
            tol=tol, restart=int(ell["c_elliptic_restart"]), maxiter=self.pressure_maxit,
            accept=max(1e-6, 10.0 * tol), backend=self._pressure_backend(),
            amg_rebuild=self.pressure_amg_rebuild, amg_growth=self.pressure_amg_growth))

    def _pressure_backend(self):
        """The pressure linear solver: the configured one; 'auto' = pyamg when installed."""
        if self.pressure_linear_solver != "auto":
            return self.pressure_linear_solver
        try:
            import pyamg  # noqa: F401
        except ImportError:
            return "gmres"
        return "pyamg"

    def _newton(self):
        """Newton `(tol, maxit)` for a nonlinear pressure constraint, `None` for the linear
        solve.
        """
        return (None if self.pressure_solve == "linear" else
                (sp.Float(self.pressure_newton_tol), int(self.pressure_newton_maxit)))

    def _print(self):
        """Print the Chorin template to Python source; returns `(name, source)`."""
        Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
        template = ChorinTemplate(self.nsm, self.press, self.corr,
                                  pressure_newton=self._newton(),
                                  projection=self.projection, **self._config())
        if self.couple:
            check_overlap(template, self.couple["overlap"])
            return "chorin_precice", PyPrinter().procedure(
                "chorin_precice", (Q, Qaux, p, time, mesh), template.precice_march)
        return "chorin", PyPrinter().procedure(
            "chorin", (Q, Qaux, p, time, time_end, mesh), template.march)
