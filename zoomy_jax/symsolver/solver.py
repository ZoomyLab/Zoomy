"""Driver of the JAX backend: prints the solver template (hyperbolic, Chorin) of a
NumericalSystemModel and runs it against `zoomy_jax.symsolver.runtime`.  `ad_mode=AdMode.reverse`
checkpoints the time loop for `jax.grad`, `couple=` makes the march a preCICE participant,
`comm=` (mpi4py) runs it distributed, one rank per device.

References:
    Bradbury et al. 2018, https://github.com/jax-ml/jax
"""
from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import param
import sympy as sp

from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.mesh.geometry_xp import build_topology
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.symsolver import PyPrinter, default_wave_speed
from zoomy_core.misc.options import OptionParam
from zoomy_core.solver.constants import elliptic_constants
from zoomy_core.symsolver.settings import ChorinSettings, HyperbolicSettings, LinearSolver
from zoomy_core.symsolver.strategies.chorin import ChorinTemplate
from zoomy_core.symsolver.strategies.hyperbolic import CoupledFace, HyperbolicTemplate
from zoomy_core.systemmodel.padding import pad_to_square
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.mesh.mesh import convert_mesh_to_jax
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.runtime import AdMode  # noqa: F401
from zoomy_jax.transformation.jax_runtime import JaxRuntime


class SymbolicHyperbolicSolverJax(HyperbolicSettings):
    """Finite-volume solver on JAX: prints `HyperbolicTemplate` for a NumericalSystemModel, builds
    the runtime and marches the jitted program.  Use `setup(mesh, model)`, then `run(Q, Qaux)`.
    """
    fold_geometry = param.Boolean(
        default=False,
        doc="Bake the mesh geometry into the compiled loop as constants "
            "(XLA folds them; compile time grows with the mesh) instead of "
            "passing it as an input.")
    on_write = param.Callable(default=None, allow_None=True,
                              doc="on_write(time, Q, Qaux), forward mode only.")
    output = param.String(default=None, allow_None=True,
                          doc="HDF5 file for the io.write snapshots (the "
                              "/mesh + fields/iteration_k layout of the "
                              "post-processing readers).")
    ad_mode = OptionParam(AdMode, default=AdMode.forward,
                          doc="forward: jvp through the time loop; reverse: a checkpointed loop for "
                              "jax.grad.")
    ad_checkpoints = param.Integer(default=64, bounds=(1, None))
    couple = param.Dict(default=None, allow_None=True,
                        doc="preCICE: dict(adapter=, tag=, side=[, face=CoupledFace, "
                            "frozen_mass_row=bool]) (`HyperbolicTemplate.precice_march`; "
                            "a `ColumnPreciceAdapter` towards a VOF peer).")
    comm = param.Parameter(default=None, doc=(
        "mpi4py communicator: a distributed run, each rank on its piece of "
        "the mesh (owned cells + halo).  setup() takes the global mesh and "
        "returns the local state; partition.gather() collects it."))
    subcycling = param.Boolean(default=False,
                               doc="preCICE: adaptive steps inside a window "
                                   "(not exactly conservative).  False = one "
                                   "step per window, exactly conservative.")

    # ── setup ──────────────────────────────────────────────────────────────
    def _bind(self):
        """Distributed: this rank's GPU, before anything touches JAX."""
        if self.comm is not None and self.comm.Get_size() > 1:
            from zoomy_jax.symsolver.parallel import bind_device
            bind_device(self.comm)

    def setup(self, mesh, model):
        """Lift `model` to an NSM if needed, then build the solver on `mesh`.

        Args:
            mesh: a mesh, or the path of a mesh file
            model: SystemModel or NumericalSystemModel
        Returns:
            initial `(Q, Qaux)` (distributed: the local piece)
        """
        self._bind()
        nsm = (model if isinstance(model, NumericalSystemModel)
               else NumericalSystemModel.from_system_model(model))
        return self._setup(mesh, nsm)

    def _setup(self, mesh, nsm, systems=None, linsolve=None):
        """Distribute or read the mesh, build topology, runtime and the jitted program; returns the
        initial `(Q, Qaux)`.  Called by `setup` and by the Chorin subclass (which adds
        `systems` and `linsolve`).
        """
        from zoomy_core.mesh import distributed
        bcs = getattr(nsm.sm, "_bc_source", None)
        self.partition, self.global_mesh = None, mesh
        self.mesh_file = mesh if isinstance(mesh, (str, os.PathLike)) else None
        partners = None
        if self.comm is not None and self.comm.Get_size() > 1:
            local, self.partition = distributed.distribute(mesh, self.comm, bcs)
            mesh = ensure_lsq_mesh(local, nsm)
            self.partition.resolve_periodic(mesh)
            partners = np.asarray(mesh.boundary_face_cells)
        else:
            if isinstance(mesh, (str, os.PathLike)):
                mesh = distributed.read_mesh(mesh)
            mesh = ensure_lsq_mesh(mesh, nsm)
            if hasattr(mesh, "resolve_periodic_bcs") and bcs is not None:
                mesh.resolve_periodic_bcs(bcs)
            self.global_mesh = mesh
        sm, nc = nsm.sm, int(mesh.n_inner_cells)
        Q = np.zeros((len(sm.state), nc))
        Qaux = np.zeros((len(sm.aux_state), nc))
        if sm.initial_conditions is not None:
            Q = sm.initial_conditions.apply(mesh.cell_centers[:, :nc], Q)
        if sm.aux_initial_conditions is not None:
            Qaux = sm.aux_initial_conditions.apply(mesh.cell_centers[:, :nc],
                                                   Qaux)

        self.nsm, self.mesh_np = nsm, mesh
        self.topo = build_topology(
            mesh, nsm=nsm, partners=partners,
            periodic_ids=self.partition.periodic_ids if self.partition else None,
            systems=[sub for sub, _ in (systems or {}).values()])
        # the stencils are the topology's (periodic seams: the wrapped rings)
        self.mesh_j = dataclasses.replace(
            convert_mesh_to_jax(mesh),
            lsq_neighbors=jnp.asarray(self.topo.lsq_neighbors),
            lsq_boundary_face_neighbors=jnp.asarray(self.topo.lsq_boundary_face_neighbors))
        self.X = jnp.asarray(np.asarray(mesh.vertex_coordinates)[:mesh.dimension])
        # the same vectorised geometry in numpy (JAX op by op is slow here)
        self.mesh = R.Geometry(*map(jnp.asarray, R.geometry_of(
            self.topo, np.asarray(mesh.vertex_coordinates)[:mesh.dimension])))
        self.mesh_j = dataclasses.replace(self.mesh_j, lsq_gradQ=self.mesh.lsq_gradQ)
        default_wave_speed(nsm)
        self.rt = JaxRuntime.from_nsm(nsm)
        order = int(nsm.reconstruction.order)
        self.mood_active = bool(self.mood and order >= 2
                                and nsm.call.admissible is not None)
        on_save, on_load = self._restart()
        self.module = R.runtime(nsm, self.rt, mesh, self.mesh_j, self.topo,
                                ad_mode=self.ad_mode,
                                checkpoints=self.ad_checkpoints,
                                on_write=self._writer(), on_save=on_save,
                                on_load=on_load, couple=self.couple,
                                systems=systems, partition=self.partition,
                                implicit=dict(tol=self.implicit_tol,
                                              maxiter=self.implicit_maxiter,
                                              gmres_tol=self.gmres_tol,
                                              gmres_maxiter=self.gmres_maxiter,
                                              linsolve=linsolve))
        if self.imex and nsm.source_derivative_specs:
            raise NotImplementedError(
                "IMEX with derivative-dependent sources needs the global "
                "(Newton-GMRES) source solve; only the cell-local one exists")
        name, self.source = self._print()
        fn, topo = R.build(self.source, name, self.module), self.topo
        # the mesh enters as vertex coordinates; geometry is computed inside
        self._jit = R.hoisted_jit(lambda *a: fn(*a[:-2], R.geometry_of(topo, a[-2]),
                                                a[-1]))
        # no vertex-coordinate derivative: the geometry is an input, or with
        # fold_geometry a compile-time constant XLA folds into the loop
        geometry = self.mesh
        if self.fold_geometry:
            self._jit_fixed = jax.jit(lambda *a: fn(*a[:-1], geometry, a[-1]))
        else:
            jit_g = R.hoisted_jit(lambda *a: fn(*a))
            self._jit_fixed = lambda *a: jit_g(*a[:-1], geometry, a[-1])
        Q, Qaux = jnp.asarray(Q), jnp.asarray(Qaux)
        return Q, self.initial_aux(Q, Qaux)

    def _config(self):
        """The configuration of the solver template."""
        return dict(cfl=self.cfl, dimension=int(self.mesh_np.dimension),
                    mood=self.mood_active, dt_write=self.dt_write,
                    imex=self.imex, cfl_check_every=self.cfl_check_every,
                    wave_speed=self.cfl_wave_speed, dt_constant=self.dt,
                    time_order=self.time_order, dt_checkpoint=self.dt_checkpoint)

    def _print(self):
        """Print the template to Python source; returns `(name, source)`."""
        Q, Qaux, p, time, time_end, mesh = sp.symbols(
            "Q Qaux p time time_end mesh")
        template = HyperbolicTemplate(self.nsm, **self._config())
        if self.couple:
            c = self.couple
            return "hyperbolic_precice", PyPrinter().procedure(
                "hyperbolic_precice", (Q, Qaux, p, time, mesh),
                lambda *a: template.precice_march(
                    *a, side=c.get("side", 0), subcycling=self.subcycling,
                    face=c.get("face", CoupledFace.shared),
                    frozen_mass_row=c.get("frozen_mass_row", False)))
        return "hyperbolic", PyPrinter().procedure(
            "hyperbolic", (Q, Qaux, p, time, time_end, mesh), template.march)

    def _writer(self):
        """`on_write`, plus the HDF5 snapshots when `output` is set (distributed: gathered and
        written by rank 0 on the global mesh).
        """
        if not self.output:
            return self.on_write
        from zoomy_core.misc.io import _save_fields_to_hdf5, write_mesh_to_hdf5
        from zoomy_core.misc import misc
        part, names = self.partition, [str(v) for v in self.nsm.state]
        path = os.path.join(misc.get_main_directory(), self.output)  # as io.py
        root = part is None or part.rank == 0
        blocks = part is not None and part.parallel_io()
        if blocks:                     # every rank writes its own block
            part.write_mesh(path, names)
        elif root and self.mesh_file is not None:
            import h5py                # a mesh file: its /mesh as is
            with h5py.File(self.mesh_file, "r") as src, \
                    h5py.File(path, "w") as f:
                src.copy(src["mesh"], f, "mesh")
                f.create_group("fields").attrs["names"] = np.array(names, dtype="S")
        elif root:
            write_mesh_to_hdf5(self.output, self.global_mesh, names=names)
        count, user = [0], self.on_write

        def write(time, Q, Qaux):
            """Write snapshot number `count` and call the user's `on_write`."""
            if blocks:
                part.write_fields(path, count[0], time, Q, Qaux)
            else:
                Qg, Qauxg = ((Q, Qaux) if part is None
                             else (part.gather(Q), part.gather(Qaux)))
                if root:
                    _save_fields_to_hdf5(self.output, count[0], float(time),
                                         np.asarray(Qg), np.asarray(Qauxg))
            count[0] += 1
            if user is not None:
                user(time, Q, Qaux)
        return write

    def _restart(self):
        """The restart pair of `io.save_state` / `io.load_state`: one HDF5 file
        (`zoomy_core.symsolver.restart`), written by rank 0, each rank reading its cells.
        """
        from zoomy_core.symsolver import restart as rs
        if self.couple and (self.restart_from is not None or self.dt_checkpoint is not None):
            raise NotImplementedError("restart / checkpoints with preCICE coupling")
        part, names = self.partition, [str(v) for v in self.nsm.state]
        path = rs.restart_path(self.checkpoint, self.output)
        on_save = None
        if path is not None:
            def on_save(time, Q, Qaux):
                """Gather the march state and write it on rank 0."""
                if part is not None:
                    Q, Qaux = part.gather(np.asarray(Q)), part.gather(np.asarray(Qaux))
                if part is None or part.rank == 0:
                    rs.write_state(path, time, Q, Qaux, names)
        if self.restart_from is None:
            return on_save, None
        rs.check_state(self.restart_from, names,
                       part.n_global if part else int(self.mesh_np.n_inner_cells))

        def on_load(time, Q, Qaux):
            """The saved state, this rank's cells."""
            t, Q, Qaux = rs.read_state(self.restart_from)
            return (t, Q, Qaux) if part is None else (t, part.scatter(Q), part.scatter(Qaux))
        return on_save, on_load

    def initial_aux(self, Q, Qaux, parameters=None, X=None):
        """The aux of the initial state: the pointwise formula plus the derivative aux with the BC
        ghosts and the boundary closure.
        """
        from zoomy_core.symsolver.discretization.residual import derivative_rows
        p = self.parameters if parameters is None else parameters
        mesh = self.mesh if X is None else R.geometry_of(self.topo, X)
        m, rt = self.module, self.rt
        if rt.update_aux_variables is not None:
            Qaux = m["cells"](rt.update_aux_variables, Q, Qaux, p, 0.0,
                              1.0, mesh)
        if not self.nsm.aux_derivatives:
            return Qaux
        rows, functions = derivative_rows(self.nsm)
        ghost = (rt.boundary_ghost_state
                 if self.nsm.call.boundary_ghost_state is not None else None)
        if functions or ghost is None:
            Qaux = m["derivative_aux"](rt.aux_derivatives, Q, Qaux, mesh)
        if rows and ghost is not None:
            Qg = m["boundary_ghosts"](ghost, Q, Qaux, p, 0.0, mesh)
            Qaux = m["derivative_aux"](rt.aux_derivatives, Q, Qaux, mesh, Qg,
                                       rows)
        return Qaux

    @property
    def parameters(self):
        """The runtime's parameter vector."""
        return self.rt.parameters

    # ── run ────────────────────────────────────────────────────────────────
    def run(self, Q, Qaux, parameters=None, time0=None, X=None):
        """March from `time0` (default `time_start`; a restart: the saved state) to `time_end`;
        returns `(time, Q, Qaux)`.  `parameters` and the vertex coordinates `X`
        `(dim, n_vertices)` may be traced for AD.
        """
        p = self.parameters if parameters is None else parameters
        time0 = self.time_start if time0 is None else time0
        tok = jnp.int64(0)
        args = (Q, Qaux, p, time0) + (() if self.couple else (self.time_end,))
        out = (self._jit_fixed(*args, tok) if X is None
               else self._jit(*args, X, tok))
        if not isinstance(out[1], jax.core.Tracer):     # traced (AD): the caller waits
            jax.block_until_ready(out)                    # dispatch is asynchronous: return
            jax.effects_barrier()                         # when the march and its writes are done
        return out[0], out[1], out[2]


class SymbolicChorinSolverJax(SymbolicHyperbolicSolverJax, ChorinSettings):
    """The Chorin projection solver on JAX (`ChorinTemplate`): `setup(mesh, pred, press, corr)`
    takes the predictor NSM on the full state and the pressure and corrector systems.  The split
    must use zoomy's time-step symbol `dt` (positive).

    References:
        Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
    """
    pressure_linear_solver = OptionParam(
        LinearSolver, default=LinearSolver.auto,
        members=[LinearSolver.auto, LinearSolver.gmres, LinearSolver.amgx, LinearSolver.pyamg,
                 LinearSolver.hypre],
        doc="amgx (GPU, optional JAX-AMG), pyamg (CPU, optional PyAMG), hypre (CPU, serial or "
            "MPI: BoomerAMG + FlexGMRES, the shared library of build_hypre_cuda.sh cpu-env) or "
            "gmres (matrix-free, always available); auto: amgx on a GPU, hypre on the CPU, else "
            "pyamg, when installed, else gmres.")
    pressure_amg_rebuild = param.Integer(
        default=10, bounds=(1, None),
        doc="'pyamg' / 'hypre': rebuild the AMG hierarchy at least every this "
            "many pressure solves; in between it preconditions the current "
            "matrix (a lagged preconditioner).")
    pressure_amg_growth = param.Number(
        default=2.0, bounds=(1, None),
        doc="'pyamg' / 'hypre': rebuild early once the GMRES iterations exceed "
            "this factor times those right after the last rebuild.")
    pressure_amgx = param.Dict(
        default=dict(solver="FGMRES", gmres_n_restart=30, max_iters=500,
                     preconditioner={"solver": "AMG"}),
        doc="'amgx': AmgX solver configuration passed to jaxamg.solve.")

    pressure_coarse = param.Boolean(
        default=True,
        doc="Distributed 'pyamg': add a coarse correction to "
            "the per-rank block preconditioner, one unknown per rank and "
            "pressure row (two-level additive Schwarz); without it the "
            "iterations grow with the number of ranks.")

    def setup(self, mesh, pred, press, corr):
        """Check the split, build the pressure and corrector runtimes and the linear-solver
        settings, then `_setup`; returns the initial `(Q, Qaux)`.
        """
        self._bind()
        if self.couple:
            raise NotImplementedError("Chorin with preCICE coupling")
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
                raise ValueError(f"{name}: split with zoomy's dt symbol "
                                 f"(positive=True), found {stray}")
            if list(sm.parameters.keys()) != list(pred.parameters.keys()):
                raise ValueError(f"{name}: parameters differ from pred's")
            nsm = (sm if isinstance(sm, NumericalSystemModel)
                   else NumericalSystemModel.from_system_model(sm))
            nsm._symbolic_name = name
            systems[name] = (nsm, JaxRuntime.from_nsm(default_wave_speed(nsm)))
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
            tol=tol, restart=int(ell["c_elliptic_restart"]),
            maxiter=self.pressure_maxit, accept=max(1e-6, 10.0 * tol),
            backend=self._pressure_backend(), amgx=self.pressure_amgx,
            coarse=self.pressure_coarse,
            amg_rebuild=self.pressure_amg_rebuild, amg_growth=self.pressure_amg_growth))

    def _pressure_backend(self):
        """The pressure linear solver: the configured one; for 'auto' amgx on a GPU, hypre on the
        CPU (its library built), else pyamg, when installed, else gmres.
        """
        if self.pressure_linear_solver != "auto":
            return self.pressure_linear_solver
        gpu = jax.default_backend() == "gpu"
        if not gpu:
            from zoomy_jax.symsolver import hypre
            if hypre.available():
                return "hypre"
        name = "amgx" if gpu else "pyamg"
        try:
            __import__({"amgx": "jaxamg", "pyamg": "pyamg"}[name])
        except ImportError:
            return "gmres"
        return name

    def _newton(self):
        """Newton `(tol, maxit)` for a nonlinear pressure constraint, `None` for the linear
        solve.
        """
        return (None if self.pressure_solve == "linear" else
                (sp.Float(self.pressure_newton_tol), int(self.pressure_newton_maxit)))

    def _print(self):
        """Print the Chorin template to Python source; returns `(name, source)`."""
        Q, Qaux, p, time, time_end, mesh = sp.symbols(
            "Q Qaux p time time_end mesh")
        template = ChorinTemplate(
            self.nsm, self.press, self.corr, pressure_newton=self._newton(),
            projection=self.projection,
            **self._config())
        return "chorin", PyPrinter().procedure(
            "chorin", (Q, Qaux, p, time, time_end, mesh), template.march)
