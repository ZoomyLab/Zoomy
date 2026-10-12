"""Driver: print the hyperbolic templates for JAX and run them.

    solver = SymbolicHyperbolicSolverJax(time_end=0.5)       # adaptive CFL step
    solver = SymbolicHyperbolicSolverJax(time_end=0.5, dt=1e-3)  # constant step
    Q, Qaux = solver.setup(mesh, nsm)
    time, Q, Qaux = solver.run(Q, Qaux)                     # parameters= for AD

``ad_mode="reverse"`` swaps the time loop for a checkpointed one (``jax.grad``);
``couple=`` turns the march into one preCICE participant; ``comm=`` (an mpi4py
communicator) runs it distributed, one rank per device
(:mod:`zoomy_jax.symsolver.parallel`).
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
from zoomy_core.symsolver import PyPrinter
from zoomy_core.fvm.solver_chorin_vam_numpy import _pad_to_square
from zoomy_core.solver.constants import ELLIPTIC_RTOL, elliptic_constants
from zoomy_core.symsolver import chorin as C
from zoomy_core.symsolver import hyperbolic as T
from zoomy_core.systemmodel.system_model import _DT_SYMBOL
from zoomy_jax.mesh.mesh import convert_mesh_to_jax
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.transformation.jax_runtime import JaxRuntime


class SymbolicHyperbolicSolverJax(param.Parameterized):
    time_end = param.Number(default=None, allow_None=True,
                            doc="End time (an uncoupled run).")
    cfl = param.Number(default=0.9, bounds=(0, None),
                       doc="CFL number of the adaptive step (and of the check).")
    dt = param.Number(default=None, allow_None=True,
                      doc="Constant time step; None = adaptive CFL step.  A "
                          "constant step above the CFL step warns once.")
    cfl_check_every = param.Integer(default=10, bounds=(0, None),
                                    doc="Constant dt: test it against the CFL "
                                        "step every this many steps (1: each "
                                        "step, 0: never).")
    cfl_wave_speed = param.Selector(
        default="faces", objects=["faces", "cells"],
        doc="CFL wave speeds: 'faces' from the first RK stage's Riemann "
            "problem (the interface states; no extra eigensolve), 'cells' "
            "from the cell states of each face (an eigensolve of its own).")
    spectral_radius = param.Selector(
        default="auto", objects=["auto", "eig", "power"],
        doc="Wave speed of a model without closed-form eigenvalues: 'eig' "
            "the numerical eigenvalues (CPU-only in JAX), 'power' power "
            "iteration on A^2 (any device), 'auto' eig on the CPU, power on "
            "a GPU.  An NSM that sets spectral_radius_method keeps it.")
    spectral_power_safety = param.Number(default=1.1, bounds=(1, None),
                                         doc="'power': factor on the estimate.")
    spectral_power_steps = param.Integer(default=10, bounds=(1, None))
    fold_geometry = param.Boolean(
        default=False,
        doc="Bake the mesh geometry into the compiled loop as constants "
            "(XLA folds them; compile time grows with the mesh) instead of "
            "passing it as an input.")
    mood = param.Boolean(default=True,
                         doc="A-posteriori MOOD at order >= 2 (cells failing "
                             "the model's admissibility redo at order 1).  "
                             "What is reconstructed, and how it is limited, "
                             "is the NSM's ReconstructionSpec.")
    dt_write = param.Number(default=None, allow_None=True,
                            doc="Output interval; None = no output.")
    on_write = param.Callable(default=None, allow_None=True,
                              doc="on_write(time, Q, Qaux), forward mode only.")
    output = param.String(default=None, allow_None=True,
                          doc="HDF5 file for the io.write snapshots (the "
                              "/mesh + fields/iteration_k layout of the "
                              "post-processing readers).")
    ad_mode = param.Selector(default="forward", objects=["forward", "reverse"])
    ad_checkpoints = param.Integer(default=64, bounds=(1, None))
    couple = param.Dict(default=None, allow_None=True,
                        doc="preCICE: dict(adapter=, tag=, side=).")
    imex = param.Boolean(default=False,
                         doc="IMEX (Lie): explicit flux step, then implicit "
                             "diffusion and an implicit-Euler source.")
    implicit_tol = param.Number(default=1e-8, bounds=(0, None),
                                doc="Newton tolerance ||R||_2 (implicit solves).")
    implicit_maxiter = param.Integer(default=6, bounds=(1, None))
    gmres_tol = param.Number(default=1e-7, bounds=(0, None))
    gmres_maxiter = param.Integer(default=30, bounds=(1, None))
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
        self._bind()
        nsm = (model if isinstance(model, NumericalSystemModel)
               else NumericalSystemModel.from_system_model(model))
        return self._setup(mesh, nsm)

    def _setup(self, mesh, nsm, systems=None, linsolve=None):
        """``mesh``: a mesh, or the path of a mesh file
        (:func:`zoomy_core.mesh.distributed.write_mesh`)."""
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
            periodic_ids=self.partition.periodic_ids if self.partition else None)
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
        self._choose_spectral_radius(nsm)
        self.rt = JaxRuntime.from_nsm(nsm)
        order = int(nsm.reconstruction.order)
        self.mood_active = bool(self.mood and order >= 2
                                and nsm.call.admissible is not None)
        self.module = R.runtime(nsm, self.rt, mesh, self.mesh_j,
                                ad_mode=self.ad_mode,
                                checkpoints=self.ad_checkpoints,
                                on_write=self._writer(), couple=self.couple,
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

    def _choose_spectral_radius(self, nsm):
        """Set the NSM's spectral-radius method from ``spectral_radius``
        (before its kernels are built)."""
        if "spectral_radius_method" in vars(nsm):        # the NSM's own choice
            return
        method = self.spectral_radius
        if method == "auto":
            method = "power" if jax.default_backend() == "gpu" else "eig"
        nsm.spectral_radius_method = method
        nsm.spectral_power_safety = float(self.spectral_power_safety)
        nsm.spectral_power_steps = int(self.spectral_power_steps)
        nsm._call = None                                  # rebuild the table

    def _print(self):
        Q, Qaux, p, time, time_end, mesh = sp.symbols(
            "Q Qaux p time time_end mesh")
        kw = dict(cfl=self.cfl, dimension=int(self.mesh_np.dimension),
                  mood=self.mood_active, imex=self.imex,
                  cfl_check_every=self.cfl_check_every,
                  wave_speed=self.cfl_wave_speed,
                  dt_constant=None if self.dt is None else sp.Float(self.dt))
        if self.couple:
            return "hyperbolic_precice", PyPrinter().procedure(
                "hyperbolic_precice", (Q, Qaux, p, time, mesh),
                lambda *a: T.hyperbolic_precice(
                    self.nsm, *a, side=self.couple["side"],
                    subcycling=self.subcycling, dt_write=self.dt_write, **kw))
        return "hyperbolic", PyPrinter().procedure(
            "hyperbolic", (Q, Qaux, p, time, time_end, mesh),
            lambda *a: T.hyperbolic(self.nsm, *a, dt_write=self.dt_write, **kw))

    def _writer(self):
        """``on_write``, plus the HDF5 snapshots when ``output`` is set.
        Distributed: ``on_write`` sees each rank's local state; the snapshot
        is gathered and written by rank 0, on the global mesh."""
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

    def initial_aux(self, Q, Qaux, parameters=None, X=None):
        """The aux of the initial state: pointwise formula + derivative aux."""
        p = self.parameters if parameters is None else parameters
        mesh = self.mesh if X is None else R.geometry_of(self.topo, X)
        m = self.module
        if self.rt.update_aux_variables is not None:
            Qaux = m["cells"](self.rt.update_aux_variables, Q, Qaux, p, 0.0,
                              1.0, mesh)
        if self.nsm.aux_derivatives:
            Qaux = m["derivative_aux"](self.nsm.aux_derivatives, Q, Qaux, mesh)
        return Qaux

    @property
    def parameters(self):
        return self.rt.parameters

    # ── run ────────────────────────────────────────────────────────────────
    def run(self, Q, Qaux, parameters=None, time0=0.0, X=None):
        """March; returns ``(time, Q, Qaux)``.  ``parameters`` and the vertex
        coordinates ``X`` (``(dim, n_vertices)``) may be traced for AD."""
        p = self.parameters if parameters is None else parameters
        tok = jnp.int64(0)
        args = (Q, Qaux, p, time0) + (() if self.couple else (self.time_end,))
        out = (self._jit_fixed(*args, tok) if X is None
               else self._jit(*args, X, tok))
        return out[0], out[1], out[2]


class SymbolicChorinSolverJax(SymbolicHyperbolicSolverJax):
    """The Chorin projection solver on JAX (``zoomy_core.symsolver.chorin``).

        solver = SymbolicChorinSolverJax(time_end=0.5, dt=2**-7)
        Q, Qaux = solver.setup(mesh, pred, press, corr)   # the model's split
        time, Q, Qaux = solver.run(Q, Qaux)

    ``pred`` is the predictor NSM (or SystemModel, then default numerics) on
    the full state; ``press`` / ``corr`` the pressure and corrector systems.
    The split must use zoomy's time-step symbol ``dt`` (positive).
    """
    time_order = param.Integer(default=1, bounds=(1, 2),
                               doc="Outer SSP-RK stages, each a full "
                                   "projection stage.")
    pressure_tol = param.Number(default=ELLIPTIC_RTOL, bounds=(0, None))
    pressure_maxit = param.Integer(default=100, bounds=(1, None),
                                   doc="GMRES restarts.")
    pressure_restart = param.Integer(default=None, allow_None=True,
                                     bounds=(1, None),
                                     doc="None: from the block size.")
    pressure_solve = param.Selector(
        default="linear", objects=["linear", "newton"],
        doc="'linear': the constraint is affine in the pressure, one linear "
            "solve.  'newton': a nonlinear constraint, Newton iterations.")
    pressure_newton_tol = param.Number(default=1e-10, bounds=(0, None),
                                       doc="Newton: stop at max|R| <= tol.")
    pressure_newton_maxit = param.Integer(default=10, bounds=(1, None))
    pressure_linear_solver = param.Selector(
        default="auto", objects=["auto", "gmres", "amgx", "pyamg"],
        doc="AMG-preconditioned GMRES: 'amgx' on a GPU (NVIDIA AmgX through "
            "the optional JAX-AMG package), 'pyamg' on the CPU (the optional "
            "PyAMG, smoothed aggregation on the host).  'gmres': matrix-free "
            "restarted GMRES (jax.scipy), always available.  'auto': amgx on "
            "a GPU, pyamg on the CPU, when installed; else gmres.")
    pressure_amg_rebuild = param.Integer(
        default=10, bounds=(1, None),
        doc="'pyamg': rebuild the AMG hierarchy at least every this many "
            "pressure solves; in between it preconditions the current matrix "
            "(a lagged preconditioner).")
    pressure_amg_growth = param.Number(
        default=2.0, bounds=(1, None),
        doc="'pyamg': rebuild early once the GMRES iterations exceed this "
            "factor times those right after the last rebuild.")
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
        self._bind()
        if self.couple:
            raise NotImplementedError("Chorin with preCICE coupling")
        if not isinstance(pred, NumericalSystemModel):
            pred = NumericalSystemModel.from_system_model(
                _pad_to_square(pred), additional_systems=[press, corr])
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
            systems[name] = (nsm, JaxRuntime.from_nsm(nsm))
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
        if self.pressure_linear_solver != "auto":
            return self.pressure_linear_solver
        name = "amgx" if jax.default_backend() == "gpu" else "pyamg"
        try:
            __import__({"amgx": "jaxamg", "pyamg": "pyamg"}[name])
        except ImportError:
            return "gmres"
        return name

    def _newton(self):
        return (None if self.pressure_solve == "linear" else
                (sp.Float(self.pressure_newton_tol), int(self.pressure_newton_maxit)))

    def _print(self):
        Q, Qaux, p, time, time_end, mesh = sp.symbols(
            "Q Qaux p time time_end mesh")
        return "chorin", PyPrinter().procedure(
            "chorin", (Q, Qaux, p, time, time_end, mesh),
            lambda *a: C.chorin(
                self.nsm, self.press, self.corr, *a, cfl=self.cfl,
                dimension=int(self.mesh_np.dimension), mood=self.mood_active,
                imex=self.imex, time_order=self.time_order,
                dt_write=self.dt_write, cfl_check_every=self.cfl_check_every,
                wave_speed=self.cfl_wave_speed, pressure_newton=self._newton(),
                dt_constant=None if self.dt is None else sp.Float(self.dt)))
