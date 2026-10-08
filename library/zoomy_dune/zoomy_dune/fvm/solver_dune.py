"""Explicit DG solver on dune-fem.

The weak form is assembled from the operators zoomy_core lowers to UFL; the
solver owns the grid, the spaces, the time loop and the output.

    solver = HyperbolicSolver(settings=settings, time_end=10.0)
    solver.setup_simulation(mesh_path, model, initial_condition)
    solver.run_simulation()
"""
import os
import shutil
import xml.etree.ElementTree as ET

import meshio
import numpy as np
import param
import ufl
from ufl import FacetNormal, TestFunction, TrialFunction, as_vector, dS, ds, dot, dx

import dune.fem
from dune.alugrid import aluSimplexGrid
from dune.fem.operator import molGalerkin
from dune.fem.space import finiteVolume, lagrange
from dune.fem.view import adaptiveLeafGridView
from dune.generator import algorithm
from dune.grid import reader

from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_core.misc.misc import Settings, Zstruct
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.transformation.to_ufl import UFLRuntimeModel


def vec(expr):
    """Lowered operators are (n, 1) columns; UFL does not collapse them."""
    shape = expr.ufl_shape
    if len(shape) == 1:
        return expr
    return as_vector([expr[i, 0] for i in range(shape[0])])


def scalar(expr):
    shape = expr.ufl_shape
    return expr if not shape else (expr[0] if len(shape) == 1 else expr[0, 0])


class HyperbolicSolver(param.Parameterized):
    """Explicit SSP-RK march of a hyperbolic system on an unstructured grid."""

    time_end = param.Number(1.0, bounds=(0, None), doc="Simulation end time")
    CFL = param.Number(0.9, bounds=(0.0, 1.0), doc="Courant number")
    dg_degree = param.Integer(0, bounds=(0, 2), doc="DG polynomial degree")
    threads = param.Integer(4, bounds=(1, None), doc="OpenMP threads")
    settings = param.Parameter(default=None, doc="Settings (output directory, snapshots)")
    riemann_solver_cls = param.Parameter(default=PositiveNonconservativeHLL,
                                         doc="Symbolic Riemann solver class")
    max_level = param.Integer(0, bounds=(0, None), doc="AMR refinement cap; 0 disables")
    min_level = param.Integer(0, bounds=(0, None), doc="Level the dry region relaxes to")
    front_level = param.Integer(2, bounds=(0, None), doc="Level held at the wet/dry front")
    froude_tolerance = param.Number(0.10, bounds=(0.0, None), doc="Froude number: refine above")
    shock_level = param.Integer(2, bounds=(0, None), doc="Level held on a steep surface")
    flow_level = param.Integer(1, bounds=(0, None), doc="Level held on mild activity")
    adapt_interval = param.Integer(3, bounds=(1, None), doc="Steps between adaptations")
    buffer_cells = param.Integer(2, bounds=(0, None), doc="Lookahead rings around a target")
    refine_tolerance = param.Number(0.10, bounds=(0.0, None), doc="Normalised slope: refine above")
    coarsen_tolerance = param.Number(0.02, bounds=(0.0, None), doc="Normalised slope: coarsen below")
    surface_tolerance = param.Number(0.01, bounds=(0.0, None), doc="Absolute |d eta| gate [m]")
    wet_tolerance = param.Number(1e-2, bounds=(0.0, None), doc="Depth counted as wet")
    depth_scale_floor = param.Number(0.1, bounds=(0.0, None), doc="Normalisation depth floor [m]")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        defaults = Settings.default()
        defaults.output.update(Zstruct(snapshots=10, filename="dg"))
        if self.settings is not None:
            defaults.update(self.settings)
        self.settings = defaults
        dune.fem.threading.use = self.threads
        self.time = 0.0
        self._times = []
        self._perm = None
        self._grid_cache = None


    def setup_simulation(self, mesh_path, model, initial_condition=None):
        """``initial_condition``: callable ``(solver) -> UFL vector`` of length
        ``n_var``, evaluated once the grid and spaces exist."""
        self.mesh_path = mesh_path
        self._load_grid(mesh_path)
        self._lower_model(model)
        self._build_spaces()
        self._set_initial_condition(initial_condition)
        self._build_form()
        self._build_timestep()
        self._build_output()
        return self

    def _load_grid(self, mesh_path):
        grid = aluSimplexGrid((reader.gmsh, mesh_path), dimgrid=2)
        self.hierarchicalGrid = grid.hierarchicalGrid
        self.gridView = adaptiveLeafGridView(self.hierarchicalGrid)
        self.hierarchicalGrid.loadBalance()
        self.comm = self.gridView.comm
        self.log(f"{self.gridView.size(0)} cells, {self.comm.size} rank(s)")

    def _lower_model(self, model):
        self.system_model = SystemModel.from_model(model)
        nsm = NumericalSystemModel.from_system_model(
            self.system_model,
            reconstruction=ReconstructionSpec(order=self.dg_degree + 1))
        self.nsm = nsm
        self.runtime = UFLRuntimeModel.from_nsm(nsm)
        # The indicator reads the spectrum once per adaptation over every cell.
        # Interpolating it as UFL cost more than a full operator application;
        # the numpy runtime evaluates the same expression on the raw arrays.
        self.runtime_numpy = NumpyRuntimeModel.from_system_model(self.system_model)
        self.numerics = self.riemann_solver_cls(self.system_model).to_runtime_ufl()
        self.parameters = self.runtime.parameters
        names = [str(n) for n in self.system_model.state]
        for required in ("b", "h"):
            if required not in names:
                raise ValueError(
                    f"the AMR indicator needs a '{required}' state row; this model has {names}")
        self._row_b, self._row_h = names.index("b"), names.index("h")
        self.n_var = len(self.system_model.state)
        self.n_aux = len(self.system_model.aux_state)
        self._report_boundary(model)

    def _report_boundary(self, model):
        """Name the boundary treatment in the log, every run.

        A model carrying no boundary conditions is not boundary-free: the
        lowered kernel then returns Q itself, so every exterior facet is a free
        in/outflow against an infinite reservoir. That is invisible at a lake
        at rest -- the transmissive ghost equals a wall's while u = 0 -- and on
        a closed basin it grows without bound once water reaches the edge. It
        cost one 600 s Malpasset run, which tripled its water volume.
        """
        bcs = getattr(model, "boundary_conditions", None)
        tags = list(getattr(bcs, "list_sorted_function_names", []) or [])
        if tags:
            self.log(f"boundary: {len(tags)} condition(s) [{', '.join(tags)}]")
        else:
            self.log("boundary: TRANSMISSIVE -- the model carries no "
                     "boundary_conditions, so the ghost state is the interior "
                     "state and mass crosses the exterior freely")

    def _build_spaces(self):
        gv = self.gridView
        if self.dg_degree == 0:
            self.V = finiteVolume(gv, dimRange=self.n_var)
            self.Vaux = finiteVolume(gv, dimRange=self.n_aux)
            self.Vscalar = finiteVolume(gv, dimRange=1)
        else:
            from dune.fem.space import dgonb
            self.V = dgonb(gv, dimRange=self.n_var, order=self.dg_degree)
            self.Vaux = dgonb(gv, dimRange=self.n_aux, order=self.dg_degree)
            self.Vscalar = finiteVolume(gv, dimRange=1)

    def nodal_fields(self, names):
        """P1 grid functions from the gmsh ``$NodeData`` arrays. DUNE ignores
        ``$NodeData`` and ALUGrid renumbers vertices, so meshio supplies the
        values and a coordinate hash maps file order onto local order."""
        if self._perm is None:
            mesh = meshio.read(self.mesh_path)
            key = lambda x, y: (np.float64(x).tobytes(), np.float64(y).tobytes())
            lookup = {key(x, y): i for i, (x, y) in enumerate(mesh.points[:, :2])}
            self._V1 = lagrange(self.gridView, order=1)
            # Interpolating the coordinate field gives one point per degree of
            # freedom in the space's own numbering. Walking gridView.vertices
            # instead leaves the ghost rows of a parallel run untouched, and the
            # uninitialised memory then misses the lookup.
            V1vec = lagrange(self.gridView, order=1, dimRange=2)
            positions = V1vec.interpolate(ufl.SpatialCoordinate(V1vec),
                                          name="positions")
            coords = positions.as_numpy.reshape(-1, 2)
            self._perm = np.fromiter((lookup[key(x, y)] for x, y in coords),
                                     dtype=np.int64, count=len(coords))
            self._point_data = mesh.point_data
        fields = {}
        for name in names:
            f = self._V1.function(name=name)
            f.as_numpy[:] = self._point_data[name].astype(float)[self._perm]
            fields[name] = f
        return fields

    def _set_initial_condition(self, initial_condition):
        if initial_condition is None:
            raise ValueError(
                "setup_simulation needs an initial_condition callable: the "
                "meaning of the state rows belongs to the case, not the solver")
        self.Q = self.V.interpolate(initial_condition(self), name="Q")
        self.Qaux = self.Vaux.interpolate(as_vector([0.0] * self.n_aux), name="Qaux")
        self.update_aux()


    def _build_form(self):
        if self.dg_degree > 0:
            raise NotImplementedError(
                f"dg_degree={self.dg_degree}: this residual carries face terms "
                "and the source only -- there is no volume flux term, so every "
                "degree above 0 would silently drop div(F). Use FemDGSolver, "
                "whose F_c supplies it.")
        v = TestFunction(self.V)
        u = TrialFunction(self.V)
        n = FacetNormal(self.V)
        Qaux, params, n_var = self.Qaux, self.parameters, self.n_var
        rn = self.numerics

        flux = vec(rn.numerical_flux(u("+"), u("-"), Qaux("+"), Qaux("-"), params, n("+")))
        fluct = rn.numerical_fluctuations(u("-"), u("+"), Qaux("-"), Qaux("+"),
                                          params, n("-"))

        form = dot(v("+") - v("-"), flux) * dS
        form += dot(v("+"), as_vector([fluct[0, i] for i in range(n_var)])) * dS
        form += dot(v("-"), as_vector([fluct[1, i] for i in range(n_var)])) * dS
        form += self._boundary_form(v, u, n)
        form -= dot(v, vec(self.runtime.source(u, Qaux, params))) * dx

        self.operator = molGalerkin(form)
        self._residual = self.Q.copy()
        self._previous = self.Q.copy()

    def _boundary_form(self, v, u, n):
        """Exterior facets; the ghost state comes from the model's lowered
        ``boundary_conditions`` kernel, never from a state layout written here."""
        x = ufl.SpatialCoordinate(self.V)
        x3 = as_vector([x[0], x[1], 0.0])
        u_bnd = vec(self.runtime.boundary_conditions(
            0, self.time, x3, x[0], u, self.Qaux, self.parameters, n))
        flux = vec(self.numerics.numerical_flux(u, u_bnd, self.Qaux, self.Qaux,
                                                self.parameters, n))
        fluct = self.numerics.numerical_fluctuations(u, u_bnd, self.Qaux, self.Qaux,
                                                     self.parameters, n)
        incoming = as_vector([fluct[1, i] for i in range(self.n_var)])
        return dot(v, flux) * ds + dot(v, incoming) * ds


    def _build_timestep(self):
        rn, params = self.numerics, self.parameters
        self._wave_speed = ufl.max_value(
            scalar(rn.local_max_abs_eigenvalue(self.Q, self.Qaux, params,
                                               as_vector([1.0, 0.0]))),
            scalar(rn.local_max_abs_eigenvalue(self.Q, self.Qaux, params,
                                               as_vector([0.0, 1.0]))),
        )
        self._speed = self.Vscalar.interpolate(as_vector([self._wave_speed]),
                                               name="speed")
        self._indicator = self.Vscalar.interpolate(as_vector([0.0]), name="indicator")
        self._refinement = self.Vscalar.interpolate(as_vector([0.0]), name="level")
        self._adaptive_functions = [self.Q, self.Qaux, self._residual,
                                    self._previous, self._speed, self._indicator,
                                    self._refinement]
        self._build_grid_cache()

    def _build_grid_cache(self):
        """Per-cell size, refinement level and the face adjacency the indicator
        walks. All are indexed by indexSet, so all die on adaptation.

        The pass itself is compiled: done from Python it is one interpreter
        callback per cell and per intersection, which on Malpasset cost more
        than the flux.
        """
        if self._grid_cache is None:
            self._grid_cache = algorithm.load(
                "gridCache",
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "grid_cache.hh"),
                self.gridView)
        size, level, a, b, n_interior = self._grid_cache(self.gridView)
        self._n_interior = int(n_interior)
        self._cell_size = np.asarray(size, dtype=float)
        self._level = np.asarray(level, dtype=np.int32)
        self._faces = np.column_stack((np.asarray(a, dtype=np.int64),
                                       np.asarray(b, dtype=np.int64)))
        self._refinement.as_numpy[:] = self._level

    def compute_indicator(self):
        """Target refinement level per cell, as a mark value for fem.mark.

        Three indicators, none of which fires on a lake at rest, and no rule
        that refines simply because a cell is wet:

        surface deformation
            ``|d eta|`` across a face over the water that carries it, with
            eta = b + h and never h alone -- over varying bathymetry a lake at
            rest has constant eta but strongly varying h, so an h-gradient
            refines the whole reservoir.
        flow activity
            the Froude number, read off the model's own eigenvalues rather
            than a momentum row, so it carries over to any hyperbolic system:
            the acoustic and advective speeds are the half-difference and
            half-sum of the extreme eigenvalues.
        wet/dry front
            a wet cell beside a dry one, held at ``front_level`` whatever the
            flow is doing -- that is where positivity and mass are lost.

        The first two are graded: mild departure from rest reaches
        ``flow_level``, a shock or a steep surface ``shock_level``, so the mesh
        does not jump from macro straight to the finest level.
        """
        i, j = self._faces[:, 0], self._faces[:, 1]
        n_cell = len(self._cell_size)
        Q = self.Q.as_numpy.reshape(-1, self.n_var)
        bed, depth = Q[:, self._row_b], Q[:, self._row_h]
        eta = bed + depth
        wet = depth > self.wet_tolerance

        both = wet[i] & wet[j]
        scale = np.maximum(np.maximum(depth[i], depth[j]), self.depth_scale_floor)
        jump = np.where(both, np.abs(eta[i] - eta[j]), 0.0)
        slope = np.where(both, jump / scale, 0.0)

        jump_K = np.zeros(n_cell)
        slope_K = np.zeros(n_cell)
        for a in (i, j):
            np.maximum.at(jump_K, a, jump)
            np.maximum.at(slope_K, a, slope)
        front = np.zeros(n_cell, dtype=bool)
        mismatch = wet[i] != wet[j]
        front[i[mismatch]] = True
        front[j[mismatch]] = True

        froude = self.compute_froude()

        deformed = wet & (jump_K > self.surface_tolerance)
        active = wet & ((deformed & (slope_K > self.refine_tolerance))
                        | (froude > self.froude_tolerance))
        mild = wet & ((deformed & (slope_K > self.coarsen_tolerance))
                      | (froude > 0.25 * self.froude_tolerance))
        hold = mild

        target = np.full(n_cell, self.min_level, dtype=np.int32)
        np.maximum(target, np.where(front, self.front_level, self.min_level), out=target)
        np.maximum(target, np.where(mild, self.flow_level, self.min_level), out=target)
        np.maximum(target, np.where(active, self.shock_level, self.min_level), out=target)
        np.clip(target, self.min_level, self.max_level, out=target)

        for _ in range(self.buffer_cells):
            previous = target.copy()
            np.maximum.at(target, i, previous[j])
            np.maximum.at(target, j, previous[i])

        keep = 0.5 * (self.refine_tolerance + self.coarsen_tolerance)
        self._indicator.as_numpy[:] = np.where(
            self._level < target, 1.0,
            np.where((self._level == target) | hold, keep, 0.0))
        return self._indicator

    def compute_froude(self):
        """|u| / c per cell, from the model's own spectrum.

        The advective and acoustic speeds are the half-sum and half-difference
        of the extreme eigenvalues, so this needs no momentum row and carries
        over to any hyperbolic system.
        """
        state = self.Q.as_numpy.reshape(-1, self.n_var).T
        aux = self.Qaux.as_numpy.reshape(-1, self.n_aux).T
        n_cell = state.shape[1]
        froude = np.zeros(n_cell)
        for axis in range(2):
            normal = np.zeros((2, n_cell))
            normal[axis] = 1.0
            spectrum = np.asarray(
                self.runtime_numpy.eigenvalues(state, aux, self.parameters, normal),
                dtype=float).reshape(self.n_var, n_cell)
            high, low = spectrum.max(axis=0), spectrum.min(axis=0)
            acoustic = np.maximum(0.5 * (high - low), 1e-12)
            froude += (0.5 * (high + low) / acoustic) ** 2
        return np.sqrt(froude)

    def adapt(self):
        """Mark on the indicator and adapt, carrying every discrete function."""
        self.compute_indicator()
        n_refine, n_coarsen = dune.fem.mark(
            self._indicator,
            refineTolerance=self.refine_tolerance,
            coarsenTolerance=self.coarsen_tolerance,
            minLevel=self.min_level, maxLevel=self.max_level,
            markNeighbors=True, statistics=False)
        if n_refine == 0 and n_coarsen == 0:
            return 0, 0
        dune.fem.adapt(self._adaptive_functions)
        self._build_grid_cache()
        self.update_aux()
        return n_refine, n_coarsen

    @property
    def n_cell(self):
        """Cells in the GLOBAL mesh.

        ``gridView.size(0)`` counts this rank's ghosts too, so summing it over
        ranks double-counts every cell on a partition boundary -- it reported
        28746 for the 26000-cell mesh on 16 ranks. The compiled grid cache
        returns the owned count instead.
        """
        return int(self.comm.sum(self._n_interior))

    @property
    def adaptive(self):
        return self.max_level > self.min_level

    def settle_grid(self):
        """Lift the initial mesh to the indicator's target before t = 0; one
        adapt only ever moves a cell by one level."""
        if not self.adaptive:
            self.log("uniform mesh (max_level == min_level)")
            return
        for _ in range(self.max_level - self.min_level):
            if self.adapt() == (0, 0):
                break
        self.log(f"start mesh: {self.n_cell} cells")

    def compute_dt(self):
        """CFL over every cell this rank sees, then the global minimum.

        Guarded against a zero cell size: a single unwritten row makes the
        division nan, `.max()` propagates it, and `comm.max` then hands the nan
        to every rank -- which is exactly what a ghost row did before the grid
        cache covered Partitions::all.
        """
        self._speed.interpolate(as_vector([self._wave_speed]))
        size = np.maximum(self._cell_size, 1e-12)
        local = float((np.abs(self._speed.as_numpy) / size).max())
        if not np.isfinite(local):
            raise FloatingPointError(
                "compute_dt: non-finite CFL ratio. Either the state carries "
                "NaN or a cell size is unset (see grid_cache.hh).")
        return self.CFL / max(self.comm.max(local), 1e-12)

    def update_aux(self):
        self.Qaux.interpolate(
            self.runtime.update_aux_variables(self.Q, self.Qaux, self.parameters, 0.0))

    def step(self, dt):
        """One SSP-RK2 (Heun) step."""
        self._previous.assign(self.Q)
        self.operator(self.Q, self._residual)
        self.Q.axpy(-dt, self._residual)
        self.update_aux()
        self.operator(self.Q, self._residual)
        self.Q.axpy(-dt, self._residual)
        self.Q *= 0.5
        self.Q.axpy(0.5, self._previous)
        self.update_aux()
        self.time += dt

    def run_simulation(self):
        interval = self.time_end / self.settings.output.snapshots
        next_output = interval
        self.settle_grid()
        self.write_snapshot()
        n_step = 0
        while self.time < self.time_end:
            if self.adaptive and n_step % self.adapt_interval == 0:
                self.adapt()
            dt = min(self.compute_dt(), self.time_end - self.time)
            self.step(dt)
            n_step += 1
            if self.time >= next_output:
                self.write_snapshot()
                next_output += interval
                self.log(f"  t = {self.time:8.3f}  dt = {dt:.4e}"
                         f"  {self.n_cell} cells")
        self.write_snapshot()
        self.log(f"done, t = {self.time:.3f} -> {self.output_dir}")
        return self.Q, self.Qaux


    def _build_output(self):
        self.output_dir = self.settings.output.directory
        if (self.settings.output.clean_directory and self.comm.rank == 0
                and os.path.isdir(self.output_dir)):
            shutil.rmtree(self.output_dir)
        self.comm.barrier()
        os.makedirs(self.output_dir, exist_ok=True)
        celldata = {str(name): self.Q[i]
                    for i, name in enumerate(self.system_model.state)}
        celldata.update({str(name): self.Qaux[i]
                         for i, name in enumerate(self.system_model.aux_state)})
        celldata["level"] = self._refinement
        self._write_vtk = self.gridView.sequencedVTK(
            os.path.join(self.output_dir, self.settings.output.filename),
            celldata=celldata, number=0)

    def write_snapshot(self):
        """A .vtu plus a .pvd carrying the real time (dune records none)."""
        self._write_vtk()
        self._times.append(self.time)
        if self.comm.rank != 0:
            return
        name = self.settings.output.filename
        suffix = ".pvtu" if self.comm.size > 1 else ".vtu"
        prefix = ("s%04d-" % self.comm.size) if self.comm.size > 1 else ""
        root = ET.Element("VTKFile", type="Collection", version="0.1",
                          byte_order="LittleEndian")
        collection = ET.SubElement(root, "Collection")
        for k, t in enumerate(self._times):
            ET.SubElement(collection, "DataSet", timestep=repr(t), part="0",
                          file=f"{prefix}{name}{k:05d}{suffix}")
        ET.ElementTree(root).write(os.path.join(self.output_dir, name + ".pvd"))

    def log(self, message):
        if self.comm.rank == 0:
            print(f"[{self.settings.name}] {message}", flush=True)
