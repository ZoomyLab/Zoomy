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
from dune.grid import reader

from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_core.misc.misc import Settings, Zstruct
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel import SystemModel
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
        self.gridView = aluSimplexGrid((reader.gmsh, mesh_path), dimgrid=2)
        self.gridView.hierarchicalGrid.loadBalance()
        self.comm = self.gridView.comm
        self.log(f"{self.gridView.size(0)} cells, {self.comm.size} rank(s)")

    def _lower_model(self, model):
        self.system_model = SystemModel.from_model(model)
        nsm = NumericalSystemModel.from_system_model(
            self.system_model,
            reconstruction=ReconstructionSpec(order=self.dg_degree + 1))
        self.runtime = UFLRuntimeModel.from_nsm(nsm)
        self.numerics = self.riemann_solver_cls(self.system_model).to_runtime_ufl()
        self.parameters = self.runtime.parameters
        self.n_var = len(self.system_model.state)
        self.n_aux = len(self.system_model.aux_state)

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
            coords = np.asarray(self.gridView.coordinates())
            self._perm = np.fromiter((lookup[key(x, y)] for x, y in coords),
                                     dtype=np.int64, count=len(coords))
            self._point_data = mesh.point_data
            self._V1 = lagrange(self.gridView, order=1)
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
        index_set = self.gridView.indexSet
        self._cell_size = np.empty(len(self._speed.as_numpy), dtype=float)
        for cell in self.gridView.elements:
            perimeter = sum(i.geometry.volume
                            for i in self.gridView.intersections(cell))
            self._cell_size[index_set.index(cell)] = cell.geometry.volume / perimeter

    def compute_dt(self):
        self._speed.interpolate(as_vector([self._wave_speed]))
        local = float((np.abs(self._speed.as_numpy) / self._cell_size).max())
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
        self.write_snapshot()
        while self.time < self.time_end:
            dt = min(self.compute_dt(), self.time_end - self.time)
            self.step(dt)
            if self.time >= next_output:
                self.write_snapshot()
                next_output += interval
                self.log(f"  t = {self.time:8.3f}  dt = {dt:.4e}")
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
