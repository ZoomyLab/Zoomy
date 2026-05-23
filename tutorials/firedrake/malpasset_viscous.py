#!/usr/bin/env python3
"""malpasset_viscous.py — baseline + DG(0) TPFA viscous-friction flux.

Adds ∇·(ν h ∇u) to the momentum equation as a two-point central diffusion
flux on interior facets. For each interior facet (+/- sides) and momentum
component i ∈ {2, 3}:

    Ĝ_visc · n("+") = ν · h_face · (u_i("-") − u_i("+")) / d
    h_face = ½(h("+") + h("-"))
    d      = 2 · avg(CellVolume / FacetArea)

Contribution to the weak form (consistent with the convective interior-facet
block in firedrake_solver.py:353-368):

    weak_form += (v_i("+") - v_i("-")) · (ν h_face / d) · (u_i("+") - u_i("-")) dS

ν read from MALPASSET_NU env var (default 1.0 m^2/s).
Other env vars: see malpasset_baseline.py.
"""
import os
import time

import numpy as np
from sympy import Matrix, sqrt, Piecewise
import sympy as sp
from attr import define, field

from zoomy_core.fvm.solver_numpy import Settings
from zoomy_core.model.basemodel import Model
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.misc.misc import Zstruct, ZArray
import zoomy_core.misc.misc as misc
import zoomy_firedrake.firedrake_solver as dg


_MANNING_N = float(os.environ.get("MALPASSET_MANNING", "0.033"))
_EPS_WD = float(os.environ.get("MALPASSET_EPS_WD", "1e-2"))
_H_FRICTION_FLOOR = float(os.environ.get("MALPASSET_H_FRICTION", "0.5"))


@define(frozen=True, slots=True, kw_only=True)
class SWE(Model):
    dimension: int = 2
    variables: Zstruct = field(init=False)
    aux_variables: Zstruct = field(default=1)
    _default_parameters: dict = field(
        init=False,
        factory=lambda: {"g": 9.81, "ex": 0.0, "ey": 0.0, "ez": 1.0,
                         "rho": 1000.0, "n": _MANNING_N, "eps": _EPS_WD,
                         "nu": 0.0},
    )

    def __attrs_post_init__(self):
        object.__setattr__(self, "variables", self.dimension + 2)
        super().__attrs_post_init__()

    def get_primitives(self):
        dim = self.dimension
        b = self.variables[0]
        h = self.variables[1]
        hinv = 1 / h
        U = Matrix([hu * hinv for hu in self.variables[2:2 + dim]])
        return b, h, U, hinv

    def flux(self):
        dim = self.dimension
        b, h, U, hinv = self.get_primitives()
        F = Matrix.zeros(self.variables.length(), dim)
        F[1, :] = sp.Matrix(self.variables[2:2 + dim]).T
        F[2:, :] = h * U * U.T
        return ZArray(F)

    def nonconservative_matrix(self):
        dim = self.dimension
        _, h, _, _ = self.get_primitives()
        g = self.parameters.g
        N = ZArray.zeros(self.n_variables, self.n_variables, dim)
        for d in range(dim):
            N[2 + d, 0, d] = g * h
            N[2 + d, 1, d] = g * h
        return ZArray(N)

    def source(self):
        # Manning bed friction with corrected sign + bounded h^(-1/3)
        # divisor (see baseline for rationale).
        eps = 1e-4
        _, h, U, _ = self.get_primitives()
        hinv = self.aux_variables[0]
        g = self.parameters.g
        n = self.parameters.n
        abs_u = sqrt(U.dot(U) + eps)
        h_floor = _H_FRICTION_FLOOR
        friction_div = Piecewise((h**(-sp.Rational(1, 3)), h > h_floor),
                                  (h_floor**(-sp.Rational(1, 3)), True))
        S = Matrix.zeros(self.n_variables, 1)
        S[2:, 0] = -n**2 * g * friction_div * U[:, 0] * abs_u
        return ZArray(S).reshape(self.n_variables,)


@define(frozen=True, slots=True, kw_only=True)
class NumericSWE(SWE):
    disable_differentiation: bool = False

    def get_primitives(self):
        dim = self.dimension
        b = self.variables[0]
        h = self.variables[1]
        hinv = self.aux_variables[0]
        U = Matrix([hu * hinv for hu in self.variables[2:2 + dim]])
        return b, h, U, hinv

    def eigenvalues(self):
        ev = super().eigenvalues()
        h = self.variables[1]
        return sp.Function('conditional')(h > self.parameters.eps, ev,
                                          ZArray.zeros(*ev.shape))

    def source(self):
        h = self.variables[1]
        S = super().source()
        zeros = ZArray.zeros(*S.shape)
        return sp.Function('conditional')(h > self.parameters.eps, S, zeros)

    def source_jacobian_wrt_aux_variables(self):
        return ZArray.zeros(self.n_variables)

    def source_jacobian_wrt_variables(self):
        return ZArray.zeros(self.n_variables)


main_dir = misc.get_main_directory()
input_mesh = os.path.join(main_dir, "data", "malpasset", "geo_malpasset-small.msh")
import meshio
meshio_mesh = meshio.read(input_mesh)


def build_vertex_permutation(fd_mesh, meshio_mesh, decimal=12):
    """See malpasset_baseline.py for description."""
    dim = fd_mesh.geometric_dimension()
    coords_fd = np.round(fd_mesh.coordinates.dat.data_ro[:, :dim], decimal)
    coords_mio = np.round(meshio_mesh.points[:, :dim], decimal)
    lookup = {tuple(c): i for i, c in enumerate(coords_mio)}
    perm = np.empty(coords_fd.shape[0], dtype=np.int64)
    for j, c in enumerate(coords_fd):
        perm[j] = lookup[tuple(c)]
    return perm


bcs = BC.BoundaryConditions([
    BC.Extrapolation(tag="wall"),
    BC.Extrapolation(tag="inflow"),
    BC.Extrapolation(tag="outflow"),
])

model = NumericSWE(dimension=2, boundary_conditions=bcs)

settings = Settings(
    name="Firedrake",
    output=Zstruct(
        directory="outputs/firedrake_viscous" +
        (("_" + os.environ["MALPASSET_OUT_TAG"]) if os.environ.get("MALPASSET_OUT_TAG") else ""),
        snapshots=100, filename='dg', clean_directory=True),
)


import ufl
import firedrake as fd
from zoomy_core.transformation.to_ufl import UFLRuntimeModel


# --- MPI halo-exchange workaround --------------------------------------------
# See malpasset_baseline.py for context. Firedrake's halo SF calls use
# rootdata==leafdata; PETSc 3.20+ rejects this. Copy-into-temp workaround.
def _patch_firedrake_halo_for_mpi():
    from firedrake import halo as _halo
    from firedrake.petsc import PETSc as _PETSc
    if _PETSc.COMM_WORLD.Get_size() == 1:
        return
    import numpy as _np
    from mpi4py import MPI as _MPI
    try:
        import pyop2.types.access as _OP2
    except Exception:
        from pyop2 import op2 as _OP2

    _orig_get_mtype = _halo._get_mtype

    def _bcast(self, dat):
        mtype, _ = _orig_get_mtype(dat)
        src = _np.array(dat._data, copy=True)
        self.sf.bcastBegin(mtype, src, dat._data, _MPI.REPLACE)
        self.sf.bcastEnd(mtype, src, dat._data, _MPI.REPLACE)

    def _reduce(self, dat, op):
        mtype, _ = _orig_get_mtype(dat)
        src = _np.array(dat._data, copy=True)
        self.sf.reduceBegin(mtype, src, dat._data, op)
        self.sf.reduceEnd(mtype, src, dat._data, op)

    def global_to_local_begin(self, dat, insert_mode):
        assert insert_mode is _OP2.WRITE
        if self.comm.size == 1:
            return
        _bcast(self, dat)

    def global_to_local_end(self, dat, insert_mode):
        return

    def local_to_global_begin(self, dat, insert_mode):
        if self.comm.size == 1:
            return
        mtype, builtin = _orig_get_mtype(dat)
        op = {(False, _OP2.INC): _MPI.SUM,
              (True, _OP2.INC): _MPI.SUM,
              (False, _OP2.MIN): _halo._contig_min_op,
              (True, _OP2.MIN): _MPI.MIN,
              (False, _OP2.MAX): _halo._contig_max_op,
              (True, _OP2.MAX): _MPI.MAX}[(builtin, insert_mode)]
        _reduce(self, dat, op)

    def local_to_global_end(self, dat, insert_mode):
        return

    _halo.Halo.global_to_local_begin = global_to_local_begin
    _halo.Halo.global_to_local_end = global_to_local_end
    _halo.Halo.local_to_global_begin = local_to_global_begin
    _halo.Halo.local_to_global_end = local_to_global_end


_patch_firedrake_halo_for_mpi()

IdentityMatrix = ufl.as_tensor(
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


# Optional PETSc override for the LinearVariationalSolver: monkey-patch the
# parent class so subclasses inherit. lu is the upstream default.
_pc = os.environ.get("MALPASSET_PC", "lu")
_ksp = os.environ.get("MALPASSET_KSP", "preonly" if _pc == "jacobi" else "bcgs")

# Monkey-patch by wrapping LinearVariationalSolver construction in the module.
_orig_LVS = fd.LinearVariationalSolver
def _LVS_patched(*args, **kwargs):
    sp = kwargs.get("solver_parameters", {}) or {}
    sp = {**sp, "ksp_type": _ksp, "pc_type": _pc}
    if _pc == "bjacobi":
        sp.setdefault("sub_pc_type", "lu")
    kwargs["solver_parameters"] = sp
    return _orig_LVS(*args, **kwargs)
fd.LinearVariationalSolver = _LVS_patched


@define(frozen=True, slots=True, kw_only=True)
class MySolver(dg.FiredrakeHyperbolicSolver):
    def set_initial_condition(self, Q, mesh, meshio_mesh):
        perm = build_vertex_permutation(mesh, meshio_mesh)
        Q.dat.data[:, 0] = meshio_mesh.point_data['B'][perm]
        Q.dat.data[:, 1] = meshio_mesh.point_data['H'][perm]
        Q.dat.data[:, 2] = (meshio_mesh.point_data['H'] *
                            meshio_mesh.point_data['U'])[perm]
        Q.dat.data[:, 3] = (meshio_mesh.point_data['H'] *
                            meshio_mesh.point_data['V'])[perm]

    def _get_weak_form_source(self, runtime_model, Q_star, Qnp1, Qaux_star,
                              Qaux_np1, n, mesh,
                              map_boundary_tag_to_function_index,
                              sim_time, dt, x, x_3d, theta=1.0):
        # See malpasset_baseline.py for rationale: upstream's source weak form
        # gives b (i=0) a zero diff_restricted row -> singular Jacobian row ->
        # MPI corrupts b. Here every component gets the identity diff row.
        mom_var = [1, 2, 3]
        V = Qnp1.function_space()
        test_q = fd.TestFunction(V)
        ncomp = V.value_size
        Q_theta = theta * Qnp1 + (1.0 - theta) * Q_star
        Qaux_theta = theta * Qaux_np1 + (1.0 - theta) * Qaux_star
        source_full = runtime_model.source(Q_theta, Qaux_theta,
                                           runtime_model.parameters)
        zero = fd.Constant(0.0)
        diff_restricted = []
        source_restricted = []
        for i in range(ncomp):
            diff_restricted.append(Qnp1[i] - Q_star[i])
            source_restricted.append(source_full[i] if i in mom_var else zero)
        weak_form = fd.dot(test_q,
                           fd.as_vector(diff_restricted) / dt) * fd.dx
        weak_form -= fd.dot(test_q, fd.as_vector(source_restricted)) * fd.dx
        return weak_form

    def update_Qaux(self, Q, Qaux):
        eps = 1e-4
        h_safe = np.maximum(Q.dat.data_ro[:, 1], eps)
        a = Qaux.dat.data
        if a.ndim == 1:
            a[:] = 1.0 / h_safe
        else:
            a[:, 0] = 1.0 / h_safe

    def update_Q(self, Q, Qaux):
        eps = 1e-4
        h_dry = float(os.environ.get("MALPASSET_H_DRY", "1e-2"))
        max_vel_cap = float(os.environ.get("MALPASSET_V_MAX", "30.0"))
        data = Q.dat.data
        h = data[:, 1]
        h_safe = np.maximum(h, eps)
        wet = (h > h_dry).astype(np.float64)
        for i in (2, 3):
            u = data[:, i] / h_safe
            u_new = wet * np.sign(u) * np.minimum(np.abs(u), max_vel_cap)
            data[:, i] = h_safe * u_new

    def _setup(self, mshfile, model):
        mesh = fd.Mesh(mshfile)
        runtime_model = UFLRuntimeModel(model)
        V, Vaux, Qnp1, Qs, Qn, Qaux_np1, Qaux_s, Qaux_n = \
            self._get_functionspaces(mesh, runtime_model)
        V_CG = fd.VectorFunctionSpace(mesh, "CG", 1,
                                      dim=runtime_model.n_variables)
        Q_CG = fd.Function(V_CG)
        self.set_initial_condition(Q_CG, mesh, meshio_mesh)
        Qn = fd.project(Q_CG, V)
        Qs = fd.project(Q_CG, V)
        Qnp1 = fd.project(Q_CG, V)
        self.update_Qaux(Qn, Qaux_n)
        self.update_Qaux(Qs, Qaux_s)
        self.update_Qaux(Qnp1, Qaux_np1)
        self.update_Q(Qn, Qaux_n)
        self.update_Q(Qs, Qaux_s)
        self.update_Q(Qnp1, Qaux_np1)
        map_boundary_tag_to_function_index = \
            self.get_map_boundary_tag_to_boundary_function_index(
                model, mshfile, mesh)
        return (mesh, runtime_model, V, Vaux, Qn, Qs, Qnp1,
                Qaux_n, Qaux_s, Qaux_np1, map_boundary_tag_to_function_index)

    def _get_weak_form_convective(self, runtime_model, Qn, Qnp1, Qaux_n,
                                  Qaux_np1, n, mesh,
                                  map_boundary_tag_to_function_index,
                                  sim_time, dt, x, x_3d):
        # Start from the inviscid convective weak form (mass term + LLF flux +
        # non-conservative path + boundary fluxes).
        weak_form = super()._get_weak_form_convective(
            runtime_model, Qn, Qnp1, Qaux_n, Qaux_np1, n, mesh,
            map_boundary_tag_to_function_index, sim_time, dt, x, x_3d)

        # DG(0) two-point central diffusion flux on interior facets, for
        # momentum components 2 (hu) and 3 (hv).
        Q = Qn
        test_q = fd.TestFunction(Qn.function_space())

        nu = fd.Constant(_NU)
        eps = fd.Constant(1e-4)
        h_p = fd.max_value(Q("+")[1], eps)
        h_m = fd.max_value(Q("-")[1], eps)
        h_face = 0.5 * (h_p + h_m)
        d = 2.0 * fd.avg(fd.CellVolume(mesh) / fd.FacetArea(mesh))

        for i in (2, 3):
            u_p = Q("+")[i] / h_p
            u_m = Q("-")[i] / h_m
            weak_form += (test_q("+")[i] - test_q("-")[i]) * \
                         (nu * h_face / d) * (u_p - u_m) * fd.dS

        return weak_form


_time_end = float(os.environ.get("MALPASSET_TIME_END", "20.0"))
_cfl = float(os.environ.get("MALPASSET_CFL", "0.5"))
_NU = float(os.environ.get("MALPASSET_NU", "1.0"))

solver = MySolver(settings=settings, time_end=_time_end, CFL=_cfl,
                  IdentityMatrix=IdentityMatrix)


if __name__ == "__main__":
    try:
        from firedrake.petsc import PETSc
        rank = PETSc.COMM_WORLD.Get_rank()
        size = PETSc.COMM_WORLD.Get_size()
    except Exception:
        rank, size = 0, 1
    if rank == 0:
        print(f"[malpasset] viscous nu={_NU} time_end={_time_end} CFL={_cfl} "
              f"ksp={_ksp} pc={_pc} mpi_size={size}", flush=True)
    t0 = time.perf_counter()
    solver.solve(input_mesh, model)
    t1 = time.perf_counter()
    if rank == 0:
        print(f"[malpasset] wall_time={t1 - t0:.3f}s", flush=True)
