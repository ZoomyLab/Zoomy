"""Gate 1a-2 — boundary GHOST convention (decision D5) + ghost aux (D6).

For every BC class the new ``boundary_ghost`` kernel must return the GHOST
state at the LSQ virtual point ``x_c + 2(x_f − x_c)``:

* on a linear field ``u = a + B·x`` that satisfies the BC, the ghost equals
  ``u`` at that point EXACTLY and the LSQ boundary-cell gradient is ``B``;
* the batched kernel (one evaluation per BC branch over its faces, routed by
  tag NAME) equals the per-face evaluation of the indexed kernel;
* the ghost aux is the model's aux update rule applied to the ghost state.

Meshes: ``create_1d``, ``create_2d`` and a gmsh triangle mesh whose physical
tags are numbered in NON-alphabetical order (bottom, right, top, left).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh
from zoomy_core.mesh.geometry_xp import (
    build_topology, compute_geometry, lsq_gradient)
from zoomy_core.model.models import SWE
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.systemmodel import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel

pytestmark = [pytest.mark.template_gate]

TIME = 0.3
TOL = 1e-11


# ── meshes ───────────────────────────────────────────────────────────────

def write_tri_msh(path, nx=6, ny=5, seed=3):
    """A jittered triangle mesh of the unit square, gmsh 2.2, physical tags
    numbered bottom=1, right=2, top=3, left=4 (NOT alphabetical)."""
    import meshio
    rng = np.random.default_rng(seed)
    xs, ys = np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1)
    P = np.array([[x, y] for x in xs for y in ys])

    def vid(i, j):
        return i * (ny + 1) + j
    for i in range(nx + 1):
        for j in range(ny + 1):
            k = vid(i, j)
            if 0 < i < nx:
                P[k, 0] += rng.uniform(-0.2, 0.2) / nx
            if 0 < j < ny:
                P[k, 1] += rng.uniform(-0.2, 0.2) / ny
    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris += ([[a, b, c], [a, c, d]] if (i + j) % 2
                     else [[a, b, d], [b, c, d]])
    ids = {"bottom": 1, "right": 2, "top": 3, "left": 4}
    lines, tags = [], []
    for i in range(nx):
        lines += [[vid(i, 0), vid(i + 1, 0)], [vid(i, ny), vid(i + 1, ny)]]
        tags += [ids["bottom"], ids["top"]]
    for j in range(ny):
        lines += [[vid(0, j), vid(0, j + 1)], [vid(nx, j), vid(nx, j + 1)]]
        tags += [ids["left"], ids["right"]]
    pts = np.hstack([P, np.zeros((P.shape[0], 1))])
    phys = [np.array(tags), np.full(len(tris), 10)]
    mesh = meshio.Mesh(
        pts, [("line", np.array(lines)), ("triangle", np.array(tris))],
        cell_data={"gmsh:physical": phys, "gmsh:geometrical": phys},
        field_data={**{k: np.array([v, 1]) for k, v in ids.items()},
                    "domain": np.array([10, 2])})
    meshio.write(path, mesh, file_format="gmsh22", binary=False)
    return path


@pytest.fixture(scope="module")
def gmsh_path(tmp_path_factory):
    return write_tri_msh(str(tmp_path_factory.mktemp("msh") / "tri.msh"))


def make_mesh(kind, gmsh_path=None):
    if kind == "1d":
        return LSQMesh.create_1d((0.0, 1.0), 12)
    if kind == "2d":
        return LSQMesh.create_2d((0.0, 1.0, 0.0, 1.0), 6, 5)
    return LSQMesh.from_msh(gmsh_path)


# ── linear fields (rows b, h, hu[, hv]) ─────────────────────────────────

A0 = np.array([0.1, 1.0, 0.4, 0.3])


def field_x(dim, grad=(0.0, 0.3, 0.5, 0.0)):
    """u = a + g·x (x only): zero normal gradient on bottom/top."""
    n = dim + 2
    a, g = A0[:n].copy(), np.array(grad[:n])

    def u(x):
        return a[:, None] + g[:, None] * x[0][None, :]
    B = np.zeros((n, dim))
    B[:, 0] = g
    return u, B


def field_y(dim, grad=(0.0, 0.3, 0.5, 0.2)):
    """u = a + g·y (y only; constant in 1-D): zero normal gradient on
    left/right."""
    n = dim + 2
    a = A0[:n].copy()
    if dim == 1:
        return (lambda x: np.repeat(a[:, None], x.shape[1], axis=1),
                np.zeros((n, 1)))
    g = np.array(grad[:n])

    def u(x):
        return a[:, None] + g[:, None] * x[1][None, :]
    B = np.zeros((n, dim))
    B[:, 1] = g
    return u, B


def wall_field(dim, x_wall, tangential=0.3):
    """Normal momentum vanishing LINEARLY at the wall ``x = x_wall``; depth,
    bed and the tangential momentum constant (``tangential``; a free-slip wall
    keeps it, a rough wall damps it — use 0 there)."""
    n = dim + 2
    a = A0[:n].copy()
    a[2] = 0.0
    if dim == 2:
        a[3] = tangential
    s = 0.7

    def u(x):
        out = np.repeat(a[:, None], x.shape[1], axis=1)
        out[2] = s * (x[0] - x_wall)
        return out
    B = np.zeros((n, dim))
    B[2, 0] = s
    return u, B


# ── machinery ────────────────────────────────────────────────────────────

class Setup:
    def __init__(self, kind, bcs, gmsh_path=None):
        self.mesh = make_mesh(kind, gmsh_path)
        self.dim = int(self.mesh.dimension)
        model = SWE(dimension=self.dim,
                    boundary_conditions=BC.BoundaryConditions(bcs))
        self.nsm = NumericalSystemModel.from_system_model(
            SystemModel.from_model(model))
        self.topo = build_topology(self.mesh, nsm=self.nsm)
        X = np.asarray(self.mesh.vertex_coordinates, dtype=float)
        self.geom = compute_geometry(self.topo, X)
        self.rt = NumpyRuntimeModel.from_system_model(self.nsm)
        self.p = np.array(list(self.nsm.parameter_values.values()), float)
        t = self.topo
        names = list(self.mesh.boundary_conditions_sorted_names)
        self.face_tag = np.array(
            [names[k] for k in np.asarray(
                self.mesh.boundary_face_function_numbers)])
        cc = np.asarray(self.geom["cell_centers"])[:self.dim]
        self.cc = cc
        fc = np.asarray(self.geom["face_centers"]).T[:self.dim]
        self.fc_b = fc[:, t.boundary_faces]
        self.pos = np.asarray(self.geom["face_centers"])[t.boundary_faces].T
        self.nrm = np.asarray(self.geom["face_normals"])[:, t.boundary_faces]
        self.dist = np.asarray(self.geom["d_normal"])
        owner = t.boundary_owner
        self.x_virtual = cc[:, owner] + 2 * (self.fc_b - cc[:, owner])

    def aux(self, Q):
        uav = getattr(self.rt, "update_aux_variables", None)
        Qa0 = np.zeros((len(self.nsm.aux_state), Q.shape[1]))
        if uav is None:
            return Qa0
        return np.asarray(uav(Q, Qa0, self.p, TIME, np.zeros((3, Q.shape[1]))),
                          float).reshape(Qa0.shape)

    def ghosts(self, u, bdata=None):
        nc = self.topo.n_inner_cells
        Qc = u(self.cc[:, :nc])
        Qa = self.aux(Qc)
        cells = self.topo.bc_input_cells
        Qin, Qain = Qc[:, cells], Qa[:, cells]
        ghost = self.rt.boundary_ghost(self.topo.bc_branch)
        Qg, Ag = ghost(TIME, self.pos, self.dist, Qin, Qain, self.p, self.nrm,
                       bdata)
        return Qc, Qa, Qin, Qain, Qg, Ag

    def checked_cells(self, tags):
        """Inner cells with a boundary face, ALL of whose boundary faces carry
        one of ``tags``."""
        owner = self.topo.boundary_owner
        ok = {}
        for i, c in enumerate(owner):
            ok.setdefault(int(c), True)
            ok[int(c)] &= self.face_tag[i] in tags
        return np.array(sorted(c for c, v in ok.items() if v), dtype=int)


def tags_for(dim):
    return ("left", "right") if dim == 1 else ("left", "right", "bottom", "top")


def _others(dim, tested=("left", "right")):
    return [BC.Extrapolation(tag=t) for t in tags_for(dim) if t not in tested]


# Each case: builder(dim, setup-free) -> (bcs, field, checked tags).

def case_extrapolation(dim):
    return ([BC.Extrapolation(tag=t) for t in tags_for(dim)],
            field_y(dim), ("left", "right"))


def case_zero_neumann(dim):
    return ([BC.ZeroNeumann(tag=t) for t in tags_for(dim)],
            field_y(dim), ("left", "right"))


def case_dirichlet(dim):
    u, B = field_x(dim)
    h = lambda x: float(u(np.array([[x]] + [[0.5]] * (dim - 1)))[1, 0])
    bcs = [BC.Dirichlet("left", on="h", value=h(0.0)),
           BC.Dirichlet("right", on="h", value=h(1.0))] + _others(dim)
    # only h varies: momentum constant so its extrapolation is exact too
    u2, B2 = field_x(dim, grad=(0.0, 0.3, 0.0, 0.0))
    return bcs, (u2, B2), tags_for(dim)


def case_inflow_outflow(dim):
    u, B = field_x(dim)

    def at(x):
        return u(np.array([[x]] + [[0.5]] * (dim - 1)))[:, 0]
    bcs = [BC.InflowOutflow(tag="left",
                            prescribe_fields={1: float(at(0.0)[1]),
                                              2: {"mode": "inlet_outlet",
                                                  "value": float(at(0.0)[2])}}),
           BC.InflowOutflow(tag="right",
                            prescribe_fields={1: float(at(1.0)[1]),
                                              2: float(at(1.0)[2])})] \
        + _others(dim)
    return bcs, (u, B), tags_for(dim)


def case_lambda(dim):
    u, B = field_x(dim)
    a, g = A0, np.array([0.0, 0.3, 0.5, 0.0])
    lam = {k: (lambda kk: (lambda t, X, dX, Q, Qa, p, n: a[kk] + g[kk] * X[0]))(k)
           for k in (1, 2)}
    bcs = [BC.Lambda(tag="left", prescribe_fields=lam),
           BC.Lambda(tag="right", prescribe_fields=lam)] + _others(dim)
    return bcs, (u, B), tags_for(dim)


def case_from_data(dim):
    u, B = field_x(dim, grad=(0.0, 0.3, 0.0, 0.0))

    def h(x):
        return float(u(np.array([[x]] + [[0.5]] * (dim - 1)))[1, 0])
    tl = np.array([0.0, 1.0])
    bcs = [BC.FromData(tag="left", timeline=tl,
                       prescribe_fields={1: np.array([h(0.0)] * 2)}),
           BC.FromData(tag="right", timeline=tl,
                       prescribe_fields={1: np.array([h(1.0)] * 2)})] \
        + _others(dim)
    return bcs, (u, B), tags_for(dim)


def case_periodic(dim):
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left")] \
        + _others(dim)
    return bcs, field_y(dim), ("left", "right")


def case_wall(dim, cls=BC.Wall, tangential=0.3):
    def build(x_wall):
        return wall_field(dim, x_wall, tangential)
    return [cls(tag=t) for t in tags_for(dim)], build, ("left",)


def case_rough_wall(dim):
    return case_wall(dim, BC.RoughWall, tangential=0.0)


def case_from_model_wall(dim):
    bcs = [BC.FromModel(tag=t, definition="wall") for t in tags_for(dim)]
    return bcs, case_wall(dim)[1], ("left",)


def case_flux(dim):
    u, B = field_x(dim, grad=(0.0, 0.3, 0.0, 0.0))
    g = 0.3
    bcs = [BC.Flux("left", on="h", gradient=-g),
           BC.Flux("right", on="h", gradient=g)] + _others(dim)
    return bcs, (u, B), tags_for(dim)


def case_wind_stress(dim):
    u, B = field_x(dim, grad=(0.0, 0.3, 0.5, 0.0))

    def at(x):
        return u(np.array([[x]] + [[0.5]] * (dim - 1)))[:, 0]
    bcs = [BC.WindStress(tag="left", prescribe_values={1: float(at(0.0)[1])},
                         prescribe_gradients={2: -0.5}),
           BC.WindStress(tag="right", prescribe_values={1: float(at(1.0)[1])},
                         prescribe_gradients={2: 0.5})] + _others(dim)
    return bcs, (u, B), tags_for(dim)


def case_per_field(dim):
    """Different BCs per field at ONE tag (merged into a PerFieldBoundary):
    Dirichlet depth + reflecting x-momentum at the left wall."""
    def build(x_wall):
        u0, B = wall_field(dim, x_wall)

        def u(x):
            out = u0(x)
            out[1] = 1.0 + 0.3 * x[0]
            return out
        B = B.copy()
        B[1, 0] = 0.3
        return u, B
    bcs = [BC.Dirichlet("left", on="h", value=1.0),
           BC.Wall("left", on="hu")] + [BC.Extrapolation(tag=t) for t in
                                        tags_for(dim) if t != "left"]
    return bcs, build, ("left",)


def case_coupled(dim):
    bcs = [BC.Coupled(tag="left", mesh_name="peer"),
           BC.Coupled(tag="right", mesh_name="peer")] + _others(dim)
    return bcs, field_x(dim), tags_for(dim)


CASES = {
    "Extrapolation": case_extrapolation,
    "ZeroNeumann": case_zero_neumann,
    "Dirichlet": case_dirichlet,
    "InflowOutflow": case_inflow_outflow,
    "Lambda": case_lambda,
    "FromData": case_from_data,
    "Periodic": case_periodic,
    "Wall": case_wall,
    "RoughWall": case_rough_wall,
    "FromModel(wall)": case_from_model_wall,
    "Flux": case_flux,
    "WindStress": case_wind_stress,
    "PerFieldBoundary": case_per_field,
    "Coupled": case_coupled,
}
MESHES = ("1d", "2d", "gmsh")


# Zero-normal-gradient ghosts (``ghost = Q_inner``) are exact for a field
# that is constant along the LSQ virtual offset ``2(x_f − x_c)``.  On the
# skewed gmsh triangles that offset has a TANGENTIAL component, so a
# tangentially varying field is not reproducible by ANY zero-gradient ghost
# there; those cases use a constant field on the gmsh mesh.
NEUMANN_TYPE = ("Extrapolation", "ZeroNeumann", "Periodic")


def _run_case(case, kind, gmsh_path):
    dim = 1 if kind == "1d" else 2
    bcs, field, checked = CASES[case](dim)
    if kind == "gmsh":
        # bottom/top carry zero-gradient (extrapolation) ghosts of an
        # x-varying field there: not reproducible on skewed cells (above).
        checked = tuple(t for t in checked if t in ("left", "right"))
    if kind == "gmsh" and case in NEUMANN_TYPE:
        n = dim + 2
        field = ((lambda x: np.repeat(A0[:n, None], x.shape[1], axis=1)),
                 np.zeros((n, dim)))
    st = Setup(kind, bcs, gmsh_path)
    if callable(field):                       # wall-type: field per wall
        field = field(0.0)
    u, B = field
    bdata = None
    if case == "Coupled":                     # the peer delivers the ghost
        bdata = u(st.x_virtual)
    Qc, Qa, Qin, Qain, Qg, Ag = st.ghosts(u, bdata)
    return st, u, B, checked, Qc, Qa, Qin, Qain, Qg, Ag, bdata


@pytest.mark.parametrize("kind", MESHES)
@pytest.mark.parametrize("case", list(CASES))
def test_ghost_exact_on_linear_field(case, kind, gmsh_path):
    st, u, B, checked, Qc, Qa, Qin, Qain, Qg, Ag, _ = _run_case(
        case, kind, gmsh_path)
    sel = np.isin(st.face_tag, checked)
    assert sel.any()
    exact = u(st.x_virtual)
    err = np.abs(Qg[:, sel] - exact[:, sel]).max()
    assert err < TOL, f"{case}/{kind}: ghost error {err:.2e}"
    # The LSQ boundary-cell gradient consumes the ghosts: exact for a linear
    # field at every cell whose boundary faces all satisfy the BC.
    cells = st.checked_cells(checked)
    assert cells.size
    grad = np.asarray(lsq_gradient(st.topo, st.geom, Qc, Qg))
    gerr = np.abs(grad[:, :, cells] - B[:, :, None]).max()
    assert gerr < 1e-9, f"{case}/{kind}: LSQ boundary gradient error {gerr:.2e}"


@pytest.mark.parametrize("kind", MESHES)
@pytest.mark.parametrize("case", list(CASES))
def test_batched_equals_per_face_and_ghost_aux(case, kind, gmsh_path):
    st, u, B, checked, Qc, Qa, Qin, Qain, Qg, Ag, bdata = _run_case(
        case, kind, gmsh_path)
    bd = Qin if bdata is None else bdata
    per_face = np.stack([
        st.rt.boundary_ghost_state(
            int(st.topo.bc_branch[i]), TIME, st.pos[:, i], st.dist[i],
            Qin[:, i], Qain[:, i], st.p, st.nrm[:, i], bd[:, i])
        for i in range(Qin.shape[1])], axis=1)
    assert np.abs(per_face - Qg).max() <= 1e-14 * max(1.0, np.abs(Qg).max())
    whole = st.rt.boundary_ghost_state(
        st.topo.bc_branch, TIME, st.pos, st.dist, Qin, Qain, st.p, st.nrm, bd)
    assert np.abs(whole - Qg).max() <= 1e-14 * max(1.0, np.abs(Qg).max())
    # D6: the ghost aux IS the model's aux update rule at the ghost state
    uav = st.rt.update_aux_variables
    ref = np.asarray(uav(Qg, Qain, st.p, TIME, st.pos), float).reshape(Ag.shape)
    assert np.array_equal(Ag, ref)


def test_coupled_without_data_extrapolates():
    st = Setup("1d", [BC.Coupled(tag="left", mesh_name="peer"),
                      BC.Extrapolation(tag="right")])
    u, _ = field_x(1)
    Qc, Qa, Qin, Qain, Qg, Ag = st.ghosts(u, None)
    assert np.array_equal(Qg, Qin)


def test_routing_by_tag_name_on_nonalphabetical_gmsh(gmsh_path):
    """Each boundary face is served by the branch of ITS tag name (the mesh
    numbers bottom, right, top, left; the kernel branches are alphabetical)."""
    vals = {"bottom": 1.1, "left": 1.2, "right": 1.3, "top": 1.4}
    bcs = [BC.Dirichlet(t, on="h", value=v) for t, v in vals.items()]
    st = Setup("gmsh", bcs, gmsh_path)
    u, _ = field_x(2, grad=(0.0, 0.0, 0.0, 0.0))
    Qc, Qa, Qin, Qain, Qg, Ag = st.ghosts(u)
    want = np.array([2 * vals[t] - 1.0 for t in st.face_tag])
    assert np.abs(Qg[1] - want).max() < 1e-14
    assert set(np.asarray(st.mesh.boundary_conditions_sorted_names)) == set(vals)
    assert list(st.nsm._bc_source.list_sorted_function_names) == sorted(vals)


def _A_normal(nsm, q, n):
    """Normal-projected quasilinear matrix at state q (numpy)."""
    rt = NumpyRuntimeModel.from_system_model(nsm)
    p = np.array(list(nsm.parameter_values.values()), float)
    qa = np.zeros(len(nsm.aux_state))
    if len(qa):
        qa = np.asarray(rt.update_aux_variables(q, qa, p, 0.0, np.zeros(3)),
                        float).ravel()
    A = np.asarray(rt.quasilinear_matrix(q, qa, p), float)
    return np.einsum("ijd,d->ij", A, n[:A.shape[2]])


def _projectors(A):
    lam, R = np.linalg.eig(A)
    lam, R = lam.real, R.real
    L = np.linalg.inv(R)
    Pm = R @ np.diag((lam < 0).astype(float)) @ L
    return Pm, np.eye(A.shape[0]) - Pm


@pytest.mark.parametrize("kind", ("1d", "2d"))
def test_characteristic_supercritical_inflow_exact(kind):
    """Characteristic ghost (incoming-characteristic content of the target in
    the ghost): all-incoming supercritical inflow with the target evaluated at
    the virtual point reproduces the linear field exactly."""
    dim = 1 if kind == "1d" else 2
    a = np.array([0.1, 1.0, 5.0, 0.0][:dim + 2])
    g = np.array([0.0, 0.1, 0.5, 0.0][:dim + 2])

    def target(kk):
        return lambda t, X, dX, Q, Qa, p, n: a[kk] + g[kk] * (X[0] + dX * n[0])
    bcs = [BC.Characteristic(tag="left",
                             prescribe_fields={k: target(k) for k in (1, 2)})] \
        + [BC.Extrapolation(tag=t) for t in tags_for(dim) if t != "left"]
    st = Setup(kind, bcs)

    def u(x):
        return a[:, None] + g[:, None] * x[0][None, :]
    Qc, Qa, Qin, Qain, Qg, Ag = st.ghosts(u)
    sel = st.face_tag == "left"
    exact = u(st.x_virtual)
    assert np.abs(Qg[:, sel] - exact[:, sel]).max() < 1e-10
    grad = np.asarray(lsq_gradient(st.topo, st.geom, Qc, Qg))
    B = np.zeros((dim + 2, dim))
    B[:, 0] = g
    left_only = st.checked_cells(("left",))
    assert np.abs(grad[:, :, left_only] - B[:, :, None]).max() < 1e-9


@pytest.mark.parametrize("cls", ["Characteristic", "CharacteristicWall"])
def test_characteristic_ghost_semantics(cls):
    """Subcritical face: the ghost carries the TARGET's incoming content
    (``P⁻(ghost − target) = 0``) and the interior's outgoing content
    (``P⁺(ghost − Q) = 0``) — the Riemann problem then reproduces the
    characteristic face state exactly (no doubling)."""
    tgt = np.array([0.1, 1.2, 0.2])
    if cls == "Characteristic":
        bc = BC.Characteristic(tag="left", prescribe_fields={
            k: (lambda kk: (lambda *args: tgt[kk]))(k) for k in (1, 2)})
    else:
        bc = BC.CharacteristicWall(tag="left")
    st = Setup("1d", [bc, BC.Extrapolation(tag="right")])
    q = np.array([0.1, 1.0, 0.4])
    n = np.array([-1.0, 0.0, 0.0])
    qa = st.aux(q[:, None])[:, 0]
    i_left = int(np.flatnonzero(st.face_tag == "left")[0])
    branch = int(st.topo.bc_branch[i_left])
    g = st.rt.boundary_ghost_state(branch, 0.0, np.zeros(3), 0.05, q, qa,
                                   st.p, n)
    target = tgt.copy() if cls == "Characteristic" else np.array([q[0], q[1], -q[2]])
    Pm, Pp = _projectors(_A_normal(st.nsm, q, n))
    assert np.abs(Pm @ (g - target)).max() < 1e-12
    assert np.abs(Pp @ (g - q)).max() < 1e-12
