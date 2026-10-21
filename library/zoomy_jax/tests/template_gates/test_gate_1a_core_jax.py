"""Gate 1a-2 (jax side) — the new runtime slots and the geometry on jax.

* ``admissible`` lowered for numpy and jax agree (free surface, a declared
  Euler list, an empty list);
* the batched D5 ghost kernel + D6 ghost aux: jax == numpy for every BC
  class, jax batched == jax per-face, and differentiable (finite gradient);
* a state named ``T`` works in the (old and new) jax BC kernels;
* ``geometry_xp`` on ``jax.numpy`` == numpy, and ``jax.grad`` w.r.t. the node
  coordinates is finite (1-D and 2-D, incl. the LSQ operators) and matches a
  central finite difference.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import jax
import jax.numpy as jnp

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh
from zoomy_core.mesh.geometry_xp import build_topology, compute_geometry
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models import SWE
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.systemmodel import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_jax.transformation.jax_runtime import JaxRuntime

pytestmark = [pytest.mark.template_gate]

jax.config.update("jax_enable_x64", True)


def _tags(dim):
    return ("left", "right") if dim == 1 else ("left", "right", "bottom", "top")


def _bc_sets(dim):
    tl = np.array([0.0, 1.0])
    lam = {1: lambda t, X, dX, Q, Qa, p, n: 1.0 + 0.2 * X[0] + t}
    others = [BC.Extrapolation(tag=t) for t in _tags(dim)[2:]]
    return {
        "Extrapolation": [BC.Extrapolation(tag=t) for t in _tags(dim)],
        "Dirichlet+Wall": [BC.Dirichlet("left", on="h", value=1.3),
                           BC.Wall(tag="right")] + others,
        "InflowOutflow+RoughWall": [
            BC.InflowOutflow(tag="left", prescribe_fields={
                1: 1.1, 2: {"mode": "inlet_outlet", "value": 0.4}}),
            BC.RoughWall(tag="right")] + others,
        "Lambda+FromData": [
            BC.Lambda(tag="left", prescribe_fields=lam),
            BC.FromData(tag="right", timeline=tl,
                        prescribe_fields={1: np.array([1.0, 2.0])})] + others,
        "Periodic": [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
                     BC.Periodic(tag="right", periodic_to_physical_tag="left")]
        + others,
        "FromModel+Flux": [BC.FromModel(tag="left", definition="wall"),
                           BC.Flux("right", on="h", gradient=0.3)] + others,
        "Coupled+WindStress": [
            BC.Coupled(tag="left", mesh_name="peer"),
            BC.WindStress(tag="right", prescribe_values={1: 1.2},
                          prescribe_gradients={2: 0.1})] + others,
        "PerField": [BC.Dirichlet("left", on="h", value=1.2),
                     BC.Wall("left", on="hu"), BC.Extrapolation(tag="right")]
        + others,
        "Characteristic": [
            BC.Characteristic(tag="left", prescribe_fields={
                1: lambda *a: 1.2, 2: lambda *a: 0.3}),
            BC.CharacteristicWall(tag="right")] + others,
    }


def _setup(dim, bcs):
    mesh = (LSQMesh.create_1d((0.0, 1.0), 8) if dim == 1
            else LSQMesh.create_2d((0.0, 1.0, 0.0, 1.0), 4, 3))
    nsm = NumericalSystemModel.from_system_model(SystemModel.from_model(
        SWE(dimension=dim, boundary_conditions=BC.BoundaryConditions(bcs))))
    topo = build_topology(mesh, nsm=nsm)
    geom = compute_geometry(topo, np.asarray(mesh.vertex_coordinates, float))
    rng = np.random.default_rng(7)
    n = dim + 2
    nf = mesh.n_boundary_faces
    Q = np.vstack([rng.uniform(0.0, 0.1, nf), rng.uniform(0.8, 1.2, nf)]
                  + [rng.uniform(-0.5, 0.5, nf) for _ in range(n - 2)])
    p = np.array(list(nsm.parameter_values.values()), float)
    rt_np = NumpyRuntimeModel.from_system_model(nsm)
    Qa = np.asarray(rt_np.update_aux_variables(
        Q, np.zeros((len(nsm.aux_state), nf)), p, 0.0, np.zeros((3, nf))),
        float).reshape(len(nsm.aux_state), nf)
    args = dict(
        time=0.4,
        position=np.asarray(geom["face_centers"])[topo.boundary_faces].T,
        distance=np.asarray(geom["d_normal"]),
        Q=Q, Qaux=Qa, p=p,
        normal=np.asarray(geom["face_normals"])[:, topo.boundary_faces],
        bdata=Q[::-1].copy() * 0.5 + 0.5)
    return nsm, topo, rt_np, JaxRuntime.from_nsm(nsm), args


CASES = [(d, k) for d in (1, 2) for k in _bc_sets(d)]


@pytest.mark.parametrize("dim,case", CASES)
def test_ghost_jax_equals_numpy_and_is_differentiable(dim, case):
    nsm, topo, rt_np, rt_jx, a = _setup(dim, _bc_sets(dim)[case])
    g_np = rt_np.boundary_ghost(topo.bc_branch)
    g_jx = rt_jx.boundary_ghost(topo.bc_branch)
    pos, dist, Q, Qa, p, nrm, bd = (a["position"], a["distance"], a["Q"],
                                    a["Qaux"], a["p"], a["normal"], a["bdata"])
    Qg_n, Ag_n = g_np(a["time"], pos, dist, Q, Qa, p, nrm, bd)
    f = jax.jit(lambda Q_, Qa_: g_jx(a["time"], jnp.asarray(pos),
                                     jnp.asarray(dist), Q_, Qa_,
                                     jnp.asarray(p), jnp.asarray(nrm),
                                     jnp.asarray(bd)))
    Qg_j, Ag_j = f(jnp.asarray(Q), jnp.asarray(Qa))
    sc = max(1.0, np.abs(Qg_n).max())
    assert np.abs(np.asarray(Qg_j) - Qg_n).max() <= 1e-13 * sc
    assert np.abs(np.asarray(Ag_j) - Ag_n).max() <= 1e-13 * max(1.0, np.abs(Ag_n).max())
    # per-face jax == batched
    i = Q.shape[1] - 1
    one = rt_jx.boundary_ghost_state(int(topo.bc_branch[i]), a["time"],
                                     pos[:, i], dist[i], Q[:, i], Qa[:, i],
                                     p, nrm[:, i], bd[:, i])
    assert np.abs(np.asarray(one) - Qg_n[:, i]).max() <= 1e-13 * sc
    # differentiable through every branch (no inactive-branch NaN).  KNOWN
    # LIMIT: the characteristic family reads the OPAQUE eigensystem kernel
    # (a jax pure_callback — no JVP rule), so it is forward-only.
    if case == "Characteristic":
        return
    grad = jax.grad(lambda Q_: jnp.sum(f(Q_, jnp.asarray(Qa))[0] ** 2)
                    + jnp.sum(f(Q_, jnp.asarray(Qa))[1]))(jnp.asarray(Q))
    assert np.all(np.isfinite(np.asarray(grad)))


def _euler():
    t, x = sp.Symbol("t", real=True), sp.Symbol("x", real=True)
    rho, m, E = sp.symbols("rho m E", real=True)
    gam = sp.Symbol("gamma", positive=True)
    pz = Zstruct(gamma=gam)
    pz._symbolic_name = "p"
    pr = (gam - 1) * (E - m ** 2 / (2 * rho))
    sm = SystemModel(
        time=t, space=[x], state=[rho, m, E], aux_state=[], parameters=pz,
        flux=sp.Matrix([[m], [m ** 2 / rho + pr], [(E + pr) * m / rho]]),
        hydrostatic_pressure=sp.zeros(3, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(3, 3, 1),
        source=sp.zeros(3, 1), mass_matrix=sp.eye(3),
        parameter_values=Zstruct(gamma=1.4))
    sm.attach_boundary_conditions(BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]))
    sm.declare_admissible([rho, pr])
    return sm


def _tracer(name):
    t, x = sp.Symbol("t", real=True), sp.Symbol("x", real=True)
    q, a = sp.Symbol(name, real=True), sp.Symbol("a", real=True)
    pz = Zstruct(a=a)
    pz._symbolic_name = "p"
    sm = SystemModel(
        time=t, space=[x], state=[q], aux_state=[], parameters=pz,
        flux=sp.Matrix([[a * q]]), hydrostatic_pressure=sp.zeros(1, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(1, 1, 1),
        source=sp.zeros(1, 1), mass_matrix=sp.eye(1),
        parameter_values=Zstruct(a=1.0))
    sm.attach_boundary_conditions(BC.BoundaryConditions(
        [BC.Dirichlet("left", value=2.0), BC.Extrapolation(tag="right")]))
    return sm


@pytest.mark.parametrize("which", ["swe", "euler", "tracer"])
def test_admissible_numpy_equals_jax(which):
    if which == "swe":
        nsm = NumericalSystemModel.from_system_model(
            SystemModel.from_model(SWE(dimension=2)))
        Q = np.array([[0.0, 0.1, 0.2], [1.0, -0.2, 0.3],
                      [0.3, 0.0, 0.1], [0.0, 0.2, 0.1]])
        want_rows = 1
    elif which == "euler":
        nsm = NumericalSystemModel.from_system_model(_euler())
        Q = np.array([[1.0, 0.5, 2.0], [0.2, 0.1, -1.0], [2.5, 1e-3, 3.0]])
        want_rows = 2
    else:
        nsm = NumericalSystemModel.from_system_model(_tracer("h"))
        Q = np.array([[1e-3, 0.5, 2.0]])
        want_rows = 0
    p = np.array(list(nsm.parameter_values.values()), float)
    Qa = np.ones((len(nsm.aux_state), Q.shape[1]))
    a_np = NumpyRuntimeModel.from_system_model(nsm).admissible(Q, Qa, p)
    a_jx = np.asarray(JaxRuntime.from_nsm(nsm).admissible(
        jnp.asarray(Q), jnp.asarray(Qa), jnp.asarray(p)))
    assert a_np.shape == a_jx.shape == (want_rows, Q.shape[1])
    assert np.abs(a_np - a_jx).max(initial=0.0) <= 1e-15


def test_state_named_T_in_jax_bc_kernels():
    """The old jax BC kernel picked ``Q.T`` (the array transpose) for a state
    named ``T`` and crashed; both the old kernel and the new ghost kernel must
    index ``Q[i]``."""
    nsm = NumericalSystemModel.from_system_model(_tracer("T"))
    rt = JaxRuntime.from_nsm(nsm)
    q = jnp.array([0.7])
    p, n = jnp.array([1.0]), jnp.array([-1.0, 0.0, 0.0])
    left = rt.boundary_conditions(0, 0.0, jnp.zeros(3), 0.1, q, jnp.zeros(0), p, n)
    assert np.allclose(np.asarray(left).ravel(), [2.0])        # face value
    g = rt.boundary_ghost(np.array([0, 1]))
    Qg, _ = g(0.0, jnp.zeros((3, 2)), jnp.array([0.1, 0.1]),
              jnp.array([[0.7, 0.4]]), jnp.zeros((0, 2)), p,
              jnp.array([[-1.0, 1.0], [0, 0], [0, 0]]))
    assert np.allclose(np.asarray(Qg), [[2 * 2.0 - 0.7, 0.4]])   # D5 ghosts


def _geom_scalar(topo, X):
    g = compute_geometry(topo, X)
    keys = ("cell_centers", "cell_volumes", "cell_inradius", "face_normals",
            "face_volumes", "face_centers", "face_inradius", "d_face",
            "d_normal", "tpfa_gA", "tpfa_bnd_g", "lsq_gradQ")
    return sum(jnp.sum(jnp.sin(1.0 + jnp.asarray(g[k]))) for k in keys)


@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_geometry_jax_equals_numpy_and_grad_wrt_nodes(kind):
    if kind == "1d":
        mesh = LSQMesh.create_1d((0.0, 1.0), 9)
    else:
        mesh = LSQMesh.create_2d((0.0, 1.0, 0.0, 0.8), 4, 3)
    mesh._build_lsq_stencil(2)
    topo = build_topology(mesh, lsq_degree=2)
    dim = int(mesh.dimension)
    X = np.asarray(mesh.vertex_coordinates, float)[:dim]
    # perturb the nodes so the cells are generic (not axis-aligned)
    rng = np.random.default_rng(3)
    X = X + 0.02 * rng.uniform(-1, 1, X.shape) * (1.0 / X.shape[1])
    g_np = compute_geometry(topo, X)
    g_jx = compute_geometry(topo, jnp.asarray(X))
    for k, v in g_np.items():
        v = np.asarray(v)
        assert np.abs(np.asarray(g_jx[k]) - v).max(initial=0.0) <= \
            1e-13 * max(1.0, np.abs(v).max(initial=0.0)), k
    grad = np.asarray(jax.grad(lambda Y: _geom_scalar(topo, Y))(jnp.asarray(X)))
    assert np.all(np.isfinite(grad))
    # directional derivative == central finite difference
    d = rng.normal(size=X.shape)
    eps = 1e-6
    fd = (float(_geom_scalar(topo, jnp.asarray(X + eps * d)))
          - float(_geom_scalar(topo, jnp.asarray(X - eps * d)))) / (2 * eps)
    assert abs(float(np.sum(grad * d)) - fd) <= 1e-6 * max(1.0, abs(fd))
