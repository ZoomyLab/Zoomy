"""Gate 1a — the LSQ stencils are UNISOLVENT (``geometry_xp._lsq_stencils``).

A degree-p fit reproduces every derivative of a polynomial of degree <= p in
every inner cell, when the ghosts carry the exact value at the mirror image
of the cell centre (the samples the fit actually uses, REQ-174b decoupling
included).  The meshes are the rank-deficient cases of a count-based
selection: a 3x3x3 ring (x**3 = x), triangles whose rings lie on a conic,
jittered corners whose interior samples miss xy, one-boundary-face triangles
at p = 3.  Plus: compact wall stencils stay compact (quads p = 2: vertex ring
+ ghost, never extended into the domain).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from zoomy_core.mesh import LSQMesh
from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.geometry_xp import (
    build_topology, compute_geometry, lsq_derivatives)

pytestmark = [pytest.mark.template_gate]


def _triangles(n=8):
    xs = np.linspace(0.0, 1.0, n + 1)
    X = np.array([[x, y] for x in xs for y in xs]).T
    v = lambda i, k: i * (n + 1) + k                                  # noqa: E731
    tris = []
    for i in range(n):
        for k in range(n):
            a, b, c, d = v(i, k), v(i + 1, k), v(i + 1, k + 1), v(i, k + 1)
            tris += [(a, b, c), (a, c, d)] if (i + k) % 2 else [(a, b, d), (b, c, d)]
    return BaseMesh.from_arrays(X, np.array(tris).T, "triangle")


def _jittered(n=8, amp=0.25, seed=7):
    xs = np.linspace(0.0, 1.0, n + 1)
    X = np.array([[x, y] for x in xs for y in xs]).T
    inner = np.array([0 < i < n and 0 < k < n for i in range(n + 1) for k in range(n + 1)])
    X[:, inner] += np.random.default_rng(seed).uniform(-amp / n, amp / n, (2, int(inner.sum())))
    v = lambda i, k: i * (n + 1) + k                                  # noqa: E731
    q = [[v(i, k), v(i + 1, k), v(i + 1, k + 1), v(i, k + 1)] for i in range(n) for k in range(n)]
    return BaseMesh.from_arrays(X, np.array(q).T, "quad")


MESHES = {
    "line": lambda: LSQMesh.create_1d((0.0, 1.0), 12),
    "quad": lambda: LSQMesh.create_2d((0.0, 1.0, 0.0, 1.0), 8, 8),
    "jitter": lambda: LSQMesh.from_fvm(FVMMesh.from_base(_jittered())),
    "tri": lambda: LSQMesh.from_fvm(FVMMesh.from_base(_triangles())),
    "hex": lambda: LSQMesh.create_3d((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), 5, 5, 5),
}


def _poly(dim, p, seed):
    """A random polynomial of total degree p on the unit box and its
    derivatives: ``f(X, beta)`` at the points ``X`` ``(dim, n)``."""
    mis = [a for a in itertools.product(range(p + 1), repeat=dim) if sum(a) <= p]
    c = np.random.default_rng(seed).uniform(-1.0, 1.0, len(mis))

    def f(X, beta=None):
        beta = beta or (0,) * dim
        out = np.zeros(X.shape[1])
        for a, ca in zip(mis, c):
            if any(ai < bi for ai, bi in zip(a, beta)):
                continue
            t = np.full(X.shape[1], ca)
            for i in range(dim):
                t = t * (math.factorial(a[i]) // math.factorial(a[i] - beta[i])
                         * (X[i] - 0.5) ** (a[i] - beta[i]))
            out += t
        return out
    return f


@pytest.mark.parametrize("kind,p", [("line", 2), ("line", 3), ("quad", 3),
                                    ("jitter", 2), ("jitter", 3), ("tri", 2),
                                    ("tri", 3), ("hex", 3)])
def test_lsq_fit_reproduces_polynomials(kind, p):
    m = MESHES[kind]()
    dim, nc = int(m.dimension), int(m.n_inner_cells)
    topo = build_topology(m, lsq_degree=p)
    g = compute_geometry(topo, np.asarray(m.vertex_coordinates, float))
    cc = np.asarray(g["cell_centers"])[:dim, :nc]
    owner = topo.boundary_owner
    nrm = np.asarray(g["face_normals"])[:dim, topo.boundary_faces]
    nrm = nrm / np.linalg.norm(nrm, axis=0)
    mirror = cc[:, owner] + 2 * np.asarray(g["d_normal"]) * nrm
    f = _poly(dim, p, 100 + 10 * p + dim)
    got = np.asarray(lsq_derivatives(topo, g, f(cc)[None], f(mirror)[None]))[0]
    for k, mi in enumerate(topo.lsq_monomial_multi_index):
        ref = f(cc, tuple(int(a) for a in mi))
        err = np.abs(got[:, k] - ref).max() / max(np.abs(ref).max(), 1.0)
        assert err < 1e-9, (kind, p, mi, err)


def test_compact_wall_stencils_stay_compact():
    """Quads, p = 2: a wall cell's vertex ring (5 cells) + its ghost already
    determine the fit (the interior samples cannot tell y from y**2, but
    those are the ghost's pure-normal rows): no extension into the domain."""
    m = LSQMesh.create_2d((0.0, 1.0, 0.0, 1.0), 6, 6)
    topo = build_topology(m, lsq_degree=2)
    n = topo.lsq_neighbors[:m.n_inner_cells]
    size = np.array([len(set(r) - {c}) for c, r in enumerate(n)])
    n_ghost = (topo.lsq_boundary_face_neighbors[:m.n_inner_cells] >= 0).sum(axis=1)
    assert set(size[n_ghost == 0]) == {8}
    assert set(size[n_ghost == 1]) == {5}
    assert set(size[n_ghost == 2]) == {3}
