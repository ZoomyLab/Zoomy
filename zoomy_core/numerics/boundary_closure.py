"""Which rows a boundary condition FIXES, per boundary face (ANALYSIS 40).

The LSQ derivative fits close at the boundary by constraints, not by ghost
samples (:mod:`zoomy_core.mesh.geometry_xp`): a boundary face adds, per row,

* DIRICHLET  ``p(x_foot) = v``             (the value at the foot point),
* NEUMANN    ``n . grad p(x_foot) = g``    (the normal derivative there),
* GHOST      the ghost as a fit sample at the mirror point (today's rule).

Fits of degree >= 2 only: a linear fit's gradient is one constant, so a
constraint would pin it to the wall value, while the ghost sample is exact
for linear data and centred (``geometry_xp.build_topology``).

The data come from the EXISTING D5 ghost kernel ``boundary_ghost_state`` --
no new BC API.  For an affine BC the ghost is ``ghost = J Q + c`` per face; a
row ``r`` whose ``J`` row is ``-e_r`` is Dirichlet-type with
``v = (ghost_r + Q_r) / 2``, one whose ``J`` row is ``+e_r`` is
Neumann-type with ``g = (ghost_r - Q_r) / (2 d_n)``.  Both are a fixed
multiple of ``ghost_r - Q_r``, so the runtime's input (the ghost) does not
change -- only the fit table does.  Every other row stays GHOST: rows the
kernel couples (a wall reflection on a face that is not normal to an axis --
the momentum VECTOR; a frame rotation is the clean follow-up), nonlinear or
data-dependent ``J`` (characteristic BCs, a slip length).

The test, per (space, branch, row): ``ghost_r + W_r`` (Dirichlet) or
``ghost_r - W_r`` (Neumann) must not depend on the state.  Its gradient in
the state is built symbolically once; per boundary face it is evaluated with
the face normal and random values of every other argument (state, aux,
parameters, time, position, distance, boundary data; positive and mixed
signs) -- zero in every draw (NaN is not zero) or the row stays GHOST.
Wrongly calling a row GHOST costs accuracy, never correctness.

Spaces: ``"state"`` -- the rows of ``Q`` (derivative aux); ``"reconstruction"``
-- the rows of the reconstructed variables ``W(Q)`` of
:func:`zoomy_core.symsolver.discretization.reconstruction.reconstruct` (the
O2 gradient), whose ghost is ``W`` of the ghost state with its own aux.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

GHOST, DIRICHLET, NEUMANN = 0, 1, 2
KIND_NAMES = {GHOST: "ghost", DIRICHLET: "dirichlet", NEUMANN: "neumann"}

#: Random draws per test (the first half positive, the rest mixed signs).
_DRAWS = 4
#: |d(ghost +- W)/dQ| below this in every draw: no state dependence.
_ZERO = 1e-9
_SEED = 20261012


def _flat(expr):
    return list(sp.flatten(expr)) if expr is not None else []


def _symbols(fn):
    """Argument groups of the ghost kernel ``fn`` (lists of symbols)."""
    a = fn.args

    def grp(k):
        v = a[k]
        return list(v.get_list()) if hasattr(v, "get_list") else [v]
    return {k: grp(k) for k in a.keys() if k != "idx"}


def _subs_op(op, Q, A, p):
    """The definition of NSM operator ``op`` (args variables, aux_variables,
    parameters) at the state ``Q`` and aux ``A`` (lists of expressions)."""
    args = op.args
    m = dict(zip(args.variables.get_list(), Q))
    m.update(zip(args.aux_variables.get_list(), A))
    if hasattr(args, "parameters"):
        m.update(zip(args.parameters.get_list(), p))
    return [sp.sympify(e).xreplace(m) for e in _flat(op.definition)]


def _ghost_pairs(nsm, space):
    """``(W, [Wg_k per branch], symbols)``: the row expressions of ``space``
    at the inner state and at the ghost of each BC branch, in the ghost
    kernel's argument symbols."""
    fn, branches = nsm.boundary_ghost_kernel()
    sym = _symbols(fn)
    Q, A, p = sym["variables"], sym["aux_variables"], sym["parameters"]
    upd = nsm.operator("update_aux_variables")

    def aux(X, Aprev):
        if upd is None or not _flat(upd.definition):
            return list(Aprev)
        return _subs_op(upd, X, Aprev, p)
    A_in = aux(Q, A)                           # the inner cell's aux
    G = [[sp.sympify(e).xreplace(dict(zip(A, A_in))) for e in b] for b in branches]
    if space == "state":
        return list(Q), G, sym
    if space != "reconstruction":
        raise ValueError(f"closure space {space!r}: 'state' or 'reconstruction'")
    from zoomy_core.symsolver.discretization.reconstruction import bed_rows, maps
    to_w, _ = maps(nsm)
    if to_w is None:
        W, Wg = list(Q), [list(g) for g in G]
    else:
        op = nsm.operator("reconstruction_variables")
        W = _subs_op(op, Q, A_in, p)
        Wg = [_subs_op(op, g, aux(g, A_in), p) for g in G]
    rows = bed_rows(nsm) if to_w is not None else None
    if rows is not None:                       # h in the bed's slot
        bed, depth = rows
        W[bed] = Q[depth]
        for wg, g in zip(Wg, G):
            wg[bed] = g[depth]
    return W, Wg, sym


def _tests(nsm, space):
    """Per branch, per row: ``(dirichlet, neumann)`` evaluators of the state
    gradient of ``ghost + W`` / ``ghost - W`` (``None``: identically zero);
    cached on the NSM."""
    cache = nsm.__dict__.setdefault("_closure_tests", {})
    if space in cache:
        return cache[space]
    W, Wg, sym = _ghost_pairs(nsm, space)
    Q, n = sym["variables"], sym["normal"]
    others = [s for k, v in sym.items() if k not in ("normal",) for s in v]
    out = []
    for g in Wg:
        per_row = []
        for r, w in enumerate(W):
            pair = []
            for sign in (1, -1):               # ghost + W, ghost - W
                e = g[r] + sign * w
                d = [sp.diff(e, q) for q in Q]
                if all(x == 0 for x in d):
                    pair.append(None)
                    continue
                try:
                    f = sp.lambdify([n, others], d, "numpy")
                except Exception:              # not printable: never zero
                    f = False
                pair.append(f)
            per_row.append(tuple(pair))
        out.append(per_row)
    cache[space] = (out, len(n), len(others), len(W))
    return cache[space]


def _vanishes(f, normals, n_others, rng):
    """Is the lambdified gradient ``f`` zero at every face (columns of
    ``normals``) in every random draw?"""
    if f is None:
        return np.ones(normals.shape[1], dtype=bool)
    if f is False:
        return np.zeros(normals.shape[1], dtype=bool)
    ok = np.ones(normals.shape[1], dtype=bool)
    for k in range(_DRAWS):
        vals = rng.uniform(0.5, 1.5, n_others)
        if k >= _DRAWS // 2:
            vals = vals * rng.choice((-1.0, 1.0), n_others)
        with np.errstate(all="ignore"):
            try:
                d = f(list(normals), list(vals))
            except Exception:
                return np.zeros(normals.shape[1], dtype=bool)
        for x in d:
            x = np.broadcast_to(np.asarray(x, dtype=float), ok.shape)
            ok &= np.abs(x) <= _ZERO                    # NaN: False
    return ok


def face_kinds(nsm, space, branch, normals):
    """``(n_rows, n_faces)`` int8 kinds (GHOST / DIRICHLET / NEUMANN) of the
    rows of ``space`` on the boundary faces with BC branch ``branch``
    (``(n_faces,)``; -1: no BC, e.g. a partition cut -> GHOST) and outward
    unit ``normals`` (``(dim_n, n_faces)``, padded rows ignored)."""
    tests, n_dim, n_others, n_rows = _tests(nsm, space)
    branch = np.asarray(branch, dtype=int)
    normals = np.asarray(normals, dtype=float)
    kinds = np.full((n_rows, branch.size), GHOST, dtype=np.int8)
    rng = np.random.default_rng(_SEED)
    for k in np.unique(branch):
        if k < 0 or k >= len(tests):
            continue
        faces = np.flatnonzero(branch == k)
        nk = np.zeros((n_dim, faces.size))
        m = min(n_dim, normals.shape[0])
        nk[:m] = normals[:m, faces]
        for r, (fd, fn_) in enumerate(tests[k]):
            dirichlet = _vanishes(fd, nk, n_others, rng)
            neumann = _vanishes(fn_, nk, n_others, rng) & ~dirichlet
            kinds[r, faces[dirichlet]] = DIRICHLET
            kinds[r, faces[neumann]] = NEUMANN
    return kinds


def closure_spaces(nsm, systems=()):
    """``[(key, system, space)]``: every row space whose derivative fits a
    solver template closes -- the state of ``nsm`` and of each of
    ``systems`` (derivative aux), and the reconstruction variables of
    ``nsm`` at order >= 2 (the O2 gradient).  ``key = (owner, space)``, the
    owner the system's printed name (``_symbolic_name``, default ``nsm``)."""
    def owner(s):
        return getattr(s, "_symbolic_name", None) or "nsm"
    out = [((owner(nsm), "state"), nsm, "state")]
    if int(getattr(nsm.reconstruction, "order", 1)) >= 2:
        out.append(((owner(nsm), "reconstruction"), nsm, "reconstruction"))
    for s in systems:
        out.append(((owner(s), "state"), s, "state"))
    return out


def closure_kinds(nsm, systems, branch, normals):
    """``{key: (n_rows, n_bf) kinds}`` for :func:`closure_spaces` -- the
    ``closure`` argument of :func:`geometry_xp.build_topology`.  Systems
    without boundary conditions are skipped (no ghost kernel: GHOST)."""
    out = {}
    for key, system, space in closure_spaces(nsm, systems):
        if getattr(system, "boundary_conditions", None) is None:
            continue
        out[key] = face_kinds(system, space, branch, normals)
    return out


__all__ = ["GHOST", "DIRICHLET", "NEUMANN", "KIND_NAMES", "face_kinds",
           "closure_spaces", "closure_kinds"]
