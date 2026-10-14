"""Classify, per boundary face and row, whether a boundary condition fixes the value (Dirichlet),
the normal derivative (Neumann) or neither (ghost), from the ghost-state kernel.
Feeds the constrained LSQ derivative fits of `geometry_xp.build_topology`; a row misclassified
as ghost costs accuracy, never correctness.

A vector BC (a wall reflecting the normal momentum) couples the rows of a vector group on a face
not aligned with an axis: no row is Dirichlet or Neumann by itself.  In the face's normal frame
(normal component n.q, tangential components) the same ghost is Dirichlet in the normal and
Neumann in the tangential components; `face_closure` classifies the groups there wherever that
fixes more rows than the Cartesian frame (`RowClosure.rotated`).

References:
    Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    """Flatten a sympy expression into a list (empty for `None`)."""
    return list(sp.flatten(expr)) if expr is not None else []


def _symbols(fn):
    """Argument groups of the ghost kernel `fn` as lists of symbols."""
    a = fn.args

    def grp(k):
        """Symbols of argument group `k` as a list."""
        v = a[k]
        return list(v.get_list()) if hasattr(v, "get_list") else [v]
    return {k: grp(k) for k in a.keys() if k != "idx"}


def _subs_op(op, Q, A, p):
    """Definition of operator `op` evaluated at state `Q`, aux `A` and parameters `p`."""
    args = op.args
    m = dict(zip(args.variables.get_list(), Q))
    m.update(zip(args.aux_variables.get_list(), A))
    if hasattr(args, "parameters"):
        m.update(zip(args.parameters.get_list(), p))
    return [sp.sympify(e).xreplace(m) for e in _flat(op.definition)]


def _ghost_pairs(nsm, space):
    """Row expressions of `space` at the inner state and at the ghost of each BC branch, in the
    ghost kernel's symbols.
    """
    fn, branches = nsm.boundary_ghost_kernel()
    sym = _symbols(fn)
    Q, A, p = sym["variables"], sym["aux_variables"], sym["parameters"]
    upd = nsm.operator("update_aux_variables")

    def aux(X, Aprev):
        """Aux values at state `X` from `update_aux_variables` (`Aprev` if the system has
        none).
        """
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
    """Per branch and row, evaluators of the state gradient of `ghost + W` (Dirichlet) and
    `ghost - W` (Neumann), cached on the NSM.
    """
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
    """Per face, whether the lambdified gradient `f` is zero in every random draw."""
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
    """Kinds (`GHOST` / `DIRICHLET` / `NEUMANN`) of the rows of `space` on the boundary
    faces; a `Coupled` face's rows are `GHOST` (its ghost is the coupling's data).

    Args:
        branch: BC branch per face (-1: no BC, e.g. a partition cut, gives ghost).
        normals: outward unit normals, shape `(dim_n, n_faces)`.

    Returns:
        `(n_rows, n_faces)` int8 array.
    """
    tests, n_dim, n_others, n_rows = _tests(nsm, space)
    # a coupled face's ghost is the coupling's data, no closed-form condition: a ghost row
    from zoomy_core.model.boundary_conditions import Coupled
    bcs = getattr(getattr(nsm, "_bc_source", None), "boundary_conditions_list", None) or []
    coupled = {k for k, bc in enumerate(bcs) if isinstance(bc, Coupled)}
    branch = np.asarray(branch, dtype=int)
    normals = np.asarray(normals, dtype=float)
    kinds = np.full((n_rows, branch.size), GHOST, dtype=np.int8)
    rng = np.random.default_rng(_SEED)
    for k in np.unique(branch):
        if k < 0 or k >= len(tests) or k in coupled:
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


@dataclass(frozen=True)
class RowClosure:
    """Closure kinds of the rows of one row space on the boundary faces.

    Attributes:
        kinds: `(n_rows, n_bf)` int8 kinds.  On a face where `rotated[g]`, the rows of vector
            group `g` carry the kinds of the group's components in the face's normal frame: its
            first row the normal component `n . q`, the others the tangential components.
        groups: vector row groups `((r_x, r_y[, r_z]), ...)`, components in axis order.
        rotated: `(n_groups, n_bf)` bool, the faces whose group rows are in the normal frame.
    """

    kinds: np.ndarray
    groups: tuple = ()
    rotated: np.ndarray = field(default=None)


def _bc_list(bcs):
    """Every BC object of the container `bcs`, slot and channel BCs included."""
    out, todo = [], list(getattr(bcs, "boundary_conditions_list", []) or [])
    while todo:
        bc = todo.pop()
        if any(bc is b for b in out):
            continue
        out.append(bc)
        todo += list(getattr(bc, "_slot_bc", {}).values())
        if hasattr(bc, "channels"):
            todo += list(bc.channels().values())
    return out


def vector_groups(nsm):
    """The vector row groups the BCs of `nsm` act on as vectors (`momentum_field_indices` of
    walls), components in axis order; groups of one row and groups that share a row with a
    different group are dropped.
    """
    found = []
    for bc in _bc_list(getattr(nsm, "_bc_source", None)):
        for g in getattr(bc, "momentum_field_indices", None) or ():
            g = tuple(int(r) for r in g)
            if len(g) >= 2 and len(set(g)) == len(g) and g not in found:
                found.append(g)
    rows = [r for g in found for r in g]
    return tuple(g for g in found if all(rows.count(r) == 1 for r in g))


def _rotated_tests(nsm, space, groups):
    """Per branch and group, evaluators of the state gradient of the normal component of
    `ghost + W` / `ghost - W` and of its tangential components (2-D: `t = (-n_y, n_x)`; 3-D the
    rows of `I - n n^T`), cached on the NSM.
    """
    cache = nsm.__dict__.setdefault("_closure_rotated_tests", {})
    key = (space, groups)
    if key in cache:
        return cache[key]
    W, Wg, sym = _ghost_pairs(nsm, space)
    Q, n = sym["variables"], sym["normal"]
    others = [s for k, v in sym.items() if k not in ("normal",) for s in v]

    def frame(v, d):
        """Normal and tangential components of the vector `v` (length `d`)."""
        nv = n[:d]
        vn = sum(nv[i] * v[i] for i in range(d))
        if d == 2:
            return vn, [-nv[1] * v[0] + nv[0] * v[1]]
        return vn, [v[i] - nv[i] * vn for i in range(d)]

    def test(exprs):
        """Lambdified state gradient of `exprs` (None: identically zero, False: not
        printable).
        """
        d = [sp.diff(e, q) for e in exprs for q in Q]
        if all(x == 0 for x in d):
            return None
        try:
            return sp.lambdify([n, others], d, "numpy")
        except Exception:
            return False
    out = []
    for g in Wg:
        per_group = []
        for grp in groups:
            d = len(grp)
            if d > len(n) or max(grp) >= len(W):
                per_group.append(None)
                continue
            wn, wt = frame([W[r] for r in grp], d)
            gn, gt = frame([g[r] for r in grp], d)
            per_group.append(((test([gn + wn]), test([gn - wn])),
                              (test([a + b for a, b in zip(gt, wt)]),
                               test([a - b for a, b in zip(gt, wt)]))))
        out.append(per_group)
    cache[key] = (out, len(others))
    return cache[key]


def face_closure(nsm, space, branch, normals):
    """`RowClosure` of the rows of `space` on the boundary faces: the Cartesian `face_kinds`,
    and on each face where a vector group's rows are coupled, the group's kinds in the face's
    normal frame if that fixes more rows.

    Args:
        branch: BC branch per face (-1: no BC).
        normals: outward unit normals, shape `(dim_n, n_faces)`.
    """
    kinds = face_kinds(nsm, space, branch, normals)
    n_rows = kinds.shape[0]
    groups = tuple(g for g in vector_groups(nsm) if max(g) < n_rows)
    branch = np.asarray(branch, dtype=int)
    normals = np.asarray(normals, dtype=float)
    rotated = np.zeros((len(groups), branch.size), dtype=bool)
    if not groups:
        return RowClosure(kinds, groups, rotated)
    tests, n_others = _rotated_tests(nsm, space, groups)
    n_dim = _tests(nsm, space)[1]
    rng = np.random.default_rng(_SEED + 1)
    for k in np.unique(branch):
        if k < 0 or k >= len(tests):
            continue
        faces = np.flatnonzero(branch == k)
        nk = np.zeros((n_dim, faces.size))
        m = min(n_dim, normals.shape[0])
        nk[:m] = normals[:m, faces]
        for gi, grp in enumerate(groups):
            t = tests[k][gi]
            if t is None:
                continue
            cand = (kinds[list(grp)][:, faces] == GHOST).any(axis=0)
            if not cand.any():
                continue
            f, nc = faces[cand], nk[:, cand]
            kn = np.full(f.size, GHOST, dtype=np.int8)
            kt = np.full(f.size, GHOST, dtype=np.int8)
            for kk, (fd, fn_) in ((kn, t[0]), (kt, t[1])):
                dirichlet = _vanishes(fd, nc, n_others, rng)
                neumann = _vanishes(fn_, nc, n_others, rng) & ~dirichlet
                kk[dirichlet], kk[neumann] = DIRICHLET, NEUMANN
            d = len(grp)
            fixed_rot = (kn != GHOST).astype(int) + (d - 1) * (kt != GHOST)
            fixed_cart = (kinds[list(grp)][:, f] != GHOST).sum(axis=0)
            use = fixed_rot > fixed_cart
            if not use.any():
                continue
            fu = f[use]
            rotated[gi, fu] = True
            kinds[grp[0], fu] = kn[use]
            for r in grp[1:]:
                kinds[r, fu] = kt[use]
    return RowClosure(kinds, groups, rotated)


def closure_spaces(nsm, systems=()):
    """Row spaces whose derivative fits a solver template closes: the state of `nsm` and
    `systems`, and the reconstruction variables at order >= 2.

    Returns:
        List of `(key, system, space)` with `key = (owner, space)`.
    """
    def owner(s):
        """Printed name of system `s` (`_symbolic_name`, default `nsm`)."""
        return getattr(s, "_symbolic_name", None) or "nsm"
    out = [((owner(nsm), "state"), nsm, "state")]
    if int(getattr(nsm.reconstruction, "order", 1)) >= 2:
        out.append(((owner(nsm), "reconstruction"), nsm, "reconstruction"))
    for s in systems:
        out.append(((owner(s), "state"), s, "state"))
    return out


def closure_kinds(nsm, systems, branch, normals, frames=False):
    """`{key: (n_rows, n_bf) kinds}` over `closure_spaces`, the `closure` argument of
    `geometry_xp.build_topology`; systems without BCs are skipped.  With `frames` the values are
    `RowClosure`s (vector groups in the faces' normal frames, `face_closure`).
    """
    out = {}
    for key, system, space in closure_spaces(nsm, systems):
        if getattr(system, "boundary_conditions", None) is None:
            continue
        out[key] = (face_closure if frames else face_kinds)(system, space, branch, normals)
    return out


__all__ = ["GHOST", "DIRICHLET", "NEUMANN", "KIND_NAMES", "RowClosure", "face_kinds",
           "face_closure", "vector_groups", "closure_spaces", "closure_kinds"]
