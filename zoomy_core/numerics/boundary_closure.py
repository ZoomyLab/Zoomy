"""Classify, per boundary face and row, whether a boundary condition fixes the value (Dirichlet),
the normal derivative (Neumann) or neither (ghost), from the ghost-state kernel.
Feeds the constrained LSQ derivative fits of `geometry_xp.build_topology`; a row misclassified
as ghost costs accuracy, never correctness.

References:
    Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
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


def closure_kinds(nsm, systems, branch, normals):
    """`{key: (n_rows, n_bf) kinds}` over `closure_spaces`, the `closure` argument of
    `geometry_xp.build_topology`; systems without BCs are skipped.
    """
    out = {}
    for key, system, space in closure_spaces(nsm, systems):
        if getattr(system, "boundary_conditions", None) is None:
            continue
        out[key] = face_kinds(system, space, branch, normals)
    return out


__all__ = ["GHOST", "DIRICHLET", "NEUMANN", "KIND_NAMES", "face_kinds",
           "closure_spaces", "closure_kinds"]
