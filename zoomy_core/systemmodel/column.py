"""The model's vertical column maps in NumPy: `interpolate_to_3d` (a cell state -> the canonical
`[b, h, u, v, w, p]` column at `nz` heights) and its inverse `project_from_3d` (a sampled
column -> the state), the exchange of a coupling to a resolved free-surface solver (VOF).

The heights are the cell-centre nodes `zeta_k = (k + 1/2) / nz` the model bakes into
`project_from_3d` (`SME(project_nz=nz)`), so `project(interpolate(q))` is the model's own
quadrature of its profile and a peer sampling its column on the same nodes needs no
interpolation.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

#: the canonical column fields, in `interpolate_to_3d` row order
FIELDS = ("b", "h", "u", "v", "w", "p")


def _scalar(entry):
    """The expression of a slot row (a `(1,)` row of an `(n, 1)` slot, or a scalar)."""
    vals = list(sp.flatten([entry])) if not isinstance(entry, sp.Basic) else [entry]
    if len(vals) != 1:
        raise ValueError(f"expected one expression per row, got {len(vals)}")
    return sp.sympify(vals[0])


class ColumnMap:
    """`interpolate(Q, Qaux, p)` and `project(C)` of a SystemModel (or NSM) `sm` on `nz` nodes.

    `Q` is `(n_state, n)`, `Qaux` `(n_aux, n)` in the aux order of `aux` (default `sm`'s; a
    NumericalSystemModel's appends its own rows, which the maps do not read), a column `C`
    `(6, nz, n)` in `FIELDS` order.
    """

    def __init__(self, sm, nz, aux=None):
        self.nz = int(nz)
        self.nodes = (np.arange(self.nz) + 0.5) / self.nz
        state = [sp.Symbol(str(s)) for s in sm.state]
        aux_names = [str(s) for s in (aux if aux is not None else sm.aux_state)]
        self.n_aux = len(aux_names)
        params = [sp.Symbol(str(s)) for s in sm.parameters.get_list()]
        self.n_state = len(state)

        rows = [_scalar(sm.interpolate_to_3d[r]) for r in range(len(FIELDS))]
        by_name = {}
        for e in rows:
            for s in e.free_symbols:
                by_name.setdefault(str(s), s)
        z = by_name.get("z", sp.Symbol("z"))
        missing = sorted(n for n in by_name if n not in {"z", *map(str, state), *aux_names,
                                                         *map(str, params)})
        if missing:
            raise ValueError(f"interpolate_to_3d reads {missing}, not a state, aux or "
                             "parameter of the model")
        # bind by name: the slot's symbols may carry other assumptions than the model's
        arg = lambda n: by_name.get(n, sp.Symbol(n))  # noqa: E731
        self._interp = sp.lambdify(
            [[arg(str(s)) for s in state], [arg(n) for n in aux_names],
             [arg(str(s)) for s in params], z], rows, "numpy")

        p3 = [_scalar(sm.project_from_3d[r]) for r in range(self.n_state)]
        samples = set().union(*(e.atoms(AppliedUndef) for e in p3))
        nodes = sorted({float(a.args[0]) for a in samples})
        if len(nodes) != self.nz or not np.allclose(nodes, self.nodes, atol=1e-12):
            raise ValueError(f"project_from_3d samples {len(nodes)} nodes, not the "
                             f"{self.nz} cell-centre nodes: build the model with "
                             f"project_nz={self.nz}")
        constants = {n: sp.Symbol(f"P3_{n}") for n in FIELDS}
        heads = {}
        for a in samples:
            heads.setdefault(a.func.__name__, a.func)
        cols = {name: [sp.Symbol(f"_{name}{k}") for k in range(self.nz)] for name in heads}
        subs = {a: cols[a.func.__name__][int(np.argmin(np.abs(self.nodes - float(a.args[0]))))]
                for a in samples}
        rows = [e.xreplace(subs) for e in p3]
        by_name = {str(s): s for e in rows for s in e.free_symbols}
        self._const = [f for f in FIELDS if f"P3_{f}" in by_name]
        self._heads = sorted(heads)                      # P3_u, P3_v
        self._project = sp.lambdify(
            [[by_name.get(f"P3_{f}", constants[f]) for f in self._const]]
            + [cols[h] for h in self._heads], rows, "numpy")

    def interpolate(self, Q, Qaux, p):
        """The column `(6, nz, n)` of the states `Q` with aux `Qaux`, parameters `p`."""
        Q, Qaux = np.atleast_2d(np.asarray(Q, float)), np.atleast_2d(np.asarray(Qaux, float))
        n = Q.shape[1]
        out = np.empty((len(FIELDS), self.nz, n))
        for k, zk in enumerate(self.nodes):
            vals = self._interp(list(Q), list(Qaux), list(np.asarray(p, float)), zk)
            for r, v in enumerate(vals):
                out[r, k] = np.broadcast_to(np.asarray(v, float), (n,))
        return out

    def project(self, C):
        """The states `(n_state, n)` of the columns `C` `(6, nz, n)` (depth-constant fields
        from the first node)."""
        C = np.asarray(C, float)
        const = [C[FIELDS.index(f), 0] for f in self._const]
        prof = [list(C[FIELDS.index(h.removeprefix("P3_"))]) for h in self._heads]
        vals = self._project(const, *prof)
        n = C.shape[2]
        return np.stack([np.broadcast_to(np.asarray(v, float), (n,)) for v in vals])
