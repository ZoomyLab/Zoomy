"""The kernels of a NumericalSystemModel in numpy: every entry of `nsm.call` lambdified once
and evaluated on whole columns (cells or faces) by broadcasting, the numpy counterpart of
`zoomy_jax`'s vmapped `JaxRuntime` for the printed solver templates.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from zoomy_core.fvm.userfunctions import numpy_module

#: column operators stored as `(n, 1)`: the trailing 1 is dropped (as `JaxRuntime`)
_SQUEEZE = ("source", "source_explicit", "update_variables", "update_aux_variables")
#: argument groups that vary per column; the others (parameters, time, dt) are scalars
_COLUMN = ("variables", "aux_variables", "normal", "position", "reconstruction",
           "boundary_data", "q_minus", "q_plus", "aux_minus", "aux_plus")


def _symbols(group):
    """The symbols of one signature group: a Zstruct, a list or a bare symbol."""
    if isinstance(group, sp.Basic):
        return [group]
    if hasattr(group, "values") and callable(group.values):
        return list(group.values())
    return list(group)


def _array(definition):
    """`(flat entries, shape)` of a kernel definition."""
    if isinstance(definition, sp.MatrixBase):
        return list(definition), tuple(definition.shape)
    if hasattr(definition, "shape") and hasattr(definition, "tolist"):
        shape = tuple(definition.shape)
        return list(sp.flatten(definition.tolist())) if shape else [definition], shape
    if isinstance(definition, (list, tuple)):
        return list(sp.flatten(definition)), (len(definition),)
    return [definition], ()


class Lowered:
    """One kernel definition lambdified against signature groups; called with one array per
    group (column groups `(rows, n)`, scalars otherwise), returns `shape + (n,)`.
    """

    def __init__(self, definition, groups, module):
        """Lambdify `definition` over `groups` (lists of symbols) with `module`."""
        self.flat, self.shape = _array(definition)
        self.counts = [len(g) for g in groups]
        args = [s for g in groups for s in g]
        self.fn = sp.lambdify(args, self.flat, modules=[module, "numpy"], cse=True)

    def __call__(self, *values):
        """Evaluate on the columns of `values`; every entry broadcast to the column count."""
        flat = []
        for v, k in zip(values, self.counts):
            if k == 0:
                continue
            v = np.asarray(v, dtype=float)
            if k == 1 and v.ndim <= 1:          # a scalar, or one column (distance)
                flat.append(v)
            else:                               # rows of a group (parameters: scalars)
                flat.extend(v[i] for i in range(k))
        n = max([e.shape[-1] for e in flat if e.ndim] + [1])
        res = self.fn(*flat)
        out = np.empty((len(res), n))
        for i, e in enumerate(res):
            out[i] = e                          # broadcasts a constant entry
        return out.reshape(self.shape + (n,))


class Indexed:
    """An indexed boundary kernel (a `Piecewise` over `idx` of state vectors), called
    `bc(branch, time, position, distance, q, qaux, p, normal)` with one branch per column;
    each branch is evaluated on its own columns only.
    """

    def __init__(self, function, n_state, module):
        """Lambdify every branch of `function` (a `Function` whose definition is the
        `Piecewise`)."""
        a = function.args
        self.idx = a["idx"]
        self.keys = [k for k in a.keys() if k != "idx"]
        groups = [_symbols(a[k]) for k in self.keys]
        d = function.definition
        pieces = d.args if isinstance(d, sp.Piecewise) else ((d, sp.true),)
        self.pieces = [(Lowered(v, groups, module), c) for v, c in pieces]
        self.n, self.branch = n_state, {}

    def _piece(self, k):
        """The lowered branch taken at `idx = k`."""
        if k not in self.branch:
            for f, c in self.pieces:
                if bool(sp.sympify(c).subs(self.idx, k)):
                    self.branch[k] = f
                    break
            else:
                raise ValueError(f"boundary kernel: no branch for idx {k}")
        return self.branch[k]

    def __call__(self, branch, time, position, distance, q, qaux, p, normal):
        """`(n_state, n)` of the kernel per column; `boundary_data` is `q`."""
        branch = np.asarray(branch, dtype=int)
        named = dict(time=time, position=position, distance=distance, variables=q,
                     aux_variables=qaux, parameters=p, normal=normal, boundary_data=q)
        out = np.zeros((self.n, branch.size))
        for k in np.unique(branch):
            sel = np.flatnonzero(branch == k)
            vals = [np.asarray(named[key])[..., sel] if key in _COLUMN or key == "distance"
                    else named[key] for key in self.keys]
            out[:, sel] = self._piece(int(k))(*vals).reshape(self.n, sel.size)
        return out


class Slot:
    """A cell operator `k(Q, Qaux, p, *trailing, **named)` of a declared signature; the trailing
    groups bind in declared order or by name, `time` / `dt` default to 0 and `position` to the
    origin (as `JaxRuntime`).
    """

    def __init__(self, function, sm, module, squeeze):
        """Lambdify `function` (declared `args`); a `dt` that is also a parameter is dropped."""
        a = function.args
        dt_param = bool(sm.parameters.contains("dt"))
        self.declared = list(a.keys())
        self.keys = [k for k in self.declared if not (k == "dt" and dt_param)]
        self.low = Lowered(function.definition, [_symbols(a[k]) for k in self.keys], module)
        self.squeeze = squeeze and len(self.low.shape) == 2 and self.low.shape[1] == 1

    def __call__(self, Q, Qaux, p, *args, **named):
        """The operator on the columns of `Q`."""
        first = self.declared[0] if self.declared[0] == "reconstruction" else "variables"
        extra = dict(zip([k for k in self.declared
                          if k not in (first, "aux_variables", "parameters")], args))
        extra.update({k: v for k, v in named.items() if v is not None})
        base = {first: Q, "aux_variables": Qaux, "parameters": p}
        vals = []
        for k in self.keys:
            if k in base:
                vals.append(base[k])
            elif k in extra:
                vals.append(extra[k])
            elif k in ("time", "dt"):
                vals.append(0.0)
            elif k == "position":
                vals.append(np.zeros((3, np.shape(Q)[-1])))
            else:
                raise TypeError(f"kernel: argument group {k!r} not supplied")
        out = self.low(*vals)
        return out.reshape(out.shape[0], -1) if self.squeeze else out


class Face:
    """A numerics face function `k(qL, qR, aL, aR, p, normal)`; the `_speed` variant returns
    the two sides and the speed `(Wm, Wp, S)` (as `zoomy_jax`'s `_face_kernel`).
    """

    def __init__(self, function, module):
        """Lambdify `function` over its six groups."""
        a = function.args
        self.low = Lowered(function.definition, [_symbols(a[k]) for k in a.keys()], module)
        self.speed = function.name.endswith("_speed")

    def __call__(self, qL, qR, aL, aR, p, normal):
        """The function per face."""
        out = self.low(qL, qR, aL, aR, p, normal)
        return (out[0], out[1], out[2, 0]) if self.speed else out


class NumpyKernels:
    """Every kernel of `nsm.call` as a numpy callable, by slot name (`None` where empty);
    `aux_derivatives` is set by the runtime (the derivative-aux spec).
    """

    def __init__(self, nsm):
        """Lambdify the call table of `nsm` (a NumericalSystemModel)."""
        self.nsm, self.sm = nsm, nsm.sm
        self.n_state = int(self.sm.n_state)
        module = numpy_module()
        call = nsm.call
        for name in call.keys():
            kc = call[name]
            if kc is None or not hasattr(kc, "functions"):
                setattr(self, name, None)
                continue
            f = kc.functions[kc.name]
            if "idx" in f.args.keys():
                k = Indexed(f, self.n_state, module)
            elif "q_minus" in f.args.keys():
                k = Face(f, module)
            else:
                k = Slot(f, self.sm, module, name in _SQUEEZE)
            setattr(self, name, k)
        self.aux_derivatives = None

    @property
    def parameters(self):
        """The current parameter values, `(n_parameters,)`."""
        return np.asarray([float(v) for v in self.sm.parameter_values.values()], dtype=float)
