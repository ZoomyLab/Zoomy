"""The jax backend primitive object ``B`` for the template walker.

``JaxBackend(mode="forward")``  loops lower to ``lax.while_loop``
                                (forward-mode AD: jvp).
``JaxBackend(mode="reverse", checkpoints=K)``  loops lower to
                                ``equinox.internal.while_loop(kind=
                                "checkpointed")`` (reverse-mode AD: grad/vjp).

Lowerings:
  Select    -> ``lax.cond`` (Python ``if`` on a concrete predicate)
  Repeat    -> ``lax.fori_loop`` with a static trip count (a scan)
  Root      -> :func:`.linalg.newton_root` (``custom_root``, IFT AD)
  LinSolve  -> ``custom_linear_solve`` + the kind's solver (GMRES with the
               injectable inner product ``B.inner``)
  Guard     -> ``eqx.error_if`` attached to live values, preceded by an
               ordered debug print of the reported values (failing branch
               only); a raise on concrete values
  Callback  -> idempotent / returns: ordered ``io_callback`` inside a
               ``custom_jvp`` with a zero tangent (replayed by reverse-mode
               recomputation, hence only replay-safe kinds); stateful: a
               direct HOST call (forward backend only -- the reverse backend
               has NO lowering for it)
  reduce_min -> ``jnp.min`` (AD-safe); under SPMD (``axis_name``) an
               ``all_gather`` + ``min`` (``lax.pmin`` has no AD rule)
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import io_callback

from zoomy_core.solver.template.errors import GuardError, format_guard
from zoomy_jax.template import linalg

MODES = ("forward", "reverse")


def _traced(*xs) -> bool:
    for x in jax.tree.leaves(xs):
        if isinstance(x, jax.core.Tracer):
            return True
    return False


def _strong(tree):
    """Arrays with explicit (non-weak) dtypes -- loop carries must keep one
    type across iterations."""
    def f(x):
        a = jnp.asarray(x)
        return jnp.asarray(a, dtype=a.dtype)
    return jax.tree.map(f, tree)


def _like(out, ref):
    """Strip weak types of ``out`` where its dtype matches ``ref``."""
    def f(o, r):
        o = jnp.asarray(o)
        r = jnp.asarray(r)
        return jnp.asarray(o, dtype=r.dtype) if o.dtype == r.dtype else o
    return jax.tree.map(f, out, ref)


_SCALAR_DTYPES = {"real": lambda: jnp.asarray(0.0).dtype,
                  "int": lambda: jnp.asarray(0).dtype,
                  "bool": lambda: jnp.bool_}


def _count(x):
    return jax.lax.stop_gradient(x).astype(jnp.asarray(0).dtype)


def _as_outputs(value, n):
    if n == 0:
        return ()
    if n == 1:
        return (value,)
    return tuple(value)


class JaxBackend:
    """Backend primitive object for jax (see the module docstring)."""

    compiles = True
    xp = jnp
    #: set per instance in forward mode only
    callback_stateful = None

    def __init__(self, mode: str = "forward", checkpoints: int = 64, *,
                 axis_name=None):
        if mode not in MODES:
            raise ValueError(f"JaxBackend mode must be one of {MODES}")
        self.mode = mode
        self.checkpoints = int(checkpoints)
        self.axis_name = axis_name
        self.name = "jax_fwd" if mode == "forward" else "jax_rev"
        if mode == "forward":
            self.callback_stateful = self._callback_stateful
        if mode == "reverse":
            import equinox.internal as eqxi  # noqa: F401  (fail early)

    def __repr__(self):
        return f"JaxBackend(mode={self.mode!r}, checkpoints={self.checkpoints})"

    # -- compile ----------------------------------------------------------------
    @staticmethod
    def jit(f):
        return jax.jit(f)

    # -- control flow -----------------------------------------------------------
    def while_loop(self, cond, body, carry):
        init = _strong(carry)

        def body_s(c):
            return _like(body(c), c)

        if self.mode == "reverse":
            import equinox.internal as eqxi
            return eqxi.while_loop(cond, body_s, init, kind="checkpointed",
                                   checkpoints=self.checkpoints)
        return jax.lax.while_loop(cond, body_s, init)

    @staticmethod
    def repeat(n, body, carry):
        init = _strong(carry)
        return jax.lax.fori_loop(0, int(n), lambda i, c: _like(body(i, c), c),
                                 init)

    foreach = repeat

    @staticmethod
    def cond(pred, then, otherwise, operands):
        if not _traced(pred):
            return then(*operands) if bool(np.asarray(pred)) else otherwise(*operands)
        ops = _strong(tuple(operands))
        return jax.lax.cond(pred, lambda *o: _strong(then(*o)),
                            lambda *o: _strong(otherwise(*o)), *ops)

    # -- checks -------------------------------------------------------------------
    @staticmethod
    def guard(ok, msg, names, values, anchors):
        anchors = tuple(anchors)
        if not _traced(ok, values, anchors):
            if not bool(np.all(np.asarray(ok))):
                raise GuardError(format_guard(msg, names, values))
            return anchors
        import equinox as eqx
        bad = jnp.logical_not(jnp.all(ok))

        def report(*vals):
            import sys
            print("[Guard] " + format_guard(msg, names, vals), file=sys.stderr,
                  flush=True)

        if not anchors:
            def raise_(*vals):
                raise GuardError(format_guard(msg, names, vals))
            jax.lax.cond(bad, lambda: jax.debug.callback(
                raise_, *values, ordered=True), lambda: None)
            return ()

        def failing(a):
            jax.debug.callback(report, *values, ordered=True)
            return a

        anchors = jax.lax.cond(bad, failing, lambda a: a, anchors)
        return tuple(eqx.error_if(anchors, bad, msg))

    @staticmethod
    def warn(ok, msg, names, values):
        if not _traced(ok, values):
            if not bool(np.all(np.asarray(ok))):
                import warnings
                warnings.warn(format_guard(msg, names, values), RuntimeWarning)
            return None

        def report(*vals):
            import sys
            print("[Warning] " + format_guard(msg, names, vals), file=sys.stderr,
                  flush=True)
        jax.lax.cond(jnp.logical_not(jnp.all(ok)),
                     lambda: jax.debug.callback(report, *values, ordered=True),
                     lambda: None)
        return None

    # -- callbacks ----------------------------------------------------------------
    @staticmethod
    def callback_idempotent(fn, args):
        args = tuple(args)
        if not _traced(args):
            fn(*args)
            return None

        def host(*a):
            fn(*a)
            return None

        @jax.custom_jvp
        def cb(*a):
            io_callback(host, None, *a, ordered=True)
            return jnp.zeros(())

        @cb.defjvp
        def _cb_jvp(primals, tangents):
            return cb(*primals), jnp.zeros(())

        cb(*args)
        return None

    @staticmethod
    def callback_returns(fn, args, like):
        args = tuple(args)
        n = len(like)
        if not _traced(args):
            return _as_outputs(fn(*args), n)
        shapes = getattr(fn, "result_shape_dtypes", None)
        if shapes is not None:
            shapes = tuple(shapes(*args))
        else:
            shapes = []
            for x in like:
                if isinstance(x, str):          # an unbound output: its kind
                    if x not in _SCALAR_DTYPES:
                        raise ValueError(
                            f"callback {getattr(fn, '__name__', fn)!r} "
                            f"(kind='returns') inside a compiled unit: output "
                            f"of kind {x!r} needs a shape template (bind it "
                            "before the callback) or fn.result_shape_dtypes")
                    shapes.append(jax.ShapeDtypeStruct((), _SCALAR_DTYPES[x]()))
                else:
                    shapes.append(jax.ShapeDtypeStruct(jnp.shape(x),
                                                       jnp.result_type(x)))
            shapes = tuple(shapes)

        def host(*a):
            out = _as_outputs(fn(*a), n)
            return tuple(np.asarray(o, dtype=s.dtype).reshape(s.shape)
                         for o, s in zip(out, shapes))

        @jax.custom_jvp
        def cb(*a):
            return io_callback(host, shapes, *a, ordered=True)

        @cb.defjvp
        def _cb_jvp(primals, tangents):
            out = cb(*primals)
            return out, tuple(jnp.zeros(s.shape, s.dtype) for s in shapes)

        return tuple(cb(*args))

    @staticmethod
    def _callback_stateful(fn, args, n_out):
        args = tuple(args)
        if _traced(args):
            raise RuntimeError(
                f"stateful callback {getattr(fn, '__name__', fn)!r} reached a "
                "compiled unit: the host/compiled boundary puts every loop "
                "containing one on the host")
        return _as_outputs(fn(*args), n_out)

    # -- solves -------------------------------------------------------------------
    def root(self, residual, x0, *, tol, maxiter, jac="ad", jvp=None,
             linsolve=None):
        lin = dict(linsolve or {"kind": "cellwise"})
        solve = linalg.make_solver(inner=self.inner, **lin)
        if jac == "ad":
            def jv(x, v):
                return jax.jvp(residual, (x,), (v,))[1]
        elif jac == "fd":
            jv = linalg.fd_jvp(residual)
        elif jac == "analytic":
            jv = jvp
        else:
            raise ValueError(f"unknown jac strategy {jac!r}")
        x, (rnorm, iters) = linalg.newton_root(
            residual, jnp.asarray(x0), jv, solve, tol=tol, maxiter=int(maxiter),
            inner=self.inner)
        return x, rnorm <= tol, rnorm, _count(iters)

    def linsolve(self, matvec, rhs, *, kind, tol, restart=30, maxiter=50,
                 precond=None, nullspace=None, bandwidth=None, symmetric=False):
        rhs = jnp.asarray(rhs)
        common = dict(kind=kind, tol=tol, restart=int(restart),
                      maxiter=int(maxiter), nullspace=nullspace,
                      bandwidth=bandwidth, inner=self.inner)
        solve = linalg.make_solver(precond=precond, **common)
        if precond is None or symmetric:
            solve_t = solve
        else:
            def precond_t(r):
                return jax.linear_transpose(precond, rhs)(r)[0]
            solve_t = linalg.make_solver(precond=precond_t, **common)
        x, (rel, iters) = jax.lax.custom_linear_solve(
            matvec, rhs, solve, transpose_solve=solve_t, symmetric=symmetric,
            has_aux=True)
        return x, rel <= tol, rel, _count(iters)

    # -- array primitives -----------------------------------------------------------
    @staticmethod
    def scatter_add(target, index, values):
        return jnp.asarray(target).at[..., index].add(values)

    @staticmethod
    def scatter_min(target, index, values):
        return jnp.asarray(target).at[..., index].min(values)

    @staticmethod
    def scatter_max(target, index, values):
        return jnp.asarray(target).at[..., index].max(values)

    def _gather(self, x):
        return jax.lax.all_gather(x, self.axis_name)

    def reduce_min(self, x):
        m = jnp.min(x)
        return m if self.axis_name is None else jnp.min(self._gather(m))

    def reduce_max(self, x):
        m = jnp.max(x)
        return m if self.axis_name is None else jnp.max(self._gather(m))

    def reduce_sum(self, x):
        s = jnp.sum(x)
        return s if self.axis_name is None else jax.lax.psum(s, self.axis_name)

    def reduce_any(self, x):
        a = jnp.any(x)
        return a if self.axis_name is None else jnp.any(self._gather(a))

    def reduce_all(self, x):
        a = jnp.all(x)
        return a if self.axis_name is None else jnp.all(self._gather(a))

    def inner(self, a, b):
        s = jnp.vdot(jnp.ravel(a), jnp.ravel(b))
        return s if self.axis_name is None else jax.lax.psum(s, self.axis_name)

    @staticmethod
    def set_at(x, index, value):
        return jnp.asarray(x).at[index].set(value)

    @staticmethod
    def safe_div(a, b, fill):
        ok = b != 0
        return jnp.where(ok, a / jnp.where(ok, b, 1.0), fill)

    @staticmethod
    def as_int(x):
        return jnp.asarray(x).astype(jnp.asarray(0).dtype)


__all__ = ["JaxBackend", "MODES"]
