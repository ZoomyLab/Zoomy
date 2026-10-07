"""SCRATCH: four new statement kinds + a ProcedureBuilder subclass lowering them.

    Select(pred, then, otherwise, outputs)   RUNTIME branch
        jax  : lax.cond over `outputs`; branches close over the env
        numpy: python if
    Guard(ok, message, on)                    FATAL check
        jax  : env[on] = eqx.error_if(env[on], ~ok, message)
        numpy: raise FloatingPointError(message.format(**env))
    Callback(fn, args, anchor)                host side effect (io)
        jax  : ordered io_callback inside a custom_jvp with identity tangent on
               `anchor` (zero tangent contribution from the callback)
        numpy: direct call
    Root(target, residual, x0, args, tol, maxiter)
        jax  : zoomy_jax.fvm.implicit_ad.newton_root (custom_root, IFT AD)
        numpy: plain Newton, cell-block finite-difference Jacobian

The While additionally lowers (jax) to equinox.internal.while_loop(
kind="checkpointed") so the WHOLE march is reverse-differentiable.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import sympy as sp

from zoomy_core.solver.arg_slots import SOLVER_ARG_MAPPING
from zoomy_core.solver.ir import (Assign, Call, IfStatic, Statement, While,
                                  split_target)
from zoomy_core.transformation.procedure_python import ProcedureBuilder


# ── the IR nodes ────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Select(Statement):
    pred: Any                    # sympy Boolean / Symbol over env names
    then: tuple = ()
    otherwise: tuple = ()
    outputs: tuple = ()          # names both branches yield (the cond carry)

    def __post_init__(self):
        object.__setattr__(self, "then", tuple(self.then))
        object.__setattr__(self, "otherwise", tuple(self.otherwise))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if not self.outputs:
            raise ValueError("Select needs explicit outputs (the cond carry)")

    def children(self):
        return (self.then, self.otherwise)


@dataclasses.dataclass(frozen=True)
class Guard(Statement):
    ok: Any                      # sympy Boolean that must HOLD
    message: str
    on: str                      # value the check is attached to


@dataclasses.dataclass(frozen=True)
class Callback(Statement):
    fn: str                      # host body (externals table)
    args: tuple = ()
    anchor: str = "time"         # value threaded through (ordering + tangent)

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))


@dataclasses.dataclass(frozen=True)
class Root(Statement):
    target: str
    residual: str                # body: R(x, *args) -> same shape as x
    x0: str
    args: tuple = ()
    tol: float = 1e-12
    maxiter: int = 30

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))


def written_names_ext(stmts) -> tuple:
    found = []

    def add(n):
        if n not in found:
            found.append(n)

    for s in stmts:
        if isinstance(s, Assign):
            add(split_target(s.target)[0])
        elif isinstance(s, Call):
            for r in s.results:
                add(SOLVER_ARG_MAPPING.get(r, r))
        elif isinstance(s, Select):
            for n in s.outputs:
                add(SOLVER_ARG_MAPPING.get(n, n))
        elif isinstance(s, Guard):
            add(SOLVER_ARG_MAPPING.get(s.on, s.on))
        elif isinstance(s, Callback):
            add(SOLVER_ARG_MAPPING.get(s.anchor, s.anchor))
        elif isinstance(s, Root):
            add(s.target)
        elif isinstance(s, IfStatic):
            for n in written_names_ext(s.then + s.otherwise):
                add(n)
        else:
            for child in s.children():
                for n in written_names_ext(child):
                    add(n)
    return tuple(found)


def _m(n):
    return SOLVER_ARG_MAPPING.get(n, n)


# ── the builder ─────────────────────────────────────────────────────────────

class ProcedureBuilderExt(ProcedureBuilder):
    """ProcedureBuilder + Select/Guard/Callback/Root + checkpointed While."""

    def __init__(self, backend="numpy", externals=None, *,
                 while_kind="checkpointed", checkpoints=4, max_steps=None):
        super().__init__(backend, externals)
        self.while_kind = while_kind
        self.checkpoints = checkpoints
        self.max_steps = max_steps
        self._cb_cache = {}

    def _compile(self, stmt):
        if isinstance(stmt, Select):
            return self._compile_select(stmt)
        if isinstance(stmt, Guard):
            return self._compile_guard(stmt)
        if isinstance(stmt, Callback):
            return self._compile_callback(stmt)
        if isinstance(stmt, Root):
            return self._compile_root(stmt)
        return super()._compile(stmt)

    # -- Select -------------------------------------------------------------
    def _compile_select(self, stmt: Select):
        then_b = self._compile_block(stmt.then)
        else_b = self._compile_block(stmt.otherwise)
        pnames, pf = self._lambdify(stmt.pred)
        outs = tuple(_m(o) for o in stmt.outputs)

        if self.backend == "numpy":
            def run(env):
                pred = bool(pf(*[env[n] for n in pnames]))
                out = (then_b if pred else else_b)(dict(env))
                return {**env, **{k: out[k] for k in outs}}
            return run

        lax = self._jax.lax

        def run(env):
            pred = pf(*[env[n] for n in pnames])

            def branch(block):
                def f(_):
                    out = block(dict(env))
                    return {k: out[k] for k in outs}
                return f

            res = lax.cond(pred, branch(then_b), branch(else_b), None)
            return {**env, **res}
        return run

    # -- Guard --------------------------------------------------------------
    def _compile_guard(self, stmt: Guard):
        onames, okf = self._lambdify(stmt.ok)
        on = _m(stmt.on)

        if self.backend == "numpy":
            def run(env):
                if not bool(okf(*[env[n] for n in onames])):
                    raise FloatingPointError(
                        stmt.message.format(**{k: v for k, v in env.items()
                                               if np.ndim(v) == 0}))
                return env
            return run

        import equinox as eqx
        jnp = self.xp
        # jax: a STATIC message (eqx.error_if); the env values are not
        # formattable at trace time.
        static_msg = stmt.message.split("{")[0].strip() or stmt.message

        jax = self._jax
        fmt_names = tuple(sorted(set(
            n.split("}")[0].split(":")[0]
            for n in stmt.message.split("{")[1:])))

        def _diag(**vals):
            print("[Guard] " + stmt.message.format(
                **{k: float(v) for k, v in vals.items()}), flush=True)

        def run(env):
            ok = okf(*[env[n] for n in onames])
            bad = jnp.logical_not(ok)
            # dynamic diagnostic (values) via an ordered debug callback that
            # fires only on the failing branch; then the static error_if.
            vals = {n: env[n] for n in fmt_names if n in env}
            jax.lax.cond(bad, lambda: jax.debug.callback(_diag, ordered=True,
                                                          **vals),
                         lambda: None)
            env = dict(env)
            env[on] = eqx.error_if(env[on], bad, static_msg)
            return env
        return run

    # -- Callback -----------------------------------------------------------
    def _compile_callback(self, stmt: Callback):
        if stmt.fn not in self.externals:
            raise KeyError(f"no host body for callback {stmt.fn!r}")
        host = self.externals[stmt.fn]
        in_names = tuple(_m(a) for a in stmt.args)
        anchor = _m(stmt.anchor)

        if self.backend == "numpy":
            def run(env):
                host(*[env[n] for n in in_names])
                return env
            return run

        jax = self._jax
        from jax.experimental import io_callback

        def _host(*a):
            host(*a)
            return None

        @jax.custom_jvp
        def cb(anc, *args):
            io_callback(_host, None, *args, ordered=True)
            return anc

        @cb.defjvp
        def cb_jvp(primals, tangents):
            # zero tangent from the side effect; the anchor's tangent passes
            return cb(*primals), tangents[0]

        def run(env):
            env = dict(env)
            # integer args carry float0 tangents; custom_jvp handles them.
            env[anchor] = cb(env[anchor], *[env[n] for n in in_names])
            return env
        return run

    # -- Root ---------------------------------------------------------------
    def _compile_root(self, stmt: Root):
        if stmt.residual not in self.externals:
            raise KeyError(f"no body for residual {stmt.residual!r}")
        R = self.externals[stmt.residual]
        arg_names = tuple(_m(a) for a in stmt.args)
        x0n = _m(stmt.x0)
        tol, maxiter = float(stmt.tol), int(stmt.maxiter)

        if self.backend == "numpy":
            def run(env):
                a = [env[n] for n in arg_names]
                x = np.array(env[x0n], dtype=float)
                f = lambda y: np.asarray(R(y, *a), dtype=float)  # noqa: E731
                r = f(x)
                it = 0
                while it < maxiter and np.linalg.norm(r) > tol:
                    # cell-block FD Jacobian (n_vars x n_vars per cell)
                    nv, nc = x.shape
                    eps = 1e-7 * np.maximum(1.0, np.abs(x))
                    J = np.empty((nc, nv, nv))
                    for k in range(nv):
                        e = np.zeros_like(x)
                        e[k] = eps[k]
                        J[:, :, k] = ((f(x + e) - f(x - e)) / (2 * eps[k])).T
                    dx = np.linalg.solve(J, -r.T[:, :, None])[:, :, 0].T
                    x = x + dx
                    r = f(x)
                    it += 1
                env = dict(env)
                # keep the array namespace of x0 (bodies may be jax functions)
                if hasattr(env[x0n], "at"):
                    import jax.numpy as _jnp
                    x = _jnp.asarray(x)
                env[stmt.target] = x
                env[stmt.target + "_iters"] = it
                return env
            return run

        jax = self._jax
        from zoomy_jax.fvm.implicit_ad import cellwise_solve, newton_root

        def run(env):
            a = [env[n] for n in arg_names]
            res = lambda y: R(y, *a)  # noqa: E731
            jvp = lambda y, v: jax.jvp(res, (y,), (v,))[1]  # noqa: E731
            x = newton_root(res, env[x0n], jvp, cellwise_solve,
                            tol=tol, maxiter=maxiter)
            env = dict(env)
            env[stmt.target] = x
            return env
        return run

    # -- While: checkpointed on jax ------------------------------------------
    def _compile_while(self, stmt: While):
        if self.backend == "numpy" or self.while_kind == "lax":
            return super()._compile_while(stmt)
        import equinox.internal as eqxi
        body = self._compile_block(stmt.body)
        cond_names, cond_f = self._lambdify(stmt.condition)
        carry_names = tuple(_m(c) for c in stmt.carry) or \
            written_names_ext(stmt.body)
        kind, ckpt, ms = self.while_kind, self.checkpoints, self.max_steps

        def run(env):
            closure = {k: v for k, v in env.items() if k not in carry_names}

            def cond(carry):
                full = {**closure, **carry}
                return cond_f(*[full[n] for n in cond_names])

            def step(carry):
                out = body({**closure, **carry})
                return {k: out[k] for k in carry_names}

            carry = eqxi.while_loop(cond, step,
                                    {k: env[k] for k in carry_names},
                                    kind=kind, checkpoints=ckpt,
                                    max_steps=ms)
            return {**env, **carry}
        return run
