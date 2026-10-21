"""Walker (b): the python `ProcedureBuilder`.

numpy and jax have no text printer — their emission target is a CALLABLE
(`to_numpy.py` lambdifies straight into `runtime_functions`).  So the
second walker over the Procedure/Statement IR builds callables instead of
source:

    fn = ProcedureBuilder("numpy").build(proc)
    env = fn(**{... argument slot -> array/scalar ...})

`fn` returns the environment dict after execution, so a block's written
outputs (the stored face arrays, `Q_cand`, `troubled`, `dt`) are read
off by name — the same "written arguments" contract the C walker emits.

Backend differences, and only these:

* **numpy** — `ForEachFace` / `While` are plain python loops, indexed
  assignment mutates in place.
* **jax** — `While` lowers to `jax.lax.while_loop` and `ForEachFace`
  to `jax.lax.fori_loop`, both over the explicit carry (functional
  update via `.at[idx].set`), so a built procedure is `jit`-safe.

Expressions are lambdified ONCE at build time (that is the "builder" in the
name); statements are interpreted, which keeps the walker small and keeps all
lowering knowledge in sympy's printers rather than duplicated here.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.solver.arg_slots import SOLVER_ARG_MAPPING
from zoomy_core.solver.ir import (
    Assign,
    Call,
    ForEachFace,
    IfStatic,
    Procedure,
    While,
    expression_names,
    require_resolved,
    split_target,
    written_names,
)


class ProcedureBuilder:
    """Build python callables from solver Procedures.

    `backend` is `"numpy"` or `"jax"`.  `externals` maps
    `ExternalProcedure` names to the
    backend's body — a `Call` to a name absent from the table RAISES at
    BUILD time (missing body = red test, never a NameError mid-march).
    """

    def __init__(self, backend: str = "numpy", externals=None):
        if backend not in ("numpy", "jax"):
            raise ValueError(
                f"unknown ProcedureBuilder backend {backend!r} (numpy | jax)")
        self.backend = backend
        self.externals = dict(externals or {})
        if backend == "jax":
            import jax
            import jax.numpy as jnp
            self._jax, self.xp, self._modules = jax, jnp, "jax"
        else:
            import numpy as np
            self._jax, self.xp, self._modules = None, np, "numpy"

    # ── entry point ────────────────────────────────────────────────────────

    def build(self, proc: Procedure):
        """Compile `proc` to `fn(**args) -> env` (dict)."""
        require_resolved(proc)
        arg_names = tuple(SOLVER_ARG_MAPPING[a] for a in proc.args)
        body = self._compile_block(proc.stmts)

        def fn(**kwargs):
            missing = [a for a in arg_names if a not in kwargs]
            if missing:
                raise TypeError(
                    f"{proc.name}() missing argument(s) {missing} — solver "
                    "arguments are bound by NAME; there is no positional "
                    "default")
            env = dict(kwargs)
            return body(env)

        fn.__name__ = proc.name
        fn.__doc__ = proc.doc or f"Built solver procedure {proc.name}."
        fn.arg_names = arg_names
        return fn

    # ── statement compilation ──────────────────────────────────────────────

    def _compile_block(self, stmts):
        compiled = [self._compile(s) for s in stmts]

        def run(env):
            for step in compiled:
                env = step(env)
            return env

        return run

    def _compile(self, stmt):
        if isinstance(stmt, Assign):
            return self._compile_assign(stmt)
        if isinstance(stmt, Call):
            return self._compile_call(stmt)
        if isinstance(stmt, ForEachFace):
            return self._compile_foreach(stmt)
        if isinstance(stmt, While):
            return self._compile_while(stmt)
        if isinstance(stmt, IfStatic):
            raise ValueError(
                "IfStatic reached the python builder — resolve() it at build "
                "time")
        raise TypeError(f"unknown statement kind {type(stmt).__name__}")

    # -- Assign ------------------------------------------------------------

    def _lambdify(self, expr):
        """Lambdify one expression against its free NAMES; returns
        `(names, callable)` so the interpreter can pull them from env."""
        expr = sp.sympify(expr)
        names = expression_names(expr)
        syms = [sp.Symbol(n) for n in names]
        return names, sp.lambdify(syms, expr, self._modules)

    def _compile_assign(self, stmt: Assign):
        base, index = split_target(stmt.target)
        if isinstance(stmt.expr, sp.Piecewise):
            # Piecewise is a VALUE selection, not control flow: lower it the
            # same way the C printer does semantically, but branch-free so the
            # jax path stays traceable.
            expr = sp.sympify(stmt.expr)
        elif stmt.shape:
            expr = sp.Matrix(list(sp.flatten(
                list(stmt.expr) if isinstance(stmt.expr, sp.MatrixBase)
                else stmt.expr)))
        else:
            expr = sp.sympify(stmt.expr)
        names, f = self._lambdify(expr)
        shape = stmt.shape
        idx_names, idx_f = ((), None)
        if index is not None:
            idx_names, idx_f = self._lambdify(sp.sympify(index))

        def run(env):
            missing = [n for n in names if n not in env]
            if missing:
                raise NameError(
                    f"assignment to {stmt.target!r} reads unbound name(s) "
                    f"{missing} — every state row is indexed by NAME, an "
                    "unmapped row raises")
            value = f(*[env[n] for n in names])
            if shape:
                value = self.xp.asarray(value).reshape(shape)
            if index is None:
                env = dict(env)
                env[base] = value
                return env
            i = idx_f(*[env[n] for n in idx_names])
            env = dict(env)
            if self.backend == "jax":
                env[base] = env[base].at[i].set(value)
            else:
                target = env[base]
                target[i] = value
                env[base] = target
            return env

        return run

    # -- Call --------------------------------------------------------------

    def _compile_call(self, stmt: Call):
        if stmt.procedure not in self.externals:
            raise KeyError(
                f"no {self.backend} body registered for procedure "
                f"{stmt.procedure!r} — supply it in ProcedureBuilder("
                "externals=...) (REQUIRED_PROCEDURES contract)")
        body = self.externals[stmt.procedure]
        in_names = tuple(SOLVER_ARG_MAPPING.get(a, a) for a in stmt.args)
        out_names = tuple(SOLVER_ARG_MAPPING.get(r, r) for r in stmt.results)

        def run(env):
            out = body(*[env[n] for n in in_names])
            if not out_names:
                return env
            values = out if isinstance(out, tuple) else (out,)
            if len(values) != len(out_names):
                raise ValueError(
                    f"{stmt.procedure!r} returned {len(values)} value(s) but "
                    f"declares results {out_names}")
            env = dict(env)
            env.update(dict(zip(out_names, values)))
            return env

        return run

    # -- loops -------------------------------------------------------------

    def _compile_foreach(self, stmt: ForEachFace):
        body = self._compile_block(stmt.body)
        count_name = SOLVER_ARG_MAPPING[stmt.count]
        carry_names = written_names(stmt.body)
        index = stmt.index

        if self.backend == "numpy":
            def run(env):
                for i in range(int(env[count_name])):
                    env = body({**env, index: i})
                    env.pop(index, None)
                return env
            return run

        lax = self._jax.lax

        def run(env):
            closure = {k: v for k, v in env.items() if k not in carry_names}

            def step(i, carry):
                out = body({**closure, **carry, index: i})
                return {k: out[k] for k in carry_names}

            carry = {k: env[k] for k in carry_names}
            carry = lax.fori_loop(0, env[count_name], step, carry)
            return {**env, **carry}

        return run

    def _compile_while(self, stmt: While):
        body = self._compile_block(stmt.body)
        cond_names, cond_f = self._lambdify(stmt.condition)
        carry_names = tuple(SOLVER_ARG_MAPPING.get(c, c) for c in stmt.carry) \
            or written_names(stmt.body)

        if self.backend == "numpy":
            def run(env):
                while bool(cond_f(*[env[n] for n in cond_names])):
                    env = body(env)
                return env
            return run

        lax = self._jax.lax

        def run(env):
            closure = {k: v for k, v in env.items() if k not in carry_names}

            def cond(carry):
                full = {**closure, **carry}
                return cond_f(*[full[n] for n in cond_names])

            def step(carry):
                out = body({**closure, **carry})
                return {k: out[k] for k in carry_names}

            carry = lax.while_loop(cond, step,
                                   {k: env[k] for k in carry_names})
            return {**env, **carry}

        return run
