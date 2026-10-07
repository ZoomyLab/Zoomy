"""Statements and procedures of the solver template IR.

The template is a tree of :class:`Procedure` objects whose bodies are
statements.  The IR holds STRUCTURE only; lowering knowledge lives in the
walkers (:mod:`.walker_python`, :mod:`.walker_c`) and the backend primitive
objects.

Statement kinds
---------------
``Seq(body)``                           sequence (grouping only)
``Assign(target, expr, kind=None)``     scalar or whole-array elementwise
                                        assignment; ``target`` may be indexed
                                        (``"lam[f]"``) inside ``ForEachFace``
``Call(callee, args, out)``             call a registered block or a Procedure
``Hook(name, inputs, outputs, default)``named extension point; plugins fill it,
                                        otherwise ``default`` runs
``IfStatic(flag, then, otherwise)``     BUILD-time branch, resolved away
``While(cond, body, carry=None)``       runtime loop, dynamic trip count
``Repeat(n, body, index, carry)``       runtime loop, STATIC trip count
``ForEachFace(index, count, body)``     face traversal, STATIC count
``Select(pred, then, otherwise)``       runtime branch
``Root(residual, x0, args, out, opts)`` nonlinear solve R(x; args) = 0;
                                        ``residual`` is a Procedure (closure)
``LinSolve(matvec, rhs, args, out, opts)``  linear solve A(x; args) = rhs
``Guard(ok, msg, values)``              fatal check
``Callback(fn, inputs, outputs, kind)`` host side effect: ``idempotent`` |
                                        ``stateful`` | ``returns``

Scope rules (shared by the type checker and every walker)
----------------------------------------------------------
* Names are bound by procedure inputs, ``Assign`` targets, ``Call`` /
  ``Callback`` outputs, ``Root`` / ``LinSolve`` outputs and stats.
* ``Seq`` and ``Hook`` bodies are inline: their bindings stay visible.
* A loop (``While`` / ``Repeat`` / ``ForEachFace``) updates only names bound
  BEFORE it (its carry); names first bound inside the body are body-local.
* After a ``Select`` the names written in a branch that were bound before keep
  the branch value; a name first bound inside is visible afterwards only when
  BOTH branches bind it.
* Carries and Select outputs are inferred (written AND bound before AND live)
  unless given explicitly.
"""
from __future__ import annotations

import dataclasses
import os
import re
import sys
from typing import Any, Optional

import sympy as sp

from zoomy_core.solver.template.expr import as_expr
from zoomy_core.solver.template.kinds import Port, ports

_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
_IR_FILES = {os.path.join(_TEMPLATE_DIR, "ir.py"), "<string>"}


def _origin() -> str:
    """``file:line`` of the code that built a statement (first frame outside
    this module and the dataclass-generated ``__init__``)."""
    f = sys._getframe(1)
    while f is not None and (f.f_code.co_filename in _IR_FILES
                             or f.f_code.co_filename.startswith("<")):
        f = f.f_back
    if f is None:
        return ""
    fn = f.f_code.co_filename
    try:
        rel = os.path.relpath(fn, os.path.dirname(_TEMPLATE_DIR))
        fn = rel if not rel.startswith("..") else os.path.basename(fn)
    except ValueError:
        fn = os.path.basename(fn)
    return f"{fn}:{f.f_lineno}"


def _block(body) -> tuple:
    """Normalise a statement block: flatten nested lists/tuples, drop None."""
    out = []

    def add(x):
        if x is None:
            return
        if isinstance(x, (list, tuple)):
            for y in x:
                add(y)
            return
        if not isinstance(x, Statement):
            raise TypeError(f"not a template statement: {x!r}")
        out.append(x)

    add(body)
    return tuple(out)


def _names(seq) -> tuple:
    if seq is None:
        return ()
    if isinstance(seq, str):
        seq = seq.split()
    seq = tuple(seq)
    for n in seq:
        if not isinstance(n, str) or not n.isidentifier():
            raise ValueError(f"expected a name (identifier), got {n!r}")
    return seq


class IRError(ValueError):
    """A malformed template statement or procedure."""


# ── statements ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Statement:
    """Base class.  ``note`` is printed next to the statement in generated
    code and in ``show()``; ``origin`` records the builder line."""

    note: str = dataclasses.field(default="", kw_only=True)
    origin: str = dataclasses.field(default="", kw_only=True, compare=False,
                                    repr=False)

    def __post_init__(self):
        if not self.origin:
            object.__setattr__(self, "origin", _origin())

    # nested statement blocks: tuple of (label, block)
    def blocks(self) -> tuple:
        return ()

    def with_blocks(self, new: dict) -> "Statement":
        return dataclasses.replace(self, **new)

    # referenced procedures: tuple of (label, Procedure)
    def procedures(self) -> tuple:
        return ()

    def with_procedures(self, new: dict) -> "Statement":
        return dataclasses.replace(self, **new)

    @property
    def kind_name(self) -> str:
        return type(self).__name__


@dataclasses.dataclass(frozen=True)
class Seq(Statement):
    body: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "body", _block(self.body))
        super().__post_init__()

    def blocks(self):
        return (("body", self.body),)


_INDEXED = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[(.+)\]\s*$")


@dataclasses.dataclass(frozen=True)
class Assign(Statement):
    """``target = expr``.  ``kind`` declares the target kind explicitly
    (otherwise it is inferred from ``expr``).  A whole-array target with an
    array-valued ``expr`` is an ELEMENTWISE assignment; an indexed target
    ``"x[i]"`` writes one element (``ForEachFace`` bodies)."""

    target: str
    expr: Any
    kind: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.target, str):
            raise IRError(f"Assign.target must be a str, got {self.target!r}")
        base, idx = self.split_target()
        if not base.isidentifier():
            raise IRError(f"Assign.target {self.target!r} is not a name")
        object.__setattr__(self, "expr", as_expr(self.expr))
        if idx is not None:
            object.__setattr__(self, "_index", as_expr(sp.sympify(idx)))
        super().__post_init__()

    def split_target(self):
        m = _INDEXED.match(self.target)
        return (m.group(1), m.group(2)) if m else (self.target, None)

    @property
    def base(self) -> str:
        return self.split_target()[0]

    @property
    def index(self):
        return getattr(self, "_index", None)


@dataclasses.dataclass(frozen=True)
class Call(Statement):
    """Call ``callee`` (a registered block name or a :class:`Procedure`) with
    argument NAMES; bind its outputs to ``out``."""

    callee: Any
    args: tuple = ()
    out: tuple = ()

    def __post_init__(self):
        if not isinstance(self.callee, (str, Procedure)):
            raise IRError(f"Call.callee must be a block name or a Procedure, "
                          f"got {self.callee!r}")
        object.__setattr__(self, "args", _names(self.args))
        object.__setattr__(self, "out", _names(self.out))
        super().__post_init__()

    @property
    def callee_name(self) -> str:
        return self.callee if isinstance(self.callee, str) else self.callee.name

    def procedures(self):
        return (("callee", self.callee),) if isinstance(self.callee, Procedure) \
            else ()


@dataclasses.dataclass(frozen=True)
class Hook(Statement):
    """Named extension point.  ``inputs`` / ``outputs`` declare what a filling
    body may read / must leave bound (it may write ONLY ``outputs`` among the
    names bound before it).  ``body`` is the plugin fill (``None`` = unfilled,
    the ``default`` body runs)."""

    name: str
    inputs: tuple = ()
    outputs: tuple = ()
    default: tuple = ()
    body: Optional[tuple] = None
    filled_by: str = ""

    def __post_init__(self):
        object.__setattr__(self, "inputs", _names(self.inputs))
        object.__setattr__(self, "outputs", _names(self.outputs))
        object.__setattr__(self, "default", _block(self.default))
        if self.body is not None:
            object.__setattr__(self, "body", _block(self.body))
        super().__post_init__()

    @property
    def effective(self) -> tuple:
        return self.default if self.body is None else self.body

    def blocks(self):
        if self.body is None:
            return (("default", self.default),)
        return (("body", self.body),)


@dataclasses.dataclass(frozen=True)
class IfStatic(Statement):
    """BUILD-time branch on a flag; :func:`resolve_static` splices the chosen
    branch.  No walker lowers an unresolved ``IfStatic``."""

    flag: str
    then: tuple = ()
    otherwise: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "then", _block(self.then))
        object.__setattr__(self, "otherwise", _block(self.otherwise))
        super().__post_init__()

    def blocks(self):
        return (("then", self.then), ("otherwise", self.otherwise))


@dataclasses.dataclass(frozen=True)
class While(Statement):
    cond: Any
    body: tuple = ()
    carry: Optional[tuple] = None

    def __post_init__(self):
        object.__setattr__(self, "cond", as_expr(self.cond))
        object.__setattr__(self, "body", _block(self.body))
        if self.carry is not None:
            object.__setattr__(self, "carry", _names(self.carry))
        super().__post_init__()

    def blocks(self):
        return (("body", self.body),)


@dataclasses.dataclass(frozen=True)
class Repeat(Statement):
    """Loop with a STATIC trip count ``n`` (an int or the name of an int
    constant).  ``index`` (optional) is bound to 0..n-1 inside the body."""

    n: Any
    body: tuple = ()
    index: Optional[str] = None
    carry: Optional[tuple] = None

    def __post_init__(self):
        if not isinstance(self.n, (int, str)) or isinstance(self.n, bool):
            raise IRError(f"Repeat.n must be a static int or constant name, "
                          f"got {self.n!r}")
        object.__setattr__(self, "body", _block(self.body))
        if self.carry is not None:
            object.__setattr__(self, "carry", _names(self.carry))
        super().__post_init__()

    def blocks(self):
        return (("body", self.body),)


@dataclasses.dataclass(frozen=True)
class ForEachFace(Statement):
    """Face traversal ``for index in range(count)``; ``count`` is STATIC (an
    int or the name of an int constant)."""

    index: str
    count: Any
    body: tuple = ()
    carry: Optional[tuple] = None

    def __post_init__(self):
        if not isinstance(self.count, (int, str)) or isinstance(self.count, bool):
            raise IRError(f"ForEachFace.count must be static, got {self.count!r}")
        object.__setattr__(self, "body", _block(self.body))
        if self.carry is not None:
            object.__setattr__(self, "carry", _names(self.carry))
        super().__post_init__()

    def blocks(self):
        return (("body", self.body),)


@dataclasses.dataclass(frozen=True)
class Select(Statement):
    pred: Any
    then: tuple = ()
    otherwise: tuple = ()
    outputs: Optional[tuple] = None

    def __post_init__(self):
        object.__setattr__(self, "pred", as_expr(self.pred))
        object.__setattr__(self, "then", _block(self.then))
        object.__setattr__(self, "otherwise", _block(self.otherwise))
        if self.outputs is not None:
            object.__setattr__(self, "outputs", _names(self.outputs))
        super().__post_init__()

    def blocks(self):
        return (("then", self.then), ("otherwise", self.otherwise))


@dataclasses.dataclass(frozen=True)
class LinSolveOpts:
    """Options of a linear solve.

    ``kind``       ``krylov`` (restarted GMRES with the backend inner product)
                   | ``dense`` | ``cellwise`` (block-diagonal over the trailing
                   cell axis) | ``banded`` (block-banded over the cell axis,
                   half-bandwidth ``bandwidth`` cells)
    ``tol``        relative residual tolerance (number or constant name)
    ``restart``    Krylov restart length; ``maxiter`` restart cycles
    ``precond``    Procedure ``(r, *args) -> z`` (right preconditioner, linear)
    ``nullspace``  ``None`` | ``"constant"`` (whole vector) |
                   ``"constant_per_row"`` (per leading row, over the cell axis):
                   rhs and solution are projected to zero mean
    ``on_fail``    ``guard`` (D9, default) | ``warn`` | ``ignore``
    """

    kind: str = "krylov"
    tol: Any = 1e-10
    restart: Any = 30
    maxiter: Any = 50
    precond: Optional["Procedure"] = None
    nullspace: Optional[str] = None
    bandwidth: Optional[int] = None
    symmetric: bool = False
    on_fail: str = "guard"

    def __post_init__(self):
        if self.kind not in ("krylov", "dense", "cellwise", "banded"):
            raise IRError(f"unknown LinSolve kind {self.kind!r}")
        if self.kind == "banded" and (self.bandwidth is None
                                      or int(self.bandwidth) < 0):
            raise IRError("banded LinSolve needs bandwidth >= 0 (cells)")
        if self.nullspace not in (None, "constant", "constant_per_row"):
            raise IRError(f"unknown nullspace option {self.nullspace!r}")
        if self.on_fail not in ("guard", "warn", "ignore"):
            raise IRError(f"unknown on_fail {self.on_fail!r}")


@dataclasses.dataclass(frozen=True)
class RootOpts:
    """Options of a nonlinear (Newton) solve.

    ``jac``         ``ad`` | ``fd`` | ``analytic`` -- how the Newton update
                    applies J(x)v; ``analytic`` uses ``jac_proc`` =
                    Procedure ``(x, v, *args) -> Jv``.  Derivatives of the root
                    (jax) always use the exact AD linearisation (implicit
                    function theorem).
    ``linsolve``    :class:`LinSolveOpts` for the Newton update systems
    ``aux_policy``  ``frozen`` | ``local`` | ``full``: how the residual treats
                    aux fields; recorded here, realised by the stage that
                    builds the residual Procedure
    ``on_fail``     ``guard`` (D9, default) | ``warn`` | ``ignore``
    """

    tol: Any = 1e-12
    maxiter: Any = 50
    jac: str = "ad"
    jac_proc: Optional["Procedure"] = None
    linsolve: LinSolveOpts = dataclasses.field(
        default_factory=lambda: LinSolveOpts(kind="cellwise"))
    aux_policy: str = "frozen"
    on_fail: str = "guard"

    def __post_init__(self):
        if self.jac not in ("ad", "fd", "analytic"):
            raise IRError(f"unknown Root jac strategy {self.jac!r}")
        if (self.jac == "analytic") != (self.jac_proc is not None):
            raise IRError("jac='analytic' needs jac_proc (and only it does)")
        if self.aux_policy not in ("frozen", "local", "full"):
            raise IRError(f"unknown aux policy {self.aux_policy!r}")
        if self.on_fail not in ("guard", "warn", "ignore"):
            raise IRError(f"unknown on_fail {self.on_fail!r}")


@dataclasses.dataclass(frozen=True)
class Root(Statement):
    """``out`` = root x of ``residual(x, *args)`` starting from ``x0``.

    ``stats`` optionally names ``(ok, rnorm, iters)`` (bool, real, int)."""

    residual: "Procedure"
    x0: str
    args: tuple = ()
    out: Optional[str] = None
    opts: RootOpts = dataclasses.field(default_factory=RootOpts)
    stats: tuple = ()

    def __post_init__(self):
        if not isinstance(self.residual, Procedure):
            raise IRError("Root.residual must be a Procedure")
        object.__setattr__(self, "args", _names(self.args))
        object.__setattr__(self, "stats", _names(self.stats))
        if self.out is None:
            object.__setattr__(self, "out", self.x0)
        if len(self.stats) not in (0, 3):
            raise IRError("Root.stats names (ok, rnorm, iters) or nothing")
        super().__post_init__()

    def procedures(self):
        out = [("residual", self.residual)]
        if self.opts.jac_proc is not None:
            out.append(("jac_proc", self.opts.jac_proc))
        if self.opts.linsolve.precond is not None:
            out.append(("precond", self.opts.linsolve.precond))
        return tuple(out)

    def with_procedures(self, new):
        opts = self.opts
        if "jac_proc" in new:
            opts = dataclasses.replace(opts, jac_proc=new["jac_proc"])
        if "precond" in new:
            opts = dataclasses.replace(opts, linsolve=dataclasses.replace(
                opts.linsolve, precond=new["precond"]))
        kw = {"opts": opts}
        if "residual" in new:
            kw["residual"] = new["residual"]
        return dataclasses.replace(self, **kw)


@dataclasses.dataclass(frozen=True)
class LinSolve(Statement):
    """``out`` = x with ``matvec(x, *args) = rhs``.

    ``stats`` optionally names ``(ok, rel_resid, iters)``."""

    matvec: "Procedure"
    rhs: str
    args: tuple = ()
    out: str = "x"
    opts: LinSolveOpts = dataclasses.field(default_factory=LinSolveOpts)
    stats: tuple = ()

    def __post_init__(self):
        if not isinstance(self.matvec, Procedure):
            raise IRError("LinSolve.matvec must be a Procedure")
        object.__setattr__(self, "args", _names(self.args))
        object.__setattr__(self, "stats", _names(self.stats))
        if len(self.stats) not in (0, 3):
            raise IRError("LinSolve.stats names (ok, rel_resid, iters) or "
                          "nothing")
        super().__post_init__()

    def procedures(self):
        out = [("matvec", self.matvec)]
        if self.opts.precond is not None:
            out.append(("precond", self.opts.precond))
        return tuple(out)

    def with_procedures(self, new):
        kw = {}
        if "matvec" in new:
            kw["matvec"] = new["matvec"]
        if "precond" in new:
            kw["opts"] = dataclasses.replace(self.opts, precond=new["precond"])
        return dataclasses.replace(self, **kw)


@dataclasses.dataclass(frozen=True)
class Guard(Statement):
    """Fatal check: ``ok`` must hold.  ``values`` are reported on failure.
    ``on`` (optional) names the live values the check is attached to in
    traced lowerings; by default the walker picks live names itself."""

    ok: Any
    msg: str
    values: tuple = ()
    on: Optional[tuple] = None

    def __post_init__(self):
        object.__setattr__(self, "ok", as_expr(self.ok))
        object.__setattr__(self, "values", _names(self.values))
        if self.on is not None:
            object.__setattr__(self, "on", _names(self.on))
        super().__post_init__()


CALLBACK_KINDS = ("idempotent", "stateful", "returns")


@dataclasses.dataclass(frozen=True)
class Callback(Statement):
    """Host side effect ``fn`` (a registered callback block).

    ``idempotent``  replay-safe (e.g. a write keyed by i_snapshot); allowed
                    inside compiled loops (jax: ordered io_callback with a
                    zero tangent), replayed by reverse-mode recomputation
    ``returns``     returns data that is a deterministic function of its inputs
                    (zero tangent); allowed inside compiled loops
    ``stateful``    advances host state (preCICE advance, a counter): never
                    replayed, so it forces its enclosing loop onto the host;
                    the reverse jax backend has no lowering for it
    """

    fn: str
    inputs: tuple = ()
    outputs: tuple = ()
    kind: str = "idempotent"

    def __post_init__(self):
        object.__setattr__(self, "inputs", _names(self.inputs))
        object.__setattr__(self, "outputs", _names(self.outputs))
        if self.kind not in CALLBACK_KINDS:
            raise IRError(f"Callback kind must be one of {CALLBACK_KINDS}, "
                          f"got {self.kind!r}")
        if self.kind == "idempotent" and self.outputs:
            raise IRError("an idempotent Callback has no outputs (use "
                          "kind='returns')")
        super().__post_init__()


# ── procedures ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, eq=False)
class Procedure:
    """``name(inputs) -> outputs`` with a statement ``body``.

    ``outputs`` are the procedure's RESULTS: names bound at the end of the
    body, returned to the caller (a bare value for one output, a tuple for
    several, nothing for none -- the same convention as block bodies).
    Procedures may be passed to ``Root`` / ``LinSolve`` (closures over the
    caller's scope via their extra ``args``).

    Equality is identity-free structural equality on (name, ports, body)."""

    name: str
    inputs: tuple = ()
    outputs: tuple = ()
    body: tuple = ()
    doc: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise IRError(f"procedure name must be an identifier, got {self.name!r}")
        object.__setattr__(self, "inputs", ports(self.inputs))
        object.__setattr__(self, "outputs", ports(self.outputs))
        object.__setattr__(self, "body", _block(self.body))

    def __eq__(self, other):
        return (isinstance(other, Procedure) and self.name == other.name
                and self.inputs == other.inputs and self.outputs == other.outputs
                and self.body == other.body)

    def __hash__(self):
        return hash((self.name, self.inputs, self.outputs))

    @property
    def input_names(self) -> tuple:
        return tuple(p.name for p in self.inputs)

    @property
    def output_names(self) -> tuple:
        return tuple(p.name for p in self.outputs)

    @property
    def results(self) -> tuple:
        """Alias of ``outputs`` (the values returned to the caller)."""
        return self.outputs

    def resolve(self, flags) -> "Procedure":
        return resolve_static(self, flags)

    def show(self) -> str:
        from zoomy_core.solver.template.show import show_procedure
        return show_procedure(self)


# ── traversal / rewriting ───────────────────────────────────────────────────


def transform(proc: Procedure, fn, memo=None) -> Procedure:
    """Top-down rewrite of ``proc`` and every procedure it references.

    ``fn(stmt)`` returns ``None`` (keep ``stmt`` and descend into it) or a
    replacement statement / sequence, which is spliced in place and itself
    transformed again.  Procedures are memoised by identity, so a procedure
    referenced several times maps to ONE rewritten object."""
    memo = {} if memo is None else memo
    key = id(proc)
    if key in memo:
        return memo[key]
    new = dataclasses.replace(proc, body=_transform_block(proc.body, fn, memo))
    memo[key] = new
    return new


def _transform_block(block, fn, memo) -> tuple:
    out = []
    for s in block:
        r = fn(s)
        if r is None:
            out.append(_descend(s, fn, memo))
        else:
            rs = _block(r)
            out.extend(_transform_block(rs, fn, memo))
    return tuple(out)


def _descend(s, fn, memo):
    blocks = s.blocks()
    if blocks:
        s = s.with_blocks({label: _transform_block(b, fn, memo)
                           for label, b in blocks})
    procs = s.procedures()
    if procs:
        s = s.with_procedures({label: transform(p, fn, memo)
                               for label, p in procs})
    return s


def walk(block, fn, path=()):
    """Pre-order visit ``fn(stmt, path)`` over a block (not into referenced
    procedures)."""
    for i, s in enumerate(block):
        p = path + (i,)
        fn(s, p)
        for label, b in s.blocks():
            walk(b, fn, p + (label,))


def iter_statements(block):
    """All statements of a block, recursively (not into procedures)."""
    out = []
    walk(block, lambda s, p: out.append(s))
    return out


def resolve_static(proc: Procedure, flags) -> Procedure:
    """Splice the chosen branch of every :class:`IfStatic`, recursively into
    every nested block and every referenced procedure.

    A flag referenced by a REACHED IfStatic but absent from ``flags`` raises:
    a build-time branch has no default."""
    flags = dict(flags or {})

    def fn(s):
        if isinstance(s, IfStatic):
            if s.flag not in flags:
                raise KeyError(
                    f"IfStatic flag {s.flag!r} is not in the build flags "
                    f"{sorted(flags)} (a build-time branch has no default)")
            return s.then if flags[s.flag] else s.otherwise
        return None

    return transform(proc, fn)


def static_flags(proc: Procedure) -> tuple:
    """Every IfStatic flag referenced anywhere (sorted)."""
    found = set()

    def visit(p, seen):
        if id(p) in seen:
            return
        seen.add(id(p))

        def f(s, path):
            if isinstance(s, IfStatic):
                found.add(s.flag)
            for _, q in s.procedures():
                visit(q, seen)
        walk(p.body, f)

    visit(proc, set())
    return tuple(sorted(found))


def is_resolved(proc: Procedure) -> bool:
    return not static_flags(proc)


def collect_procedures(root: Procedure) -> dict:
    """``name -> Procedure`` for ``root`` and every procedure it references,
    callees first (a valid definition order).  Two DIFFERENT procedures with
    one name raise."""
    order: list = []
    by_name: dict = {}

    def visit(p, stack):
        if p.name in by_name:
            if by_name[p.name] is not p and by_name[p.name] != p:
                raise IRError(f"two different procedures are named {p.name!r}")
            return
        if p.name in stack:
            raise IRError(f"recursive procedure reference through {p.name!r}")
        stack = stack + (p.name,)

        def f(s, path):
            for _, q in s.procedures():
                visit(q, stack)
        walk(p.body, f)
        by_name[p.name] = p
        order.append(p.name)

    visit(root, ())
    return {n: by_name[n] for n in order}


def fill_hooks(proc: Procedure, fills: dict) -> Procedure:
    """Fill hooks by name: ``fills[name] = [(filled_by, statements), ...]``;
    several fills of one hook run in order.  Already-filled hooks are kept."""

    def fn(s):
        if isinstance(s, Hook) and s.body is None and s.name in fills:
            parts = fills[s.name]
            body = []
            for _, stmts in parts:
                body.extend(_block(stmts))
            by = ", ".join(name for name, _ in parts)
            return dataclasses.replace(s, body=tuple(body), filled_by=by)
        return None

    return transform(proc, fn)


def hook_names(proc: Procedure) -> tuple:
    found = []

    def visit(p, seen):
        if id(p) in seen:
            return
        seen.add(id(p))

        def f(s, path):
            if isinstance(s, Hook) and s.name not in found:
                found.append(s.name)
            for _, q in s.procedures():
                visit(q, seen)
        walk(p.body, f)

    visit(proc, set())
    return tuple(found)


__all__ = [
    "Statement", "Seq", "Assign", "Call", "Hook", "IfStatic", "While",
    "Repeat", "ForEachFace", "Select", "Root", "LinSolve", "Guard",
    "Callback", "RootOpts", "LinSolveOpts", "Procedure", "Port", "IRError",
    "CALLBACK_KINDS", "transform", "walk", "iter_statements",
    "resolve_static", "static_flags", "is_resolved", "collect_procedures",
    "fill_hooks", "hook_names",
]
