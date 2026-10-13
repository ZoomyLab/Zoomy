"""Procedure / Statement IR — the ONE core extension the solver-unification
program needs (design 2026-07-20-solver-flowcharts-phase1.md, v5 amendment 13;
validation appendix D13).

Today the emitters hand-build string lists (`generic_c.py`,
`to_openfoam.py`); there is no object a *statement-position, side-effecting*
solver block can be expressed in.  This module adds exactly that and nothing
more: a `Procedure` (name, ordered argument keys, statements) and five
statement kinds —

`Assign`       bind a name to a symbolic expression (scalar or array-shaped)
`Call`         invoke another Procedure or an `ExternalProcedure`
`IfStatic`     a **BUILD-time** branch on an NSM flag — resolved away before
                 either walker runs, so it can never become a runtime branch
`ForEachFace`  the face traversal (the one loop shape every backend has)
`While`        used exactly once, for the march

The IR carries no lowering knowledge.  Two walkers consume it:
`zoomy_core.transformation.procedure_c` (C-family text, reusing
`convert_expression_body` / CSE / Piecewise / `wrap_function_signature` /
`_sm_arg_decl`) and
`zoomy_core.transformation.procedure_python` (numpy/jax callables; the
`While` lowers to `lax.while_loop` on jax).

Guards (design §"Guards"): every Procedure / ExternalProcedure name MUST start
with `solver_`, and registration asserts the name collides with neither the
`c_functions` print-map nor `_NON_RESOLVABLE` nor the model operator
kernels — a solver block and a model kernel can never shadow each other.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping, Sequence

import sympy as sp

from zoomy_core.solver.arg_slots import SOLVER_ARG_KINDS, SOLVER_ARG_MAPPING

# ── name guard ──────────────────────────────────────────────────────────────

#: Mandatory prefix for every solver-level procedure name.  Model kernels
#: (``flux`` / ``source`` / ``update_aux_variables`` / …) never carry it, so
#: the two namespaces are disjoint by construction.
SOLVER_PREFIX = "solver_"


class ProcedureNameError(ValueError):
    """A Procedure / ExternalProcedure name violates the `solver_` guard or
    collides with an existing kernel / print-map name."""


def _reserved_names() -> frozenset:
    """Names a solver procedure must not take: the C-family print-map
    (`c_functions`), the never-resolvable external kernels
    (`_NON_RESOLVABLE`) and the declared model operator slots.  Imported
    lazily so the IR module stays importable on its own."""
    from zoomy_core.transformation.generic_c import GenericCppBase
    from zoomy_core.systemmodel.system_model import OPERATOR_ARG_SLOTS

    return frozenset(
        set(GenericCppBase.c_functions)
        | set(GenericCppBase._NON_RESOLVABLE)
        | set(OPERATOR_ARG_SLOTS)
    )


def check_procedure_name(name: str) -> str:
    """Validate one solver-procedure name; return it unchanged.

    RAISES `ProcedureNameError` — never warns, never renames.
    """
    if not isinstance(name, str) or not name:
        raise ProcedureNameError(f"procedure name must be a non-empty str, got {name!r}")
    if name == SOLVER_PREFIX:
        raise ProcedureNameError(
            f"{SOLVER_PREFIX!r} is the prefix, not a procedure name")
    if not name.startswith(SOLVER_PREFIX):
        raise ProcedureNameError(
            f"solver procedure {name!r} must start with {SOLVER_PREFIX!r} "
            "(guard: solver blocks and model kernels share one emitted namespace)"
        )
    reserved = _reserved_names()
    if name in reserved:
        raise ProcedureNameError(
            f"solver procedure {name!r} collides with an existing kernel / "
            "print-map entry"
        )
    return name


def check_arg_keys(name: str, args: Sequence[str]) -> tuple:
    """Validate an ordered argument-key list against the solver argument
    vocabulary (`zoomy_core.solver.arg_slots.SOLVER_ARG_KINDS`).

    An unknown key RAISES: there is no positional default and no silent
    fallthrough — an unmapped slot must be added to the vocabulary explicitly.
    """
    args = tuple(args)
    unknown = [a for a in args if a not in SOLVER_ARG_KINDS]
    if unknown:
        raise ProcedureNameError(
            f"procedure {name!r} declares unknown argument slot(s) {unknown!r}; "
            f"add them to SOLVER_ARG_KINDS (known: {sorted(SOLVER_ARG_KINDS)})"
        )
    dupes = [a for a in set(args) if args.count(a) > 1]
    if dupes:
        raise ProcedureNameError(
            f"procedure {name!r} repeats argument slot(s) {sorted(dupes)!r}")
    return args


# ── statements ──────────────────────────────────────────────────────────────


class Statement:
    """Base class of the five statement kinds.  Deliberately empty: the IR
    holds structure, the walkers hold lowering."""

    __slots__ = ()

    def children(self) -> tuple:
        """Nested statement blocks, as a tuple of tuples (for traversal)."""
        return ()


@dataclasses.dataclass(frozen=True)
class Assign(Statement):
    """`target = expr`.

    `target`  lvalue name.  A bare identifier (`dt`) or an indexed lvalue
                spelled by the caller (`lam_f[f]`) — the walkers print it
                verbatim on the left of `=`.
    `expr`    sympy expression, or an iterable of expressions when
                `shape` is non-empty (array-valued assign).
    `shape`   `()` for a scalar; otherwise the array shape.
    `declare` True emits a declaration (`T dt = …;` / a fresh array);
                False assigns into an existing name.
    `ctype`   optional C scalar type for a declaring assign (`"int"` for a
                counter); default is the backend's `real_type`.  Ignored by
                the python walker, which is dynamically typed.
    """

    target: str
    expr: Any
    shape: tuple = ()
    declare: bool = True
    ctype: str = ""

    def __post_init__(self):
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("Assign.target must be a non-empty str")
        object.__setattr__(self, "shape", tuple(self.shape))


@dataclasses.dataclass(frozen=True)
class Call(Statement):
    """Invoke `procedure` (a `Procedure` or an
    `ExternalProcedure`) with argument NAMES.

    `args`     ordered names bound in the enclosing scope.
    `results`  ordered out-names the call binds.  Empty for a pure
                 side-effecting block (`solver_write_fields`).

    Call statements deliberately bypass `SymbolicRegistrar.proxy_caller`:
    that path is expression-position only (`basefunction.py` builds a
    `sp.Function` node), and these blocks are statement-position and
    side-effecting.
    """

    procedure: str
    args: tuple = ()
    results: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "results", tuple(self.results))


@dataclasses.dataclass(frozen=True)
class IfStatic(Statement):
    """A BUILD-TIME branch on an NSM/scheme flag.

    `flag` is looked up in the flag mapping handed to
    `Procedure.resolve`; the selected branch is spliced in and the node
    disappears.  Both walkers REFUSE to lower an unresolved `IfStatic` — a
    build-time decision can never leak into the emitted march as a runtime
    branch (design v5: "resolved at BUILD time from NSM flags, never a runtime
    branch").
    """

    flag: str
    then: tuple = ()
    otherwise: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "then", tuple(self.then))
        object.__setattr__(self, "otherwise", tuple(self.otherwise))

    def children(self):
        return (self.then, self.otherwise)


@dataclasses.dataclass(frozen=True)
class ForEachFace(Statement):
    """Traverse the faces: `for (index = 0; index < count; ++index) body`.

    `count` is an argument NAME (`n_faces`), not a literal — the face
    count is a MeshRT quantity.
    """

    index: str
    count: str
    body: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "body", tuple(self.body))

    def children(self):
        return (self.body,)


@dataclasses.dataclass(frozen=True)
class While(Statement):
    """The march loop: `while (condition) body`.

    `condition` is a sympy Boolean expression over names live in the
    enclosing scope (typically the `solver_proceed` predicate applied to the
    march state).  `carry` names the values the loop updates — required by
    the python walker, whose jax lowering needs an explicit
    `lax.while_loop` carry; the C walker ignores it (mutation in place).
    """

    condition: Any
    body: tuple = ()
    carry: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "body", tuple(self.body))
        object.__setattr__(self, "carry", tuple(self.carry))

    def children(self):
        return (self.body,)


# ── procedure ───────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Procedure:
    """A named, emittable solver block: ordered argument slots + statements.

    `args` are keys of the solver argument vocabulary
    (`SOLVER_ARG_KINDS`); each walker spells
    them in its own dialect.  The IR never invents or routes an argument.

    `returns` names ONE scalar slot returned by value — the shape
    `reduce_dt` / `should_write` / `proceed` have.  Blocks that
    communicate through written arrays leave it empty (`void`).
    """

    name: str
    args: tuple = ()
    stmts: tuple = ()
    returns: str = ""
    doc: str = ""

    def __post_init__(self):
        object.__setattr__(self, "name", check_procedure_name(self.name))
        object.__setattr__(self, "args", check_arg_keys(self.name, self.args))
        object.__setattr__(self, "stmts", tuple(self.stmts))
        if self.returns:
            kind = SOLVER_ARG_KINDS.get(self.returns)
            if kind not in ("scalar_real", "scalar_int"):
                raise ProcedureNameError(
                    f"procedure {self.name!r} returns {self.returns!r}, which "
                    "is not a declared SCALAR slot — array results are passed "
                    "as written arguments")

    # -- build-time resolution --------------------------------------------

    def resolve(self, flags: Mapping[str, bool]) -> "Procedure":
        """Collapse every `IfStatic` against `flags` (build time).

        A flag referenced by the body but absent from `flags` RAISES — no
        silent default (the user law: an unmapped slot raises).
        """
        return dataclasses.replace(self, stmts=_resolve_block(self.stmts, flags))

    def is_resolved(self) -> bool:
        """True when no `IfStatic` survives anywhere in the body."""
        return not _find_static(self.stmts)

    def calls(self) -> tuple:
        """Every procedure name this body calls, in first-appearance order."""
        found: list = []

        def walk(block):
            for s in block:
                if isinstance(s, Call) and s.procedure not in found:
                    found.append(s.procedure)
                for child in s.children():
                    walk(child)

        walk(self.stmts)
        return tuple(found)


def _resolve_block(block, flags):
    out = []
    for s in block:
        if isinstance(s, IfStatic):
            if s.flag not in flags:
                raise KeyError(
                    f"IfStatic references build flag {s.flag!r} which is not in "
                    f"the supplied flag map {sorted(flags)!r} — solver build "
                    "flags have no default"
                )
            chosen = s.then if flags[s.flag] else s.otherwise
            out.extend(_resolve_block(chosen, flags))
        elif isinstance(s, ForEachFace):
            out.append(dataclasses.replace(s, body=_resolve_block(s.body, flags)))
        elif isinstance(s, While):
            out.append(dataclasses.replace(s, body=_resolve_block(s.body, flags)))
        else:
            out.append(s)
    return tuple(out)


def _find_static(block) -> bool:
    for s in block:
        if isinstance(s, IfStatic):
            return True
        for child in s.children():
            if _find_static(child):
                return True
    return False


def require_resolved(proc: Procedure) -> Procedure:
    """Walker entry guard: an unresolved build-time branch is a hard error."""
    if not proc.is_resolved():
        raise ValueError(
            f"procedure {proc.name!r} still contains IfStatic nodes — call "
            "Procedure.resolve(flags) at BUILD time; a static branch must "
            "never be lowered into the emitted march"
        )
    return proc


# ── expression helpers shared by both walkers ───────────────────────────────


_LVALUE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[(.+)\]\s*$")


def split_target(target: str):
    """`"lam_f[f]"` -> `("lam_f", "f")`; `"dt"` -> `("dt", None)`.

    An `Assign` target may be an indexed lvalue; both walkers need the same
    split (C writes `lam_f[f] = …`, jax writes `lam_f.at[f].set(…)`).
    """
    m = _LVALUE.match(target)
    return (m.group(1), m.group(2)) if m else (target, None)


def written_names(stmts) -> tuple:
    """Base names a statement block WRITES, in first-appearance order.

    Drives two things: the C walker's per-procedure const-correctness (a face
    array is `T*` in the block that writes it and `const T*` everywhere
    else) and the python walker's implicit loop carry.
    """
    found: list = []

    def add(n):
        if n not in found:
            found.append(n)

    for s in stmts:
        if isinstance(s, Assign):
            add(split_target(s.target)[0])
        elif isinstance(s, Call):
            for r in s.results:
                add(SOLVER_ARG_MAPPING.get(r, r))
        elif isinstance(s, IfStatic):
            for n in written_names(s.then + s.otherwise):
                add(n)
        else:
            for child in s.children():
                for n in written_names(child):
                    add(n)
    return tuple(found)


def expression_names(expr) -> tuple:
    """Free names an expression reads: plain Symbols plus the labels of any
    `IndexedBase` it indexes.  Sorted, deterministic."""
    if expr is None:
        return ()
    names = set()
    try:
        expr = sp.sympify(expr)
    except (sp.SympifyError, TypeError):
        return ()
    for b in expr.atoms(sp.IndexedBase):
        names.add(str(b.label))
    for s in expr.atoms(sp.Symbol):
        names.add(str(s))
    for f in expr.atoms(sp.Function):
        names.discard(f.func.__name__)
    return tuple(sorted(names))
