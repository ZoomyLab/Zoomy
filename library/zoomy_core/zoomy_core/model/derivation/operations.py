"""Derivation operations that do more than a substitution: SolveFor, SolveLinearSystem,
ChangeOfVariables. They follow the `Operation` protocol of `zoomy_core.model.operations` and act
on a Model before it is lifted to a SystemModel.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation


__all__ = ["SolveFor", "SolveLinearSystem", "ChangeOfVariables",
           "Granularity", "granularity_of"]


# ── SolveFor ───────────────────────────────────────────────────────────────


class SolveFor(Operation):
    """Reorient an equation in place into `variable = solution`. Pipeline counterpart of
    `equation.solve_for(var)`, which returns a new equation.
    """

    def __init__(self, variable, name=None):
        """Store the variable to isolate."""
        self._var = variable
        super().__init__(name=name or f"solve_for[{variable}]")

    def apply_to_equation(self, eq):
        """Replace the equation by the oriented relation `variable = solution`."""
        solved = eq.solve_for(self._var)        # oriented Equation
        eq.expr = solved.expr                   # residual  var - solution
        eq._as_relation = dict(solved._as_relation)
        return eq

    def __repr__(self):
        """Return `SolveFor(variable)`."""
        return f"SolveFor({self._var})"


# ── SolveLinearSystem ──────────────────────────────────────────────────────


class SolveLinearSystem:
    """Solve a coupled linear system `A v = b` for a family of variables with `sympy.linsolve`.
    `solve()` returns an indexable MomentFamily of oriented rows `v_i = solution_i` that can be
    passed to `.apply`.
    """

    def __init__(self, equations, variables, name="solve_linear_system"):
        """Store the equations and the unknown variables."""
        self._eqs = list(equations)
        self._vars = list(variables)
        self.name = name

    def solve(self):
        """Solve the system, hiding opaque Galerkin brackets behind symbols during the solve.

        Returns:
            MomentFamily whose component `i` is `variables[i] = solution_i`.
        """
        from zoomy_core.model.derivation.model import MomentFamily
        from zoomy_core.model.equation import Equation
        from zoomy_core.model.operations import is_bracket

        exprs = [sp.expand(sp.sympify(getattr(e, "expr", e))) for e in self._eqs]
        vars_ = list(self._vars)
        n = len(vars_)
        # Hide opaque Galerkin brackets behind symbols so the solver sees clean
        # (rational / numeric) coefficients; restore them in the solution.
        igs = sorted({ig for ex in exprs for ig in ex.atoms(sp.Integral)
                      if is_bracket(ig)}, key=sp.srepr)
        hide = {ig: sp.Symbol(f"_Br{i}") for i, ig in enumerate(igs)}
        restore = {s: ig for ig, s in hide.items()}
        # ``variables`` are applied Functions (``ŵ_j(t,x)``), not Symbols —
        # linsolve wants Symbols, so swap in placeholders for the solve.
        ph = [sp.Symbol(f"_v{i}") for i in range(n)]
        vsub = {v: ph[i] for i, v in enumerate(vars_)}
        back = {ph[i]: vars_[i] for i in range(n)}
        H = [ex.xreplace(hide).xreplace(vsub) for ex in exprs]
        sols = sp.linsolve(H, ph)
        if not sols:
            raise ValueError(
                "SolveLinearSystem: inconsistent system (linsolve returned ∅).")
        tup = list(sols)[0]
        comps, relmap = [], {}
        for i, v in enumerate(vars_):
            si = sp.expand(tup[i].xreplace(restore).xreplace(back))
            comps.append(Equation(sp.Eq(v, si), name=f"{self.name}_{i}"))
            relmap[v] = si
        fam = MomentFamily(name=self.name, modes=list(range(n)), components=comps)
        # Combined relation so ``eq.apply(sol)`` substitutes the whole family
        # (the ``.apply`` dispatch duck-types on ``_as_relation``).
        fam._as_relation = relmap
        return fam

    def __repr__(self):
        """Return the equation and variable counts."""
        return (f"SolveLinearSystem({len(self._eqs)} eqs, "
                f"{len(self._vars)} vars)")


# ── granularity ──────────────────────────────────────────────────────────


class Granularity:
    """Structural level an Operation acts on: `LEAF`, `TERM`, `EQUATION` or `SYSTEM`."""

    LEAF = "LEAF"
    TERM = "TERM"
    EQUATION = "EQUATION"
    SYSTEM = "SYSTEM"


def granularity_of(op) -> str:
    """Return the Granularity of an operation from its `system_level` / `whole_model_op` /
    `single_term_only` flags.

    Args:
        op: an Operation instance.
    """
    if getattr(op, "system_level", False):
        return Granularity.SYSTEM
    if getattr(op, "whole_model_op", False):
        return Granularity.EQUATION
    if getattr(op, "single_term_only", False):
        return Granularity.TERM
    return Granularity.LEAF


# ── ChangeOfVariables ────────────────────────────────────────────────────


class ChangeOfVariables(Operation):
    """Rename a coefficient family `a(i, ...) -> relation(q(i, ...))` across the model and swap
    `a` for `q` in the unknowns. The unknown swap (`model.redeclare_unknown`) is what
    distinguishes it from a plain substitution.

    Args:
        old: name of the family currently in `Q`.
        new: name of the replacing family.
        relation: callable mapping the applied `new` function to the substituted expression.
    """

    # A change of variables grows / rewrites the whole equation set and swaps
    # Q — it is a model-level op routed through ``apply_to_model``.
    whole_model_op = True
    transforms_bcs = True

    def __init__(self, old, new, relation, name=None):
        """Store the family names and the relation."""
        self._old = old
        self._new = new
        self._relation = relation
        super().__init__(
            name=name or f"change_of_variables[{old}->{new}]",
            description=f"Change of variables {old}_i -> relation({new}_i)",
        )

    def apply_to_model(self, model):
        """Rewrite every equation and relation, then redeclare the unknowns `old -> new`."""
        # Match the old family by NAME — reconstructing ``sp.Function(self._old)``
        # would drop the field's assumptions (``real=True``) and match nothing.
        # The real head (with assumptions) is recovered from the equations and
        # handed to ``redeclare_unknown`` so the Q swap matches too.
        old_head = next(
            (f.func for eq in model._equations.values()
             for f in eq.expr.atoms(sp.Function)
             if f.func.__name__ == self._old),
            sp.Function(self._old, real=True))
        # Mint the new head with the SAME assumptions as the old family — a real
        # ``a`` gives a real ``q``, a plain ``a`` a plain ``q``.  Hardcoding
        # ``real=True`` would create a ``q`` that ``expr.has`` cannot match
        # against an assumption-free ``Function(self._new)``.
        new_head = sp.Function(
            self._new, **getattr(old_head, "_explicit_class_assumptions", {}))

        def _rewrite(expr):
            """Replace each `old` application in `expr` by `relation(new(...))`."""
            if not hasattr(expr, "replace"):
                return expr
            return expr.replace(
                lambda e: (isinstance(e, sp.Function)
                           and e.func.__name__ == self._old),
                lambda e: self._relation(new_head(*e.args)))

        for eq in model._equations.values():
            eq.expr = _rewrite(eq.expr)
            # An ORIENTED relation (e.g. the w-reconstruction ``w̃ = …``) keeps a
            # separate ``_as_relation`` — rewrite its lhs/rhs too.
            rel = getattr(eq, "_as_relation", None)
            if rel:
                eq._as_relation = {_rewrite(k): _rewrite(v)
                                   for k, v in rel.items()}

        # Swap the unknown family in Q (and let the model recompute Qaux).  A
        # change of variables is a genuine family RENAME, so the logical Q key
        # follows the new family (``a → q``), not the old one.
        model.redeclare_unknown(old_head, new_head, rename_key=True)
        model._refresh_unknowns()
        return model

    def __repr__(self):
        """Return `ChangeOfVariables(old -> new, relation)`."""
        return (f"ChangeOfVariables({self._old!r} -> {self._new!r}, "
                f"{self._relation!r})")
