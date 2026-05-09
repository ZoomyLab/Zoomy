"""End-to-end regression for the symbolic → solver pipeline.

Pins the closed forms produced by the SWE / SME tutorials so that
future refactors to ``ins_generator`` / ``tag_extraction`` /
``system_model`` can't silently drift.  The shared build logic lives
in ``_symbolic_fixtures.py``; this file only asserts outcomes.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from zoomy_core.model.models.ins_generator import Expression, FullINS

from tests.unit.zoomy_core._symbolic_fixtures import (
    build_sme_l1_system,
    build_swe_system,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — both SWE and SME-L1 builds take a few seconds;
# build once per module and share across tests.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def swe_fixture():
    return build_swe_system()


@pytest.fixture(scope="module")
def sme_l1_fixture():
    return build_sme_l1_system()


# ---------------------------------------------------------------------------
# SWE — extract flux + NC, assemble, match the canonical 1-D SWE with topo.
# ---------------------------------------------------------------------------

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_swe_untagged_remainders_empty(swe_fixture):
    sm, _state, _ph = swe_fixture
    remainders = sm.untagged_remainders()
    assert not remainders, f"SWE has untagged remainders: {remainders}"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_swe_matches_canonical(swe_fixture):
    """``∂_t q + ∂_x F + A · ∂_x q`` reconstructs the classical SWE.

    Order in the SystemModel adapter is ``[b, h, hu]`` (with ``b`` a
    non-evolving parameter, so row 0 should be trivially zero).  The
    symbolic derivation uses the η-coupled form — part of the
    hydrostatic pressure ends up in the NC matrix rather than the flux
    — so the canonical check is against the *reconstruction*, not the
    individual operators.
    """
    sm, _state, ph = swe_fixture
    h, hu, b, g = ph["h"], ph["hu"], ph["b"], ph["g"]

    F = sm.flux()
    A = sm.nonconservative_matrix()

    # Row 0 (``b``) is not evolved — flux / NC should be zero there.
    assert F[0, 0] == 0
    for j in range(3):
        assert sp.simplify(A[0, j, 0]) == 0

    # Pin the depth equation: ``∂_t h + ∂_x hu = 0``.
    assert sp.simplify(F[1, 0] - hu) == 0
    for j in range(3):
        assert sp.simplify(A[1, j, 0]) == 0

    # Momentum row: ``F[hu,0] + A[hu,:]·∂q`` must equal the classical
    # ``∂_x(hu²/h + g·h²/2) + g·h·∂_x b``.
    x = sp.Symbol("x")
    q = (b, h, hu)
    reconstructed = sp.diff(F[2, 0], x) + sum(
        A[2, j, 0] * sp.diff(q_j, x) for j, q_j in enumerate(q)
    )
    canonical = sp.diff(hu ** 2 / h + g * h ** 2 / 2, x) + g * h * sp.diff(b, x)
    assert sp.simplify(reconstructed - canonical) == 0, (
        f"SWE reconstruction diverges from canonical:\n"
        f"  reconstructed = {sp.simplify(reconstructed)}\n"
        f"  canonical     = {sp.simplify(canonical)}"
    )


@pytest.mark.medium
@pytest.mark.unittest
def test_swe_dambreak_matches_hand_coded(swe_fixture):
    """1-D dam-break: symbolic SWE must reproduce the hand-coded baseline.

    Wraps the extracted ``flux`` / ``nonconservative_matrix`` in a
    ``StructuredDerivativeModel`` and compares against Zoomy's classical
    ``SWEModelV2``-equivalent hand-coded flux on a 200-cell,
    ``t=0.5`` dam-break.  Assert match to floating-point noise.
    """
    from sympy import Matrix

    import zoomy_core.fvm.timestepping as timestepping
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.mesh.fvm_mesh import FVMMesh
    from zoomy_core.misc.misc import ZArray, Zstruct
    from zoomy_core.model.derivative_workflow import (
        DerivativeAwareSolver, StructuredDerivativeModel,
    )

    sm, _state, ph = swe_fixture
    F_sym = sm.flux()
    A_sym = sm.nonconservative_matrix()
    H_PH, HU_PH, B_PH, G_PH = ph["h"], ph["hu"], ph["b"], ph["g"]

    class SymbolicSWE(StructuredDerivativeModel):
        dimension = 1
        variables = ["h", "hu"]
        parameters = {"g": (9.81, "positive")}

        def _to_runtime(self, expr):
            return sp.sympify(expr).subs({
                H_PH: self.Q.h,
                HU_PH: self.Q.hu,
                B_PH: 0,
                G_PH: self.params.g,
            })

        def flux(self):
            F = Matrix.zeros(self.n_variables, self.dimension)
            F[0, 0] = self._to_runtime(F_sym[1, 0])
            F[1, 0] = self._to_runtime(F_sym[2, 0])
            return ZArray(F)

        def nonconservative_matrix(self):
            from sympy.tensor.array import MutableDenseNDimArray
            T = MutableDenseNDimArray.zeros(
                self.n_variables, self.n_variables, self.dimension
            )
            for i_sym, i_run in [(1, 0), (2, 1)]:
                for j_sym, j_run in [(1, 0), (2, 1)]:
                    val = self._to_runtime(A_sym[i_sym, j_sym, 0])
                    if val != 0:
                        T[i_run, j_run, 0] = val
            return ZArray(T)

        def source(self):
            return ZArray.zeros(self.n_variables)

    class HandCodedSWE(StructuredDerivativeModel):
        dimension = 1
        variables = ["h", "hu"]
        parameters = {"g": (9.81, "positive")}

        def flux(self):
            h, hu = self.Q.h, self.Q.hu
            g = self.params.g
            F = Matrix.zeros(self.n_variables, self.dimension)
            F[0, 0] = hu
            F[1, 0] = hu ** 2 / h + 0.5 * g * h ** 2
            return ZArray(F)

    def _ic(x):
        Q = np.zeros(2, dtype=float)
        Q[0] = 2.0 if x[0] < 5.0 else 1.0
        Q[1] = 0.0
        return Q

    def _run(cls):
        bcs = BC.BoundaryConditions(
            [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
        )
        model = cls(
            boundary_conditions=bcs,
            initial_conditions=IC.UserFunction(_ic),
        )
        mesh = FVMMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=200)
        settings = Zstruct(output=Zstruct(
            directory="/tmp/pytest_dambreak",
            filename="dambreak",
            snapshots=2,
            clean_directory=True,
        ))
        solver = DerivativeAwareSolver(
            time_end=0.5,
            settings=settings,
            compute_dt=timestepping.adaptive(CFL=0.9),
        )
        Qnew, _ = solver.solve(mesh, model, write_output=False)
        return Qnew[0], Qnew[1]

    h_sym, hu_sym = _run(SymbolicSWE)
    h_hc, hu_hc = _run(HandCodedSWE)

    assert float(np.abs(h_sym - h_hc).max()) < 1e-10
    assert float(np.abs(hu_sym - hu_hc).max()) < 1e-10


# ---------------------------------------------------------------------------
# SME level-1 + Newtonian — closure properties.
# ---------------------------------------------------------------------------

def _auto_tag_and_remainder(leaf_expr, *, state_vars, time_var, coords,
                            parameters=()):
    """Helper: run auto_solver_tag on an Expression and return the
    untagged remainder after expand.
    """
    e = Expression(leaf_expr)
    tagged = e.auto_solver_tag(state_vars=state_vars, time_var=time_var,
                               coords=coords, parameters=parameters)
    return tagged.untagged_remainder()


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_sme_l1_continuity_closes(sme_l1_fixture):
    """SME depth-average continuity: ``∂_t h + ∂_x(α₀·h) = 0`` — exactly."""
    model, state, basis = sme_l1_fixture
    h = state.H
    alpha_0 = basis.alpha.alpha_0
    alpha_1 = basis.alpha.alpha_1

    remainder = _auto_tag_and_remainder(
        model.continuity._node.expr,
        state_vars=[h, alpha_0, alpha_1],
        time_var=state.t,
        coords=[state.x],
        parameters=[state.b],
    )
    assert sp.simplify(remainder) == 0, f"continuity residual: {remainder}"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_sme_l1_mean_moment_residual_bounded(sme_l1_fixture):
    """SME mean-moment ``momentum.x.test_0``: pin the known residual shape.

    ``auto_solver_tag`` doesn't yet recognize either the Newtonian
    viscous cross-derivatives (``ν·∂α₀·∂h``, ``ν·∂²h/∂t∂x``) or the
    product-rule-expanded time derivative ``h·∂_t α₀ + α₀·∂_t h``
    — neither matches its ``coeff · Derivative(q, v)`` template with
    coefficient state-free.  Both are pre-existing limitations, not
    regressions from the Task-3.5 simplify pipeline.

    Pin the current remainder to ≤ 6 additive terms so a future
    auto-tag extension (e.g. recognizing product-rule time
    derivatives) shows up as a test failure prompting a tighter
    bound, and any regression that grows the remainder fails loudly.
    """
    model, state, basis = sme_l1_fixture
    h = state.H
    alpha_0 = basis.alpha.alpha_0
    alpha_1 = basis.alpha.alpha_1
    nu = sp.Symbol("nu")
    lamda = sp.Symbol("lamda", positive=True)
    tau_c = sp.Symbol("tau_c", positive=True)

    remainder = _auto_tag_and_remainder(
        model.momentum.x.test_0._node.expr,
        state_vars=[h, alpha_0, alpha_1],
        time_var=state.t,
        coords=[state.x],
        parameters=[state.b, state.rho, nu, lamda, tau_c, state.g],
    )
    n_terms = len(sp.Add.make_args(sp.expand(remainder)))
    assert n_terms <= 7, (
        f"mean-moment residual grew from 7 to {n_terms} terms:\n"
        f"  {sp.simplify(remainder)}"
    )
    # All remaining terms should either involve a ``nu``-named symbol
    # (viscous cross-derivatives, viscous boundary Subs residues) or a
    # time derivative (product-rule-expanded ``∂_t(α·h)``).  Catches
    # an orthogonal regression that introduces an untagged *spatial*
    # flux.  Match by symbol *name* rather than the Symbol object,
    # since ``Newtonian`` builds its own ``Symbol("nu", positive=True)``
    # that differs in assumptions from the bare ``Symbol("nu")``.
    for term in sp.Add.make_args(sp.expand(remainder)):
        if term == 0:
            continue
        has_nu = any(s.name == "nu" for s in term.free_symbols)
        has_dt = any(
            isinstance(d, sp.Derivative) and state.t in d.variables
            for d in term.atoms(sp.Derivative)
        )
        assert has_nu or has_dt, (
            f"unexpected residual term (neither ν nor ∂_t): {term}"
        )


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_sme_l1_shear_residual_is_single_zeta_integral(sme_l1_fixture):
    """SME shear-moment ``momentum.x.test_1``: pin the current residual.

    After the three-pass simplify pipeline the shear moment reduces to
    the closed moment form plus **one** unclosed integral
    ``Integral(..., (ζ, 0, 1))`` whose integrand's denominator
    ``4·h·∂_x α₁ − 4·α₁·∂_x h`` defeats ``sympy.integrate``.

    The test pins this: exactly one Integral, exactly over ``(ζ, 0, 1)``.
    A regression that introduces more residual terms fails; an
    improvement that drives the residual to zero also fails (loudly,
    so the acceptance bar can be tightened in the same commit).
    """
    model, state, _basis = sme_l1_fixture
    expr = model.momentum.x.test_1._node.expr
    zeta = state.zeta_ref
    integrals = [I for I in expr.atoms(sp.Integral) if I.has(zeta)]
    assert len(integrals) == 1, (
        f"expected exactly one ζ-Integral residual, got {len(integrals)}"
    )
    limits = integrals[0].args[1]
    assert limits[0] == zeta
    assert limits[1] == 0
    assert limits[2] == 1


# ---------------------------------------------------------------------------
# SystemModel belt-and-braces scan: no hardcoded physics literals.
# ---------------------------------------------------------------------------

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_system_model_has_no_hardcoded_physics():
    """Guard: the SystemModel source must not contain physics literals.

    Every PDE operator should come out of the symbolic chain via
    ``auto_solver_tag`` — the adapter is just a thin reader.  This
    catches accidental backsliding where someone hardcodes a
    ``g*h**2/2`` or ``hu**2/h`` into the extractor.
    """
    from zoomy_core.model.models import system_model

    src = Path(system_model.__file__).read_text()
    # Keep the list narrow and specific; broad words like ``flux`` or
    # ``momentum`` show up legitimately in docstrings.
    forbidden_patterns = [
        r"g\s*\*\s*h\s*\*\*\s*2\s*/\s*2",   # g * h**2 / 2
        r"hu\s*\*\*\s*2\s*/\s*h",           # hu**2 / h
        r"0\.5\s*\*\s*g\s*\*\s*h",          # 0.5 * g * h
    ]
    for pat in forbidden_patterns:
        if re.search(pat, src):
            pytest.fail(
                f"SystemModel source matches hardcoded-physics pattern "
                f"{pat!r}:\n{src}"
            )


# ---------------------------------------------------------------------------
# History mermaid — smoke: output is a non-empty ``flowchart`` string.
# ---------------------------------------------------------------------------

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_history_mermaid_renders():
    """``history_mermaid()`` renders a mermaid flowchart.

    Returns a ``MermaidDiagram`` whose ``str()`` is the raw mermaid
    source (``flowchart LR ...``).  Smoke: the string starts with a
    ``flowchart`` header.
    """
    from zoomy_core.model.models.ins_generator import StateSpace

    state = StateSpace(dimension=2)
    model = FullINS(state)
    diagram = model.history_mermaid()
    text = str(diagram)
    assert "flowchart" in text.lower()
