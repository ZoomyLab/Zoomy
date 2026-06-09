"""Pin-down: tag-driven solver-tag API + extraction rules.

Guards the low-level symbolic pipeline:
  * catalog + Expression.solver_tag API
  * per-term survival of solver tags across .apply()
  * collect_solver_tag per-canonical extraction rules

The end-to-end symbolic → solver path now lives in
``test_symbolic_pipeline.py`` via ``SystemModel``; the legacy
``SMEModelTagged`` / ``VAMModelTagged`` classes (removed) are
replaced by that adapter.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from sympy import Derivative

from zoomy_core.model.models.ins_generator import Expression
from zoomy_core.model.derivation.tag_catalog import (
    canonical_solver_tag, register_alias, CANONICAL_SOLVER_TAGS,
)
from zoomy_core.model.derivation.tag_extraction import collect_solver_tag


# ──────────────────────────────────────────────────────────────────────────────
# Tag catalog + Expression.solver_tag API
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_catalog_canonicalizes_aliases():
    assert canonical_solver_tag("convection") == "flux"
    assert canonical_solver_tag("convective") == "flux"
    assert canonical_solver_tag("ncp") == "nonconservative_flux"
    assert canonical_solver_tag("viscous") == "implicit_diffusion"
    assert canonical_solver_tag("pressure") == "hydrostatic_pressure"
    assert canonical_solver_tag("temporal") == "time_derivative"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_catalog_unknown_alias_raises():
    with pytest.raises(ValueError, match="Unknown solver tag"):
        canonical_solver_tag("not_a_real_tag")


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_register_alias_roundtrip():
    register_alias("my_custom_flux", "flux")
    assert canonical_solver_tag("my_custom_flux") == "flux"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_solver_tag_basic():
    t, x = sp.symbols("t x")
    u = sp.Function("u")(t, x)
    f = sp.Function("f")(t, x)
    flux_term = Derivative(u**2 / 2, x)
    src_term = -f
    eq = Expression(flux_term + src_term, name="burgers")
    eq = eq.solver_tag(flux=flux_term, source=src_term)
    assert eq.get_solver_tag("convection") == flux_term  # alias
    assert eq.get_solver_tag("flux") == flux_term
    assert eq.get_solver_tag("source") == src_term
    assert eq.untagged_remainder() == 0


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_solver_tag_survives_untouched_term():
    t, x = sp.symbols("t x")
    u = sp.Function("u")(t, x)
    f = sp.Function("f")(t, x)
    eq = Expression(Derivative(u**2 / 2, x) - f).solver_tag(
        flux=Derivative(u**2 / 2, x), source=-f,
    )
    # Substitute only u → stays in flux, drops source? f is in source, u is in flux.
    eq_after_u = eq.apply({u: 2 * u})
    assert "implicit_source" in eq_after_u.solver_tags  # source doesn't reference u → survives
    assert "flux" not in eq_after_u.solver_tags  # flux references u → dropped

    eq_after_f = eq.apply({f: 2 * f})
    assert "flux" in eq_after_f.solver_tags
    assert "implicit_source" not in eq_after_f.solver_tags


# ──────────────────────────────────────────────────────────────────────────────
# collect_solver_tag per-canonical extraction rules
# ──────────────────────────────────────────────────────────────────────────────

class _FakeSystem:
    def __init__(self, equations):
        self.equations = equations


def _swe_tagged_system():
    """Minimal SWE-with-topo tagged system for extractor tests."""
    b, h, hu = sp.symbols("b h hu")
    t, x = sp.symbols("t x")
    g = sp.Symbol("g", positive=True)

    mass = Expression(Derivative(h, t) + Derivative(hu, x), name="mass").solver_tag(
        time_derivative=Derivative(h, t),
        flux=Derivative(hu, x),
    )
    xmom = Expression(
        Derivative(hu, t)
        + Derivative(hu ** 2 / h, x)
        + Derivative(g * h ** 2 / 2, x)
        + g * h * Derivative(b, x),
        name="xmom",
    ).solver_tag(
        time_derivative=Derivative(hu, t),
        flux=Derivative(hu ** 2 / h, x),
        hydrostatic_pressure=Derivative(g * h ** 2 / 2, x),
        nonconservative_flux=g * h * Derivative(b, x),
    )
    return _FakeSystem({"mass": mass, "xmom": xmom}), (b, h, hu), [x], g


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_collect_flux_strips_outer_derivative():
    sys_, (b, h, hu), coords, g = _swe_tagged_system()
    F = collect_solver_tag(sys_, "flux",
                           variable_map={"mass": [1], "xmom": [2]},
                           n_variables=3, n_directions=1,
                           coords=coords, state_variables=(b, h, hu),
                           policy="strict")
    assert F[0, 0] == 0
    assert F[1, 0] == hu
    assert sp.simplify(F[2, 0] - hu ** 2 / h) == 0


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_collect_nc_reads_state_derivative():
    sys_, (b, h, hu), coords, g = _swe_tagged_system()
    B = collect_solver_tag(sys_, "nonconservative_flux",
                           variable_map={"mass": [1], "xmom": [2]},
                           n_variables=3, n_directions=1,
                           state_variables=(b, h, hu), coords=coords,
                           policy="strict")
    # g*h * d b/dx  →  B[row=2, col=0 (b is state var 0), dir=0] = g*h
    assert sp.simplify(B[2, 0, 0] - g * h) == 0


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_collect_rejects_nonconservative_flux_tag():
    """A state-variable in the coefficient of a flux tag is flagged."""
    b, h, hu = sp.symbols("b h hu")
    x = sp.Symbol("x")
    # h * d(hu)/dx is non-conservative; tagging it as flux should raise.
    bad = Expression(h * Derivative(hu, x)).solver_tag(flux=h * Derivative(hu, x))
    with pytest.raises(ValueError, match="non-conservative"):
        collect_solver_tag(_FakeSystem({"eq": bad}), "flux",
                           variable_map={"eq": [0]}, n_variables=1,
                           coords=[x], state_variables=(b, h, hu),
                           policy="strict")


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_collect_strict_policy_raises_on_remainder():
    h, hu = sp.symbols("h hu")
    t, x = sp.symbols("t x")
    # solver_tag only covers time_derivative; flux term left untagged
    eq = Expression(Derivative(h, t) + Derivative(hu, x)).solver_tag(
        time_derivative=Derivative(h, t)
    )
    with pytest.raises(ValueError, match="untagged remainder"):
        collect_solver_tag(_FakeSystem({"eq": eq}), "time_derivative",
                           variable_map={"eq": [0]}, n_variables=1,
                           policy="strict")


# ──────────────────────────────────────────────────────────────────────────────
# End-of-file.
#
# Model parity (hand-coded SME vs. tag-driven) now lives in
# ``test_symbolic_pipeline.py`` against the ``SystemModel`` adapter.
# ──────────────────────────────────────────────────────────────────────────────
