"""Catalog of canonical solver tags, their aliases and their SystemModel slots. Solver tags route
symbolic terms into SystemModel operator slots; physical display tags are not normalized here.
"""

from __future__ import annotations

from typing import Iterable


CANONICAL_SOLVER_TAGS: set[str] = {
    "flux",
    "nonconservative_flux",
    "hydrostatic_pressure",
    "time_derivative",
    "explicit_source",
    "implicit_source",
    "explicit_diffusion",
    "implicit_diffusion",
}


# Map every canonical solver tag to the corresponding SystemModel slot
# name.  This is the *only* explicit translation layer between the
# tag namespace (the model author's authoring vocabulary) and the
# SystemModel namespace (the solver's storage).
SOLVER_TAG_TO_SLOT: dict[str, str] = {
    "flux":                 "flux",
    "nonconservative_flux": "nonconservative_matrix",
    "hydrostatic_pressure": "hydrostatic_pressure",
    "time_derivative":      "mass_matrix",
    "implicit_source":      "source",
    "explicit_source":      "source_explicit",
    "implicit_diffusion":   "diffusion_matrix",
    "explicit_diffusion":   "diffusion_matrix_explicit",
}


_SOLVER_TAG_ALIASES: dict[str, str] = {
    # ── flux (convective / advective) ──────────────────────────────
    "flux":                 "flux",
    "convection":           "flux",
    "convective":           "flux",
    "convective_flux":      "flux",
    "advection":            "flux",
    "advective":            "flux",
    "advective_flux":       "flux",

    # ── non-conservative product ───────────────────────────────────
    "nonconservative_flux": "nonconservative_flux",
    "nc_flux":              "nonconservative_flux",
    "ncp":                  "nonconservative_flux",
    "nonconservative":      "nonconservative_flux",

    # ── time derivative ────────────────────────────────────────────
    "time_derivative":      "time_derivative",
    "temporal":             "time_derivative",
    "d_dt":                 "time_derivative",

    # ── hydrostatic pressure (flux-like column) ────────────────────
    "hydrostatic_pressure": "hydrostatic_pressure",
    "pressure":             "hydrostatic_pressure",
    "hydro":                "hydrostatic_pressure",

    # ── source — implicit / explicit treatment ─────────────────────
    "implicit_source":      "implicit_source",
    "source_implicit":      "implicit_source",
    "source":               "implicit_source",        # back-compat
    "explicit_source":      "explicit_source",
    "source_explicit":      "explicit_source",

    # ── diffusion (A tensor in div(A:∇Q)) — implicit / explicit ────
    "implicit_diffusion":            "implicit_diffusion",
    "diffusion_implicit":            "implicit_diffusion",
    "diffusion":                     "implicit_diffusion",        # default
    "diffusive":                     "implicit_diffusion",
    "diffusive_flux":                "implicit_diffusion",
    "viscous":                       "implicit_diffusion",
    "viscous_flux":                  "implicit_diffusion",
    "diffusion_matrix":              "implicit_diffusion",        # back-compat
    "diffusion_matrix_implicit":     "implicit_diffusion",
    "explicit_diffusion":            "explicit_diffusion",
    "diffusion_explicit":            "explicit_diffusion",
    "viscous_explicit":              "explicit_diffusion",
    "diffusion_matrix_explicit":     "explicit_diffusion",        # back-compat
}


def canonical_solver_tag(name: str) -> str:
    """Resolve an alias to its canonical solver tag.

    Raises:
        ValueError: Unknown alias; the message lists all known aliases.
    """
    try:
        return _SOLVER_TAG_ALIASES[name]
    except KeyError:
        known = ", ".join(
            f"{a}→{c}" for a, c in sorted(_SOLVER_TAG_ALIASES.items())
        )
        raise ValueError(
            f"Unknown solver tag {name!r}. Known aliases: {known}"
        ) from None


def solver_tag_slot(canonical: str) -> str:
    """Return the SystemModel slot name of a canonical solver tag.

    Raises:
        KeyError: Not a registered canonical tag; normalize aliases with canonical_solver_tag first.
    """
    return SOLVER_TAG_TO_SLOT[canonical]


def register_alias(alias: str, canonical: str) -> None:
    """Register an alias for an already registered canonical tag."""
    if canonical not in CANONICAL_SOLVER_TAGS:
        raise ValueError(
            f"Canonical name {canonical!r} is not registered. "
            f"Known canonicals: {sorted(CANONICAL_SOLVER_TAGS)}"
        )
    _SOLVER_TAG_ALIASES[alias] = canonical


def register_canonical_tag(name: str, aliases: Iterable[str] = (),
                           *, slot: str | None = None) -> None:
    """Register a new canonical tag with optional aliases and the SystemModel slot (default: the tag
    name) that stores its operator.
    """
    CANONICAL_SOLVER_TAGS.add(name)
    _SOLVER_TAG_ALIASES[name] = name
    for a in aliases:
        _SOLVER_TAG_ALIASES[a] = name
    SOLVER_TAG_TO_SLOT[name] = slot if slot is not None else name


def known_aliases() -> dict[str, str]:
    """Return a copy of the alias to canonical mapping."""
    return dict(_SOLVER_TAG_ALIASES)
