"""Symbolic Domain / Boundary / Normal layer.

Sympy-side atoms for representing continuous integration regions and
their boundaries.  These are *not* the numerical mesh in
`zoomy_core.mesh` — they are symbolic descriptors used to drive
weak-form derivations (`DivergenceTheorem`, `MapToReferenceElement`)
without committing to a specific discretisation.

Pattern:

* `Domain` and its subclasses (`Interval`, `Simplex`, `Box`) are
  Python descriptors carrying coordinate symbols plus geometry data.
* `BoundaryIntegral(integrand, domain)` is an opaque sympy atom — the
  surface integral `∫_∂D f ds`.  It is manufactured per `Domain`
  via the established `phi_fn` pattern (one `sympy.Function`
  subclass per domain instance, with `_domain` back-reference).
* `NormalVector(domain)` returns the `sympy.Function` subclass for
  the outward unit normal.  Calls `n(0)`, `n(1)`, … give opaque
  component atoms.

Volume integrals stay as native `sympy.Integral` — the existing 1D
pipeline keeps consuming them unchanged.
"""

from __future__ import annotations

from typing import Sequence

import sympy as sp


# ---------------------------------------------------------------------------
# Domain hierarchy.
# ---------------------------------------------------------------------------


class Domain:
    """Continuous integration region in symbolic form.

    Subclasses fill in geometry (vertices, intervals) and override
    `boundary`, `reference`, `affine_map`, `normal` as
    appropriate for their shape.
    """

    coords: tuple[sp.Symbol, ...]
    name: str

    def __init__(self, coords: Sequence[sp.Symbol], name: str):
        self.coords = tuple(coords)
        self.name = name
        # Atom families and shape-derived domains (cached per-instance —
        # see module docstring).  Caching `boundary()` and `reference()`
        # is essential: the manufactured Function classes carry the
        # `_domain` back-ref, so two fresh `boundary()` calls would
        # produce two distinct atom families and structurally-equal
        # `BoundaryIntegral` atoms would compare unequal.
        self._boundary_integral_fn: type | None = None
        self._normal_fn: type | None = None
        self._boundary_cache: "Domain | None" = None
        self._reference_cache: "Domain | None" = None

    @property
    def dim(self) -> int:
        return len(self.coords)

    # --- shape-dependent hooks ----------------------------------------------

    def boundary(self) -> "Domain":
        raise NotImplementedError(
            f"{type(self).__name__} does not define a boundary domain.")

    def reference(self) -> "Domain":
        raise NotImplementedError(
            f"{type(self).__name__} does not define a reference element.")

    def affine_map(self) -> tuple[sp.Matrix, sp.Matrix]:
        """Return `(B, V0)` such that `x = V0 + B · ξ` maps the
        reference element to `self`.  Defined only on volume domains.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define an affine map.")

    def normal(self) -> type:
        """Return the `sympy.Function` class for the outward unit
        normal on `self`.  Defined only on boundary domains.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not a boundary; normals are "
            "defined on its boundary, not on the volume itself.")

    # --- atom factories (cached) --------------------------------------------

    @property
    def boundary_integral_fn(self) -> type:
        """`sympy.Function` subclass for `∫_∂(self) f ds` atoms.

        One class per `Domain` instance.  The back-reference
        `cls._domain` lets operations recover the surrounding domain.
        Renders in LaTeX as `∮_{<name>} f \\, dS` via the `_latex`
        method.
        """
        if self._boundary_integral_fn is None:
            domain_name_latex = self._latex_name()

            def _latex(self_atom, printer, exp=None):
                inner = printer.doprint(self_atom.args[0])
                tex = rf"\oint_{{{domain_name_latex}}} {inner} \, dS"
                if exp is not None:
                    tex = rf"\left({tex}\right)^{{{exp}}}"
                return tex

            self._boundary_integral_fn = type(
                f"BoundaryIntegral_{self.name}",
                (sp.Function,),
                {"_domain": self, "nargs": 1, "_latex": _latex},
            )
        return self._boundary_integral_fn

    def _latex_name(self) -> str:
        """LaTeX-friendly version of the domain's name (override in subclasses
        to inject `\\partial` / `\\hat` and similar)."""
        # ``hat_K`` → ``\\hat{K}``, ``hat_∂K`` → ``\\hat{\\partial K}``.
        if self.name.startswith("hat_"):
            inner = self.name[4:]
            inner = inner.replace("∂", r"\partial ")
            return rf"\hat{{{inner}}}"
        return self.name.replace("∂", r"\partial ")

    @property
    def normal_fn(self) -> type:
        """`sympy.Function` subclass for outward unit normal components.

        Calls `n(i)` carry the back-reference `cls._domain` so
        downstream operations can verify "this normal belongs to that
        boundary".  Renders in LaTeX as `n_<name>^{(i)}`.
        """
        if self._normal_fn is None:
            domain_name_latex = self._latex_name()

            def _latex(self_atom, printer, exp=None):
                idx_tex = printer.doprint(self_atom.args[0])
                tex = rf"n_{{{domain_name_latex}}}^{{({idx_tex})}}"
                if exp is not None:
                    tex = rf"\left({tex}\right)^{{{exp}}}"
                return tex

            # Sanitised class name so sympy's default repr doesn't carry
            # the unicode ``∂`` (which the LaTeX printer can't tokenise).
            class_name = f"n_{self.name}".replace("∂", "boundary_")
            self._normal_fn = type(
                class_name,
                (sp.Function,),
                {"_domain": self, "nargs": 1, "_latex": _latex},
            )
        return self._normal_fn

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Volume domains.
# ---------------------------------------------------------------------------


class Interval(Domain):
    """1D interval `[a, b]` parametrised by a single `var`."""

    def __init__(self, var: sp.Symbol, a, b, name: str | None = None):
        super().__init__((var,), name or f"I_{var}")
        self.a = sp.sympify(a)
        self.b = sp.sympify(b)

    def boundary(self) -> "PointSet":
        if self._boundary_cache is None:
            self._boundary_cache = PointSet(
                self.coords[0], (self.a, self.b), name=f"∂{self.name}")
        return self._boundary_cache  # type: ignore[return-value]

    def reference(self) -> "Interval":
        if self._reference_cache is None:
            ref_var = sp.Symbol(f"hat_{self.coords[0]}", real=True)
            self._reference_cache = Interval(
                ref_var, sp.S.Zero, sp.S.One, name=f"hat_{self.name}")
        return self._reference_cache  # type: ignore[return-value]

    def affine_map(self) -> tuple[sp.Matrix, sp.Matrix]:
        return sp.Matrix([[self.b - self.a]]), sp.Matrix([[self.a]])


class Simplex(Domain):
    """`n`-dim simplex from a vertex tuple.

    `vertices` is a sequence of `n+1` points, each a tuple/Matrix
    of length `n` (matching `coords`).  `B = [V₁−V₀ | V₂−V₀ | …]`,
    `V₀` is the offset, and the affine map is `x = V₀ + B · ξ`.
    """

    def __init__(self, vertices: Sequence, coords: Sequence[sp.Symbol],
                 name: str | None = None):
        super().__init__(coords, name or "K")
        self.vertices = tuple(
            sp.Matrix(v) if not isinstance(v, sp.MatrixBase)
            else sp.Matrix(v)
            for v in vertices
        )
        if len(self.vertices) != self.dim + 1:
            raise ValueError(
                f"{self.dim}-simplex needs {self.dim + 1} vertices; "
                f"got {len(self.vertices)}.")
        for i, v in enumerate(self.vertices):
            if v.shape != (self.dim, 1):
                raise ValueError(
                    f"vertex {i} has shape {v.shape}, expected "
                    f"({self.dim}, 1).")

    def boundary(self) -> "SimplexBoundary":
        if self._boundary_cache is None:
            self._boundary_cache = SimplexBoundary(parent=self)
        return self._boundary_cache  # type: ignore[return-value]

    def reference(self) -> "Simplex":
        if self._reference_cache is None:
            ref_coords = tuple(sp.Symbol(f"xi_{i}", real=True, positive=True)
                               for i in range(self.dim))
            # Canonical simplex vertices: origin + standard unit vectors.
            ref_vertices = [sp.zeros(self.dim, 1)]
            for i in range(self.dim):
                v = sp.zeros(self.dim, 1)
                v[i, 0] = sp.S.One
                ref_vertices.append(v)
            self._reference_cache = Simplex(
                ref_vertices, ref_coords, name=f"hat_{self.name}")
        return self._reference_cache  # type: ignore[return-value]

    def affine_map(self) -> tuple[sp.Matrix, sp.Matrix]:
        V0 = self.vertices[0]
        cols = [self.vertices[i + 1] - V0 for i in range(self.dim)]
        B = sp.Matrix.hstack(*cols)
        return B, V0


class Box(Domain):
    """Tensor product of `Interval` factors."""

    def __init__(self, intervals: Sequence[Interval], name: str | None = None):
        coords = tuple(I.coords[0] for I in intervals)
        super().__init__(coords, name or "Box")
        self.intervals = tuple(intervals)

    def boundary(self) -> "BoxBoundary":
        if self._boundary_cache is None:
            self._boundary_cache = BoxBoundary(parent=self)
        return self._boundary_cache  # type: ignore[return-value]

    def reference(self) -> "Box":
        if self._reference_cache is None:
            ref_intervals = [I.reference() for I in self.intervals]
            self._reference_cache = Box(
                ref_intervals, name=f"hat_{self.name}")
        return self._reference_cache  # type: ignore[return-value]

    def affine_map(self) -> tuple[sp.Matrix, sp.Matrix]:
        diag = [I.b - I.a for I in self.intervals]
        offset = sp.Matrix([[I.a] for I in self.intervals])
        B = sp.diag(*diag)
        return B, offset


# ---------------------------------------------------------------------------
# Boundary domains.
# ---------------------------------------------------------------------------


class _BoundaryDomain(Domain):
    """Base for boundary-of-volume domains.

    Boundary domains expose `normal()` but do not have their own
    `boundary()` (we don't model boundary-of-boundary in this layer).
    """

    parent: Domain

    def __init__(self, parent: Domain, name: str | None = None):
        # Boundary inherits the parent's coords for the purpose of
        # writing integrands; its dim is one less than the parent's.
        super().__init__(parent.coords, name or f"∂{parent.name}")
        self.parent = parent

    def _latex_name(self) -> str:
        return rf"\partial {self.parent._latex_name()}"

    @property
    def dim(self) -> int:
        return self.parent.dim - 1

    def boundary(self) -> "Domain":
        raise NotImplementedError(
            "boundary-of-boundary is not modelled in this layer.")

    def affine_map(self) -> tuple[sp.Matrix, sp.Matrix]:
        # Boundary parameterisations (curve / surface charts) are not
        # part of this plan — only the volume affine map is.
        raise NotImplementedError(
            f"{type(self).__name__} does not define an affine map; "
            "boundary parameterisations are out of scope for this layer.")

    def normal(self) -> type:
        return self.normal_fn


class PointSet(_BoundaryDomain):
    """Degenerate 0-dim boundary — the boundary of an `Interval`.

    Bridge to the existing `Subs(f, var, bound)` plumbing in
    `ins_generator`: a `BoundaryIntegral` over a `PointSet` is
    semantically `f|_{var=b} − f|_{var=a}` and resolves through that
    convention rather than producing a new symbolic kind.
    """

    def __init__(self, var: sp.Symbol, points: Sequence,
                 name: str | None = None):
        # Construct without going through _BoundaryDomain.__init__'s
        # parent-coupling, since a PointSet can also be created free-
        # standing for tests.
        Domain.__init__(self, (var,), name or f"∂I_{var}")
        self.parent = None  # type: ignore[assignment]
        self.points = tuple(sp.sympify(p) for p in points)

    @property
    def dim(self) -> int:
        return 0


class SimplexBoundary(_BoundaryDomain):
    """Opaque boundary of a `Simplex` — one symbolic atom covers
    `∂K` (no per-face decomposition in this layer).
    """


class BoxBoundary(_BoundaryDomain):
    """Opaque boundary of a `Box` — same convention as
    `SimplexBoundary`.
    """


# ---------------------------------------------------------------------------
# Atom factories — user-facing.
# ---------------------------------------------------------------------------


def BoundaryIntegral(integrand: sp.Expr, domain: Domain) -> sp.Expr:
    """Construct a symbolic `∫_∂D f ds` atom.

    `domain` should be a boundary domain (e.g. `K.boundary()`).  The
    result is an opaque `sympy.Function` call whose class carries
    `_domain` back to `domain` — operations like
    `DivergenceTheorem` and `MapToReferenceElement` walk for
    instances of this class to find boundary integrals.
    """
    return domain.boundary_integral_fn(integrand)


def NormalVector(domain: Domain) -> type:
    """Return the `sympy.Function` class for outward unit normal
    components on `domain`.

    Usage:

    ```
        n = NormalVector(K.boundary())
        flux = sp.diff(u, x) * n(0) + sp.diff(u, y) * n(1)
    ```
    """
    return domain.normal_fn
