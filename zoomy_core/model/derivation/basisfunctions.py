"""Concrete polynomial bases in ``z`` (Legendre, Chebyshev, splines, layered, BC-aware) for the
moment/Galerkin derivation. They resolve the opaque Galerkin brackets of a Model into numbers on the
way to a SystemModel.
"""

from copy import deepcopy

import numpy as np
import sympy
from sympy import Symbol, bspline_basis_set, diff, integrate, lambdify, legendre
from sympy.abc import z
from sympy.functions.special.polynomials import chebyshevu


class Basisfunction:
    """Base class of a concrete basis: polynomials ``basis`` plus opaque heads ``phi_fn(k, zeta)``
    and ``c_fn(zeta)`` that point back to the instance. ResolveBasis routes Galerkin brackets built
    from these heads to :meth:`resolve`.

    Args:
        level: highest basis index (``level + 1`` functions).
        symbol: name of the opaque ``phi_fn`` Function class.
    """
    name = "Basisfunction"

    def bounds(self):
        """Return the integration interval ``[0, 1]``."""
        return [0, 1]

    def basis_definition(self):
        """Return the monomials ``z**k`` for ``k = 0..level``."""
        z = Symbol("z")
        b = lambda k, z: z**k
        return [b(k, z) for k in range(self.level + 1)]

    def weight(self, z):
        """Return the weight ``omega(z) = 1``."""
        return 1

    def mean_coefficients(self):
        """Return coefficients ``c_k`` with ``sum(c_k phi_k) = 1``, so the depth mean is ``sum(c_k
        alpha_k)``. Default ``[1, 0, ...]`` assumes ``phi_0 = 1``; override when the constant is a
        combination (e.g. B-splines).
        """
        c = [sympy.Integer(0)] * (self.level + 1)
        c[0] = sympy.Integer(1)
        return c

    def weight_eval(self, z):
        """Evaluate ``weight`` numerically via lambdify."""
        z = Symbol("z")
        f = sympy.lambdify(z, self.weight(z))
        return f(z)

    def __init__(self, level=0, symbol="phi", **kwargs):
        """Build the concrete ``basis``, the opaque heads ``phi_fn`` / ``c_fn`` and the bracket
        cache.
        """
        self.level = level
        self.symbol = symbol
        self.basis = self.basis_definition(**kwargs)
        # ONE opaque sympy Function class for this basis instance,
        # with two arguments: ``(k, arg)`` — ``k`` is the basis index,
        # ``arg`` is the ζ-coordinate.  ``_basis`` back-reference lets
        # ``EvaluateIntegrals`` route integrals to this basis without
        # any registry, and the single-class design makes the Sum-based
        # ``Expand`` representation natural:
        #
        #     Sum(amp(k) · basis.phi_fn(k, ζ), (k, 0, L))
        self.phi_fn = type(
            symbol,
            (sympy.Function,),
            {"_basis": self, "nargs": 2},
        )
        # Backwards-compat per-index accessor ``self.phi[k](arg)``.
        # Each entry is a thin lambda that pins the integer index.
        self.phi = [
            (lambda arg, _k=k, _fn=self.phi_fn: _fn(_k, arg))
            for k in range(level + 1)
        ]
        # ── ONE opaque weight Function head ``c(ζ)`` (1-arg), the counterpart
        # of ``phi_fn`` for the test weight — so a derivation can build its
        # brackets ``⟨c φ_i φ_l⟩`` from THIS basis's heads and resolve them
        # against the same instance.  ``_is_basis_head`` marks both heads so
        # :meth:`resolve` recognises them structurally.
        self.phi_fn._is_basis_head = True
        self.c_fn = type(
            self.weight_name,
            (sympy.Function,),
            {"_basis": self, "_is_basis_head": True, "nargs": 1},
        )
        # Per-instance memo of evaluated brackets (antiderivative results).
        self._bracket_cache: dict = {}

    # ``weight_name`` is the name of the opaque test-weight Function family
    # ``c(ζ)`` the derivation framework's brackets carry.  Default ``"c"``;
    # the modal-projection brackets render ``⟨φ_i, c φ_l⟩`` etc.
    weight_name = "c"

    def closed_form_bracket(self, name, args):
        """Return the closed form of a named bracket (``"Gram"`` / ``"Weight"``) in symbolic
        indices, or ``None``. The base class has none; orthogonal bases override it, e.g.
        :class:`Legendre_shifted`.

        Args:
            name: bracket family name.
            args: its index tuple.
        """
        return None

    def resolve_atoms(self, expr):
        """Replace each ``phi_fn(k, arg)`` with integer ``k`` by the polynomial ``eval(k, arg)``;
        symbolic ``k`` stays opaque. Atoms of other bases are untouched.
        """
        def _replace(k, z_arg):
            """Evaluate one ``phi_fn`` atom if its index is an integer."""
            if k.is_Integer:
                return self.eval(int(k), z_arg)
            # Symbolic index → leave opaque (un-doited Sum).
            return self.phi_fn(k, z_arg)

        return expr.replace(self.phi_fn, _replace)

    # ── Galerkin-bracket resolution: ⟨…⟩ → number (antiderivative + cache) ──

    def _concretize_bracket(self, body):
        """Replace opaque ``phi`` / ``c`` heads (integer index) in a bracket body by this basis's
        polynomials and weight, leaving numerical-quadrature brackets opaque.
        """
        def _is_phi(e):
            """Match a two-argument basis head."""
            return (isinstance(e, sympy.Function)
                    and getattr(e.func, "_is_basis_head", False)
                    and len(e.args) == 2)

        def _is_c(e):
            """Match a one-argument basis head."""
            return (isinstance(e, sympy.Function)
                    and getattr(e.func, "_is_basis_head", False)
                    and len(e.args) == 1)

        # Protect any opaque numerical bracket ``⟨…⟩^N`` (``_is_numquad``): its φ
        # are resolved at the Gauss nodes by ResolveNumQuad, NOT concretised here
        # (concretising them would defeat the whole point — the rational closure
        # would re-acquire its concrete-basis symbolic form and explode).
        _nqmap = {}

        def _protect_nq(e):
            """Swap numerical-quadrature brackets for Dummies (restored by the caller)."""
            if isinstance(e, sympy.Function) and getattr(e.func, "_is_numquad", False):
                d = sympy.Dummy("_NQ")
                _nqmap[d] = e
                return d
            if e.args:
                na = tuple(_protect_nq(a) for a in e.args)
                if any(n is not o for n, o in zip(na, e.args)):
                    return e.func(*na)
            return e
        body = _protect_nq(body)

        body = body.replace(
            _is_phi, lambda e: (self.eval(int(e.args[0]), e.args[1])
                                if e.args[0].is_Integer else e))
        body = body.replace(_is_c, lambda e: sympy.sympify(self.weight(e.args[0])))
        if _nqmap:
            body = body.xreplace(_nqmap)
        return body

    def evaluate_bracket(self, body, var, lower=0, upper=1):
        """Evaluate one Galerkin bracket ``int_lower^upper body d(var)`` to a number by concretising
        and using the antiderivative; memoised per instance.

        Args:
            body: product of opaque ``phi`` / ``c`` heads, their derivatives and nested running
                integrals.
            var: integration variable.
        """
        key = (sympy.sympify(body), var, lower, upper)
        hit = self._bracket_cache.get(key)
        if hit is not None:
            return hit
        concrete = sympy.expand(self._concretize_bracket(sympy.sympify(body)).doit())
        val = sympy.simplify(
            self.analytical_weighted_integral_bounds(concrete, var, lower, upper))
        self._bracket_cache[key] = val
        return val

    def analytical_weighted_integral_bounds(self, poly_expr, var, lower, upper):
        """Return ``int_lower^upper poly_expr d(var)`` by antiderivative."""
        try:
            anti = sympy.Poly(poly_expr, var).integrate().as_expr()
        except (sympy.GeneratorsNeeded, sympy.PolynomialError):
            anti = sympy.integrate(poly_expr, var)
        return anti.subs(var, upper) - anti.subs(var, lower)

    def resolve(self, expr, var):
        """Resolve every Galerkin bracket in ``expr`` to a number against this basis (closed forms
        for ``Gram`` / ``Weight``, :meth:`evaluate_bracket` otherwise). Only inserts values: outer
        derivatives are not evaluated and deferred integrals stay compact.

        Args:
            expr: sympy expression containing brackets.
            var: bracket integration variable.
        """
        from zoomy_core.model.operations import is_bracket

        def _named(e):
            """Return the closed form of a named bracket, else the bracket itself."""
            cf = self.closed_form_bracket(e.func.__name__, tuple(e.args))
            return cf if cf is not None else e

        expr = expr.replace(
            lambda e: (isinstance(e, sympy.Function)
                       and getattr(e.func, "_is_bracket", False)),
            _named)

        def _opaque(ig):
            """Evaluate one opaque bracket integral, leaving other integrals untouched."""
            if not is_bracket(ig):
                return ig
            v, lo, hi = ig.limits[0]
            return self.evaluate_bracket(ig.function, v, lo, hi)

        expr = expr.replace(lambda e: isinstance(e, sympy.Integral), _opaque)
        # Concretise any LOOSE φ/c left outside a bracket — the boundary
        # evaluations ``φ_i(0)``, ``c(0)`` of the wall-friction term (``φ_i(0) =
        # P_i(−1) = (−1)^i``).
        expr = self._concretize_bracket(expr)
        # Collapse only the δ-``Sum``s produced by the closed-form Gram/Weight
        # (the symbolic-``l`` case).  Deliberately NOT a bare ``expr.doit()``:
        # that would evaluate the OUTER spatial derivatives too, product-ruling
        # ``∂_x(h·a²) → a²∂_x h + 2 a h ∂_x a`` and destroying the conservative
        # structure — resolution must only INSERT VALUES, never restructure.
        expr = expr.replace(lambda e: isinstance(e, sympy.Sum),
                            lambda e: e.doit())
        # Expand ONLY outside Integral atoms.  The expand here exists solely to
        # surface the ``Derivative(const, v)`` zombies dropped just below; it must
        # NOT distribute the integrand of a deferred rational integral
        # ``∫ C_μ sk⁴/se² · … dζ`` (a non-analytic closure term left for
        # GaussQuadrature) — a plain expand fragments that into a >1M-term monster
        # that then re-expands all down the pipeline (hours of build).  Protect
        # every Integral as a Dummy, expand the rest, restore — so the rational
        # integral stays a COMPACT atom and GaussQuadrature evaluates it at nodes.
        _intmap = {}

        def _protect_integrals(e):
            """Swap Integrals and numerical-quadrature brackets for Dummies (restored by the
            caller).
            """
            if isinstance(e, sympy.Integral):
                k = sympy.Dummy("_INT")
                _intmap[k] = e
                return k
            # Opaque numerical bracket ⟨…⟩^N: protect like an Integral so expand
            # never recurses into (and fragments) the rational closure body.
            if isinstance(e, sympy.Function) and getattr(e.func, "_is_numquad", False):
                k = sympy.Dummy("_NQ")
                _intmap[k] = e
                return k
            if e.args:
                na = tuple(_protect_integrals(a) for a in e.args)
                if any(n is not o for n, o in zip(na, e.args)):
                    return e.func(*na)
            return e
        expr = sympy.expand(_protect_integrals(expr))
        if _intmap:
            expr = expr.xreplace(_intmap)
        # A basis bracket / mode that resolves to a CONSTANT inside ``∂_v(…)`` —
        # a vanishing Gram/Weight (``0``) or the constant 0th mode ``φ_0 = 1`` —
        # leaves an unevaluated ``Derivative(const, v)``: ``replace`` rebuilds the
        # derivative wrapper around the new constant without collapsing it, and
        # ``expand`` does not touch it.  ``∂_v(const) ≡ 0`` for ANY constant, so
        # drop them all (``Derivative(0, t)`` AND ``Derivative(1, ζ̂)``) — otherwise
        # zombie ``2·∂_t(0)`` / ``∫ w̃·∂_ζ(1)`` terms would reach SystemModel.
        expr = expr.replace(
            lambda e: (isinstance(e, sympy.Derivative)
                       and getattr(e.expr, "is_number", False)),
            lambda e: sympy.S.Zero)
        return expr

    def get(self, k):
        """Return the polynomial of basis function ``k``."""
        return self.basis[k]

    def gram(self, i, j):
        """Return the Galerkin mass-matrix entry ``<phi_i, phi_j> = int omega phi_i phi_j dz``, from
        the closed form if the basis has one.

        Args:
            i: row index.
            j: column index.
        """
        cf = self.closed_form_bracket("Gram", (i, j))
        if cf is not None:
            return sympy.sympify(cf)
        return self.evaluate_bracket(
            self.weight(z) * self.phi_fn(i, z) * self.phi_fn(j, z),
            z, *self.bounds())

    def eval(self, k, _z):
        """Evaluate basis function ``k`` at ``_z``."""
        return self.get(k).subs(z, _z)

    def at0(self, k):
        """Return basis function ``k`` at the bottom of the layer, ``eval(k, bounds[0])``."""
        return self.eval(k, self.bounds()[0])

    def at1(self, k):
        """Return basis function ``k`` at the free surface, ``eval(k, bounds[1])``."""
        return self.eval(k, self.bounds()[1])

    def eval_psi(self, k, _z):
        """Evaluate the running integral of basis function ``k`` from the lower bound to ``_z``."""
        z = sympy.Symbol('z')
        psi = sympy.integrate(self.get(k), (z, self.bounds()[0], z))
        return psi.subs(z, _z)

    def get_lambda(self, k):
        """Return a numeric callable of basis function ``k`` accepting scalars, lists or arrays."""
        f = lambdify(z, self.get(k))

        def lam(z):
            """Evaluate the lambdified polynomial on a scalar or array."""
            if type(z) == int or type(z) == float:
                return f(z)
            elif type(z) == list or type(z) == np.ndarray:
                return np.array([f(xi) for xi in z])
            else:
                assert False

        return lam

    def plot(self, ax):
        """Plot every basis function on ``bounds`` into the matplotlib axes ``ax``."""
        X = np.linspace(self.bounds()[0], self.bounds()[1], 1000)
        for i in range(len(self.basis)):
            f = lambdify(z, self.get(i))
            y = np.array([f(xi) for xi in X])
            ax.plot(X, y, label=f"basis {i}")

    def reconstruct_velocity_profile(self, alpha, N=100):
        """Return ``sum(alpha_i phi_i)`` on ``N`` equispaced points of ``bounds``."""
        Z = np.linspace(self.bounds()[0], self.bounds()[1], N)
        u = np.zeros_like(Z)
        for i in range(len(self.basis)):
            b = lambdify(z, self.get(i))
            u[:] += alpha[i] * b(Z)
        return u
    
    def reconstruct_velocity_profile_at(self, alpha, z):
        """Return ``sum(alpha_i phi_i)`` at the point ``z``."""
        u = 0
        for i in range(len(self.basis)):
            b = lambdify(z, self.eval(i, z))
            u += alpha[i] * b(z)
        return u

    def reconstruct_alpha(self, velocities, z):
        """Return per-mode coefficients of sampled ``velocities`` at points ``z`` by trapezoid
        quadrature (assumes an orthogonal basis).
        """
        n_basis = len(self.basis)
        alpha = np.zeros(n_basis)
        for i in range(n_basis):
            b = lambdify(z, self.eval(i, z))
            nom = np.trapz(velocities * b(z) * self.weight(z), z)
            if type(b(z)) == int:
                den = b(z) ** 2
            else:
                den = np.trapz((b(z) * b(z)).reshape(z.shape), z)
            res = nom / den
            alpha[i] = res
        return alpha
    
    def project_onto_basis(self, Y):
        """Return the unnormalised projections ``int Y phi_k omega dz`` of the samples ``Y``
        (equispaced on ``bounds``) by trapezoid quadrature.
        """
        Z = np.linspace(self.bounds()[0], self.bounds()[1], Y.shape[0])
        n_basis = len(self.basis)
        alpha = np.zeros(n_basis)
        z = Symbol("z")
        for i in range(n_basis):
            # b = lambdify(z, self.eval(i, Z))
            b = self.get_lambda(i)
            alpha[i] = np.trapz(Y * b(Z) * self.weight_eval(Z), Z)
        return alpha

    def projection_rows(self, nodes, weights, samples, norm=None):
        """Return one symbolic row per basis function that projects a profile sampled at fixed nodes
        onto the basis (discrete least squares, full inverse Gram matrix). Each row is plain
        arithmetic in ``samples``, so every code printer can lower it.

        Args:
            nodes: sample points ``zeta_j``.
            weights: quadrature weights ``w_j`` (same length).
            samples: input symbols ``field(zeta_j)`` (same length).
            norm: optional per-row scale, a callable of ``k`` or a sequence.

        Returns:
            List of ``len(basis)`` sympy rows.
        """
        if not (len(nodes) == len(weights) == len(samples)):
            raise ValueError(
                "projection_rows: nodes, weights, samples must be equal length "
                f"(got {len(nodes)}, {len(weights)}, {len(samples)})")
        n = len(nodes)
        nb = len(self.basis)
        wj = [weights[j] * self.weight(nodes[j]) for j in range(n)]      # w_j·ω
        phi = [[self.eval(k, nodes[j]) for j in range(n)]               # φ_k(ζ_j)
               for k in range(nb)]
        # Discrete Gram G[k,i] = Σ_j w_j·ω(ζ_j)·φ_k(ζ_j)·φ_i(ζ_j); its inverse
        # is the EXACT least-squares inverse of basis evaluation.  Numeric
        # (nodes/weights are concrete) — cond is tiny for the column sizes used,
        # and the rows it produces stay plain arithmetic in ``samples``.
        Gnum = np.array(
            [[float(sum(wj[j] * phi[k][j] * phi[i][j] for j in range(n)))
              for i in range(nb)] for k in range(nb)])
        Ginv = np.linalg.inv(Gnum)
        rows = []
        for k in range(nb):
            # row_k = Σ_i G⁻¹[k,i]·(Σ_j w_j·φ_i(ζ_j)·sample_j)
            #       = Σ_j (Σ_i G⁻¹[k,i]·φ_i(ζ_j))·w_j·sample_j
            coeff = [wj[j] * sum(float(Ginv[k, i]) * phi[i][j]
                                 for i in range(nb)) for j in range(n)]
            row = sum(coeff[j] * samples[j] for j in range(n))
            if norm is not None:
                nk = norm(k) if callable(norm) else norm[k]
                row = nk * row
            rows.append(sympy.expand(row))
        return rows

    def get_diff_basis(self):
        """Return the ``z``-derivatives of all basis polynomials."""
        db = [diff(b, z) for i, b in enumerate(self.basis)]
        return db


class Monomials(Basisfunction):
    """Monomial basis ``z**k`` (the base-class default)."""
    name = "Monomials"


class LayeredBasis(Basisfunction):
    """Multi-layer basis: an inner basis rescaled onto each of ``N`` sub-intervals and zero outside,
    giving ``N * (inner_level + 1)`` functions with flat index ``i*m + k``. ``inner_level=0`` gives
    piecewise constants (multi-layer SWE), ``Legendre_shifted`` multi-layer moments.

    Args:
        inner_cls: per-layer Basisfunction subclass.
        inner_level: level of the inner basis.
        interfaces: explicit breakpoints (``N + 1``), in zeta or physical z.
        n_layers: number of equal layers on ``[0, 1]`` when ``interfaces`` is not given.

    References:
        Audusse et al. 2011, https://doi.org/10.1051/m2an/2010036
    """

    name = "LayeredBasis"

    def __init__(self, inner_cls=None, inner_level=0, interfaces=None,
                 n_layers=None, **inner_kwargs):
        """Build the inner basis and the layered piecewise polynomials."""
        if inner_cls is None:
            inner_cls = Monomials
        if interfaces is None:
            if n_layers is None:
                raise ValueError(
                    "LayeredBasis requires either ``interfaces`` (explicit "
                    "breakpoints) or ``n_layers`` (uniform partition of [0, 1])."
                )
            interfaces = [sympy.Rational(i, n_layers) for i in range(n_layers + 1)]
        if len(interfaces) < 2:
            raise ValueError("LayeredBasis needs at least two breakpoints.")
        self.interfaces = list(interfaces)
        self.n_layers = len(self.interfaces) - 1
        self.inner = inner_cls(level=inner_level, **inner_kwargs)
        self.inner_n_basis = inner_level + 1
        # n_basis = level + 1 = N * m  (keeps the Legendre convention)
        self.level = self.n_layers * self.inner_n_basis - 1
        self.basis = self.basis_definition()

    def basis_definition(self):
        """Return the inner functions rescaled to each layer as Piecewise, zero outside the layer."""
        from sympy import Piecewise
        z_sym = Symbol("z")
        out = []
        for i in range(self.n_layers):
            lo, hi = self.interfaces[i], self.interfaces[i + 1]
            h_i = hi - lo
            zeta_i = (z_sym - lo) / h_i
            for k in range(self.inner_n_basis):
                phi_inner = self.inner.eval(k, zeta_i)
                out.append(Piecewise(
                    (phi_inner, (z_sym > lo) & (z_sym < hi)),
                    (0, True),
                ))
        return out

    def bounds(self):
        """Return the outer interfaces ``[first, last]``."""
        return [self.interfaces[0], self.interfaces[-1]]

    def weight(self, z):
        """Return the inner-basis weight."""
        # Weight per layer follows the inner basis; outside we treat
        # it as 1 (weight only matters inside the layer's own integral).
        return self.inner.weight(z) if hasattr(self.inner, "weight") else 1

    def mean_coefficients(self):
        """Return the inner mean coefficients repeated for every layer."""
        cm = self.inner.mean_coefficients()
        out = []
        for _ in range(self.n_layers):
            out.extend(cm)
        return out

    # Convenience accessors for layer / moment indexing --------------
    def flat_index(self, layer_idx, moment_idx):
        """Map ``(layer_idx, moment_idx)`` to the flat basis index."""
        return layer_idx * self.inner_n_basis + moment_idx

    def layer_zeta(self, layer_idx, z_value):
        """Return the local coordinate ``(z_value - lower_i) / h_i`` in layer ``layer_idx``."""
        lo = self.interfaces[layer_idx]
        hi = self.interfaces[layer_idx + 1]
        return (z_value - lo) / (hi - lo)


class Legendre_shifted(Basisfunction):
    """Shifted Legendre basis ``phi_k(z) = P_k(2z - 1)`` on ``[0, 1]``, orthogonal under unit
    weight; the standard SME moment basis.

    References:
        Kowalski, Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """
    name = "Legendre_shifted"

    def basis_definition(self):
        """Return ``legendre(k, 2z - 1)``, so ``phi_k(0) = (-1)^k`` at the bed and ``phi_k(1) = 1``
        at the free surface.
        """
        z = Symbol("z")
        return [legendre(k, 2 * z - 1) for k in range(self.level + 1)]

    def analytical_weighted_integral(self, poly_expr, var):
        """Return ``int_0^1 poly_expr d(var)`` (unit weight) by antiderivative."""
        anti = sympy.integrate(poly_expr, var)
        return anti.subs(var, 1) - anti.subs(var, 0)

    def closed_form_bracket(self, name, args):
        """Return ``Gram(i, l) = delta_il / (2l + 1)`` and ``Weight(l) = delta_l0``, or ``None`` for
        any other bracket.
        """
        if name == "Gram" and len(args) == 2:
            i, l = args
            return sympy.KroneckerDelta(i, l) / (2 * l + 1)
        if name == "Weight" and len(args) == 1:
            (l,) = args
            return sympy.KroneckerDelta(l, 0)
        return None
    
class WeightedBasis(Basisfunction):
    """Wrap an orthogonal basis with an explicit test weight (e.g. a curvilinear metric), so
    projections resolve concretely. It has no delta closed forms because the wrapped family is
    orthogonal only under unit weight.

    Args:
        inner: wrapped Basisfunction.
        weight: weight expression in the integration variable.
        level: basis level (default: that of ``inner``).
    """
    name = "WeightedBasis"

    def __init__(self, inner, weight, level=None):
        """Store the wrapped basis and the weight."""
        self._inner = inner
        self._weight = weight
        super().__init__(level=level if level is not None else inner.level)

    def basis_definition(self):
        """Return the wrapped basis polynomials unchanged."""
        return self._inner.basis_definition()

    def bounds(self):
        """Return the bounds of the wrapped basis."""
        return self._inner.bounds()

    @property
    def weight(self):
        """Return the geometric weight."""
        return self._weight

    def analytical_weighted_integral(self, poly_expr, var):
        """Return ``int_0^1 poly_expr * weight d(var)`` by antiderivative."""
        w = self._weight
        if hasattr(w, "subs"):
            w = w.subs(Symbol("z"), var) if Symbol("z") in getattr(w, "free_symbols", set()) else w
        anti = sympy.integrate(sympy.expand(poly_expr * w), var)
        return sympy.expand(anti.subs(var, 1) - anti.subs(var, 0))

    def closed_form_bracket(self, name, args):
        """Return ``None``: no delta closed form under a non-unit weight."""
        return None


class Chebyshevu(Basisfunction):
    """Chebyshev-U basis on ``[-1, 1]`` with weight ``sqrt(1 - z**2)`` and normalisation
    ``sqrt(2/pi)``.
    """
    name = "Chebyshevu"
    
    def bounds(self):
        """Return ``[-1, 1]``."""
        return [-1, 1]
    
    def weight(self, z):
        # do not forget to include the jacobian of the coordinate transformation in the weight
        """Return ``sqrt(1 - z**2)``."""
        return sympy.sqrt(1-z**2)

    def basis_definition(self):
        """Return ``sqrt(2/pi) * U_k(z)``."""
        z = Symbol("z")
        b = lambda k, z: sympy.sqrt(2 / sympy.pi) * chebyshevu(k, z)
        return [b(k, z) for k in range(self.level + 1)]
    
class Chebyshevu_shifted(Basisfunction):
    """Chebyshev-U basis ``U_k(2z - 1)`` on ``[0, 1]`` with weight ``sqrt(z(1 - z))`` and ``phi_0 =
    1``, orthogonal with norm ``pi/8``.
    """
    name = "Chebyshevu_shifted"

    def bounds(self):
        """Return ``[0, 1]``."""
        return [0, 1]

    def weight(self, z):
        """Return ``sqrt(z (1 - z))``."""
        return sympy.sqrt(z * (1 - z))

    def basis_definition(self):
        """Return ``U_k(2z - 1)``."""
        z = Symbol("z")
        return [chebyshevu(k, 2 * z - 1) for k in range(self.level + 1)]

    # Cache for the Vandermonde inverse, keyed by polynomial degree
    _vandermonde_cache: dict = {}

    def _u_vandermonde_inv(self, deg):
        """Return the cached inverse of the monomial-to-``U_k(2z - 1)`` matrix for degree ``deg``."""
        if deg in self._vandermonde_cache:
            return self._vandermonde_cache[deg]
        z_sym = Symbol("z")
        n = deg + 1
        # V[power, k] = coefficient of z^power in U_k(2z-1)
        U_polys = [sympy.Poly(chebyshevu(k, 2 * z_sym - 1), z_sym) for k in range(n)]
        V = sympy.Matrix(n, n, lambda i, j: U_polys[j].nth(i))
        Vinv = V.inv()
        self._vandermonde_cache[deg] = Vinv
        return Vinv

    def analytical_weighted_integral(self, poly_expr, var):
        """Return ``int_0^1 poly_expr * sqrt(z(1 - z)) d(var)`` via U-orthogonality, or ``None`` if
        ``poly_expr`` is not a polynomial.
        """
        expanded = sympy.expand(poly_expr)
        try:
            p = sympy.Poly(expanded, var)
        except (sympy.GeneratorsNeeded, sympy.PolynomialError):
            return None  # not a polynomial

        deg = p.degree()
        if deg < 0:
            return sympy.Integer(0)

        Vinv = self._u_vandermonde_inv(deg)
        n = deg + 1

        # Monomial coefficient vector
        p_vec = sympy.Matrix(n, 1, lambda i, _: p.nth(i))

        # U-expansion coefficients: c = Vinv @ p_vec
        c = Vinv * p_vec

        return c[0] * sympy.pi / 8

    def quadrature_nodes(self, n=None):
        """Return Gauss-Chebyshev-U nodes and weights on ``[0, 1]`` for the weight ``sqrt(z(1 -
        z))``.

        Args:
            n: node count minus one; default ``2 * (level + 1)``.

        Returns:
            Tuple ``(nodes, weights)`` of arrays.

        References:
            Golub, Welsch 1969, https://doi.org/10.1090/s0025-5718-69-99647-1
        """
        if n is None:
            n = 2 * (self.level + 1)
        import numpy as np
        k = np.arange(1, n + 2)
        nodes_ref = np.cos(k * np.pi / (n + 2))
        weights_ref = np.pi / (n + 2) * np.sin(k * np.pi / (n + 2))**2
        nodes_01 = 0.5 * (1 + nodes_ref)
        weights_01 = weights_ref / 4
        return nodes_01, weights_01


class Legendre_DN(Basisfunction):
    """Legendre-based basis on ``[-1, 1]`` satisfying no-slip at the bottom ``z = -1`` and no-stress
    at the top ``z = 1`` by construction; not an SWE-limit basis.
    """
    name = "Legendre_DN - satifying no-slip and no-stress. This is a non-SWE basis"

    def bounds(self):
        """Return ``[-1, 1]``."""
        return [-1, 1]

    def basis_definition(self):
        """Return ``P_k + alpha_k P_{k+1} + beta_k P_{k+2}`` with fixed rational ``alpha_k``,
        ``beta_k``.
        """
        z = Symbol("z")
        def b(k, z):
            """Return the ``k``-th recombined Legendre polynomial."""
            alpha = sympy.Rational((2*k+3), (k+2)**2)
            beta = -sympy.Rational((k+1),(k+2))**2
            return (legendre(k, z) ) + alpha * (legendre(k+1, z) ) + beta * (legendre(k+2, z))
        #normalizing makes no sence, as b(k, 0) = 0 by construction
        return [b(k, z) for k in range(self.level + 1)]


class GalerkinBasis(Basisfunction):
    """Boundary-condition-aware basis: ``phi_0 = 1`` and ``phi_k`` recombined from parent
    polynomials (Legendre or Chebyshev-U) to satisfy 'free', 'noslip', 'nostress' or 'slip'
    conditions at bottom and top.

    Args:
        level: highest basis index.
        parent: parent family, ``'legendre'`` or ``'chebyshev'``.
        bc_bottom: condition at ``z = 0``.
        bc_top: condition at ``z = 1``.
        slip_length: slip length for ``'slip'`` (symbolic ``slip_length`` if None).

    References:
        Shen 1994, https://doi.org/10.1137/0915089
    """
    name = "GalerkinBasis"

    def __init__(self, level=0, parent="legendre", bc_bottom="slip", bc_top="nostress",
                 slip_length=None, **kwargs):
        """Store the parent family and boundary conditions, then build the basis."""
        self._parent = parent
        self._bc_bottom = bc_bottom
        self._bc_top = bc_top
        self._slip_length = slip_length
        super().__init__(level=level, **kwargs)
        self.name = f"Galerkin_{parent}_{bc_bottom}_{bc_top}"

    def bounds(self):
        """Return ``[0, 1]``."""
        return [0, 1]

    def weight(self, z):
        """Return the unit weight."""
        return sympy.Integer(1)

    def _parent_poly(self, k, z):
        """Return the ``k``-th parent polynomial on ``[0, 1]``."""
        if self._parent == "legendre":
            return legendre(k, 2 * z - 1) * (-1) ** k
        elif self._parent == "chebyshev":
            return chebyshevu(k, 2 * z - 1)
        else:
            raise ValueError(f"Unknown parent polynomial: {self._parent}")

    def _bc_equation(self, poly, z_bc, bc_type):
        """Return the residual of the boundary condition ``bc_type`` at ``z_bc`` for ``poly``
        (``None`` for 'free').
        """
        z = Symbol("z")
        val = poly.subs(z, z_bc)
        deriv_val = diff(poly, z).subs(z, z_bc)

        if bc_type == "noslip":
            return val  # phi(z_bc) = 0
        elif bc_type == "nostress":
            return deriv_val  # phi'(z_bc) = 0
        elif bc_type == "slip":
            lam = self._slip_length if self._slip_length is not None else sympy.Symbol("slip_length")
            return deriv_val - val / lam  # phi'(z_bc) = phi(z_bc) / lambda
        elif bc_type == "free":
            return None  # no constraint
        else:
            raise ValueError(f"Unknown BC type: {bc_type}")

    def basis_definition(self):
        """Return ``[1, phi_1, ...]`` with each ``phi_k`` solved from the linear boundary conditions
        (one condition: two parents, two conditions: three).
        """
        z = Symbol("z")
        basis = [sympy.Integer(1)]  # phi_0 = 1 always

        z_bot = self.bounds()[0]
        z_top = self.bounds()[1]

        # Count constraints
        constraints = []
        if self._bc_bottom != "free":
            constraints.append(("bottom", z_bot, self._bc_bottom))
        if self._bc_top != "free":
            constraints.append(("top", z_top, self._bc_top))

        n_constraints = len(constraints)

        for k in range(1, self.level + 1):
            if n_constraints == 0:
                basis.append(self._parent_poly(k, z))
            elif n_constraints == 1:
                # Combine P_k and P_{k+1} to satisfy 1 BC
                P_k = self._parent_poly(k, z)
                P_k1 = self._parent_poly(k + 1, z)
                _, z_bc, bc_type = constraints[0]
                c_k = self._bc_equation(P_k, z_bc, bc_type)
                c_k1 = self._bc_equation(P_k1, z_bc, bc_type)
                if c_k1 == 0:
                    basis.append(P_k)
                else:
                    alpha = -c_k / c_k1
                    basis.append(sympy.simplify(P_k + alpha * P_k1))
            elif n_constraints == 2:
                # Combine P_k, P_{k+1}, P_{k+2} to satisfy 2 BCs
                P_k = self._parent_poly(k, z)
                P_k1 = self._parent_poly(k + 1, z)
                P_k2 = self._parent_poly(k + 2, z)

                _, z_bc0, bc0 = constraints[0]
                _, z_bc1, bc1 = constraints[1]

                c0_k = self._bc_equation(P_k, z_bc0, bc0)
                c0_k1 = self._bc_equation(P_k1, z_bc0, bc0)
                c0_k2 = self._bc_equation(P_k2, z_bc0, bc0)

                c1_k = self._bc_equation(P_k, z_bc1, bc1)
                c1_k1 = self._bc_equation(P_k1, z_bc1, bc1)
                c1_k2 = self._bc_equation(P_k2, z_bc1, bc1)

                A = sympy.Matrix([[c0_k1, c0_k2], [c1_k1, c1_k2]])
                b_vec = sympy.Matrix([-c0_k, -c1_k])

                try:
                    coeffs = A.solve(b_vec)
                    alpha = coeffs[0]
                    beta = coeffs[1]
                    phi_k = sympy.simplify(P_k + alpha * P_k1 + beta * P_k2)
                    basis.append(phi_k)
                except Exception:
                    basis.append(P_k)

        return basis


class SplineBasis(Basisfunction):
    """Raw B-spline hat-function basis on ``[0, 1]`` with nodal coefficients and partition of unity
    (``sum B_k = 1``). Provides :meth:`get_knot_spans` for piecewise integration.

    Args:
        level: highest basis index.
        degree: spline degree.
    """
    name = "SplineBasis"

    def __init__(self, level=0, degree=1, **kwargs):
        """Store the spline degree, then build the basis."""
        self._degree = degree
        super().__init__(level=level, **kwargs)

    def bounds(self):
        """Return ``[0, 1]``."""
        return [0, 1]

    def weight(self, z):
        """Return the unit weight."""
        return sympy.Integer(1)

    def _make_knots(self):
        """Return the clamped knot vector for ``level + 1`` functions of the given degree."""
        n = self.level + 1
        n_internal = max(n - self._degree, 1)
        internal = [sympy.Rational(i, n_internal) for i in range(n_internal + 1)]
        knots = [sympy.Rational(0)] * (self._degree + 1) + internal[1:-1] + [sympy.Rational(1)] * (self._degree + 1)
        return knots

    def basis_definition(self):
        """Return the first ``level + 1`` B-splines, padded with zeros."""
        z_sym = Symbol("z")
        n = self.level + 1
        knots = self._make_knots()

        raw_basis = bspline_basis_set(self._degree, knots, z_sym)

        # Use raw B-splines directly — phi_0 = B_0 (hat at bottom)
        result = list(raw_basis[:n])

        while len(result) < n:
            result.append(sympy.Integer(0))
        return result[:n]

    def get_knot_spans(self):
        """Return the ``(a, b)`` intervals between consecutive distinct knots."""
        knots = self._make_knots()
        unique_knots = sorted(set(float(k) for k in knots))
        spans = []
        for i in range(len(unique_knots) - 1):
            a, b = unique_knots[i], unique_knots[i + 1]
            if b > a:
                spans.append((sympy.Rational(a).limit_denominator(10000),
                              sympy.Rational(b).limit_denominator(10000)))
        return spans

    def mean_coefficients(self):
        """Return all ones, since B-splines are a partition of unity."""
        return [sympy.Integer(1)] * (self.level + 1)


class Spline(Basisfunction):
    """Legacy B-spline basis with fixed knots; use :class:`SplineBasis` instead."""
    name = "Spline"

    def basis_definition(self, degree=1, knots=[0, 0, 0.001, 1, 1]):
        """Return the B-spline set of the given degree and knots."""
        z = Symbol("z")
        basis = bspline_basis_set(degree, knots, z)
        return basis


class OrthogonalSplineWithConstant(Basisfunction):
    """B-spline basis plus a constant, orthonormalised on ``[0, 1]`` by Gram-Schmidt."""
    name = "OrthogonalSplineWithConstant"

    def basis_definition(self, degree=1, knots=[0, 0, 0.5, 1, 1]):
        """Return the Gram-Schmidt orthonormalised basis ``[1] + B-splines[:-1]``."""
        z = Symbol("z")

        def prod(u, v):
            """Return the ``L2(0, 1)`` inner product of ``u`` and ``v``."""
            return integrate(u * v, (z, 0, 1))

        basis = bspline_basis_set(degree, knots, z)
        add_basis = [1]
        # add_basis = [sympy.Piecewise((0, z<0.1), (1, True))]
        basis = add_basis + basis[:-1]
        orth = deepcopy(basis)
        for i in range(1, len(orth)):
            for j in range(0, i):
                orth[i] -= prod(basis[i], orth[j]) / prod(orth[j], orth[j]) * orth[j]

        for i in range(len(orth)):
            orth[i] /= sympy.sqrt(prod(orth[i], orth[i]))

        return orth
