"""Exact symbolic integration of basis-function products over the basis domain, with strategy
dispatch (sympy, knot spans, orthogonality, quadrature). Used by the basis matrices (mass, triple
product, derivative products) of the Galerkin projection.
"""

import warnings
import numpy as np
import sympy
from sympy import Symbol, integrate as sp_integrate, diff, Piecewise, Rational
from sympy.abc import z
from time import time as get_time


# Default timeout per single integral (seconds).  If an integral takes longer
# than this the integrator raises instead of hanging silently.
_DEFAULT_INTEGRAL_TIMEOUT = 30


class SymbolicIntegrator:
    """Integrates basis-function expressions symbolically; quadrature only when requested
    explicitly. A single integral exceeding `integral_timeout` seconds raises `TimeoutError`.
    """

    def __init__(self, basis, strategy="auto", n_workers=1,
                 integral_timeout=_DEFAULT_INTEGRAL_TIMEOUT):
        """Select the integration strategy for `basis`.

        Args:
            basis: Basisfunction providing `bounds()`, `weight(z)` and `eval(k, z)`.
            strategy: `'auto'` (knot spans, orthogonal or sympy by basis capability), `'sympy'`,
                `'knot_spans'` or `'quadrature'`.
            n_workers: Parallel workers for matrix computation (1 = serial).
            integral_timeout: Seconds allowed per integral; `None` disables the timeout.
        """
        self.basis = basis
        self.n_workers = n_workers
        self.integral_timeout = integral_timeout
        self._strategy = self._resolve_strategy(strategy)

    def _resolve_strategy(self, strategy):
        """Map `'auto'` to the best exact-symbolic strategy; quadrature is never chosen
        automatically.
        """
        if strategy != "auto":
            return strategy
        # Auto: pick the best exact-symbolic strategy for the basis.
        # Quadrature is never selected automatically — the user must opt in.
        if hasattr(self.basis, "get_knot_spans") and self.basis.get_knot_spans():
            return "knot_spans"
        if hasattr(self.basis, "analytical_weighted_integral"):
            return "orthogonal"
        return "sympy"

    @property
    def strategy(self):
        """The resolved strategy name."""
        return self._strategy

    # --- Core integration ---

    def integrate(self, expr, var, domain=None):
        """Integrate `expr` over `var` in `domain` (default: the basis bounds), exactly unless
        the strategy is `'quadrature'`.

        Returns:
            Symbolic value of the integral.
        """
        if domain is None:
            domain = tuple(self.basis.bounds())
        a, b = domain

        if self._strategy == "knot_spans":
            return self._integrate_knot_spans(expr, var, a, b)
        elif self._strategy == "orthogonal":
            return self._integrate_orthogonal(expr, var, a, b)
        elif self._strategy == "quadrature":
            warnings.warn(
                "Using numerical quadrature for integration. "
                "Results will contain approximate floats, not exact symbolics. "
                "Pass strategy='sympy' or strategy='auto' for exact results.",
                stacklevel=2,
            )
            return self._integrate_quadrature(expr, var, a, b)
        else:
            return self._integrate_sympy_timed(expr, var, a, b)

    def _integrate_sympy_timed(self, expr, var, a, b):
        """`sympy.integrate` under a SIGALRM wall-clock timeout."""
        if self.integral_timeout is None:
            return sp_integrate(expr, (var, a, b))

        import signal

        def _handler(signum, frame):
            """SIGALRM handler raising `TimeoutError`."""
            raise TimeoutError(
                f"Symbolic integration timed out after {self.integral_timeout}s. "
                f"The integrand may be too complex for exact symbolic evaluation. "
                f"Consider using strategy='knot_spans' (for splines) or "
                f"strategy='quadrature' (for numerical fallback).\n"
                f"Integrand: {str(expr)[:200]}"
            )

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(self.integral_timeout))
        try:
            result = sp_integrate(expr, (var, a, b))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result

    def _integrate_orthogonal(self, expr, var, a, b):
        """Integrate weight times polynomial by delegating the polynomial part to
        `basis.analytical_weighted_integral`, falling back to sympy.
        """
        w = self.basis.weight(var)
        # Factor out the weight: expr = w * polynomial_part
        # For safety, divide and simplify
        poly_part = sympy.cancel(expr / w)

        result = self.basis.analytical_weighted_integral(poly_part, var)
        if result is not None:
            return result

        # Fallback: the integrand isn't purely polynomial × weight.
        # Use timed sympy integration.
        warnings.warn(
            f"Orthogonal decomposition failed (integrand not polynomial × weight). "
            f"Falling back to sympy.integrate.",
            stacklevel=3,
        )
        return self._integrate_sympy_timed(expr, var, a, b)

    def _integrate_knot_spans(self, expr, var, a, b):
        """Integrate span by span, collapsing Piecewise to a polynomial in each span."""
        spans = self.basis.get_knot_spans()
        result = sympy.Integer(0)
        for span_a, span_b in spans:
            sa = max(float(span_a), float(a))
            sb = min(float(span_b), float(b))
            if sa >= sb:
                continue
            midpoint = (sa + sb) / 2
            collapsed = self._collapse_piecewise(expr, var, midpoint)
            # Fast path: antiderivative + evaluate at bounds
            ra = Rational(sa).limit_denominator(10000)
            rb = Rational(sb).limit_denominator(10000)
            anti = sympy.integrate(collapsed, var)
            chunk = anti.subs(var, rb) - anti.subs(var, ra)
            result += chunk
        return result

    def _collapse_piecewise(self, expr, var, point):
        """Evaluate Piecewise conditions at `point`, recursively, to get a plain expression."""
        if not isinstance(expr, sympy.Basic):
            return expr
        if isinstance(expr, Piecewise):
            for piece_expr, piece_cond in expr.args:
                if piece_cond == True or piece_cond.subs(var, point):
                    return self._collapse_piecewise(piece_expr, var, point)
            return sympy.Integer(0)
        if expr.args:
            new_args = [self._collapse_piecewise(a, var, point) for a in expr.args]
            return expr.func(*new_args)
        return expr

    def _integrate_quadrature(self, expr, var, a, b):
        """Numerical integration with the basis quadrature nodes, returned as a sympy number."""
        nodes, weights = self.basis.quadrature_nodes()
        # Filter to domain
        mask = (nodes >= float(a)) & (nodes <= float(b))
        nodes_in = nodes[mask]
        weights_in = weights[mask]

        from sympy import lambdify
        f = lambdify(var, expr, "numpy")
        try:
            vals = f(nodes_in)
            if np.isscalar(vals):
                vals = np.full_like(nodes_in, float(vals))
            result = float(np.sum(weights_in * vals))
            return sympy.nsimplify(result, tolerance=1e-12, rational=False)
        except Exception:
            # Fallback to sympy
            return sp_integrate(expr, (var, a, b))

    # --- Basis matrix computation ---

    def _precompute_basis(self, level):
        """Return the lists `phi_k`, `dphi_k` and antiderivatives `psi_k` for `k <= level`."""
        n = level + 1
        phi = [self.basis.eval(k, z) for k in range(n)]
        dphi = [diff(p, z) for p in phi]
        psi = [sympy.integrate(p, z) for p in phi]  # antiderivatives for B
        return phi, dphi, psi

    def mass_matrix(self, level):
        """`M[k,i] = int(phi_k phi_i w)` over the domain (symmetric)."""
        n = level + 1
        M = np.empty((n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())
        phi, _, _ = self._precompute_basis(level)
        for k in range(n):
            for i in range(k, n):
                val = self.integrate(w * phi[k] * phi[i], z, bounds)
                M[k, i] = val
                M[i, k] = val
        return M

    def triple_product(self, level):
        """`A[k,i,j] = int(phi_k phi_i phi_j w)` over the domain, fully symmetric in (k,i,j)."""
        n = level + 1
        A = np.empty((n, n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())
        phi, _, _ = self._precompute_basis(level)
        for k in range(n):
            for i in range(k, n):
                for j in range(i, n):
                    val = self.integrate(w * phi[k] * phi[i] * phi[j], z, bounds)
                    # Fill all permutations of (k, i, j)
                    for p in {(k,i,j),(k,j,i),(i,k,j),(i,j,k),(j,k,i),(j,i,k)}:
                        A[p] = val
        return A

    def derivative_product(self, level):
        """`D[k,i] = int(phi_k' phi_i' w)` over the domain (symmetric)."""
        n = level + 1
        D = np.empty((n, n), dtype=object)
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())
        _, dphi, _ = self._precompute_basis(level)
        for k in range(n):
            for i in range(k, n):
                val = self.integrate(w * dphi[k] * dphi[i], z, bounds)
                D[k, i] = val
                D[i, k] = val
        return D

    def boundary_values(self, level):
        """`phib[k] = phi_k` at the lower bound."""
        n = level + 1
        phib = np.empty(n, dtype=object)
        z_bot = self.basis.bounds()[0]
        for k in range(n):
            phib[k] = self.basis.eval(k, z_bot)
        return phib

    def compute_all_matrices(self, level):
        """All standard basis matrices (`phib, M, A, B, D, Dxi, Dxi2, DD, D1, DT`) for `level`,
        exploiting their symmetries.

        Returns:
            Dict of the matrices plus `_elapsed` and `_strategy`.
        """
        t0 = get_time()
        n = level + 1
        w = self.basis.weight(z)
        bounds = tuple(self.basis.bounds())
        phi, dphi, psi = self._precompute_basis(level)

        phib = self.boundary_values(level)
        M = self.mass_matrix(level)
        A = self.triple_product(level)
        D = self.derivative_product(level)

        # n^2 matrices (some symmetric)
        D1 = np.empty((n, n), dtype=object)
        Dxi = np.empty((n, n), dtype=object)
        Dxi2 = np.empty((n, n), dtype=object)
        DD = np.empty((n, n), dtype=object)

        for k in range(n):
            for i in range(n):
                D1[k, i] = self.integrate(w * phi[k] * dphi[i], z, bounds)
                DD[k, i] = self.integrate(w * phi[k] * diff(dphi[i], z), z, bounds)
            # Dxi, Dxi2: symmetric in (k,i) -> upper triangle
            for i in range(k, n):
                Dxi[k, i] = self.integrate(w * dphi[k] * dphi[i] * z, z, bounds)
                Dxi[i, k] = Dxi[k, i]
                Dxi2[k, i] = self.integrate(w * dphi[k] * dphi[i] * z * z, z, bounds)
                Dxi2[i, k] = Dxi2[k, i]

        # n^3 tensors
        B = np.empty((n, n, n), dtype=object)
        DT = np.empty((n, n, n), dtype=object)

        for k in range(n):
            # DT: symmetric in (k,i) -> only compute k <= i
            for i in range(k, n):
                for j in range(n):
                    val = self.integrate(w * dphi[k] * dphi[i] * phi[j], z, bounds)
                    DT[k, i, j] = val
                    DT[i, k, j] = val
            # B: no symmetry -> all entries
            for i in range(n):
                for j in range(n):
                    B[k, i, j] = self.integrate(
                        w * dphi[k] * psi[j] * phi[i], z, bounds)

        elapsed = get_time() - t0
        return {
            "phib": phib, "M": M, "A": A, "B": B, "D": D,
            "Dxi": Dxi, "Dxi2": Dxi2, "DD": DD, "D1": D1, "DT": DT,
            "_elapsed": elapsed, "_strategy": self._strategy,
        }

    def project(self, expr, test_index, weight=None):
        """Galerkin projection of `expr` onto `phi_test_index`: `int(expr phi_k w) / M[k,k]`."""
        if weight is None:
            weight = self.basis.weight(z)
        phi_k = self.basis.eval(test_index, z)
        bounds = tuple(self.basis.bounds())
        numerator = self.integrate(expr * phi_k * weight, z, bounds)
        denominator = self.integrate(phi_k * phi_k * weight, z, bounds)
        return numerator / denominator
