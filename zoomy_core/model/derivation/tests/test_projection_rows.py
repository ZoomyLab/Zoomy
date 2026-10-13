"""`Basisfunction.projection_rows` — the symbolic, fixed-node, hand-rolled
projection of a sampled profile onto the basis (the reusable core building
block behind `project_from_3d`).

Pins three properties:
1. each row is PLAIN ARITHMETIC in the input sample symbols — no `Integral`,
   no free symbols beyond `samples` — so every code printer lowers it for
   free;
2. with Gauss-Legendre nodes/weights the projection is the EXACT inverse of
   basis evaluation for a profile that lies in the basis (round-trip to
   machine precision);
3. with a uniform column + trapezoid weights it converges O(N_z^-2) — the
   realistic resolved-column (VOF) case.
"""
import numpy as np
import sympy as sp

from zoomy_core.model.derivation.basisfunctions import Legendre_shifted


def _gauss_01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return [float((xi + 1) / 2) for xi in x], [float(wi / 2) for wi in w]


def _trapezoid_01(nz):
    nodes = list(np.linspace(0.0, 1.0, nz))
    w = np.full(nz, 1.0 / (nz - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    return nodes, list(w)


def test_projection_rows_are_integral_free_arithmetic():
    b = Legendre_shifted(level=2)
    nodes, weights = _gauss_01(5)
    samples = [sp.Symbol(f"f{j}") for j in range(len(nodes))]
    rows = b.projection_rows(nodes, weights, samples)
    assert len(rows) == 3
    for r in rows:
        assert not r.has(sp.Integral), f"row still carries an Integral: {r}"
        # plain arithmetic: only the input samples remain free
        assert r.free_symbols <= set(samples), f"unexpected free symbols: {r.free_symbols}"


def test_projection_rows_exact_inverse_with_gauss():
    b = Legendre_shifted(level=2)
    z = sp.Symbol("z")
    nodes, weights = _gauss_01(5)               # exact for deg <= 9 >> 2*level
    samples = [sp.Symbol(f"f{j}") for j in range(len(nodes))]
    rows = b.projection_rows(nodes, weights, samples)

    coeffs = {0: 0.7, 1: -1.3, 2: 0.4}
    profile = sum(coeffs[k] * b.get(k) for k in coeffs)
    subs = {samples[j]: float(profile.subs(z, nodes[j])) for j in range(len(nodes))}
    recovered = [float(sp.sympify(rows[k]).subs(subs)) for k in range(3)]
    for k in range(3):
        assert abs(recovered[k] - coeffs[k]) < 1e-10, (
            f"mode {k}: recovered {recovered[k]} != {coeffs[k]}")


def test_projection_rows_trapezoid_converges_second_order():
    b = Legendre_shifted(level=2)
    z = sp.Symbol("z")
    coeffs = {0: 0.7, 1: -1.3, 2: 0.4}
    profile = sum(coeffs[k] * b.get(k) for k in coeffs)

    errs = {}
    for nz in (41, 161):
        nodes, weights = _trapezoid_01(nz)
        samples = [sp.Symbol(f"g{j}") for j in range(nz)]
        rows = b.projection_rows(nodes, weights, samples)
        subs = {samples[j]: float(profile.subs(z, nodes[j])) for j in range(nz)}
        rec = [float(sp.sympify(rows[k]).subs(subs)) for k in range(3)]
        errs[nz] = max(abs(rec[k] - coeffs[k]) for k in range(3))

    assert errs[41] < 5e-3
    # 4x points -> ~16x smaller error for O(N^-2); allow slack
    assert errs[161] < errs[41] / 8.0
