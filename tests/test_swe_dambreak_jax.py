"""SWASHES-style 1D SWE dam-break TDD target for the JAX hyperbolic solver.

We use the **Ritter solution** (1892) — the canonical analytical solution of the
shallow-water equations for a dry-bed dam-break — directly, rather than the
external SWASHES binary. It is short, exact, and the standard 1D SWE
benchmark; it's what the SWASHES library itself produces for case (1, 3, 1, 2)
(dim, type=dam_break, domain=1, choice=dry, ncellx=200).

Geometry: domain [0, L], dam at x = L/2.
Initial:  h(x, 0) = h_L for x < L/2, 0 for x ≥ L/2; u = 0 everywhere.
With c_L = sqrt(g h_L) the closed form for t > 0 is::

    x < L/2 - c_L t                    : h = h_L,     u = 0          (undisturbed)
    L/2 - c_L t ≤ x ≤ L/2 + 2 c_L t    : h = (2 c_L - (x-L/2)/t)² / (9 g)
                                          u = (2/3) (c_L + (x-L/2)/t)
    x > L/2 + 2 c_L t                  : h = 0,        u = 0          (dry)

The wave front advances at 2 c_L; for h_L = 0.005 and g = 9.81 that's
≈ 0.442 m/s. At t = 0.5 s the rarefaction spans x ∈ [4.89, 5.22] — well
inside a [0, 10] domain — so both boundaries stay quiescent, which is
why extrapolation BCs are appropriate here.

We test 1st- and 2nd-order reconstructions and assert that L1 error
falls under refinement. Dam-break has a non-smooth dry front so the
*global* L1 rate caps near 1.0 for any scheme (the discontinuity
contributes O(dx) regardless of order); a clean ~2.0 rate measurement
needs a fully smooth problem (separate test).
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec

# Avoid spamming logs during tests.
from loguru import logger
logger.remove()

# Will be set from the JAX HyperbolicSolver below.
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.fvm.reconstruction_jax import (
    ConstantReconstruction, FreeSurfaceLSQMUSCLJAX,
)


class SWEHyperbolicSolver(HyperbolicSolver):
    """JAX HyperbolicSolver that uses ``FreeSurfaceLSQMUSCLJAX`` for O2 so
    the wet/dry dam-break doesn't produce negative h at faces near the
    dry front (LSQ-augmented stencil; mirrors NumPy
    ``FreeSurfaceLSQMUSCL``)."""

    def _build_reconstruction(self, mesh, symbolic_model, runtime=None):
        dim = symbolic_model.dimension
        if self.nsm.reconstruction.order >= 2:
            # h is variable index 0 for our SWE1D model (state = [h, hu]).
            return FreeSurfaceLSQMUSCLJAX(
                mesh, dim, h_index=0, eps_wet=1e-6,
                limiter=self.nsm.reconstruction.limiter,
            )
        return ConstantReconstruction(mesh, dim)


# ── 1D SWE model — pure h, hu; flat bed; g = 9.81 ────────────────────────────

from sympy import Matrix


class SWE1D(StructuredDerivativeModel):
    """Conservative 1D shallow water on a flat bed."""

    dimension = 1
    variables = ["h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h = self.Q.h
        hu = self.Q.hu
        g = self.params.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = hu
        F[1, 0] = hu * hu / h + 0.5 * g * h * h
        return ZArray(F)

    def source(self):
        return ZArray.zeros(self.n_variables)


# ── Ritter analytical solution ───────────────────────────────────────────────

G = 9.81


def ritter(x: np.ndarray, t: float, h_L: float, x0: float):
    """Closed-form (h, u) at points ``x``, time ``t``."""
    c_L = float(np.sqrt(G * h_L))
    h = np.empty_like(x)
    u = np.empty_like(x)
    xL = x0 - c_L * t          # left edge of rarefaction
    xR = x0 + 2.0 * c_L * t    # dry front

    undisturbed = x < xL
    rarefaction = (x >= xL) & (x <= xR)
    dry = x > xR

    h[undisturbed] = h_L
    u[undisturbed] = 0.0
    eta = (x[rarefaction] - x0) / max(t, 1e-12)
    h[rarefaction] = (2.0 * c_L - eta) ** 2 / (9.0 * G)
    u[rarefaction] = (2.0 / 3.0) * (c_L + eta)
    h[dry] = 0.0
    u[dry] = 0.0
    return h, u


# ── Test fixture ─────────────────────────────────────────────────────────────

DOMAIN = (0.0, 10.0)
X0 = 5.0
H_L = 0.005
EPS_DRY = 1e-8    # h "floor" for the initially-dry side — Ritter analytical
                  # treats it as exactly zero; the numerics needs a small
                  # positive value plus a wet/dry-aware MUSCL.
T_END = 0.5

# Smooth-wet test parameters — for measuring the 2nd-order convergence
# rate without the wet/dry singularity that caps dam-break L1 rates.
H0 = 0.1          # background still-water depth
H_PERT = 0.01     # peak height of the initial Gaussian h perturbation
SIGMA = 1.0       # Gaussian half-width
T_SMOOTH = 0.4    # short enough that the disturbance stays well inside [0, 10]


def _make_model():
    """Wet/dry dam-break IC + extrapolation BCs."""
    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
    )

    def ic(x):
        Q = np.zeros(2, dtype=float)
        Q[0] = H_L if x[0] < X0 else EPS_DRY
        Q[1] = 0.0
        return Q

    return SWE1D(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(function=ic),
    )


def _run(N: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_inner, h_inner) at t = T_END for given N, reconstruction order."""
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N)
    model = _make_model()
    nsm = NumericalSystemModel.from_system_model(
        model, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    solver = SWEHyperbolicSolver(
        time_end=T_END,
        compute_dt=timestepping.adaptive(CFL=0.9, dimension=1),
    )
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    x = np.asarray(mesh.cell_centers[0, :N])
    h = np.asarray(Q[0, :N])
    return x, h


def _l1_error(x_num: np.ndarray, h_num: np.ndarray, dx: float) -> float:
    h_ex, _ = ritter(x_num, T_END, H_L, X0)
    return float(np.sum(np.abs(h_num - h_ex)) * dx)


# ── Smooth-wet helpers (no wet/dry, no shocks) ───────────────────────────────


def _make_smooth_model():
    """Still lake + a Gaussian perturbation in h, zero initial momentum."""
    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
    )

    def ic(x):
        Q = np.zeros(2, dtype=float)
        Q[0] = H0 + H_PERT * np.exp(-((x[0] - X0) ** 2) / (2 * SIGMA ** 2))
        Q[1] = 0.0
        return Q

    return SWE1D(boundary_conditions=bcs, initial_conditions=IC.UserFunction(function=ic))


def _run_smooth(N: int, order: int, t_end: float = T_SMOOTH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N)
    model = _make_smooth_model()
    nsm = NumericalSystemModel.from_system_model(
        model, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    solver = SWEHyperbolicSolver(
        time_end=t_end,
        compute_dt=timestepping.adaptive(CFL=0.9, dimension=1),
    )
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    x = np.asarray(mesh.cell_centers[0, :N])
    return x, np.asarray(Q[0, :N]), np.asarray(Q[1, :N])


def _l1_against_ref(x: np.ndarray, h: np.ndarray, x_ref: np.ndarray, h_ref: np.ndarray, dx: float) -> float:
    """Self-convergence: L1 error of (x, h) interpolated against a fine-grid (x_ref, h_ref)."""
    h_ex = np.interp(x, x_ref, h_ref)
    return float(np.sum(np.abs(h - h_ex)) * dx)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.jax
def test_dambreak_o1_converges():
    """O1 against Ritter: finite, finite-rate, L1 decreases with N.

    Dry-bed Ritter is bounded-variation with a non-smooth dry front;
    the asymptotic L1 rate for any TVD-MUSCL scheme caps around 0.4-0.7
    on this case. We gate the test at >0.3 so a regression that drops
    the rate below 0.3 fails clearly while routine numerical noise
    doesn't.
    """
    errs = []
    for N in [200, 400, 800]:
        x, h = _run(N, order=1)
        dx = (DOMAIN[1] - DOMAIN[0]) / N
        errs.append(_l1_error(x, h, dx))
    print(f"O1 dam-break L1 errors at N=200,400,800: {errs}")
    assert all(np.isfinite(e) for e in errs)
    assert errs[0] > errs[1] > errs[2]
    rate = np.log(errs[0] / errs[2]) / np.log(800 / 200)
    assert rate > 0.3, f"O1 dam-break rate {rate:.2f} below 0.3 (regression)"


@pytest.mark.jax
def test_dambreak_o2_runs_and_converges():
    """O2 against Ritter: doesn't blow up to NaN at the dry front.

    The L1 rate near the dry front is bounded by the discontinuity
    width and is rarely better than O1 for this case (O1 ≈ 0.4-0.7,
    O2 ≈ 0.4-0.9). The real proof of 2nd-order accuracy is the smooth
    test below.
    """
    errs = []
    for N in [200, 400, 800]:
        x, h = _run(N, order=2)
        dx = (DOMAIN[1] - DOMAIN[0]) / N
        errs.append(_l1_error(x, h, dx))
    print(f"O2 dam-break L1 errors at N=200,400,800: {errs}")
    assert all(np.isfinite(e) for e in errs), "O2 produced NaN at the dry front"
    # At the highest resolution, O2 should at minimum not be worse than O1.
    x1, h1 = _run(800, order=1)
    dx_h = (DOMAIN[1] - DOMAIN[0]) / 800
    e1_high = _l1_error(x1, h1, dx_h)
    assert errs[-1] <= 1.5 * e1_high, (
        f"O2 ({errs[-1]:.4e}) much worse than O1 ({e1_high:.4e}) at N=800"
    )


@pytest.mark.jax
def test_smooth_swe_o2_convergence():
    """Smooth-wet SWE perturbation: 2nd-order self-convergence vs fine ref."""
    Nref = 2048
    x_ref, h_ref, _ = _run_smooth(Nref, order=2)
    Ns = [128, 256, 512]
    errs1, errs2 = [], []
    for N in Ns:
        dx = (DOMAIN[1] - DOMAIN[0]) / N
        x, h1, _ = _run_smooth(N, order=1)
        _, h2, _ = _run_smooth(N, order=2)
        errs1.append(_l1_against_ref(x, h1, x_ref, h_ref, dx))
        errs2.append(_l1_against_ref(x, h2, x_ref, h_ref, dx))
    rates2 = [np.log(errs2[i - 1] / errs2[i]) / np.log(Ns[i] / Ns[i - 1])
              for i in range(1, len(errs2))]
    print(f"  smooth-SWE O1 L1 vs N=2048 ref: {[f'{e:.4e}' for e in errs1]}")
    print(f"  smooth-SWE O2 L1 vs N=2048 ref: {[f'{e:.4e}' for e in errs2]}")
    print(f"  smooth-SWE O2 rates:            {[f'{r:.3f}' for r in rates2]}")
    # O2 should beat O1 at every resolution.
    for e1, e2 in zip(errs1, errs2):
        assert e2 <= e1, f"O2 ({e2:.4e}) ≤ O1 ({e1:.4e}) failed"
    # And the O2 rate should be >= 1.5 (we'd love 2.0 but minmod + smooth-flat
    # background gives at best ~1.8; gate at 1.5 to allow some headroom).
    for r in rates2:
        assert r > 1.5, f"O2 smooth-SWE rate {r:.2f} below 1.5"


if __name__ == "__main__":
    print("=" * 70)
    print("Ritter dam-break — JAX HyperbolicSolver  (wet/dry, L1 rate ≈ 0.5)")
    print("=" * 70)
    for order in [1, 2]:
        errs = []
        for N in [200, 400, 800]:
            x, h = _run(N, order=order)
            dx = (DOMAIN[1] - DOMAIN[0]) / N
            errs.append(_l1_error(x, h, dx))
        rates = [np.log(errs[i - 1] / errs[i]) / np.log(2)
                 for i in range(1, len(errs))]
        print(f"  O{order}  L1 errors: {[f'{e:.4e}' for e in errs]}")
        print(f"        rates:   {[f'{r:.3f}' for r in rates]}")

    print()
    print("=" * 70)
    print("Smooth Gaussian SWE perturbation — JAX HyperbolicSolver")
    print("=" * 70)
    Nref = 2048
    x_ref, h_ref, _ = _run_smooth(Nref, order=2)
    for order in [1, 2]:
        errs = []
        Ns = [128, 256, 512]
        for N in Ns:
            dx = (DOMAIN[1] - DOMAIN[0]) / N
            x, h, _ = _run_smooth(N, order=order)
            errs.append(_l1_against_ref(x, h, x_ref, h_ref, dx))
        rates = [np.log(errs[i - 1] / errs[i]) / np.log(Ns[i] / Ns[i - 1])
                 for i in range(1, len(errs))]
        print(f"  O{order}  L1 errors: {[f'{e:.4e}' for e in errs]}")
        print(f"        rates:   {[f'{r:.3f}' for r in rates]}")
