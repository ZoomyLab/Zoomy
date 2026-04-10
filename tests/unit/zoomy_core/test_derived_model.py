"""Tests for the DerivedModel architecture.

Covers: DerivedModel base, SMEModel, chaining, mesh auto-promotion,
kernel compilation, and NumpyRuntimeModel integration.
"""

import numpy as np
import pytest

from zoomy_core.mesh import BaseMesh, FVMMesh, LSQMesh, ensure_lsq_mesh
from zoomy_core.kernel import Kernel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, NumpyRuntimeSymbolic
from zoomy_core.model.models.derived_model import DerivedModel
from zoomy_core.model.models.sme_model import SMEModel, SMEInviscid, INSModel
from zoomy_core.model.models.ins_generator import (
    StateSpace, Newtonian, HydrostaticPressure, DepthIntegrate,
)


# ── DerivedModel basics ───────────────────────────────────────────────────────

class _IntermediateModel(INSModel):
    """Intermediate: applies hydrostatic + depth integrate, but not Newtonian."""
    def derive_model(self):
        super().derive_model()
        self.apply(HydrostaticPressure(self.state))
        self.apply(DepthIntegrate(self.state))


class _ChainedModel(_IntermediateModel):
    """Final: adds Newtonian to intermediate."""
    projectable = True
    def derive_model(self):
        super().derive_model()
        self.apply(Newtonian(self.state))


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_intermediate_model_not_solver_ready():
    m = _IntermediateModel()
    assert m.projectable is False
    assert m.system is not None
    assert m.n_variables == 0


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_intermediate_describe():
    m = _IntermediateModel()
    desc = str(m.describe())
    assert "IntermediateModel" in desc
    assert "continuity" in desc or "x_momentum" in desc


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_chained_model_is_solver_ready():
    m = _ChainedModel(level=0)
    assert m.projectable is True
    assert m.n_variables == 3
    assert hasattr(m.functions, "flux")


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_chained_model_compiles():
    m = _ChainedModel(level=1)
    rt = NumpyRuntimeModel(m)
    Q = np.array([0.0, 0.5, 0.1, 0.01])
    p = np.array(m.parameter_values, dtype=float)
    F = rt.flux(Q, np.array([]), p)
    assert F.shape == (4, 1)
    assert np.isfinite(F).all()


# ── SMEModel ──────────────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
@pytest.mark.parametrize("level", [0, 1, 2])
def test_sme_model_levels(level):
    m = SMEModel(level=level)
    expected_vars = 3 + level  # b, h, + (level+1) moments
    assert m.n_variables == expected_vars
    assert m.system.name == "SMEModel"
    flux = m.flux()
    assert flux.shape == (expected_vars, 1)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_sme_inviscid():
    m = SMEInviscid(level=1)
    assert m.n_variables == 4
    assert "inviscid" in m.system.name.lower() or "Inviscid" in str(m.system.assumptions)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_sme_compile_and_evaluate():
    m = SMEModel(level=1)
    rt = NumpyRuntimeModel(m)

    Q = np.array([0.0, 0.5, 0.1, 0.01])
    p = np.array(m.parameter_values, dtype=float)
    F = rt.flux(Q, np.array([]), p)
    S = rt.source(Q, np.array([]), p)
    assert np.isfinite(F).all()
    assert np.isfinite(S).all()
    # flux[1,0] = hu0 = q2 = 0.1
    assert abs(F[1, 0] - 0.1) < 1e-10


# ── Mesh auto-promotion ──────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_ensure_lsq_mesh_from_basemesh():
    bm = BaseMesh.create_1d((-1, 1), 20)
    lm = ensure_lsq_mesh(bm)
    assert isinstance(lm, LSQMesh)
    assert lm._lsq_gradQ is not None


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_ensure_lsq_mesh_passthrough():
    lm = LSQMesh.create_1d((-1, 1), 20)
    lm2 = ensure_lsq_mesh(lm)
    assert lm2 is lm


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_create_2d_mesh():
    bm = BaseMesh.create_2d((0, 10, 0, 5), nx=5, ny=3)
    assert bm.dimension == 2
    assert bm.n_inner_cells == 15
    assert bm.type == "quad"
    assert set(bm.boundary_conditions_sorted_names) == {"left", "right", "bottom", "top"}


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_create_2d_lsq_mesh_derivatives():
    lm = LSQMesh.create_2d((0, 10, 0, 5), nx=10, ny=10, lsq_degree=1)
    # u = x, check that one derivative component ≈ 1 and the other ≈ 0
    u = lm._cell_centers[0, :lm.n_cells]
    derivs = lm.compute_derivatives(u, degree=1)
    # One of the two monomial components should be ~1 (du/dx), other ~0 (du/dy)
    interior = derivs[50:60, :]  # well away from boundary
    col_means = np.abs(interior.mean(axis=0))
    assert max(col_means) > 0.9, f"Expected one derivative ≈ 1, got {col_means}"
    assert min(col_means) < 0.1, f"Expected one derivative ≈ 0, got {col_means}"


# ── 3D Mesh ──────────────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_create_3d_mesh():
    bm = BaseMesh.create_3d((0, 1, 0, 1, 0, 1), nx=3, ny=3, nz=3)
    assert bm.dimension == 3
    assert bm.n_inner_cells == 27
    assert bm.type == "hexahedron"
    assert bm.n_faces_per_cell == 6
    assert set(bm.boundary_conditions_sorted_names) == {
        "left", "right", "front", "back", "bottom", "top"
    }
    # 6 faces × 9 boundary faces each = 54
    assert bm.n_boundary_faces == 54


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_3d_mesh_volumes():
    from zoomy_core.mesh.fvm_mesh import FVMMesh
    fm = FVMMesh.create_3d((0, 1, 0, 1, 0, 1), nx=4, ny=4, nz=4)
    # Total volume = 1.0
    assert abs(np.sum(fm.cell_volumes[:64]) - 1.0) < 1e-12
    # Each cell volume = 1/64
    assert abs(fm.cell_volumes[0] - 1.0 / 64) < 1e-12
    # Face areas = 1/16
    assert abs(fm.face_volumes.min() - 1.0 / 16) < 1e-12


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_3d_lsq_derivatives():
    lm = LSQMesh.create_3d((0, 1, 0, 1, 0, 1), nx=4, ny=4, nz=4, lsq_degree=1)
    # u = x → one derivative ≈ 1, others ≈ 0
    u = lm._cell_centers[0, :lm.n_cells]
    derivs = lm.compute_derivatives(u, degree=1)
    # 3 monomials for degree 1 in 3D
    assert derivs.shape[1] == 3
    # Interior cell
    center_cell = 4 * 4 * 2 + 4 * 2 + 2  # roughly center
    d = derivs[center_cell, :]
    # One component should be ~1 (the x-derivative)
    assert max(np.abs(d)) > 0.9
    # At least one should be ~0
    assert min(np.abs(d)) < 0.1


# ── Kernel ────────────────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_kernel_creation_from_model():
    m = SMEModel(level=0)
    k = Kernel(m)
    assert hasattr(k.functions, "safe_denominator")
    assert hasattr(k.functions, "clamp_positive")
    assert "eps" in k._param_map


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_kernel_compiles_with_numpy():
    m = SMEModel(level=0)
    k = Kernel(m)
    rt = NumpyRuntimeSymbolic(k)
    result = rt.safe_denominator(np.array([0.0, 1.0]), np.float64(1e-8))
    assert np.allclose(result, [1e-8, 1.0 + 1e-8])


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_model_compiles_with_kernel():
    m = SMEModel(level=1)
    k = Kernel(m)
    rt = NumpyRuntimeModel(m, kernel=k)
    Q = np.array([0.0, 0.5, 0.1, 0.01])
    p = np.array(m.parameter_values, dtype=float)
    F = rt.flux(Q, np.array([]), p)
    assert np.isfinite(F).all()


# ── Full pipeline ─────────────────────────────────────────────────────────────

@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_ins_2d_chorin_projection():
    """2D Taylor-Green vortex with Chorin pressure splitting."""
    from zoomy_core.model.models.ins3d_model import INS3DChorin
    from zoomy_core.fvm.solver_splitting_numpy import SplittingSolver
    import zoomy_core.fvm.timestepping as ts
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    model = INS3DChorin(dimension=2, nu=0.01)
    model.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([
            np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]),
            -np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]),
        ])
    )
    model.boundary_conditions = BC.BoundaryConditions(
        boundary_conditions_list=[
            BC.Extrapolation(tag=t) for t in ["left", "right", "bottom", "top"]
        ]
    )
    mesh = BaseMesh.create_2d((0, 1, 0, 1), nx=10, ny=10)
    solver = SplittingSolver(time_end=0.05, compute_dt=ts.adaptive(CFL=0.2))
    Q, p = solver.solve(mesh, model, write_output=False)
    nc = 100
    assert np.isfinite(Q[:, :nc]).all()
    assert np.isfinite(p[:nc]).all()
    max_speed = np.sqrt(Q[0, :nc]**2 + Q[1, :nc]**2).max()
    assert max_speed > 0.1  # velocity is evolving
    assert max_speed < 10.0  # not blowing up


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_ins_3d_chorin_projection():
    """3D Taylor-Green vortex with Chorin pressure splitting."""
    from zoomy_core.model.models.ins3d_model import INS3DChorin
    from zoomy_core.fvm.solver_splitting_numpy import SplittingSolver
    import zoomy_core.fvm.timestepping as ts
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    model = INS3DChorin(dimension=3, nu=0.1)
    model.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([
            np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]),
            -np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]),
            0.0,
        ])
    )
    model.boundary_conditions = BC.BoundaryConditions(
        boundary_conditions_list=[
            BC.Extrapolation(tag=t) for t in
            ["left", "right", "front", "back", "bottom", "top"]
        ]
    )
    mesh = BaseMesh.create_3d((0, 1, 0, 1, 0, 1), nx=5, ny=5, nz=5)
    solver = SplittingSolver(time_end=0.02, compute_dt=ts.adaptive(CFL=0.2))
    Q, p = solver.solve(mesh, model, write_output=False)
    nc = 125
    assert np.isfinite(Q[:, :nc]).all()
    assert np.isfinite(p[:nc]).all()


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_full_pipeline_1d():
    """BaseMesh → auto-promote → Kernel → NumpyRuntimeModel → evaluate."""
    model = SMEModel(level=1)
    mesh = BaseMesh.create_1d((0, 10), 20)
    lsq = ensure_lsq_mesh(mesh, model)
    kernel = Kernel(model)
    rt = NumpyRuntimeModel(model, kernel=kernel)

    Q = np.zeros((model.n_variables, lsq.n_cells))
    Q[1, :] = 0.5  # h
    Q[2, :] = 0.1  # hu0
    p = np.array(model.parameter_values, dtype=float)

    for ic in range(lsq.n_inner_cells):
        F = rt.flux(Q[:, ic], np.array([]), p)
        assert np.isfinite(F).all()


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_3d_advection_solve():
    """3D scalar advection on a cube: Gaussian pulse in x-direction."""
    from zoomy_core.model.models.advection_model import ScalarAdvection
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver
    import zoomy_core.fvm.timestepping as ts
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC

    model = ScalarAdvection(dimension=3)
    model.parameter_values = np.array([1.0, 0.0, 0.0])
    model.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([np.exp(-((x[0]-0.3)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2) / 0.01)])
    )
    model.boundary_conditions = BC.BoundaryConditions(
        boundary_conditions_list=[
            BC.Extrapolation(tag=t) for t in
            ["left", "right", "front", "back", "bottom", "top"]
        ]
    )
    mesh = BaseMesh.create_3d((0, 1, 0, 1, 0, 1), nx=8, ny=8, nz=8)
    solver = HyperbolicSolver(time_end=0.1, compute_dt=ts.adaptive(CFL=0.3))
    Q, _ = solver.solve(mesh, model, write_output=False)
    nc = mesh.n_inner_cells
    assert np.isfinite(Q[:, :nc]).all()
    assert Q[0, :nc].min() >= -0.01  # no large negative values
    lsq = ensure_lsq_mesh(mesh, model)
    mass_init = np.sum(lsq.cell_volumes[:nc]) * 0.005577  # approximate
    mass_final = np.sum(Q[0, :nc] * lsq.cell_volumes[:nc])
    assert mass_final > 0  # mass is positive


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_advection_diffusion_explicit_vs_implicit():
    """1D advection-diffusion: explicit and implicit diffusion should agree."""
    from zoomy_core.model.models.advection_model import ScalarAdvectionDiffusion
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver
    from zoomy_core.fvm.solver_imex_numpy import IMEXSolver
    import zoomy_core.fvm.timestepping as ts
    from zoomy_core.model.initial_conditions import InitialConditions

    class GaussianIC(InitialConditions):
        def apply(self, X, Q):
            Q[0, :] = np.exp(-100 * (X[0, :] - 0.3)**2)
            return Q

    mesh = BaseMesh.create_1d(domain=(0, 1), n_inner_cells=100)
    nu = 0.01

    # Explicit diffusion (HyperbolicSolver)
    m1 = ScalarAdvectionDiffusion(dimension=1, nu=nu)
    m1.parameter_values = np.array([1.0, nu])
    m1.initial_conditions = GaussianIC()
    Q1, _ = HyperbolicSolver(
        time_end=0.2, compute_dt=ts.adaptive(CFL=0.5, nu=nu)
    ).solve(mesh, m1, write_output=False)

    # Implicit diffusion (IMEXSolver — no diffusive CFL, diffusion is implicit)
    m2 = ScalarAdvectionDiffusion(dimension=1, nu=nu)
    m2.parameter_values = np.array([1.0, nu])
    m2.initial_conditions = GaussianIC()
    Q2, _ = IMEXSolver(
        time_end=0.2, compute_dt=ts.adaptive(CFL=0.5)
    ).solve(mesh, m2, write_output=False)

    nc = mesh.n_inner_cells
    # Both should produce advected+diffused Gaussian
    assert Q1[0, :nc].max() < 0.75, "explicit: peak should be diffused"
    assert Q2[0, :nc].max() < 0.75, "IMEX: peak should be diffused"
    assert Q1[0, :nc].argmax() > 40, "explicit: peak should have advected right"
    assert Q2[0, :nc].argmax() > 40, "IMEX: peak should have advected right"
    # Explicit and IMEX should produce similar results
    # (CN vs explicit diffusion differ slightly, tolerance ~5%)
    assert np.abs(Q1[0, :nc] - Q2[0, :nc]).max() < 0.05

    # IMEX with larger dt (no diffusive CFL — implicit handles it)
    m3 = ScalarAdvectionDiffusion(dimension=1, nu=nu)
    m3.parameter_values = np.array([1.0, nu])
    m3.initial_conditions = GaussianIC()
    solver3 = IMEXSolver(time_end=0.2, compute_dt=ts.adaptive(CFL=0.9))
    Q3, _ = solver3.solve(mesh, m3, write_output=False)
    assert solver3.last_stats.n_steps < 40, "IMEX should take fewer steps without diff CFL"
    assert np.abs(Q1[0, :nc] - Q3[0, :nc]).max() < 0.1


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_advection_diffusion_convergence():
    """1D advection-diffusion convergence against exact analytical solution.

    Uses IMEXSolver with convective CFL only (no diffusive CFL constraint).
    Implicit diffusion keeps the convective CFL number constant at 0.5,
    giving clean O(Δx) convergence from the Rusanov scheme.

    With explicit diffusion + diffusive CFL, the convective CFL drops to
    σ ~ Δx, inflating Rusanov numerical diffusion and degrading the rate.
    """
    from zoomy_core.model.models.advection_model import ScalarAdvectionDiffusion
    from zoomy_core.fvm.solver_imex_numpy import IMEXSolver
    import zoomy_core.fvm.timestepping as ts
    from zoomy_core.model.initial_conditions import InitialConditions

    def exact_solution(x, t, x0=0.5, a=1.0, nu=0.01, alpha=100.0):
        sigma0_sq = 1.0 / (2 * alpha)
        sigma_t_sq = sigma0_sq + 2 * nu * t
        return np.sqrt(sigma0_sq / sigma_t_sq) * np.exp(
            -(x - x0 - a * t) ** 2 / (2 * sigma_t_sq)
        )

    class GaussianIC(InitialConditions):
        def apply(self, X, Q):
            Q[0, :] = np.exp(-100 * (X[0, :] - 0.5) ** 2)
            return Q

    nu = 0.01
    t_end = 0.05  # short so Gaussian stays away from boundaries
    errors = []
    for N in [100, 200, 400]:
        mesh = BaseMesh.create_1d(domain=(0, 1), n_inner_cells=N)
        m = ScalarAdvectionDiffusion(dimension=1, nu=nu)
        m.parameter_values = np.array([1.0, nu])
        m.initial_conditions = GaussianIC()
        Q, _ = IMEXSolver(
            time_end=t_end, compute_dt=ts.adaptive(CFL=0.5)
        ).solve(mesh, m, write_output=False)
        lsq = ensure_lsq_mesh(mesh, m)
        nc = lsq.n_inner_cells
        xc = lsq.cell_centers[0, :nc]
        l2 = np.sqrt(np.sum((Q[0, :nc] - exact_solution(xc, t_end)) ** 2) / N)
        errors.append(l2)

    # First-order convergence: each refinement should halve the error
    rate1 = np.log2(errors[0] / errors[1])
    rate2 = np.log2(errors[1] / errors[2])
    assert rate1 > 0.9, f"convergence rate {rate1:.2f} too low (100→200)"
    assert rate2 > 0.9, f"convergence rate {rate2:.2f} too low (200→400)"
    assert errors[-1] < 0.01, f"L2 error {errors[-1]:.4e} too large at N=400"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_muscl_second_order_convergence():
    """MUSCL reconstruction gives ~2nd-order convergence for pure advection."""
    from zoomy_core.model.models.advection_model import ScalarAdvection
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver
    import zoomy_core.fvm.timestepping as ts
    from zoomy_core.model.initial_conditions import InitialConditions

    class GaussianIC(InitialConditions):
        def apply(self, X, Q):
            Q[0, :] = np.exp(-100 * (X[0, :] - 0.5) ** 2)
            return Q

    errors = []
    for N in [100, 200, 400]:
        mesh = BaseMesh.create_1d(domain=(0, 1), n_inner_cells=N)
        m = ScalarAdvection(dimension=1)
        m.parameter_values = np.array([1.0])
        m.initial_conditions = GaussianIC()
        Q, _ = HyperbolicSolver(
            time_end=0.02, compute_dt=ts.adaptive(CFL=0.5),
            reconstruction_order=2,
        ).solve(mesh, m, write_output=False)
        lsq = ensure_lsq_mesh(mesh, m)
        nc = lsq.n_inner_cells
        xc = lsq.cell_centers[0, :nc]
        u_exact = np.exp(-100 * (xc - 0.52) ** 2)
        errors.append(np.sqrt(np.sum((Q[0, :nc] - u_exact) ** 2) / N))

    rate1 = np.log2(errors[0] / errors[1])
    rate2 = np.log2(errors[1] / errors[2])
    # MUSCL + RK2 should give rate > 1.5 (limited by Venkatakrishnan at peak)
    assert rate1 > 1.5, f"MUSCL rate {rate1:.2f} too low (100→200)"
    assert rate2 > 1.5, f"MUSCL rate {rate2:.2f} too low (200→400)"
    # Error should be much smaller than 1st-order at same resolution
    assert errors[-1] < 1e-4, f"L2 error {errors[-1]:.4e} too large at N=400"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.core
def test_full_pipeline_2d():
    """2D BaseMesh → auto-promote → Kernel → NumpyRuntimeModel → evaluate."""
    model = SMEModel(level=0)
    mesh = BaseMesh.create_2d((0, 10, 0, 5), nx=5, ny=3)
    lsq = ensure_lsq_mesh(mesh, model)
    kernel = Kernel(model)
    rt = NumpyRuntimeModel(model, kernel=kernel)

    Q = np.zeros((model.n_variables, lsq.n_cells))
    Q[1, :] = 0.5
    Q[2, :] = 0.05
    p = np.array(model.parameter_values, dtype=float)

    for ic in range(min(5, lsq.n_inner_cells)):
        F = rt.flux(Q[:, ic], np.array([]), p)
        assert np.isfinite(F).all()
        assert F.shape == (3, 1)
