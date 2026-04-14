"""
Firedrake DG convergence study for advection and advection-diffusion.

Scalar advection equation:  du/dt + div(a*u) = 0,  a = (1,0)
DG weak form after integration by parts:
    (du/dt, v)_K = (a*u, grad(v))_K - <F_hat . n, v>_{dK}

where F_hat is the Rusanov/LLF numerical flux on facets.

IC: sin(2*pi*x) * sin(2*pi*y)
Exact: sin(2*pi*(x-t)) * sin(2*pi*y)

Uses SSP-RK3 (Shu-Osher) for time integration.
"""
import firedrake as fd
import numpy as np
import ufl
from math import pi


def run_advection(N, degree, T_end=0.1, CFL=0.2, nu=0.0, use_limiter=False):
    """
    Run scalar advection on NxN UnitSquareMesh with SSP-RK3.
    Returns: (L2_error, h, n_steps)
    """
    mesh = fd.UnitSquareMesh(N, N)
    V = fd.FunctionSpace(mesh, "DG", degree)
    x, y = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

    a = fd.Constant((1.0, 0.0))  # velocity

    # Time-dependent boundary/exact solution
    sim_time = fd.Constant(0.0)
    u_exact = fd.sin(2*pi*(x - sim_time)) * fd.sin(2*pi*y)

    u_n = fd.Function(V, name="u_n")
    u_1 = fd.Function(V, name="u_1")
    u_2 = fd.Function(V, name="u_2")
    du = fd.Function(V)

    u_n.interpolate(fd.sin(2*pi*x) * fd.sin(2*pi*y))

    # Time step
    h_mesh = 1.0 / N
    degree_factor = float(2 * degree + 1)
    dt_val = CFL * h_mesh / (degree_factor * 1.0)

    if nu > 0.0 and degree >= 1:
        dt_diff = 0.02 * h_mesh**2 / nu
        dt_val = min(dt_val, dt_diff)

    # Helper: build RHS form for a given u_src
    def make_rhs(u_src):
        v = fd.TestFunction(V)

        # du/dt + div(a*u) = 0
        # Multiply by v, integrate by parts over each cell K:
        # (du/dt, v)_K = (a*u, grad(v))_K - <F_hat . n, v>_{dK}
        #
        # So: RHS = (a*u, grad(v))_K - <F_hat . n, v>_{dK}

        # Volume: + (a*u_src, grad(v))
        rhs = fd.inner(a * u_src, fd.grad(v)) * fd.dx

        # Interior facets: - <F_hat . n, [v]>
        # Rusanov flux: F_hat . n('+') = 0.5*(a.n('+'))*(u('+)+u('-)) - 0.5*|a.n('+')|*(u('+')-u('-))
        # The contribution is: -[v] * F_hat . n('+')
        an = fd.dot(a('+'), n('+'))
        F_hat_n = 0.5 * an * (u_src('+') + u_src('-')) \
                  - 0.5 * abs(an) * (u_src('+') - u_src('-'))

        # -[v] * F_hat . n('+') over interior facets
        # [v] = v('+') - v('-')
        rhs -= (v('+') - v('-')) * F_hat_n * fd.dS

        # Exterior facets: - v * F_hat . n
        # Upwind: if a.n > 0, use u_src (outflow); else use u_exact (inflow)
        an_ext = fd.dot(a, n)
        u_bdy = fd.conditional(an_ext < 0, u_exact, u_src)
        F_hat_ext = 0.5 * an_ext * (u_src + u_bdy) \
                    - 0.5 * abs(an_ext) * (u_src - u_bdy)
        rhs -= v * F_hat_ext * fd.ds

        # Diffusion (IP-DG) for nu > 0, degree >= 1
        if nu > 0.0 and degree >= 1:
            nu_c = fd.Constant(nu)
            h_F = fd.CellDiameter(mesh)
            sigma = fd.Constant(10.0 * degree**2)

            # Volume: - nu * grad(u) . grad(v) dx  (from IBP of -div(nu*grad(u)))
            rhs -= nu_c * fd.dot(fd.grad(u_src), fd.grad(v)) * fd.dx

            # Interior faces (symmetric IP-DG)
            avg_h = (h_F('+') + h_F('-')) / 2.0
            jump_u = u_src('+') - u_src('-')
            jump_v = v('+') - v('-')

            for d in range(2):
                avg_grad_u_d = 0.5 * (fd.grad(u_src)('+')[d] + fd.grad(u_src)('-')[d])
                avg_grad_v_d = 0.5 * (fd.grad(v)('+')[d] + fd.grad(v)('-')[d])
                # Consistency: + avg(nu*grad(u)) . n * [v]
                rhs += nu_c * avg_grad_u_d * n('+')[d] * jump_v * fd.dS
                # Symmetry: + avg(nu*grad(v)) . n * [u]
                rhs += nu_c * avg_grad_v_d * n('+')[d] * jump_u * fd.dS

            # Penalty: - sigma/h * [u] . [v]
            rhs -= nu_c * (sigma / avg_h) * jump_u * jump_v * fd.dS

            # Exterior faces: Dirichlet with u_exact
            for d in range(2):
                rhs += nu_c * fd.grad(u_src)[d] * n[d] * v * fd.ds
                rhs += nu_c * fd.grad(v)[d] * n[d] * (u_src - u_exact) * fd.ds
            rhs -= nu_c * (sigma / h_F) * (u_src - u_exact) * v * fd.ds

        return rhs

    rhs_n = make_rhs(u_n)
    rhs_1 = make_rhs(u_1)
    rhs_2 = make_rhs(u_2)

    # Mass matrix
    v_test = fd.TestFunction(V)
    u_trial = fd.TrialFunction(V)
    M = fd.assemble(v_test * u_trial * fd.dx)

    solver_params = {"ksp_type": "preonly", "pc_type": "bjacobi",
                     "sub_pc_type": "ilu"}

    limiter = None
    if use_limiter and degree >= 1:
        limiter = fd.VertexBasedLimiter(V)

    # SSP-RK3
    t = 0.0
    step = 0

    while t < T_end - 1e-14:
        current_dt = min(dt_val, T_end - t)
        sim_time.assign(t)

        # Stage 1: u_1 = u_n + dt * L(u_n, t)
        b = fd.assemble(rhs_n)
        b *= current_dt
        fd.solve(M, du, b, solver_parameters=solver_params)
        u_1.assign(u_n + du)
        if limiter:
            limiter.apply(u_1)

        # Stage 2: u_2 = 3/4 u_n + 1/4 (u_1 + dt * L(u_1, t+dt))
        sim_time.assign(t + current_dt)
        b = fd.assemble(rhs_1)
        b *= current_dt
        fd.solve(M, du, b, solver_parameters=solver_params)
        u_2.assign(0.75 * u_n + 0.25 * (u_1 + du))
        if limiter:
            limiter.apply(u_2)

        # Stage 3: u_np1 = 1/3 u_n + 2/3 (u_2 + dt * L(u_2, t+dt/2))
        sim_time.assign(t + 0.5 * current_dt)
        b = fd.assemble(rhs_2)
        b *= current_dt
        fd.solve(M, du, b, solver_parameters=solver_params)
        u_n.assign((1.0/3.0) * u_n + (2.0/3.0) * (u_2 + du))
        if limiter:
            limiter.apply(u_n)

        t += current_dt
        step += 1

    # L2 error
    sim_time.assign(T_end)
    error = fd.errornorm(u_exact, u_n, norm_type="L2")

    return error, 1.0/N, step


def convergence_rate(errors, hs):
    rates = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            rate = np.log(errors[i-1] / errors[i]) / np.log(hs[i-1] / hs[i])
            rates.append(rate)
        else:
            rates.append(float('nan'))
    return rates


def run_convergence_study(degree, Ns, T_end, CFL, nu=0.0, use_limiter=False, label=""):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  degree={degree}, nu={nu}, T_end={T_end}, CFL={CFL}, limiter={use_limiter}")
    print(f"{'='*65}")

    errors = []
    hs = []

    for N in Ns:
        err, h, steps = run_advection(N, degree, T_end=T_end, CFL=CFL, nu=nu,
                                       use_limiter=use_limiter)
        errors.append(err)
        hs.append(h)
        print(f"  N={N:4d}  h={h:.5f}  L2_error={err:.6e}  steps={steps}")

    rates = convergence_rate(errors, hs)
    print(f"\n  Convergence rates:")
    for i, rate in enumerate(rates):
        print(f"    {Ns[i]:4d} -> {Ns[i+1]:4d}:  rate = {rate:.3f}")

    expected = degree + 1
    final_rate = rates[-1] if rates else float('nan')
    tol = 0.6 if use_limiter else 0.5
    status = "PASS" if abs(final_rate - expected) < tol else "CHECK"
    print(f"\n  Expected rate: ~{expected}")
    print(f"  Measured final rate: {final_rate:.3f}  [{status}]")

    return errors, hs, rates


if __name__ == "__main__":
    Ns = [8, 16, 32, 64]
    T_end = 0.1
    CFL = 0.2

    print("="*65)
    print("  ZOOMY FIREDRAKE DG CONVERGENCE STUDY")
    print("  SSP-RK3 time integration, Rusanov/LLF numerical flux")
    print(f"  IC: sin(2*pi*x)*sin(2*pi*y), velocity=(1,0), T={T_end}")
    print("="*65)

    # Test 1: Pure advection DG0
    run_convergence_study(
        degree=0, Ns=Ns, T_end=T_end, CFL=CFL, nu=0.0,
        label="TEST 1: Pure Advection DG0 (expected O(h^1))"
    )

    # Test 2: Pure advection DG1 without limiter
    run_convergence_study(
        degree=1, Ns=Ns, T_end=T_end, CFL=CFL, nu=0.0,
        label="TEST 2: Pure Advection DG1, no limiter (expected O(h^2))"
    )

    # Test 3: Pure advection DG1 with limiter
    run_convergence_study(
        degree=1, Ns=Ns, T_end=T_end, CFL=CFL, nu=0.0,
        use_limiter=True,
        label="TEST 3: Pure Advection DG1, with vertex limiter"
    )

    # Test 4: Advection-diffusion DG1
    run_convergence_study(
        degree=1, Ns=Ns, T_end=T_end, CFL=CFL, nu=0.001,
        label="TEST 4: Advection-Diffusion DG1, IP-DG (nu=0.001, expected O(h^2))"
    )

    print("\n" + "="*65)
    print("  ALL CONVERGENCE STUDIES COMPLETE")
    print("="*65)
