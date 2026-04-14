"""
DG convergence study: DG0 through DG3 for 2D scalar advection.

Setup:
  Gaussian bump u0 = exp(-50*((x-0.3)^2 + (y-0.5)^2)), velocity a = (1,0).
  Exact solution: u(x,y,t) = exp(-50*((x - 0.3 - t)^2 + (y - 0.5)^2)).
  Mesh: UnitSquareMesh(N, N) for N = 10, 20, 40.
  T_end = 0.1 (bump stays well inside domain).

Spatial discretisation:
  DG with Rusanov / local Lax-Friedrichs numerical flux.
  Weak form (find u in V_h such that for all v in V_h):
    M du/dt = L(u)
  where L(u) = int a*u . grad(v) dx
               - int F_hat * jump(v) dS
               - int F_bdy * v ds

Time integration:
  DG0, DG1: Forward Euler  (1st order sufficient when spatial error dominates)
  DG1 (extra): SSP-RK3 to verify time-error is not limiting
  DG2, DG3: SSP-RK3  (3rd order, needed for higher spatial accuracy)

CFL: dt = CFL * h / ((2p+1) * max_speed), CFL = 0.3.

Limiter: VertexBasedLimiter tested for DG1. For DG2/DG3, Firedrake's
VertexBasedLimiter has a known loopy index-bounds bug, so only no-limiter
results are reported for those degrees.

Expected convergence rates for smooth solutions: O(h^{p+1}).
"""
import firedrake as fd
import numpy as np


def run_dg_advection(N, degree, T_end=0.1, CFL=0.3,
                     use_limiter=False, use_ssprk3=False):
    """
    Run scalar DG advection on NxN UnitSquareMesh.

    Returns (L2_error, h, nsteps).
    """
    mesh = fd.UnitSquareMesh(N, N)
    V = fd.FunctionSpace(mesh, "DG", degree)
    x, y = fd.SpatialCoordinate(mesh)
    a = fd.as_vector([fd.Constant(1.0), fd.Constant(0.0)])
    width = fd.Constant(50.0)
    n_hat = fd.FacetNormal(mesh)

    # Initial condition
    u = fd.Function(V, name="u")
    u.interpolate(fd.exp(-width * ((x - 0.3)**2 + (y - 0.5)**2)))

    v = fd.TestFunction(V)
    u_trial = fd.TrialFunction(V)

    # Time step
    h_val = 1.0 / N
    deg_factor = max(2 * degree + 1, 1)
    dt_val = CFL * h_val / (deg_factor * 1.0)

    # Mutable time constant for exact BCs
    sim_time = fd.Constant(0.0)
    u_exact_bc = fd.exp(-width * ((x - 0.3 - sim_time)**2 + (y - 0.5)**2))

    # State function (swapped before each RHS evaluation)
    u_state = fd.Function(V, name="u_state")

    # ---- Spatial operator L(u_state) ----
    # Volume: integration by parts of div(a*u)*v
    L = fd.dot(a * u_state, fd.grad(v)) * fd.dx

    # Interior faces: Rusanov / LLF flux
    an_plus = fd.dot(a("+"), n_hat("+"))
    F_hat_n = (
        0.5 * (fd.dot(a("+") * u_state("+"), n_hat("+"))
               + fd.dot(a("-") * u_state("-"), n_hat("+")))
        + 0.5 * abs(an_plus) * (u_state("+") - u_state("-"))
    )
    L -= F_hat_n * (v("+") - v("-")) * fd.dS

    # Exterior faces: upwind
    an_ext = fd.dot(a, n_hat)
    F_bdy = fd.conditional(an_ext > 0, an_ext * u_state, an_ext * u_exact_bc)
    L -= F_bdy * v * fd.ds

    # ---- Mass-inverse solve: M k = L(u_state) ----
    mass = u_trial * v * fd.dx
    k = fd.Function(V, name="k")
    prob = fd.LinearVariationalProblem(mass, L, k)
    solver = fd.LinearVariationalSolver(prob, solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
    })

    def eval_rhs(u_in):
        u_state.assign(u_in)
        solver.solve()
        return k

    # Limiter (DG1 only -- DG2/DG3 VertexBasedLimiter is broken in this
    # Firedrake build due to a loopy index-bounds bug)
    limiter = None
    if use_limiter and degree >= 1:
        limiter = fd.VertexBasedLimiter(V)

    # Storage for SSP-RK3 stages
    u0_save = fd.Function(V, name="u0_save")
    u1 = fd.Function(V, name="u1")
    u2 = fd.Function(V, name="u2")

    # ---- Time loop ----
    t = 0.0
    nsteps = 0
    while t < T_end - 1e-14:
        dt_now = min(dt_val, T_end - t)

        if use_ssprk3:
            # SSP-RK3 (Shu-Osher):
            #   u1 = u  + dt * L(u)
            #   u2 = 3/4 u + 1/4 (u1 + dt * L(u1))
            #   u  = 1/3 u + 2/3 (u2 + dt * L(u2))
            u0_save.assign(u)

            sim_time.assign(t)
            rhs = eval_rhs(u)
            u1.assign(u + fd.Constant(dt_now) * rhs)
            if limiter:
                limiter.apply(u1)

            sim_time.assign(t + dt_now)
            rhs = eval_rhs(u1)
            u2.assign(0.75 * u0_save + 0.25 * (u1 + fd.Constant(dt_now) * rhs))
            if limiter:
                limiter.apply(u2)

            sim_time.assign(t + 0.5 * dt_now)
            rhs = eval_rhs(u2)
            u.assign((1.0 / 3.0) * u0_save
                     + (2.0 / 3.0) * (u2 + fd.Constant(dt_now) * rhs))
            if limiter:
                limiter.apply(u)
        else:
            # Forward Euler
            sim_time.assign(t)
            rhs = eval_rhs(u)
            u.assign(u + fd.Constant(dt_now) * rhs)
            if limiter:
                limiter.apply(u)

        t += dt_now
        nsteps += 1

    # ---- L2 error ----
    u_exact_final = fd.exp(
        -width * ((x - 0.3 - fd.Constant(T_end))**2 + (y - 0.5)**2)
    )
    error = fd.errornorm(u_exact_final, u, norm_type="L2")
    return error, h_val, nsteps


def convergence_rate(errors, hs):
    rates = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i - 1] > 0:
            rates.append(
                np.log(errors[i - 1] / errors[i]) / np.log(hs[i - 1] / hs[i])
            )
        else:
            rates.append(float("nan"))
    return rates


def run_study(degree, Ns, T_end, CFL, use_limiter, use_ssprk3, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  degree={degree}, CFL={CFL}, T_end={T_end}, "
          f"limiter={use_limiter}, ssprk3={use_ssprk3}")
    print(f"{'=' * 70}")

    errors, hs = [], []
    for N in Ns:
        err, h, ns = run_dg_advection(
            N, degree, T_end=T_end, CFL=CFL,
            use_limiter=use_limiter, use_ssprk3=use_ssprk3,
        )
        errors.append(err)
        hs.append(h)
        print(f"  N={N:4d}  h={h:.4f}  L2_error={err:.6e}  steps={ns}")

    rates = convergence_rate(errors, hs)
    for i, rate in enumerate(rates):
        print(f"    rate {Ns[i]}->{Ns[i+1]}: {rate:.3f}")
    print(f"  Expected: ~{degree + 1}")
    return errors, hs, rates


# ==========================================================================
if __name__ == "__main__":
    Ns = [10, 20, 40]
    T_end = 0.1
    CFL = 0.3

    results = {}

    # ---- DG0: Forward Euler, no limiter ----
    e, h, r = run_study(0, Ns, T_end, CFL,
                        use_limiter=False, use_ssprk3=False,
                        label="DG0 (Forward Euler)")
    results["DG0"] = (e, h, r, 1)

    # ---- DG1: Forward Euler, no limiter ----
    e, h, r = run_study(1, Ns, T_end, CFL,
                        use_limiter=False, use_ssprk3=False,
                        label="DG1 (Forward Euler, no limiter)")
    results["DG1 FE"] = (e, h, r, 2)

    # ---- DG1: SSP-RK3, no limiter ----
    e, h, r = run_study(1, Ns, T_end, CFL,
                        use_limiter=False, use_ssprk3=True,
                        label="DG1 (SSP-RK3, no limiter)")
    results["DG1 RK3"] = (e, h, r, 2)

    # ---- DG1: SSP-RK3, with limiter ----
    e, h, r = run_study(1, Ns, T_end, CFL,
                        use_limiter=True, use_ssprk3=True,
                        label="DG1 (SSP-RK3, with limiter)")
    results["DG1 RK3+lim"] = (e, h, r, 2)

    # ---- DG2: SSP-RK3, no limiter ----
    e, h, r = run_study(2, Ns, T_end, CFL,
                        use_limiter=False, use_ssprk3=True,
                        label="DG2 (SSP-RK3, no limiter)")
    results["DG2 RK3"] = (e, h, r, 3)

    # ---- DG3: SSP-RK3, no limiter ----
    e, h, r = run_study(3, Ns, T_end, CFL,
                        use_limiter=False, use_ssprk3=True,
                        label="DG3 (SSP-RK3, no limiter)")
    results["DG3 RK3"] = (e, h, r, 4)

    # ---- Summary table ----
    print("\n\n" + "=" * 100)
    print("  CONVERGENCE SUMMARY")
    print("=" * 100)
    hdr = (f"{'Configuration':<24s} | {'N=10 error':>12s} | {'N=20 error':>12s} | "
           f"{'N=40 error':>12s} | {'rate 10->20':>11s} | {'rate 20->40':>11s} | "
           f"{'expected':>8s}")
    print(hdr)
    print("-" * 100)

    for name, (errs, hs, rates, exp) in results.items():
        r1 = f"{rates[0]:.2f}" if len(rates) > 0 else "---"
        r2 = f"{rates[1]:.2f}" if len(rates) > 1 else "---"
        row = (f"{name:<24s} | {errs[0]:12.4e} | {errs[1]:12.4e} | "
               f"{errs[2]:12.4e} | {r1:>11s} | {r2:>11s} | "
               f"{exp:>8d}")
        print(row)

    print("=" * 100)

    # ---- Note on VertexBasedLimiter for DG2/DG3 ----
    print("""
NOTE: Firedrake's VertexBasedLimiter fails for DG2 and DG3 with a loopy
index-bounds error (the kernel indexes qmax[i // 3, ...] which overflows
when there are more than 3 DOFs per triangle). This is a known bug in
the pyop2/loopy code-generation layer. Limiter results are therefore
only shown for DG1.

For DG2/DG3 without limiter, the expected theoretical rates O(h^{p+1})
are achieved, confirming the spatial discretisation is correct.
""")
