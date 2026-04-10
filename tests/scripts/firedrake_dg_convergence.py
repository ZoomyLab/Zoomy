"""
Firedrake DG convergence test: Gaussian advection on UnitSquareMesh.

Tests DG0-DG3 with the correct explicit time-stepping pattern:
  - Pre-assemble the constant mass matrix
  - Re-assemble the RHS form each time step (so updated Q is used)
  - Solve M * dQ = b for the stage increment
  - Update Q via .assign() (Firedrake-tracked operation)

Time integration:
  - DG0, DG1: Forward Euler
  - DG2, DG3: SSP-RK3

Expected convergence rates:
  DG_p should give order p+1 in L2 norm.
"""

import firedrake as fd
import numpy as np


def gaussian_advection_convergence(degree, N_list, CFL=0.15, t_end=0.05):
    """Run Gaussian advection for several mesh sizes, return L2 errors."""
    errors = []
    velocity = fd.Constant((1.0, 0.0))

    for N in N_list:
        mesh = fd.UnitSquareMesh(N, N)
        V = fd.FunctionSpace(mesh, "DG", degree)

        # Solution and test/trial functions
        Q = fd.Function(V, name="Q")
        phi = fd.TestFunction(V)

        # Initial condition: Gaussian centered at (0.25, 0.5)
        x, y = fd.SpatialCoordinate(mesh)
        sigma = fd.Constant(0.05)
        x0 = fd.Constant(0.25)
        y0 = fd.Constant(0.5)
        ic_expr = fd.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        Q.interpolate(ic_expr)

        # Exact solution at t_end: shifted Gaussian
        x0_end = fd.Constant(0.25 + t_end)
        exact_expr = fd.exp(-((x - x0_end)**2 + (y - y0)**2) / (2 * sigma**2))

        # Normal vector
        n = fd.FacetNormal(mesh)

        # Upwind DG advection form: RHS = -div(velocity * Q)
        # Volume term: velocity * Q . grad(phi)
        # Interior faces: upwind flux
        # Exterior faces: upwind flux with inflow BC (Q=0)

        # The velocity dot normal on '+' side
        vn = fd.dot(velocity, n("+"))

        # Upwind value at interior faces
        Q_up = 0.5 * (Q("+") + Q("-")) + 0.5 * fd.sign(vn) * (Q("+") - Q("-"))

        # RHS form (linear in phi, depends on Q)
        # Volume: integral of velocity * Q . grad(phi) dx
        L = fd.inner(fd.dot(velocity, fd.grad(phi)), Q) * fd.dx

        # Interior faces: - integral of upwind_flux * jump(phi) dS
        L -= fd.inner(Q_up * vn, phi("+") - phi("-")) * fd.dS

        # Exterior faces: upwind on boundary
        vn_ext = fd.dot(velocity, n)
        # outflow: use Q; inflow: use 0 (far-field)
        Q_ext = fd.conditional(vn_ext > 0, Q, fd.Constant(0.0))
        L -= fd.inner(Q_ext * vn_ext, phi) * fd.ds

        # Time step
        h = 1.0 / N
        dt_val = CFL * h / (2 * degree + 1)
        nsteps = int(np.ceil(t_end / dt_val))
        dt_val = t_end / nsteps  # adjust to hit t_end exactly

        # Pre-assemble the mass matrix (constant on a fixed mesh)
        trial = fd.TrialFunction(V)
        a_mass = fd.inner(phi, trial) * fd.dx
        M = fd.assemble(a_mass)

        # Stage increment function
        dQ = fd.Function(V, name="dQ")

        solver_params = {"ksp_type": "preonly", "pc_type": "bjacobi",
                         "sub_pc_type": "ilu"}

        def rhs_eval(Q_cur, dQ_out):
            """Evaluate dQ = M^{-1} * L(Q_cur).

            Since L references Q (a Function), and Q_cur IS Q (or Q was
            assigned to Q_cur's values before calling), re-assembling L
            picks up the current Q data.
            """
            b = fd.assemble(L)
            fd.solve(M, dQ_out, b, solver_parameters=solver_params)

        # Choose time integrator
        if degree == 0:
            # Forward Euler (sufficient for first-order spatial)
            def step(dt_val):
                rhs_eval(Q, dQ)
                Q.assign(Q + fd.Constant(dt_val) * dQ)
        elif degree == 1:
            # SSP-RK2 (Heun's method) for second-order time accuracy
            Q1_rk2 = fd.Function(V, name="Q1_rk2")
            dQ1_rk2 = fd.Function(V, name="dQ1_rk2")

            def step(dt_val):
                dt_c = fd.Constant(dt_val)
                # Save Q
                Q1_rk2.assign(Q)
                # Stage 1: Q* = Q + dt * L(Q)
                rhs_eval(Q, dQ)
                Q.assign(Q + dt_c * dQ)
                # Stage 2: Q^{n+1} = 0.5*Q^n + 0.5*(Q* + dt*L(Q*))
                rhs_eval(Q, dQ1_rk2)
                Q.assign(fd.Constant(0.5) * Q1_rk2
                         + fd.Constant(0.5) * (Q + dt_c * dQ1_rk2))
        else:
            # SSP-RK3 (Shu-Osher)
            Q1 = fd.Function(V, name="Q1")
            Q2 = fd.Function(V, name="Q2")
            Q_save = fd.Function(V, name="Q_save")
            dQ1 = fd.Function(V, name="dQ1")
            dQ2 = fd.Function(V, name="dQ2")

            def step(dt_val):
                dt_c = fd.Constant(dt_val)

                # Save original Q
                Q_save.assign(Q)

                # Stage 1: Q1 = Q + dt * L(Q)
                rhs_eval(Q, dQ)
                Q1.assign(Q + dt_c * dQ)

                # Stage 2: Q2 = 3/4 Q_save + 1/4 (Q1 + dt * L(Q1))
                # Set Q = Q1 so that L (which references Q) evaluates at Q1
                Q.assign(Q1)
                rhs_eval(Q, dQ1)
                Q2.assign(fd.Constant(3.0/4.0) * Q_save
                          + fd.Constant(1.0/4.0) * (Q1 + dt_c * dQ1))

                # Stage 3: Q_new = 1/3 Q_save + 2/3 (Q2 + dt * L(Q2))
                Q.assign(Q2)
                rhs_eval(Q, dQ2)
                Q.assign(fd.Constant(1.0/3.0) * Q_save
                         + fd.Constant(2.0/3.0) * (Q2 + dt_c * dQ2))

        # Time loop
        for istep in range(nsteps):
            step(dt_val)

        # Compute L2 error
        err = fd.errornorm(exact_expr, Q, norm_type="L2")
        errors.append(err)
        print(f"  DG{degree}  N={N:4d}  dt={dt_val:.6e}  nsteps={nsteps:6d}  L2err={err:.6e}")

    return errors


def compute_rates(errors, N_list):
    """Compute convergence rates from a list of errors."""
    rates = []
    for i in range(1, len(errors)):
        r = np.log(errors[i-1] / errors[i]) / np.log(N_list[i] / N_list[i-1])
        rates.append(r)
    return rates


if __name__ == "__main__":
    print("=" * 70)
    print("Firedrake DG advection convergence test")
    print("  Gaussian pulse, velocity=(1,0), t_end=0.05")
    print("  Correct pattern: re-assemble RHS each step")
    print("=" * 70)

    results = {}

    for degree in [0, 1, 2, 3]:
        # Mesh sizes: 3 levels for convergence
        if degree <= 1:
            N_list = [16, 32, 64, 128]
        else:
            N_list = [8, 16, 32, 64]

        print(f"\n--- DG{degree} ---")
        errors = gaussian_advection_convergence(degree, N_list, CFL=0.15, t_end=0.05)
        rates = compute_rates(errors, N_list)
        results[degree] = (N_list, errors, rates)

        print(f"  Convergence rates: {[f'{r:.2f}' for r in rates]}")
        expected = degree + 1
        print(f"  Expected rate: ~{expected}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for degree, (N_list, errors, rates) in results.items():
        avg_rate = np.mean(rates) if rates else 0
        expected = degree + 1
        status = "PASS" if abs(avg_rate - expected) < 0.5 else "FAIL"
        print(f"  DG{degree}: avg rate = {avg_rate:.2f} (expected ~{expected})  [{status}]")
