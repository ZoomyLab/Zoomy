# VAM Chorin time order: Escalante et al. 2024 vs our symbolic Chorin (2026-10-13)

## 0. Bottom line
- Escalante et al. 2024 (2XP7LGXZ) use **the same approximate projection as ours**: they substitute the corrector into
  the constraints, then discretize with a compact 3-point `P_xx`. Their scheme is not idempotent either.
- Their order evidence refines space and time together at **fixed CFL 0.4**, on a solitary wave away from the walls.
  That test hides the O(dt * s) approximate-projection error (s = O(dx^2) on resolved modes, so O(dx^3) at fixed CFL).
  Our C10 is that kind of test and already gives 2.01 on JAX / foam / AMReX; C31 is a fixed-mesh test they never ran.
- Recommended: an **incremental (lagged) pressure**, ABC 2000's "version (1)". It changes the template only, keeps L_a
  and gives order 2.00 in a Fourier model at every wavenumber (sec. 3). Run the diagnostics in sec. 4, item 1 first.

## 1. Their time discretization (2XP7LGXZ, sec. 3.1-3.4)
- "a two-stage second-order TVD Runge-Kutta scheme [37] is adopted to obtain second-order accuracy in time"; "for each
  of the kth stage, k in {1,2}, a two-step projection-correction method is used".
- Eq. (11): the PVM/MUSCL hyperbolic step U^(k-1) -> ~U^(k), with **no pressure term**. Eq. (12a):
  `(U^(k) - ~U^(k))/dt + T(U^(k), d_x U^(k), P^(k), d_x P^(k), d_x b) = 0`; (12b-c): `I_1 = I_2 = 0` at U^(k).
  Final: `U^{n+1} = 1/2 U^n + 1/2 U^(2)`. So the pressure is **non-incremental** (full P^(k) per stage), with no
  Crank-Nicolson and no increment form. Friction `-tau` is semi-implicit "at the end of the second step" (Remark 2).
- Evidence (sec. 4.2): "The integration time is T = 1 s., and the CFL number is 0.4"; "a reference solution is computed
  with the described second-order scheme and 51.200 cells. Free-outflow boundary conditions ... domain is [-150, 150] m".
  L1 orders at N = 3200: h / hu0 / hw0 2.03 (Table 1); hu1 1.97, hw1 1.99, hw2 1.81 (Table 2). "the theoretical order
  of accuracy is achieved" is a joint space-time order: **they show no fixed-mesh time study.**

## 2. Their space discretization of the elliptic step
- 1-D, uniform, **collocated** ("the mid-point x_i of each cell I_i ... P_i(t) = P(x_i, t)"); FD for the pressure,
  FV for the predictor.
- **Approximate projection, built exactly like ours.** "we shall substitute U^(k) from (12a) in equations (12b)-(12c),
  resulting in the Poisson-like equations" (15) `a1 d_xx p0 + a2 d_x p0 + a3 p0 + a4 d_xx p1 + ... = RHS_1`.
  (16): `d_x P_i = (P_{i+1} - P_{i-1})/(2dx)`, `d_xx P_i = (P_{i+1} - 2P_i + P_{i-1})/dx^2`. The corrector is a central
  "second-order point value approximation in the center of the cell" (sec. 3.4); the constraint uses central `d_x U` (17).
  The composition D(G P) would be the wide `(P_{i+2} - 2P_i + P_{i-2})/(4dx^2)`; they use "compact-centered" FD.
- Solver: "an iterative method similar to the one described in [25]". In [25] (86BQYCRP, 2019, eqs. (41)-(43)),
  T_(j), C_(j) are "tridiagonal and symmetrical", solved by block Jacobi plus Thomas to tol 1e-8: the compact operator,
  never D∘G. In 1-D the compact Laplacian is SPD-like and sees the odd-even mode, so null space and indefiniteness never arise.
- Same group, other choices:
  - KXDAIVFX (2021): collocated but EXACT, mixed form `[H/dt B; B^T 0]` (20). "The matrix in (20) is symmetric due to
    the duality relation (9). However, a naive approach where Eqs. (17b) and (17c) are discretised independently from
    each other would lead to a non-symmetric matrix." B's identity blocks (21) make B^T H^-1 B SPD. They mention
    "the incremental method (see e.g. [29])" but present non-incremental results.
  - 6BVZFAHD (2023): pressure "on the staggered grid [x_i, x_{i+1}]" plus "an incremental decomposition of the
    non-hydrostatic unknowns to increase the order of accuracy of the projection method ... (see [49])" ([49] =
    Guermond-Minev-Shen 2006). Eqs. (15)-(17): `P^{n+1} = P^n + ~P^{n+1}`; T(P^n) explicit, T(~P^{n+1}) implicit.

## 3. Why their test shows 2 and our C31 shows 1
- Ours = Escalante 2024 = Almgren-Bell-Crutchfield 2000 (ABC) **version (2)**: pressure-free predictor, project the
  velocity, P = phi. ABC Table 1, version (2): `U^n = P~^n U^0 + sum_i P~^i (dt N^{n+1/2-i})`; the exact projection
  gives `U^0 + sum_i P (dt N)`. An update made i steps ago has been projected i times. ABC: version (2) "can
  accumulate the imprint of the pressure gradient from previous times".
- Mode analysis (mine). With P~ = (I - Pi) + s Pi (Pi the oblique projector onto range(G), s = 1 - (D∘G)/L_a), the
  off-manifold part settles at `c ~ dt s/(1-s) Pi F`: an **O(dt) error with constant s/(1-s)**. On resolved modes
  s ~ (k dx)^2/4, so at fixed CFL the error is O(dx^3) and invisible; modes or closures with s = O(1) are order 1.
- Our data agree: P(P(U)) - P(U) is 5.5e-2 / 1.9e-2 / 5.1e-3 of the correction (about dx^2). The C31 error peaks "at the
  closed left end" (matrix README), where the L_a and D∘G closures differ at O(1); Escalante's test has no wall.
  ABC: "the difficulty with using approximate projections is not a question of formal accuracy on smooth problems".
- Fourier model (scratchpad/toy/fourier_projection.py): linear 1-layer non-hydrostatic system, periodic, central D/G,
  SSP-RK2, upwind-dissipative predictor, T = 2, m = 100..800 steps.

  | points/wavelength (s) | exact projection | approximate (ours) | approximate L_a + incremental P |
  |-----------------------|------------------|--------------------|---------------------------------|
  | 20 (4.9e-3)           | 2.00 2.00 2.00   | 1.77 1.60 1.40     | 2.00 2.00 2.00                  |
  | 80 (3.1e-4)           | 2.00 2.00 2.00   | 1.99 1.98 1.95     | 2.00 2.00 2.00                  |
  | 4 (8.4e-2)            | 2.01 2.00 2.00   | 1.15 1.06 1.02     | 2.01 2.00 2.00                  |

  - Approximate at fixed CFL (20 -> 40 -> 80 points per wavelength, dt / 2 each): 2.05 / 2.03, Escalante's Table 1
    behaviour. Incremental with SSP-RK2-averaged P rows: still 2.00. No growth over 4000 steps at CFL 0.45 / 0.9 (stability.py).
  - Why: G P^n in the predictor removes Pi F up to O(dt), so the approximate projection only sees an O(dt^2) defect
    and c = O(s dt^2). ABC: with versions (1)/(3) "the initial pressure converges to the correct value (that which
    would be found with infinitely many iterations of an exact projection)"; (2)/(4) "the pressure does not converge".
- An exact projection per stage is order 2: Sanderse & Koren 2012, "the velocity attains the classical order of
  accuracy without further constraints". A time-dependent G (our frozen h per stage) adds conditions only from 3rd
  order on (sec. 3.5). This agrees with the prototype's 1.98.

## 4. Options for the symbolic Chorin solver, ranked (effort / risk)
1. **Diagnostics (no code changes, about half a day).** (a) Spectrum of `S = I - (D∘G) L_a^-1` on C8 quads and
   triangles: the prototype's composed R(P) and the current `press` residual as matvecs, dense eig at 1440 dofs.
   rho(S) < 1 means items 2-5 are stable and L_a-preconditioned GMRES converges at rate rho; today's residual recursion
   `r' = S(r + dt D F)` needs the same condition. (b) C31 on a periodic or far-wall setup, nx 30 / 60 / 120: if the
   O(dt) constant then shrinks like dx^2, the wall closure is what caps C31.
2. **Incremental pressure (ABC v1; GMS 2006; Escalante 2023 eqs. (15)-(17)). Effort LOW, risk MEDIUM. Do this first.**
   - chorin.py `project()` only: (i) Q1 = correct(corr, Q*) with the current P rows (= P^n); (ii) solve `press` on Q1
     for dP (same L_a and ghosts; zero Dirichlet P BCs stay homogeneous, time-dependent ones need P_b^{n+1} - P^n);
     (iii) Q = correct(corr, put(Q1, rows, dP)), then put P^n + dP.
   - Exact by linearity: T is affine in P with (h, b) coefficients, and h is frozen through the correction.
   - P^0: one non-incremental projection at t = 0. IMEX-SSP2 is unchanged (`B_i = Y_p - R_i` sees the total
     `-dt T(P_i)`). No runtime changes.
   - Risk: ABC say v1 "can produce a lower divergence ... however, it is not as robust as version (2)", and is
     "intimately coupled to the details of the particular advection method"; halving the CFL helped.
   - Gate: C31 vam ssp2 >= 1.8; C10, C8 (triangles) and C26 unchanged within tolerance on JAX / foam / AMReX.
3. **ABC version (5), fallback. Effort LOW-MEDIUM, risk LOW-MEDIUM.** v1 for the velocity, then a loose second solve
   ("reduce the residual by only one order of magnitude") for the full pressure: "fully second-order accurate for
   smooth problems ... lacks the instability of version (1)". A second `pressure()` call with a looser tolerance.
4. **Defect correction / Richardson on P~. Effort LOW; does not fix the order.** k sweeps `P += L_a^-1 (b - R_exact(P))`
   scale the O(dt) constant by s^k; the order stays 1 on a fixed mesh, and the iteration diverges if rho(S) > 1.
5. **Exact projection with L_a-preconditioned GMRES. Effort MEDIUM, risk LOW on 1-D / quads, rho(S) on triangles.**
   Needs a `LinSolve(op, b, M=...)` slot in the vocabulary and each runtime (JAX gmres M=, foam rt.H, AMReX), plus the
   prototype's fixes (solve on b/|b|; matvec |x| op(x/|x|)). AMG-factor **only L_a** and apply D∘G matrix-free, so the
   more-than-63-colour stencil-of-stencil pattern is never built. D∘G's indefiniteness does not matter if
   spec(L_a^-1 D∘G) lies in a disk around 1 (item 1a).
6. **Adjoint-consistent D/G pair (D := -M^-1 G^T, e.g. Green-Gauss face average, or KXDAIVFX's B/B^T). Effort HIGH,
   risk HIGH.** ABC: with "D and G ... skew-adjoint; i.e., D = -G^T ... the discrete projection ... is a discrete
   orthogonal projection", so D∘G is negative semidefinite on any mesh, and SPD if VAM's algebraic terms (2p1, 6(p0 - p1))
   give B full rank (cf. KXDAIVFX (21)). Prerequisite: a symbolic check that VAM's pressure terms are the constraint's
   adjoint for some energy mass matrix. Costs: adjoint operators on 3 runtimes plus SBP-like closures. Risks: G^T is
   inconsistent on skewed cells; the "2^d-dimensional kernel for G" causes "artificial compatibility constraints".
7. **Staggered / face-based (MAC / Rhie-Chow). Effort VERY HIGH; not recommended.** It needs a face-flux constraint
   (VAM has non-conservative h d_x u0 terms); Escalante 2023 staggers only in 1-D.

**Recommended path:** item 1, then item 2 (about 1-2 days including the C31 / C10 / C8 matrix reruns on 3 backends).
If 2 is fragile on triangles, use 3. Do item 5 only if a constraint residual at round-off is needed (coupling, energy).
Item 6 is the only route to an SPD exact projection on triangles.

## 5. Sources and library gaps (all DOIs checked against Crossref)
- In the library: 2XP7LGXZ (10.1016/j.jcp.2024.112882); 86BQYCRP (10.1007/s10915-018-0849-9); KXDAIVFX
  (10.1007/s10915-021-01552-6); 6BVZFAHD (10.1007/s10915-023-02299-y); Guermond-Minev-Shen 2006 J4IJXP8A
  (10.1016/j.cma.2005.10.010, **no PDF**).
- NOT in the library: Almgren-Bell-Crutchfield 2000, 10.1137/S1064827599357024 (read from the LBNL report
  ccse.lbl.gov/Publications/almgren/AlmBelCru/lblreport.ps.gz); Almgren-Bell-Szymczak 1996, 10.1137/S1064827593244213
  (abstract only); Sanderse-Koren 2012, 10.1016/j.jcp.2011.11.028 (CWI preprint ir.cwi.nl/pub/19520); Brown-Cortez-
  Minion 2001, 10.1006/jcph.2001.6715 (content NOT verified); Bell-Colella-Glaz 1989, 10.1016/0021-9991(89)90151-4;
  Brasey-Hairer 1993, 10.1137/0730025; Escalante-Morales de Luna-Castro 2018, 10.1016/j.amc.2018.06.035 (grid not checked).
