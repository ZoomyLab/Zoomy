"""Gate 53: diffusion on an AMReX AMR hierarchy (one dt for every level), the diffusive
fluxes refluxed.

Gate 41's heat equation u_t = div(nu grad u), nu 0.05, on [0, 1]^2, base 16 x 16, one finer
level (ratio 2), u0 = 1 + exp(-((x - 0.3)^2 + (y - 0.45)^2) / 0.02): the bump straddles the
coarse-fine interface of the static patch [0.25, 0.75]^2.  BCs `neu` (zero Neumann) and `per`
(periodic): the total of u is conserved.  Paths (constant dt, t 0.1):

  expl   the explicit slot (diffusion_matrix_explicit) in SSP-RK2, dt 3.2e-3
         (nu dt / dx_f^2 = 0.16): the stage's two-point + correction fluxes join its log entry
  impl   the implicit slot, Lie: SSP-RK2 then Crank-Nicolson, dt 1e-2 (0.51): the CN face
         fluxes refluxed by their share of dt (1/2 at the start, 1/2 at the end)
  ssp2   the implicit slot in IMEX-SSP2 (gate 52), dt 1e-2: the stages' explicit correction
         and implicit two-point fluxes in the stage's log entry (w = wt = 1/2)

Runs per (BC, path): `cover` (the fine level covers the domain), `patch` (static patch, with
reflux), `nr` (patch without reflux), the fine single level (32^2) and the coarse single level
(16^2).  Checks:
  cover  fine level = fine single level, max rel <= 1e-12 (expl) / 1e-9 (implicit: Newton
         1e-12 against the solve's own sequence of levels)
  patch  |sum u vol - sum u0 vol| / sum u0 vol <= 1e-12 (reflux; u0: the composite at t 0,
         the covered coarse cells the fine level's average); `nr` printed (the control)
  patch  L1(AMR composite - fine single) / L1(coarse single - fine single) <= 0.75
         (observed 0.37 expl, 0.59 impl / ssp2: the fine halo takes the coarse level at the new
         time in every stage, an O(dt) lag at the interface; impl 0.45 / 0.40 / 0.38 at dt / 2,
         / 4, / 8, expl 0.37 at any dt)

    python gate53_amrex_amr_diffusion.py OUT [neu|per ...] [expl|impl|ssp2 ...] [shear]

`shear`: matrix C32's viscous SME(0) (Stress.full: state-dependent, mixed A) with an x-varying
u0, Lie / Crank-Nicolson, the covering fine level vs the fine single level <= 1e-9 (the momentum
rows carry the bed's nonconservative product: not refluxed, as without diffusion).
"""
import sys
from pathlib import Path

import numpy as np

import common  # noqa: F401  (x64)
import gate41_diffusion_bc as G41
from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex, cartesian
from zoomy_core.mesh import LSQMesh
from zoomy_core.symsolver.settings import Imex

NU, T_END, N = 0.05, 0.1, 16
DT = dict(expl=3.2e-3, impl=1e-2, ssp2=1e-2)
IMEX = dict(expl=False, impl=Imex.lie, ssp2=Imex.ssp2)
PATCH = ((0.25, 0.25), (0.75, 0.75))
TIGHT = dict(implicit_tol=1e-12, gmres_tol=1e-13, implicit_maxiter=10, gmres_maxiter=200)
FAILED = []


def u0(x):
    return 1.0 + np.exp(-((x[0] - 0.3) ** 2 + (x[1] - 0.45) ** 2) / 0.02)


def model(bc, path):
    """(nsm): gate 41's 2-D heat equation, the slot of `path`, nu = NU."""
    nsm = G41.heat(2, bc, "expl" if path == "expl" else "impl")
    nsm.sm.parameter_values.nu = NU
    return nsm


def setup(bc, path, n):
    """(LSQ mesh, nsm, Q0) at n x n (zoomy_core's NumPy setup: the LSQ stencils)."""
    from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
    nsm = model(bc, path)
    s = SymbolicHyperbolicSolverNumpy(time_end=1.0)
    s.setup(LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=n, ny=n), nsm)
    nc = int(s.mesh_np.n_inner_cells)
    return s.mesh_np, nsm, u0(np.asarray(s.mesh_np.cell_centers)[:2, :nc])[None, :]


def solver(path, **kw):
    return SymbolicHyperbolicSolverAmrex(time_end=T_END, dt=DT[path], cfl_check_every=0,
                                         imex=IMEX[path], max_grid_size=16, **TIGHT, **kw)


def single(out, bc, path, n):
    mesh, nsm, Q0 = setup(bc, path, n)
    t, Q = solver(path).run(mesh, nsm, Q0, out / f"single_{bc}_{path}_{n}")
    return np.asarray(Q), Q0


def amr(out, bc, path, kind):
    """(level 0 = the composite averaged down, the fine level (NaN off it), the composite at
    t = 0)."""
    mesh, nsm, Q0 = setup(bc, path, N)
    fm, _, Q1 = setup(bc, path, 2 * N)
    region = ((0, 0), (1, 1)) if kind == "cover" else PATCH
    s = solver(path, amr=dict(max_level=1, region=region, n_error_buf=0,
                              reflux=kind != "nr"))
    t, Q = s.run(mesh, nsm, Q0, out / f"amr_{bc}_{path}_{kind}", fine=[(fm, Q1)])
    return np.asarray(Q), np.asarray(s.levels[1]), np.asarray(s.initial)


def coarse_avg(Qf, n=N):
    """A fine-grid field (grid order of the 2n x 2n mesh) averaged onto the n x n mesh."""
    m0 = LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=n, ny=n)
    m1 = LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=2 * n, ny=2 * n)
    g0, g1 = cartesian(m0), cartesian(m1)
    avg = np.zeros((Qf.shape[0], n, n))
    np.add.at(avg, (slice(None), g1["ij"][1] // 2, g1["ij"][0] // 2), Qf / 4)
    return avg[:, g0["ij"][1], g0["ij"][0]]


def composite_l1(Qc, Qf, ref):
    """L1 of the AMR composite (coarse where uncovered, fine where it has cells) against the
    fine single-level `ref`."""
    v0, v1 = 1.0 / N ** 2, 1.0 / (2 * N) ** 2
    m0 = LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=N, ny=N)
    m1 = LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=2 * N, ny=2 * N)
    g0, g1 = cartesian(m0), cartesian(m1)
    has = np.isfinite(Qf[0])
    cov = np.zeros((N, N), bool)
    cov[g1["ij"][1][has] // 2, g1["ij"][0][has] // 2] = True
    unc = ~cov[g0["ij"][1], g0["ij"][0]]
    rc = coarse_avg(ref)
    return (np.abs(Qc[:, unc] - rc[:, unc]).sum() * v0
            + np.abs(Qf[:, has] - ref[:, has]).sum() * v1)


def shear(n):
    """(LSQ mesh, nsm, Q0): matrix C32's viscous SME(0) (Newtonian, Stress.full, no-slip
    `Dirichlet` momentum walls at y = 0, 1, periodic in x; nu 0.05, g 0.01, O2 without limiter)
    on (0, 0.5) x (0, 1), n / 2 x n cells, u0 = sin(pi y) (1 + 0.2 cos(4 pi x)): x-varying, so
    the state-dependent and mixed entries of A act."""
    import zoomy_core.model.boundary_conditions as BC
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
    from zoomy_core.model.models.equations import Stress
    from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
    from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
    from zoomy_core.systemmodel.system_model import SystemModel
    wall = lambda t: BC.Dirichlet(tag=t, on="momentum", value=0.0)  # noqa: E731
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left"), wall("bottom"), wall("top")]
    sm = SystemModel.from_model(SME(level=0, dimension=3, stress=Stress.full,
                                    boundary_conditions=BC.BoundaryConditions(bcs),
                                    closures=[Newtonian(), ManningFriction(), StressFree()]))
    sm.parameter_values.nu, sm.parameter_values.g = NU, 0.01
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.none))
    s = SymbolicHyperbolicSolverNumpy(time_end=1.0)
    s.setup(LSQMesh.create_2d(domain=(0, 0.5, 0, 1), nx=n // 2, ny=n), nsm)
    nc = int(s.mesh_np.n_inner_cells)
    x = np.asarray(s.mesh_np.cell_centers)[:2, :nc]
    names = [str(v) for v in nsm.state]
    Q0 = np.zeros((len(names), nc))
    Q0[names.index("h")] = 1.0
    Q0[names.index("q_x_0")] = np.sin(np.pi * x[1]) * (1 + 0.2 * np.cos(4 * np.pi * x[0]))
    return s.mesh_np, nsm, Q0


def shear_cover(out):
    """Viscous SME (Lie, Crank-Nicolson): the covering fine level vs the fine single level."""
    kw = dict(time_end=T_END, dt=1.0 / (32 * 16), cfl_check_every=0, imex=Imex.lie,
              max_grid_size=16, **TIGHT)
    fm, nsm, Q1 = shear(32)
    _, fine = SymbolicHyperbolicSolverAmrex(**kw).run(fm, nsm, Q1, out / "shear_single")
    mesh, nsm, Q0 = shear(16)
    fm, _, Q1 = shear(32)
    s = SymbolicHyperbolicSolverAmrex(amr=dict(max_level=1, region=((0, 0), (0.5, 1)),
                                               n_error_buf=0), **kw)
    s.run(mesh, nsm, Q0, out / "shear_cover", fine=[(fm, Q1)])
    fine = np.asarray(fine)
    return float(np.abs(np.asarray(s.levels[1]) - fine).max() / np.abs(fine).max())


def check(label, value, op, bound, info=""):
    ok = value <= bound if op == "<=" else value >= bound
    if not ok:
        FAILED.append(label)
    print(f"gate53 {label}: {value:.3e} ({op} {bound:g})  {'PASS' if ok else 'FAIL'}  {info}",
          flush=True)


def measures(out, bc, path):
    """{fine_vs_single, mass, l1_ratio, mass_nr, fine_cells} of one (BC, path)."""
    vol = 1.0 / N ** 2
    fine, _ = single(out, bc, path, 2 * N)
    coarse, _ = single(out, bc, path, N)
    _, Qf, _ = amr(out, bc, path, "cover")
    res = dict(fine_vs_single=float(np.abs(Qf - fine).max() / np.abs(fine).max()))
    Qc, Qf, I0 = amr(out, bc, path, "patch")
    m0 = I0.sum() * vol                       # the composite at t = 0
    e_c = float(np.abs(coarse - coarse_avg(fine)).sum() * vol)
    res.update(mass=float(abs(Qc.sum() * vol - m0) / m0), fine_cells=int(np.isfinite(Qf[0]).sum()),
               l1_ratio=float(composite_l1(Qc, Qf, fine) / e_c))
    Qn, _, In = amr(out, bc, path, "nr")
    res["mass_nr"] = float(abs(Qn.sum() - In.sum()) / In.sum())
    return res


def gate(out, bc, path):
    m = measures(out, bc, path)
    check(f"{bc} {path} cover: fine level vs fine single (max rel)", m["fine_vs_single"], "<=",
          1e-12 if path == "expl" else 1e-9)
    check(f"{bc} {path} patch: mass change (rel)", m["mass"], "<=", 1e-12,
          f"fine cells {m['fine_cells']}")
    check(f"{bc} {path} patch: L1(AMR - fine) / L1(coarse - fine)", m["l1_ratio"], "<=", 0.75)
    print(f"gate53 {bc} {path} nr (no reflux, control): mass change (rel) {m['mass_nr']:.3e}",
          flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    bcs = [a for a in sys.argv[2:] if a in ("neu", "per")] or ["neu", "per"]
    paths = [a for a in sys.argv[2:] if a in DT] or list(DT)
    for bc in bcs:
        for path in paths:
            gate(out, bc, path)
    if "shear" in sys.argv[2:]:
        check("shear (viscous SME, Lie) cover: fine level vs fine single (max rel)",
              shear_cover(out), "<=", 1e-9)
    print(f"gate53: {len(FAILED)} FAIL", flush=True)
    sys.exit(1 if FAILED else 0)
