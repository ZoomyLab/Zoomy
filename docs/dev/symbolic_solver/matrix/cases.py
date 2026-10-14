"""The matrix cases, each defined once (mesh, model, IC, BCs, t_end,
numerics) and run identically on every backend.  README.md explains them."""
from dataclasses import dataclass
from functools import partial

import numpy as np

import common as C  # gates/common.py (x64)
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, NavierSlip, Newtonian, StressFree
from zoomy_core.model.models.equations import Stress
from zoomy_core.numerics import (Limiter, NumericalSystemModel, ReconstructedVariables,
                                 ReconstructionSpec)
from zoomy_core.symsolver.settings import Imex, LinearSolver, PressureSolve, Projection
from zoomy_core.symsolver.strategies.chorin import predictor_nsm
from zoomy_core.systemmodel.system_model import SystemModel

ALL = ("serial", "mpi2", "mpi4", "gpu1")
REF = ("mpi8",)        # the fine JAX references: 8 ranks (C7: = serial)


@dataclass
class Run:
    """One solver run: ``make() -> (mesh, nsm)`` or, Chorin,
    ``(mesh, (pred, press, corr))``; ``q0(names, xc)`` overrides the model's
    initial condition."""
    key: str
    make: object
    t_end: float
    cfl: float = 0.9
    mood: bool = False
    dt: float = None
    chorin: bool = False
    time_order: int = 1
    pressure_solve: PressureSolve = PressureSolve.linear   # Chorin: linear | newton
    imex: object = False                # False | Imex.lie | Imex.ssp2
    solver: dict = None                 # further solver settings (tolerances)
    q0: object = None
    configs: tuple = ("serial",)
    backends: tuple = ("jax", "foam", "amrex")
    custom: object = None               # custom(backend, config, work) -> {metric: value}
    launch: bool = True                 # False: the custom run launches its own ranks
    perturb: float = 0.0                # relative perturbation of Q0 (round-off sensitivity runs)


def tri_rect(domain, nx, ny, nsm, tags=("left", "right", "bottom", "top")):
    """Triangles: each quad of an nx x ny grid cut along alternating diagonals
    (gate 16's tri_bump on any rectangle), boundary tags as create_2d."""
    from zoomy_core.mesh.base_mesh import BaseMesh
    x0, x1, y0, y1 = domain
    vid = lambda i, k: i * (ny + 1) + k   # noqa: E731
    X = np.array([[x, y] for x in np.linspace(x0, x1, nx + 1)
                  for y in np.linspace(y0, y1, ny + 1)]).T
    tris = []
    for i in range(nx):
        for k in range(ny):
            a, b, c, d = vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)
            tris += ([(a, b, c), (a, c, d)] if (i + k) % 2 else [(a, b, d), (b, c, d)])
    ftag = {}
    for k in range(ny):
        ftag[tuple(sorted((vid(0, k), vid(0, k + 1))))] = 0
        ftag[tuple(sorted((vid(nx, k), vid(nx, k + 1))))] = 1
    for i in range(nx):
        ftag[tuple(sorted((vid(i, 0), vid(i + 1, 0))))] = 2
        ftag[tuple(sorted((vid(i, ny), vid(i + 1, ny))))] = 3
    base = BaseMesh.from_arrays(X, np.array(tris).T, "triangle", face_tags=ftag,
                                tag_names=dict(enumerate(tags)))
    return ensure_lsq_mesh(base, nsm)


def on_triangles(make, nx, ny, domain):
    """The case ``make`` with its mesh replaced by triangles."""
    def m():
        _, systems = make()
        nsm = systems[0] if isinstance(systems, tuple) else systems
        return tri_rect(domain, nx, ny, nsm), systems
    return m


def prepare(r):
    """Backend-neutral: (mesh, systems, Q0, names), the mesh with its LSQ
    stencils, Q0 in the mesh's cell order.  Every backend takes these."""
    mesh, systems = r.make()
    nsm = systems[0] if r.chorin else systems
    mesh = ensure_lsq_mesh(mesh, nsm)
    names = [str(s) for s in nsm.state]
    xc = np.asarray(mesh.cell_centers)[:, :int(mesh.n_inner_cells)]
    if r.q0 is not None:
        Q0 = r.q0(names, xc)
    else:
        Q0 = nsm.sm.initial_conditions.apply(xc, np.zeros((len(names), xc.shape[1])))
    Q0 = np.asarray(Q0, dtype=float)
    if r.perturb:
        Q0 = Q0 * (1.0 + r.perturb * np.random.default_rng(0).uniform(-1.0, 1.0, Q0.shape))
    return mesh, systems, Q0, names


# ── C3: lake at rest over a Gaussian bed ───────────────────────────────────
DT_LAKE = 0.01                          # 100 steps of a constant dt


def lake(order, nx=40):
    sm = SystemModel.from_model(C.sme(0, 3, ("left", "right", "bottom", "top"), "wall"))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        b = 0.2 * np.exp(-((x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2) / 2.0)
        q[names.index("b")], q[names.index("h")] = b, 0.5 - b
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx), nsm


#: C3 on plain SWE (the hydrostatic pressure in its conservative flux, P = 0): every
#: hydrostatic-reconstruction flux (ANALYSIS 50)
LAKE_SWE = ("PositiveHLL", "PositiveRusanov", "PositiveNonconservativeRusanov",
            "PositiveQuasilinearRusanov")


def lake_swe(order, riemann, nx=24):
    """C3's lake at rest on the hand-written ``SWE`` (pressure in the flux), ``riemann`` by name."""
    import zoomy_core.numerics as N
    from zoomy_core.model.models.swe import SWE
    tags = ("left", "right", "bottom", "top")
    sm = SystemModel.from_model(SWE(dimension=2, boundary_conditions=BC.BoundaryConditions(
        [BC.FromModel(tag=t, definition="wall") for t in tags])))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        b = 0.2 * np.exp(-((x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2) / 2.0)
        q[names.index("b")], q[names.index("h")] = b, 0.5 - b
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=getattr(N, riemann),
        reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx), nsm


# ── C4: quasi-1-D channel over a bump, steady Bernoulli flow ───────────────
G = 9.81
CHANNEL = dict(sub=dict(q=4.42, h_out=2.0, t_end=3.0),      # subcritical
               trans=dict(q=1.53, h_out=None, t_end=3.0))    # transcritical
LENGTH, WIDTH, NY = 25.0, 1.0, 2


def bed(x):
    return 0.2 * np.exp(-(x - 10.0) ** 2 / 2.0)


def channel(variant, order, nx, dim=2):
    """sub: inflow q (Dirichlet), outflow h (Dirichlet); trans: inflow q
    (InflowOutflow), outflow extrapolation; 2-D: walls on the sides (Wall)."""
    c, qx = CHANNEL[variant], "q_x_0" if dim == 2 else "q_0"
    if variant == "sub":
        ends = [BC.Dirichlet("left", on=qx, value=c["q"]),
                BC.Dirichlet("right", on="h", value=c["h_out"])]
    else:
        ends = [BC.InflowOutflow(tag="left", prescribe_fields={2: c["q"]}),
                BC.Extrapolation(tag="right")]
    sides = [BC.Wall(tag="bottom"), BC.Wall(tag="top")] if dim == 2 else []
    model = SME(level=0, dimension=dim + 1,
                closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=BC.BoundaryConditions(ends + sides))
    sm = SystemModel.from_model(model)
    names = [str(s) for s in sm.state]
    assert names[:3] == ["b", "h", qx], names     # index 2 above

    def ic(x):              # the exact steady state; the run relaxes to the discrete one
        return np.array([bed(x[0]), bernoulli(variant, np.array([x[0]]))[0], c["q"], 0.0][:len(names)])
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    if dim == 1:
        return LSQMesh.create_1d(domain=(0, LENGTH), n_inner_cells=nx), nsm
    return LSQMesh.create_2d(domain=(0, LENGTH, 0, WIDTH), nx=nx, ny=NY), nsm


def smooth_1d(order, nx):
    """C1 in 1-D: a Gaussian surface bump at rest, walls."""
    sm = SystemModel.from_model(C.sme(0, 2, ("left", "right"), "wall"))
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        q[names.index("h")] = 0.5 + 0.05 * np.exp(-(x[0] - 5.0) ** 2 / 2.0)
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod,
                                              variables=ReconstructedVariables.conservative))
    return LSQMesh.create_1d(domain=(0, 10), n_inner_cells=nx), nsm


# ── C13: periodic in x, walls in y ──────────────────────────────────────────
PERIODIC = dict(L=10.0, u=0.3, nx=40, ny=2, t_end=1.0)


def periodic(order, x0, riemann=None):
    """A Gaussian surface bump (periodic distance) at x0 in a uniform flow u; ``riemann``:
    the NSM's Riemann class (default: the free-surface default, which rebuilds ``hinv`` from
    the reconstructed depth; ``NonconservativeRusanov`` reads both cells' aux at a face)."""
    P = PERIODIC
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left"),
           BC.Wall(tag="bottom"), BC.Wall(tag="top")]
    model = SME(level=0, dimension=3, closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=BC.BoundaryConditions(bcs))
    sm = SystemModel.from_model(model)

    def ic(x):
        d = (x[0] - x0 + P["L"] / 2) % P["L"] - P["L"] / 2
        h = 0.5 + 0.05 * np.exp(-d ** 2 / 2.0)
        return np.array([0.0, h, h * P["u"], 0.0])
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, riemann=riemann, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    return LSQMesh.create_2d(domain=(0, P["L"], 0, 1.0), nx=P["nx"], ny=P["ny"]), nsm


# ── C14: IMEX, SWE + Manning friction (implicit source) ─────────────────────
MANNING = 0.5          # g n^2 |u| / h^(4/3) ~ 0.3 /s at the bump; dt ~ 0.05


def _ncr():
    from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
    return NonconservativeRusanov


def imex_bump(order, nx):
    mesh, nsm = C.bump_2d(order, nx=nx, flow=(0.1, 0.05))
    nsm.sm.parameter_values.n = MANNING
    return mesh, nsm


def bernoulli(variant, x):
    """Exact steady depth: h + q^2/(2 g h^2) + b = E.  sub: E from the
    outflow depth, subcritical branch; trans: E from critical flow at the
    crest, subcritical upstream, supercritical downstream."""
    c = CHANNEL[variant]
    q = c["q"]
    hc = (q * q / G) ** (1 / 3)
    if variant == "sub":
        E = c["h_out"] + q * q / (2 * G * c["h_out"] ** 2) + bed(LENGTH)
        sup = np.zeros_like(x, dtype=bool)
    else:
        E = bed(10.0) + 1.5 * hc
        sup = x > 10.0
    out = np.empty_like(x, dtype=float)
    for i, (xi, s) in enumerate(zip(x, sup)):
        f = lambda h: h + q * q / (2 * G * h * h) + bed(xi) - E   # noqa: E731
        lo, hi = ((1e-6, hc) if s else (hc, E + 1.0))
        if f(hc) >= 0:                  # the crest itself: critical
            out[i] = hc
            continue
        for _ in range(200):            # bisection, f changes sign on [lo, hi]
            mid = 0.5 * (lo + hi)
            if (f(mid) > 0) == (not s):
                hi = mid
            else:
                lo = mid
        out[i] = 0.5 * (lo + hi)
    return out


# ── C8: VAM(1) Chorin, the gate-21 channel ─────────────────────────────────
def vam():
    from gate21_parallel_chorin import B, NX
    pred, press, corr = B.split()
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=NX, ny=NX // 10)
    return mesh, (predictor_nsm(pred, press, corr), press, corr)


def vam_q0(names, xc):
    from gate21_parallel_chorin import B
    q = np.zeros((len(names), xc.shape[1]))
    b = B.BUMP(xc[0])
    q[names.index("b")] = b
    q[names.index("h")] = np.maximum(np.where(xc[0] < 1.0, B.H_RES - b, B.H_DRY), B.H_DRY)
    return q


# ── C10: VAM(1) Chorin convergence, a smooth wave over the bump ────────────
C10_N = (30, 60, 120)
C10_VARIANTS = ((1, 1), (2, 1), (2, 2))     # (space order, time_order); O1 = gate 7's config
WAVE = dict(H=0.34, a=0.01, x0=-0.5, s=0.3, t_end=0.2)   # waves stay off the ends


def vam_split(friction=False):
    """VAM(1) split for C10 (the gate-21 boundaries with no inflow: left
    q = r = 0 (Dirichlet), right P = 0, sides Extrapolation); ``friction``:
    Manning bed friction (C31), in the predictor's implicit source.  ~16 s; not
    cached: a second NSM built from the same split SMs after a JAX setup fails
    in deepcopy (Max(Abs(eigenvalues(0, 0, 0, 0))) not comparable)."""
    from zoomy_core.model.initial_conditions import Constant
    from zoomy_core.model.models.vam import VAM
    from zoomy_core.systemmodel.system_model import _DT_SYMBOL
    bcs = ([BC.Dirichlet("left", on=f, value=0.0) for f in ("q_x_0", "q_x_1", "r_0", "r_1")]
           + [BC.Dirichlet("right", on=f"P_{i}", value=0.0) for i in range(2)]
           + [BC.Extrapolation(tag=t) for t in ("bottom", "top")])
    model = VAM(level=1, dimension=3, boundary_conditions=bcs,
                closures=[Newtonian(), *([ManningFriction()] if friction else []),
                          StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(_DT_SYMBOL, system_model=sm)
    return r.SM_pred, r.SM_press, r.SM_corr


def vam_periodic(nx=32):
    """VAM(1) Chorin on a channel periodic in x (sides Extrapolation): the pressure couples
    across the seam (the assembled AMG solves need the periodic partners in their sparsity,
    ANALYSIS 50).  Fresh split per call (see `vam_split`)."""
    from zoomy_core.model.initial_conditions import Constant
    from zoomy_core.model.models.vam import VAM
    from zoomy_core.systemmodel.system_model import _DT_SYMBOL
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left"),
           *(BC.Extrapolation(tag=t) for t in ("bottom", "top"))]
    model = VAM(level=1, dimension=3, boundary_conditions=bcs,
                closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(_DT_SYMBOL, system_model=sm)
    mesh = LSQMesh.create_2d(domain=(0.0, 2.0, 0.0, 0.1), nx=nx, ny=2)
    return mesh, (predictor_nsm(r.SM_pred, r.SM_press, r.SM_corr), r.SM_press, r.SM_corr)


def periodic_q0(names, xc):
    q = np.zeros((len(names), xc.shape[1]))
    q[names.index("h")] = 0.5 + 0.05 * np.sin(np.pi * xc[0])
    q[names.index("q_x_0")] = 0.1 * q[names.index("h")]
    return q


def vam_wave(order, nx, friction=False):
    """Predictor numerics as gate 7 (`predictor_nsm`), reconstruction order ``order``."""
    pred, press, corr = vam_split(friction)
    nsm = predictor_nsm(pred, press, corr, None if order == 1 else
                        ReconstructionSpec(order=order, limiter=Limiter.minmod))
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.1), nx=nx, ny=2)   # quasi-1-D
    return mesh, (nsm, press, corr)


def wave_q0(names, xc):
    from gate21_parallel_chorin import B
    q = np.zeros((len(names), xc.shape[1]))
    b = B.BUMP(xc[0])
    eta = WAVE["H"] + WAVE["a"] * np.exp(-(xc[0] - WAVE["x0"]) ** 2 / (2 * WAVE["s"] ** 2))
    q[names.index("b")], q[names.index("h")] = b, eta - b
    return q


# ── C31: IMEX time order, constant dt on a fixed mesh ──────────────────────
#: scheme -> (imex, time_order); Lie with SSP-RK1 / RK2, IMEX-SSP2(2,2,2)
C31_SWE = {"lie1": (Imex.lie, 1), "lie2": (Imex.lie, 2), "ssp2": (Imex.ssp2, None)}
C31_VAM = {"t1": (Imex.lie, 1), "t2": (Imex.lie, 2), "ssp2": (Imex.ssp2, 2)}
C31_K = (0, 1, 2, 3)        # dt / 2^k
C31_DT = dict(swe=4e-3, vam=8e-3)


def vam_friction(nx, predictor=False):
    """C10's wave (space O1, gate 7's predictor) with Manning n = MANNING;
    ``predictor``: the predictor alone (no projection)."""
    mesh, (nsm, press, corr) = vam_wave(1, nx, friction=True)
    nsm.sm.parameter_values.n = MANNING
    return mesh, (nsm if predictor else (nsm, press, corr))


# ── C18-C24: compressible Euler (gates/euler_common.py), 3-D and 2-D ────────
VEL = {3: (1.0, 0.5, 0.25), 2: (1.0, 0.5)}


def sod(dim, order, n, axis, thin=2):
    import euler_common as E
    return E.box(dim, n, axis, thin=thin), E.nsm_of(E.sod_system(dim, axis), order)


def wave(dim, variant, n):
    """Periodic smooth density wave; variant o1, o2none, o2minmod."""
    import euler_common as E
    from zoomy_core.mesh import LSQMesh
    order, lim = {"o1": (1, Limiter.minmod), "o2none": (2, Limiter.none),
                  "o2minmod": (2, Limiter.minmod)}[variant]
    mesh = (LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=n, ny=n, nz=n) if dim == 3
            else LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=n, ny=n))
    return mesh, E.nsm_of(E.wave_system(dim, VEL[dim]), order, limiter=lim)


def pulse(order, n, centre=0.5, amp=1.0, width=0.02):
    """Gaussian pressure/density pulse in the walled unit cube (gates 3d c/d/f)."""
    import euler_common as E
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.systemmodel.euler import compressible_euler

    def ic(x):
        b = amp * np.exp(-sum((x[d] - centre) ** 2 for d in range(3)) / width)
        return E.conserved(1.0 + 0.5 * b, np.zeros(3), 1.0 + 2.0 * b)
    sm = compressible_euler(3, E.GAMMA, boundary_conditions=BC.BoundaryConditions(
        [BC.Wall(tag=t) for t in E.TAGS[3]]), initial_conditions=IC.UserFunction(function=ic))
    return LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=n, ny=n, nz=n), E.nsm_of(sm, order)


def _c22():
    from custom import ad3d
    return [Run(f"{k}_{p}", None, 0.0, custom=lambda b, c, w, k=k, p=p: ad3d(k, p),
                backends=("jax",)) for k in ("o1_dt", "o1_adaptive", "o2_dt") for p in ("fwd", "rev")]


EU = ("jax", "foam", "amrex")


def _viscous():
    from gate31_parallel_implicit_diffusion import viscous
    mesh, nsm = viscous(3, nx=20)
    return mesh, nsm     # AMReX 3-D raises NotImplementedError: "unsupported"


# ── C32: viscous shear decay between walls (exact: u_t = nu u_yy) ──────────
SHEAR = dict(nu=0.05, g=0.01, t_end=0.5, nx=8, ny=(20, 40, 80), steps=16)   # dt = dy / steps


def shear_exact(variant, y, t):
    """u(y, t): no-slip sin(pi y) e^(-nu pi^2 t), free-slip cos(pi y) e^(-nu pi^2 t).
    ``ctrl`` (negative control): the no-slip profile on free-slip walls.
    ``inviscid``: nu = 0 on no-slip walls, u stays sin(pi y).  ``*_split``: the same on the
    split wall."""
    f = np.cos if variant == "free" else np.sin
    nu = 0.0 if variant.startswith("inviscid") else SHEAR["nu"]
    return f(np.pi * y) * np.exp(-nu * np.pi ** 2 * t)


def shear_decay(variant, ny, order=2, limiter=Limiter.none):
    """SWE (SME(0)) + Newtonian, stress=Stress.full: h = 1, v = 0, u = u(y), periodic in x,
    walls at y = 0, 1.  Every x-derivative vanishes, so h stays 1, v stays 0 and
    the x-momentum equation is exactly u_t = nu u_yy (diffusion_matrix
    A[q_x, q_x, y, y] = nu, A[q_x, h, y, y] = -nu q_x / h; no factor 2: that
    is the xx entry).  The viscous wall is the BC's ghost: noslip =
    ``Dirichlet`` on the momentum (face value 0, ghost -q), free = ``Wall``
    (tangential momentum kept: ghost = q, a zero normal gradient); ``ctrl`` =
    the no-slip profile on ``Wall``s (the walls must act: the error is O(1)); ``inviscid``
    = nu 0 on the no-slip walls (what the Dirichlet wall's Riemann state alone does);
    ``noslip_split`` / ``inviscid_split`` = the split wall, ``Wall`` face (the reflective
    Riemann state) with a ``Dirichlet`` momentum ghost (ghost -q for the stencils).  Second
    order, no limiter: a smooth profile (minmod clips at its extremum).  g is
    small (0.01): the exact solution does not depend on it, but the Rusanov
    dissipation (wave speed sqrt(g h), a third-order error) would otherwise
    cancel part of the O(dy^2) error of the viscous flux, and the measured
    order jumps (3.4, 4.6, 1.4 at g = 9.81, nx 20 / 40 / 80 / 160)."""
    nu = 0.0 if variant.startswith("inviscid") else SHEAR["nu"]
    if variant.endswith("_split"):
        wall = lambda t: BC.Wall(tag=t, ghost=BC.Dirichlet(on="momentum", value=0.0))  # noqa: E731
    elif variant in ("noslip", "inviscid"):
        wall = lambda t: BC.Dirichlet(tag=t, on="momentum", value=0.0)  # noqa: E731
    else:
        wall = lambda t: BC.Wall(tag=t)  # noqa: E731
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left"),
           wall("bottom"), wall("top")]
    model = SME(level=0, dimension=3, boundary_conditions=BC.BoundaryConditions(bcs),
                stress=Stress.full, closures=[Newtonian(), ManningFriction(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.parameter_values.nu, sm.parameter_values.g = nu, SHEAR["g"]
    names = [str(s) for s in sm.state]
    ix, ih = names.index("q_x_0"), names.index("h")
    A = sm.diffusion_matrix            # the reduction u_t = nu u_yy rests on these two entries
    assert A[ix, ix, 1, 1] == sm.parameters.nu, A[ix, ix, 1, 1]
    assert A[ix, ih, 1, 1] == -sm.parameters.nu * sm.state[ix] / sm.state[ih], A[ix, ih, 1, 1]

    def ic(x):
        q = np.zeros(len(names))
        q[names.index("h")] = 1.0
        q[names.index("q_x_0")] = shear_exact(variant, x[1], 0.0)
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=limiter))
    nx = SHEAR["nx"]
    return LSQMesh.create_2d(domain=(0, nx / ny, 0, 1), nx=nx, ny=ny), nsm


# ── C33: vertical viscosity of the shallow moment model (exact: expm) ─────
#: SME(2) + Newtonian (bulk) + NavierSlip (bottom) + StressFree (surface),
#: uniform periodic state at h = 1: the flux divergence vanishes, the moments
#: obey q' = J q (the source is linear in q at fixed h)
MOMENT = dict(level=2, nu=0.05, lambda_s=0.5, h=1.0, q0=(0.5, 0.2, -0.1), nx=8, t_end=1.0,
              dt0=0.05, ks=(0, 1, 2))      # dt = dt0 / 2^k


def moment_decay(order=2):
    """The C33 model and mesh: 1-D, periodic, flat bed, uniform (h, q_0..q_2)."""
    M = MOMENT
    bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
           BC.Periodic(tag="right", periodic_to_physical_tag="left")]
    model = SME(level=M["level"], dimension=2, boundary_conditions=BC.BoundaryConditions(bcs),
                closures=[Newtonian(), NavierSlip(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.parameter_values.nu, sm.parameter_values.lambda_s = M["nu"], M["lambda_s"]
    names = [str(s) for s in sm.state]

    def ic(x):
        q = np.zeros(len(names))
        q[names.index("h")] = M["h"]
        for i, v in enumerate(M["q0"]):
            q[names.index(f"q_{i}")] = v
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
    return LSQMesh.create_1d(domain=(0, M["nx"]), n_inner_cells=M["nx"]), nsm   # dx 1: CFL-safe dt


def moment_jacobian():
    """J = d(source_q)/d(q_0..q_L) at h = h0 and the parameters of C33,
    by sympy differentiation of the SystemModel's source (the q rows); the
    source must be linear in q at fixed h (asserted: source_q - J q = 0)."""
    import sympy as sp
    mesh, nsm = moment_decay()
    sm = nsm.sm
    st = list(sm.state)
    names = [str(s) for s in st]
    mom = [s for s in st if str(s).startswith("q_")]
    S = sp.Matrix(sm.source)[[names.index(str(m)) for m in mom], :]
    val = {str(k): float(v) for k, v in sm.parameter_values.items()}
    val["h"], val["hinv"] = MOMENT["h"], 1.0 / MOMENT["h"]      # hinv: the NSM's aux 1/h
    sub = {s: val[s.name] for s in S.free_symbols if s.name in val}     # by name
    J = S.jacobian(mom).subs(sub)
    resid = (S - S.jacobian(mom) * sp.Matrix(mom)).subs(sub)
    assert sp.simplify(resid) == sp.zeros(len(mom), 1), f"source not linear in q: {resid}"
    return np.array(J.tolist(), dtype=float)


def moment_exact(t):
    """q(t) = expm(t J) q(0)."""
    from scipy.linalg import expm
    return expm(t * moment_jacobian()) @ np.array(MOMENT["q0"])


# ── the matrix ─────────────────────────────────────────────────────────────
def _c1():
    runs = [Run(f"o{o}_n{n}", partial(C.bump_2d, o, nx=n), 2.5,
                configs=ALL if (o, n) == (2, 40) else ("serial",))
            for o in (1, 2) for n in (20, 40, 80)]
    return runs + [Run("ref_n320", partial(C.bump_2d, 2, nx=320), 2.5,
                       configs=REF, backends=("jax",))]


def _c2():
    shock = lambda n: C.bump_2d(2, nx=n, flow=(0.3, 0.1))   # noqa: E731
    return ([Run(f"n{n}", partial(shock, n), 0.5, mood=True) for n in (40, 80)]
            + [Run("ref_n320", partial(shock, 320), 0.5, mood=True, configs=REF,
                   backends=("jax",))])


JF = ("jax", "foam")           # triangles: no AMReX (Cartesian grids only)
JFA = ("jax", "foam", "amrex")
C4_N = (25, 50, 100)
#: Q0 perturbation of the sensitivity runs (``*_pert``, JAX): the AMReX - JAX
#: tolerance is 10x the JAX - perturbed JAX difference where one exists
PERTURB = 1e-14


def _c8():
    tri = on_triangles(vam, 60, 6, (-1.5, 1.5, 0.0, 0.3))
    return [Run("vam", vam, 0.05, chorin=True, q0=vam_q0, configs=ALL),
            Run("vam_newton", vam, 0.05, chorin=True, q0=vam_q0,
                pressure_solve=PressureSolve.newton),
            Run("vam_t2", vam, 0.05, chorin=True, q0=vam_q0, time_order=2),
            Run("vam_tri", tri, 0.05, chorin=True, q0=vam_q0, backends=JF),
            # periodic channel: each backend's default pressure solve (JAX / NumPy: the
            # assembled 'auto', foam / AMReX: hypre) vs the matrix-free GMRES
            Run("per", vam_periodic, 0.05, dt=0.01, chorin=True, q0=periodic_q0),
            Run("per_auto", vam_periodic, 0.05, dt=0.01, chorin=True, q0=periodic_q0,
                backends=("jax",), solver=dict(pressure_linear_solver=LinearSolver.auto))]


def _c35():
    from custom import restart
    return [Run(f"{s}_{p}", None, 0.0, configs=("serial", "mpi2") if s == "c1" else ("serial",),
                custom=partial(lambda b, c, w, s=s, p=p: restart(s, p, b, c, w)))
            for s in ("c1", "c8") for p in ("full", "half", "restart", "typo")]


def _c15():
    from custom import ad
    return [Run(f"{k}_{p}", None, 0.0, custom=partial(lambda b, c, w, k=k, p=p: ad(k, p, _comm(c))),
                configs=("serial", "mpi2", "gpu1"), backends=("jax",))
            for k in ("hyperbolic", "imex", "chorin") for p in ("fwd", "rev")]


def _c16():
    from custom import couple
    return [Run(pair, None, 0.0, custom=partial(lambda b, c, w, pair=pair: couple(pair, _ranks(c))),
                configs=("serial", "mpi2"), backends=(pair.split("_")[0],), launch=False)
            for pair in ("jax_jax", "jax_foam", "foam_foam", "amrex_amrex", "amrex_jax")]


def _c17():
    from custom import amr
    return [Run(case, None, 0.0, custom=lambda b, c, w, case=case: amr(case, c, w),
                configs=("serial", "mpi2", "gpu1"), backends=("amrex",), launch=False)
            for case in ("cover_o2", "cover_dry", "wave16", "wave32", "box", "boxnr", "lake", "dry")]


def _c27():
    from custom import time_order, time_order_control
    runs = [Run(f"{part}_t{to}", None, 0.0, custom=lambda b, c, w, part=part, to=to:
                time_order(b, part, to, w))
            for part in ("ode", "wave") for to in (1, 2, 3)]
    return runs + [Run("ode_control", None, 0.0, custom=lambda b, c, w: time_order_control(),
                       backends=("jax",))]


def _c31():
    """SWE (C14, space O1, nx 40): dt / 2^k and the reference dt / 32;
    VAM(1) (C10's wave + Manning, nx 60): dt / 2^k (Richardson), and its
    predictor alone (``pred``).  Cross-backend at k = 0."""
    swe = [Run(f"swe_{s}_{k if k == 'ref' else f'k{k}'}", partial(imex_bump, 1, 40), 0.3,
               dt=C31_DT["swe"] / (32 if k == "ref" else 2 ** k), imex=imex,
               solver=dict(time_order=to, cfl_check_every=0),
               backends=JFA if (s, k) == ("ssp2", 0) else ("jax",))
           for s, (imex, to) in C31_SWE.items() for k in (*C31_K, "ref")]
    vam = [Run(f"vam_{s}_k{k}", partial(vam_friction, 60), WAVE["t_end"],
               dt=C31_DT["vam"] / 2 ** k, chorin=True, time_order=to, imex=imex, q0=wave_q0,
               solver=dict(cfl_check_every=0, projection=Projection.increment),
               backends=JFA if (s, k) == ("ssp2", 0) else ("jax",))
           for s, (imex, to) in C31_VAM.items() for k in C31_K]
    pred = [Run(f"pred_ssp2_k{k}", partial(vam_friction, 60, True), WAVE["t_end"],
                dt=C31_DT["vam"] / 2 ** k, imex=Imex.ssp2, q0=wave_q0,
                solver=dict(cfl_check_every=0), backends=("jax",)) for k in C31_K]
    return swe + vam + pred


DIFF_KINDS = ("1d", "quad", "tri")
DIFF_PATHS = ("expl", "impl")
DIFF_ALL = ("dir0", "neu", "dirL", "per", "steady")


def _c28():
    from custom import diffusion_orders, diffusion_run
    runs = [Run(f"orders_{k}_{p}", None, 0.0, custom=lambda b, c, w, k=k, p=p: diffusion_orders(k, p, b),
                backends=("jax",)) for k in DIFF_KINDS for p in DIFF_PATHS]
    return runs + [Run(f"{k}_{p}_{case}", None, 0.0,
                       custom=lambda b, c, w, k=k, p=p, case=case: diffusion_run(b, k, p, case, w),
                       backends=JF if k == "tri" else JFA)   # AMReX: Cartesian
                   for k in DIFF_KINDS for p in DIFF_PATHS for case in DIFF_ALL]


def _c29():
    from custom import lsq_closure
    return [Run(f"p{p}_{bc}", None, 0.0, custom=lambda b, c, w, p=p, bc=bc: lsq_closure(p, bc, b),
                backends=("jax",)) for p in (2, 3) for bc in ("dir", "neu")]


def _c30():
    from custom import lsq_polynomials
    return [Run("polynomials", None, 0.0, custom=lambda b, c, w: lsq_polynomials(b),
                backends=("jax",))]


def _c34():
    from custom import inplane_operator
    return [Run(f"n{n}", None, 0.0, custom=lambda b, c, w, n=n: inplane_operator(b, n, w),
                backends=EU if n == 16 else ("jax",)) for n in (16, 32, 64)]


IMPLICIT = dict(implicit_tol=1e-12, gmres_tol=1e-13, implicit_maxiter=10, gmres_maxiter=60)


# ── C36: the full Newtonian stress at level 1 (ANALYSIS 45h) ─────────────────
FULL1 = dict(nu=1e-2, nx=16, t_end=0.02, dt=1e-3)


def full_stress_level1(nx=FULL1["nx"]):
    """SME(1) + Newtonian with stress=Stress.full (in-plane tau_de, tau_zz, the complete
    tau_ez, the vertical balance of Marche 2007 eq. (3.5) with the pressure (3.11)) on the
    periodic [0,1]^2: smooth h, q_x_0, q_x_1, q_y_0, q_y_1, flat bed, slip-free bed
    (lambda_s 0); O2 minmod, the viscous terms implicit (IMEX)."""
    sides = (("left", "right"), ("right", "left"), ("bottom", "top"), ("top", "bottom"))
    bcs = [BC.Periodic(tag=a, periodic_to_physical_tag=b_) for a, b_ in sides]
    sm = SystemModel.from_model(SME(level=1, dimension=3, stress=Stress.full,
                                    closures=[Newtonian(), NavierSlip(), StressFree()],
                                    boundary_conditions=BC.BoundaryConditions(bcs)))
    sm.parameter_values.nu = FULL1["nu"]
    names = [str(s_) for s_ in sm.state]
    tp = 2 * np.pi

    def ic(x):
        q = np.zeros(len(names))
        h = 1.0 + 0.1 * np.sin(tp * x[0]) * np.cos(tp * x[1])
        q[names.index("h")] = h
        q[names.index("q_x_0")] = 0.1 * np.cos(tp * x[0]) * np.sin(tp * x[1]) + 0.05
        q[names.index("q_y_0")] = 0.05 * np.sin(tp * (x[0] + x[1]))
        q[names.index("q_x_1")] = 0.02 * h * np.sin(tp * x[1])
        q[names.index("q_y_1")] = -0.02 * h * np.cos(tp * x[0])
        return q
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.minmod))
    return LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=nx, ny=nx), nsm


def _c32():
    """The viscous walls (no-slip, free-slip, split) on ny 20 / 40 / 80 (JAX); foam and
    AMReX on ny 20; the controls ``ctrl`` (walls act), ``inviscid`` and ``inviscid_split``
    (JAX)."""
    return [Run(f"{v}_n{ny}", partial(shear_decay, v, ny), SHEAR["t_end"],
                dt=1.0 / (ny * SHEAR["steps"]), imex=Imex.lie, solver=IMPLICIT,
                backends=JFA if ny == SHEAR["ny"][0] else ("jax",))
            for v in ("noslip", "free", "noslip_split") for ny in SHEAR["ny"]] + [
        Run(f"{v}_n20", partial(shear_decay, v, SHEAR["ny"][0]), SHEAR["t_end"],
            dt=1.0 / (SHEAR["ny"][0] * SHEAR["steps"]), imex=Imex.lie, solver=IMPLICIT,
            backends=("jax",)) for v in ("ctrl", "inviscid", "inviscid_split")]


def _c33():
    """IMEX (implicit-Euler source) and explicit SSP-RK2 (the source in the
    explicit step) at constant dt = dt0 / 2^k."""
    M = MOMENT
    return [Run(f"{integ}_k{k}", partial(moment_decay, 2), M["t_end"], dt=M["dt0"] / 2 ** k,
                imex=Imex.lie if integ == "imex" else False, solver=IMPLICIT)
            for integ in ("imex", "rk2") for k in M["ks"]]


#: why a backend has no run of a case (measure(): the label of its n/a row)
NA = {("C29", b): "gate 44 calls the JAX runtime (derivative_aux)" for b in ("foam", "amrex")}
NA.update({("C30", b): "gate 39 calls the JAX runtime (lsq_gradient_per_field)"
           for b in ("foam", "amrex")})


def _ranks(config):
    return int(config[3:]) if config.startswith("mpi") else 1


def _comm(config):
    if not config.startswith("mpi") and config != "gpu2":
        return None
    from mpi4py import MPI
    return MPI.COMM_WORLD


ALL5 = ALL + ("gpu2",)
CASES = {
    "C1": ("smooth: bump at rest, walls, t = 2.5", _c1()),
    "C2": ("shock: bump in flow (0.3, 0.1), walls, O2 + MOOD", _c2()),
    "C3": ("lake at rest over a bed bump, 100 steps",
           [Run(f"o{o}", partial(lake, o, 24), 100 * DT_LAKE, dt=DT_LAKE) for o in (1, 2)]
           + [Run(f"swe_{f}_o{o}", partial(lake_swe, o, f), 100 * DT_LAKE, dt=DT_LAKE,
                  backends=JFA) for f in LAKE_SWE for o in (1, 2)]),
    "C4": ("channel over a bump vs Bernoulli (inflow/outflow/wall BCs)",
           [Run(f"{v}_o{o}_n{n}", partial(channel, v, o, n), CHANNEL[v]["t_end"])
            for v in CHANNEL for o in (1, 2) for n in C4_N]
           + [Run(f"{v}_o{o}_n{n}_pert", partial(channel, v, o, n), CHANNEL[v]["t_end"],
                  backends=("jax",), perturb=PERTURB)
              for v in CHANNEL for o in (1, 2) for n in C4_N]),
    "C5": ("radial dry dam break, O2 eta + MOOD",
           [Run("n48", partial(C.dry_dambreak_2d, nx=48), 0.02, cfl=0.45, mood=True,
                configs=ALL5)]),
    "C8": ("VAM(1) Chorin channel (gate 21): linear / Newton / time_order 2 / triangles", _c8()),
    "C10": ("VAM(1) Chorin convergence: smooth wave over the bump",
            [Run(f"s{o}_t{to}_n{n}", partial(vam_wave, o, n), WAVE["t_end"], chorin=True,
                 time_order=to, q0=wave_q0)
             for o, to in C10_VARIANTS for n in C10_N]),
    "C11": ("1-D: smooth wave (walls), channel vs Bernoulli (inflow/outflow)",
            [Run(f"smooth_o{o}_n{n}", partial(smooth_1d, o, n), 2.5, backends=JFA)
             for o in (1, 2) for n in (40, 80, 160)]
            + [Run("smooth_ref_n1280", partial(smooth_1d, 2, 1280), 2.5, backends=("jax",))]
            + [Run(f"sub_o{o}_n{n}", partial(channel, "sub", o, n, 1), CHANNEL["sub"]["t_end"],
                   backends=JFA) for o in (1, 2) for n in C4_N]
            + [Run(f"sub_o{o}_n{n}_pert", partial(channel, "sub", o, n, 1),
                   CHANNEL["sub"]["t_end"], backends=("jax",), perturb=PERTURB)
               for o in (1, 2) for n in C4_N]),
    "C12": ("triangles: C1 bump and C4 channel",
            [Run(f"bump_o{o}_n{n}", on_triangles(partial(C.bump_2d, o, nx=4), n, n, (0, 10, 0, 10)),
                 2.5, backends=JF) for o in (1, 2) for n in (10, 20, 40)]
            # refined in x AND y (ny = n / 25): at a fixed ny the triangles
            # stretch with n, and OpenFOAM's leastSquares gradient at the
            # walls loses its order on them (foam O2 n100 ny 2: 1.6e-3 > n50)
            + [Run(f"sub_o{o}_n{n}", on_triangles(partial(channel, "sub", o, 4), n,
                                                  max(1, n * NY // 50), (0, LENGTH, 0, WIDTH)),
                   CHANNEL["sub"]["t_end"], backends=JF) for o in (1, 2) for n in C4_N]),
    "C13": ("periodic in x: shifted bump in a uniform flow",
            [Run(f"o{o}_x{x0}", partial(periodic, o, x0), PERIODIC["t_end"],
                 configs=("serial", "mpi2", "mpi4")) for o in (1, 2) for x0 in (2.5, 7.5)]
            # the seam face reads the partner cell's aux (face_aux): NonconservativeRusanov's
            # face kernel takes hinv from both cells' aux
            + [Run(f"nc_o{o}_x{x0}", partial(periodic, o, x0, _ncr()), PERIODIC["t_end"],
                   backends=JFA) for o in (1, 2) for x0 in (2.5, 7.5)]),
    "C14": ("IMEX: SWE + Manning friction (implicit source), bump in flow, t = 0.3",
            [Run(f"o{o}_n{n}", partial(imex_bump, o, n), 0.3, imex=Imex.lie,
                 configs=ALL5 if (o, n) == (2, 40) else ("serial",))
             for o in (1, 2) for n in (20, 40, 80)]
            + [Run("ref_n160", partial(imex_bump, 2, 160), 0.3, imex=Imex.lie, backends=("jax",))]),
    "C15": ("AD (JAX): jvp vs central FD, reverse vs forward, dJ/dg", _c15()),
    "C16": ("preCICE: coupled (2 x 100 cells, window 1e-3, O1) vs single domain", _c16()),
    "C17": ("AMR (AMReX, gate 37): covering, static patch, reflux, lake, regrid", _c17()),
    "C18": ("Euler 3-D Sod along x, y, z (n x 2 x 2), HLL, t = 0.2",
            [Run(f"o{o}_n{n}_ax{a}", partial(sod, 3, o, n, a), 0.2, cfl=0.45, mood=o == 2,
                 backends=EU) for o in (1, 2) for n in (50, 100) for a in (0, 1, 2)]),
    "C19": ("Euler 3-D smooth periodic wave, [0,1]^3, t = 0.1",
            [Run(f"{v}_n{n}", partial(wave, 3, v, n), 0.1, cfl=0.45, backends=EU)
             for v in ("o1", "o2none", "o2minmod") for n in (8, 16, 24)]),
    "C20": ("Euler 3-D walled box 16^3, centred pulse, t = 0.6",
            [Run(f"o{o}", partial(pulse, o, 16), 0.6, cfl=0.45, mood=o == 2, backends=EU)
             for o in (1, 2)]),
    "C21": ("Euler 3-D parallel: pulse 12^3 (centre 0.4), Sod along y 60 x 4 x 4",
            [Run(f"pulse_o{o}", partial(pulse, o, 12, 0.4), 0.3, cfl=0.45, mood=o == 2,
                 configs=("serial", "mpi2", "mpi4"), backends=EU) for o in (1, 2)]
            + [Run("sod_y", partial(sod, 3, 2, 60, 1, 4), 0.2, cfl=0.45, mood=True,
                   configs=("serial", "mpi2", "mpi4"), backends=EU)]),
    "C22": ("Euler 3-D AD (JAX): pulse 10^3, dJ/dgamma, t = 0.1", _c22()),
    "C23": ("Euler 3-D cross-backend: acoustic pulse (amp 0.01, width 0.04), t = 0.3",
            [Run(f"o{o}_n{n}", partial(pulse, o, n, 0.4, 0.01, 0.04), 0.3, cfl=0.45, backends=EU)
             for o in (1, 2) for n in (8, 16, 24)]),
    "C24": ("Euler 2-D: Sod along x, y (n x 2); periodic wave in [0,1]^2",
            [Run(f"sod_o{o}_n{n}_ax{a}", partial(sod, 2, o, n, a), 0.2, cfl=0.45, mood=o == 2,
                 backends=EU) for o in (1, 2) for n in (50, 100) for a in (0, 1)]
            + [Run(f"{v}_n{n}", partial(wave, 2, v, n), 0.1, cfl=0.45, backends=EU)
               for v in ("o1", "o2none", "o2minmod") for n in (16, 32, 64)]),
    "C26": ("implicit diffusion (IMEX, Newtonian, Stress.full, nu 1e-3; gate 31), 20^2, t 0.05",
            [Run("visc", _viscous, 0.05, imex=Imex.lie, solver=dict(
                implicit_tol=1e-12, gmres_tol=1e-13, implicit_maxiter=10, gmres_maxiter=60),
                configs=ALL5)]),
    "C27": ("SSP-RK time order 1 / 2 / 3 (gate 40): forced SWE, exact ODE and a smooth wave", _c27()),
    "C28": ("diffusion blocks at the boundaries (gates 41, 43): heat equation, dir0 / neu / dirL / per",
            _c28()),
    "C29": ("LSQ derivatives with the boundary closure (gate 44): quads, p 2 / 3, dir / neu", _c29()),
    "C30": ("LSQ polynomial reproduction (gate 39): 1-D, quads, jittered quads, triangles, hexahedra",
            _c30()),
    "C31": ("IMEX time order (Lie vs IMEX-SSP2(2,2,2)): SWE + Manning (C14), VAM(1) wave + Manning",
            _c31()),
    "C32": ("viscous shear decay between walls (SWE + Newtonian, Stress.full): u_t = nu u_yy, exact; "
            "no-slip, free-slip and split (Wall face, Dirichlet ghost) walls", _c32()),
    "C33": ("vertical viscosity of SME(2) (Newtonian + NavierSlip): uniform state, q(t) = expm(t J) q0",
            _c33()),
    "C34": ("in-plane viscous operator vs the exact div(h tau'), mixed derivatives (gate 45)",
            _c34()),
    "C35": ("restart: 0 -> T vs 0 -> T/2, restart -> T (C1 O2 adaptive; C8 increment projection)",
            _c35()),
    "C36": ("full Newtonian stress at level 1 (SME(1), Stress.full): periodic smooth state, IMEX",
            [Run("sme1", full_stress_level1, FULL1["t_end"], dt=FULL1["dt"], imex=Imex.lie,
                 solver=IMPLICIT, backends=("jax",))]),
}


# ── the NumPy backend (serial): every serial JAX run, but the fine JAX references
#    the checks read, the round-off sensitivity runs (*_pert), the gate-40 control,
#    AD (C15, C22), preCICE (C16) and AMR (C17)
NUMPY_SKIP = ("C15", "C16", "C17", "C22")
NUMPY_REFS = {("C11", "smooth_ref_n1280"), ("C14", "ref_n160"), ("C27", "ode_control")}
for _case, (_, _runs) in CASES.items():
    for _r in _runs if _case not in NUMPY_SKIP else ():
        if ("jax" in _r.backends
                and "serial" in _r.configs and (_case, _r.key) not in NUMPY_REFS
                and not _r.key.endswith("_pert")):
            _r.backends = _r.backends + ("numpy",)
