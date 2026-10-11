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
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
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
    q0: object = None
    configs: tuple = ("serial",)
    backends: tuple = ("jax", "foam", "amrex")


def prepare(r):
    """Backend-neutral: (mesh, systems, Q0, names), the mesh with its LSQ
    stencils, Q0 in the mesh's cell order.  Every backend takes these."""
    mesh, systems = r.make()
    nsm = systems[0] if r.chorin else systems
    nsm.spectral_radius_method = "eig"     # one choice for CPU and GPU runs
    nsm._call = None
    mesh = ensure_lsq_mesh(mesh, nsm)
    names = [str(s) for s in nsm.state]
    xc = np.asarray(mesh.cell_centers)[:, :int(mesh.n_inner_cells)]
    if r.q0 is not None:
        Q0 = r.q0(names, xc)
    else:
        Q0 = nsm.sm.initial_conditions.apply(xc, np.zeros((len(names), xc.shape[1])))
    return mesh, systems, np.asarray(Q0, dtype=float), names


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
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
    return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx), nsm


# ── C4: quasi-1-D channel over a bump, steady Bernoulli flow ───────────────
G = 9.81
CHANNEL = dict(sub=dict(q=4.42, h_out=2.0, t_end=30.0),     # subcritical
               trans=dict(q=1.53, h_out=None, t_end=30.0))   # transcritical
LENGTH, WIDTH, NY = 25.0, 1.0, 2


def bed(x):
    return 0.2 * np.exp(-(x - 10.0) ** 2 / 2.0)


def channel(variant, order, nx):
    """sub: inflow q (Dirichlet), outflow h (Dirichlet); trans: inflow q
    (InflowOutflow), outflow extrapolation; walls on the sides (Wall)."""
    c = CHANNEL[variant]
    if variant == "sub":
        ends = [BC.Dirichlet("left", on="q_x_0", value=c["q"]),
                BC.Dirichlet("right", on="h", value=c["h_out"])]
    else:
        ends = [BC.InflowOutflow(tag="left", prescribe_fields={2: c["q"]}),
                BC.Extrapolation(tag="right")]
    model = SME(level=0, dimension=3,
                closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=BC.BoundaryConditions(
                    ends + [BC.Wall(tag="bottom"), BC.Wall(tag="top")]))
    sm = SystemModel.from_model(model)
    names = [str(s) for s in sm.state]
    assert names == ["b", "h", "q_x_0", "q_y_0"], names     # index 2 above

    def ic(x):              # the exact steady state; the run relaxes to the discrete one
        return np.array([bed(x[0]), bernoulli(variant, np.array([x[0]]))[0], c["q"], 0.0])
    sm.initial_conditions = IC.UserFunction(function=ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
    return LSQMesh.create_2d(domain=(0, LENGTH, 0, WIDTH), nx=nx, ny=NY), nsm


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
    from gate7_chorin import pred_nsm
    from gate21_parallel_chorin import B, NX
    pred, press, corr = B.split()
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=NX, ny=NX // 10)
    return mesh, (pred_nsm(pred, press, corr), press, corr)


def vam_q0(names, xc):
    from gate21_parallel_chorin import B
    q = np.zeros((len(names), xc.shape[1]))
    b = B.BUMP(xc[0])
    q[names.index("b")] = b
    q[names.index("h")] = np.maximum(np.where(xc[0] < 1.0, B.H_RES - b, B.H_DRY), B.H_DRY)
    return q


# ── the matrix ─────────────────────────────────────────────────────────────
def _c1():
    runs = [Run(f"o{o}_n{n}", partial(C.bump_2d, o, nx=n), 2.5,
                configs=ALL if (o, n) == (2, 80) else ("serial",))
            for o in (1, 2) for n in (40, 80, 160)]
    return runs + [Run("ref_n320", partial(C.bump_2d, 2, nx=320), 2.5,
                       configs=REF, backends=("jax",))]


def _c2():
    shock = lambda n: C.bump_2d(2, nx=n, flow=(0.3, 0.1))   # noqa: E731
    return ([Run(f"n{n}", partial(shock, n), 0.5, mood=True) for n in (40, 80)]
            + [Run("ref_n320", partial(shock, 320), 0.5, mood=True, configs=REF,
                   backends=("jax",))])


CASES = {
    "C1": ("smooth: bump at rest, walls, t = 2.5", _c1()),
    "C2": ("shock: bump in flow (0.3, 0.1), walls, O2 + MOOD", _c2()),
    "C3": ("lake at rest over a bed bump, 100 steps",
           [Run(f"o{o}", partial(lake, o), 100 * DT_LAKE, dt=DT_LAKE) for o in (1, 2)]),
    "C4": ("channel over a bump, steady Bernoulli",
           [Run(f"{v}_o{o}_n{n}", partial(channel, v, o, n), CHANNEL[v]["t_end"])
            for v in CHANNEL for o in (1, 2) for n in (50, 100, 200)]),
    "C5": ("radial dry dam break, O2 eta + MOOD",
           [Run("n48", partial(C.dry_dambreak_2d, nx=48), 0.02, cfl=0.45, mood=True,
                configs=ALL)]),
    "C8": ("VAM(1) Chorin channel (gate 21)",
           [Run("vam", vam, 0.05, chorin=True, q0=vam_q0, configs=ALL)]),
}
