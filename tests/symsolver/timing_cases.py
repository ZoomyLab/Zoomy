"""Registry of the timing cases (README.md): representative runs of the symbolic
solvers, each sized so the march takes about 1-20 s on one CPU core.

A case is a matrix ``cases.Run`` (``TimingCase.run``) marched ``steps`` constant
steps of a dyadic ``dt`` (the step count is exact on every backend).  Importing
this module imports no solver: the builders import the matrix when called."""
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import timing_paths  # noqa: F401  (the matrix and gates on sys.path)

#: implicit solves (Newton / GMRES tolerances): matrix.py's C14 and C26 AMReX - JAX bound
IMPLICIT = ("<=", 1e-8, "matrix C14 / C26: implicit Newton and GMRES tolerances")


@dataclass(frozen=True)
class Agree:
    """Agreement with the JAX serial run: ``metric`` is ``matrix.relmax`` or
    ``matrix.l1rel``; ``tol`` a matrix threshold (op, value, source) or the
    name of one in ``matrix`` (``RO``, ``PRESS``, ``TH:<key>``)."""
    metric: str
    tol: object

    def threshold(self):
        if not isinstance(self.tol, str):
            return self.tol
        import matrix
        return matrix.TH[self.tol[3:]] if self.tol.startswith("TH:") else getattr(matrix, self.tol)


RO_MAX = Agree("relmax", "RO")
GPU = {"jax:gpu1": RO_MAX}

DEFAULT_CONFIGS = {"jax": ("serial", "gpu1"), "foam": ("serial",), "amrex": ("serial", "gpu1")}


@dataclass(frozen=True)
class TimingCase:
    key: str
    title: str
    make: Callable                     # () -> (mesh, nsm) or (mesh, (pred, press, corr))
    steps: int
    dt: float
    mood: bool = False
    imex: bool = False
    chorin: bool = False
    solver: dict = None
    q0: Callable = None
    cfl: float = 0.9
    configs: dict = field(default_factory=lambda: DEFAULT_CONFIGS)
    agree: dict = field(default_factory=dict)     # "backend:config" -> Agree

    @property
    def t_end(self):
        return self.steps * self.dt

    def run(self):
        """The matrix ``cases.Run`` of this case."""
        import cases
        return cases.Run(self.key, self.make, self.t_end, cfl=self.cfl, mood=self.mood, dt=self.dt,
                         chorin=self.chorin, imex=self.imex, solver=self.solver, q0=self.q0)


# ── builders (the matrix modules are imported when called) ─────────────────
def _swe(order, nx):
    import common as C
    return C.bump_2d(order, nx=nx)


def _imex(order, nx):
    import cases
    return cases.imex_bump(order, nx)


def _vam(nx):
    from gate7_chorin import pred_nsm
    from gate21_parallel_chorin import B
    from zoomy_core.mesh import LSQMesh
    pred, press, corr = B.split()
    mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=nx, ny=nx // 10)
    return mesh, (pred_nsm(pred, press, corr), press, corr)


def _vam_q0(names, xc):
    import cases
    return cases.vam_q0(names, xc)


def _euler(order, n):
    import cases
    return cases.pulse(order, n)


def _diffusion(nx):
    from gate31_parallel_implicit_diffusion import viscous
    return viscous(3, nx=nx)


DT7, DT9, DT10 = 2.0 ** -7, 2.0 ** -9, 2.0 ** -10

CASES = {c.key: c for c in (
    TimingCase("swe_o1", "SWE bump (C1), 200^2, O1", partial(_swe, 1, 200), 50, DT7,
               agree={"foam:serial": RO_MAX, "amrex:serial": RO_MAX, "amrex:gpu1": RO_MAX, **GPU}),
    TimingCase("swe_o2", "SWE bump (C1), 200^2, O2 minmod", partial(_swe, 2, 200), 50, DT7,
               agree={"amrex:serial": RO_MAX, "amrex:gpu1": RO_MAX, **GPU}),
    TimingCase("vam1_chorin", "VAM(1) Chorin channel (C8 / L2P), 200 x 20", partial(_vam, 200), 8, DT10,
               chorin=True, q0=_vam_q0,
               agree={"foam:serial": Agree("l1rel", "TH:C8 foam"), "amrex:serial": Agree("relmax", "PRESS"),
                      "amrex:gpu1": Agree("relmax", "PRESS"), "jax:gpu1": Agree("relmax", "PRESS")}),
    TimingCase("imex_swe", "IMEX SWE + Manning (C14), 100^2, O2", partial(_imex, 2, 100), 50, DT7, imex=True,
               agree={"amrex:serial": Agree("relmax", IMPLICIT), "amrex:gpu1": Agree("relmax", IMPLICIT),
                      **GPU}),
    TimingCase("euler3d_o1", "Euler 3-D pulse box (C20), 40^3, O1", partial(_euler, 1, 40), 40, DT10,
               agree={"foam:serial": RO_MAX, "amrex:serial": RO_MAX, "amrex:gpu1": RO_MAX, **GPU}),
    TimingCase("euler3d_o2", "Euler 3-D pulse box (C20), 40^3, O2 + MOOD", partial(_euler, 2, 40), 40, DT10,
               mood=True),
    TimingCase("diffusion_impl", "implicit diffusion (C26), 48^2, IMEX", partial(_diffusion, 48), 10, DT9,
               imex=True, solver=dict(implicit_tol=1e-12, gmres_tol=1e-13, implicit_maxiter=10,
                                      gmres_maxiter=60),
               agree={"amrex:serial": Agree("relmax", IMPLICIT), "amrex:gpu1": Agree("relmax", IMPLICIT),
                      **GPU}),
)}


def selected_configs():
    """Configs to run: ZOOMY_TIMING_CONFIGS (comma list, default serial,gpu1)."""
    return tuple(c for c in os.environ.get("ZOOMY_TIMING_CONFIGS", "serial,gpu1").split(",") if c)


def params():
    """[(case, backend, config)] of every applicable, selected combination."""
    want = selected_configs()
    return [(c.key, b, cfg) for c in CASES.values() for b, cfgs in c.configs.items()
            for cfg in cfgs if cfg in want]
