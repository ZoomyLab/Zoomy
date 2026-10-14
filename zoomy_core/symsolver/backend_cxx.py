"""The C++ backend of the symbolic solvers, shared by foam and AMReX: it prints a solver
template as the host program `Solver.H`. Both runtimes include
`RUNTIME_HEADER` (`rt_common.H`), and a driver combines with a strategy by
inheritance, driver first.
"""
from pathlib import Path

import numpy as np
import param
import sympy as sp

from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.solver.constants import TPFA_COS_FLOOR, elliptic_constants
from zoomy_core.symsolver import default_wave_speed
from zoomy_core.symsolver.core import CxxPrinter
from zoomy_core.symsolver.discretization import reconstruction as R
from zoomy_core.symsolver.discretization.time import IMEX_RK, SSP_RK, Imex
from zoomy_core.symsolver.settings import ChorinSettings, HyperbolicSettings
from zoomy_core.symsolver.strategies.chorin import ChorinTemplate
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_core.symsolver.vocabulary import clean_round_off, pointwise, put, take

RUNTIME_HEADER = Path(__file__).with_name("rt_common.H")


class CxxBackend(HyperbolicSettings):
    """The settings of a hyperbolic solver on a C++ runtime and the printing of its host
    program (`_print`); the driver subclasses implement `build` and
    `run`.
    """
    couple = param.Dict(default=None, allow_None=True,
                        doc="preCICE: dict(participant=, config=, mesh=, "
                            "write=[..], read=[..], tag=, side=); side 0: this "
                            "domain is the low side of the interface.  Chorin: side= -> "
                            "overlap=layers (the mesh reaches `layers` cell rings across the "
                            "interface, its outer boundary tagged `tag`; "
                            "`ChorinTemplate.precice_march`).")
    subcycling = param.Boolean(default=False,
                               doc="preCICE: adaptive steps inside a window.")
    gmres_restart = param.Integer(default=20, bounds=(1, None))

    # ── the driver ─────────────────────────────────────────────────────────
    def build(self, nsm, dimension):
        """Print the host program (`self.source`), write the build tree and compile
        (cached); returns the binary.
        """
        raise NotImplementedError(f"{type(self).__name__}.build")

    def run(self, mesh, model, Q0, workdir):
        """March `Q0` to `time_end` (write the case into `workdir`, launch, read
        back); returns `(time, Q)` and keeps the snapshots in `self.snapshots`.

        Args:
            Q0: initial state in the mesh's order.
            model: the SystemModel or NSM (see `_prepare`).
        """
        raise NotImplementedError(f"{type(self).__name__}.run")

    def _amr_loop(self):
        """Whether the host program carries an AMR level loop's procedures."""
        return False

    # ── printing ───────────────────────────────────────────────────────────
    def _systems(self, nsm):
        """The systems whose kernels the build writes, `[(name, NSM)]`."""
        return [("nsm", nsm)]

    def _prepare(self, mesh, model):
        """The NumericalSystemModel that `run` marches (the template's NSM)."""
        return default_wave_speed(model if isinstance(model, NumericalSystemModel)
                                  else NumericalSystemModel.from_system_model(model))

    def _mood(self, nsm):
        """Whether MOOD is on: `mood`, order >= 2 and the model declares
        admissibility.
        """
        return bool(self.mood and nsm.reconstruction.order >= 2
                    and nsm.call.admissible is not None)

    def _linsolve(self, nsm):
        """Settings of the implicit linear solve (GMRES)."""
        return dict(tol=self.gmres_tol, restart=self.gmres_restart,
                    maxiter=self.gmres_maxiter, accept=-1.0)

    def _degree(self, nsm):
        """The LSQ degree of the derivative stencils."""
        return nsm.resolved_lsq_degree()

    def _config(self, nsm, dimension):
        """The template's settings."""
        return dict(cfl=self.cfl, dimension=dimension, mood=self._mood(nsm),
                    dt_write=self.dt_write, imex=self.imex,
                    cfl_check_every=self.cfl_check_every,
                    wave_speed=self.cfl_wave_speed, dt_constant=self.dt,
                    time_order=self.time_order, dt_checkpoint=self.dt_checkpoint)

    def _template(self, nsm, dimension):
        """The `HyperbolicTemplate` of `nsm`."""
        return HyperbolicTemplate(nsm, **self._config(nsm, dimension))

    def _print(self, nsm, dimension):
        """The host program of `nsm`: `run` (`HyperbolicTemplate.march`, or
        the preCICE march when coupled), `march`, `initial_aux` and
        `configure`; with an AMR level loop also its procedures.
        """
        template = self._template(nsm, dimension)
        Q, Qaux, p, time, time_end, mesh = sp.symbols("Q Qaux p time time_end mesh")
        if self.couple:
            run = CxxPrinter().procedure(
                "run", (Q, Qaux, p, time, time_end, mesh),
                lambda Q, Qaux, p, time, time_end, mesh: self._precice_march(
                    template, Q, Qaux, p, time, mesh))
        else:
            run = CxxPrinter().procedure("run", (Q, Qaux, p, time, time_end, mesh),
                                         template.march)
            if self._amr_loop():
                run += "\n\n" + _amr_procedures(template)
        run += ("\n\ninline auto march = [](auto Q, auto Qaux, auto p, double time, auto time_end, "
                "auto mesh) {\n    return run(Q, Qaux, p, time, time_end, mesh, 0L);\n};")
        aux = CxxPrinter().procedure("initial_aux", (Q, Qaux, p, mesh),
                                     template.initial_aux)
        return run + "\n\n" + aux + "\n\n" + self._configure(nsm) + "\n"

    def _precice_march(self, template, Q, Qaux, p, time, mesh):
        """The template's preCICE march with the coupling settings."""
        return template.precice_march(Q, Qaux, p, time, mesh, side=self.couple["side"],
                                      subcycling=self.subcycling)

    def _configure(self, nsm):
        """`configure()`: LSQ degree, the two-point flux floor and the implicit-solve
        tolerances.
        """
        lin = self._linsolve(nsm)
        return ("inline void configure()\n{\n"
                f"    settings().lsq_degree = {int(self._degree(nsm))};\n"
                f"    settings().tpfa_cos_floor = {float(TPFA_COS_FLOOR)!r};\n"
                f"    implicit().tol = {float(self.implicit_tol)!r};\n"
                f"    implicit().maxiter = {int(self.implicit_maxiter)};\n"
                f"    implicit().ls_tol = {float(lin['tol'])!r};\n"
                f"    implicit().ls_restart = {int(lin['restart'])};\n"
                f"    implicit().ls_maxiter = {int(lin['maxiter'])};\n"
                f"    implicit().ls_accept = {float(lin['accept'])!r};\n}}")


class CxxChorinBackend(CxxBackend, ChorinSettings):
    """The Chorin projection template (`ChorinTemplate`) on a C++ runtime;
    `run` takes `(pred, press, corr)`, the predictor and the pressure and
    corrector systems of the model's split.

    References:
        Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
    """

    def _prepare(self, mesh, model):
        """Name the split's systems as the template prints them (`pred.`, `press.`,
        `corr.`); returns the predictor.
        """
        pred, press, corr = model
        out = []
        for name, sm in (("pred", pred), ("press", press), ("corr", corr)):
            nsm = (sm if isinstance(sm, NumericalSystemModel)
                   else NumericalSystemModel.from_system_model(sm))
            nsm._symbolic_name = name
            out.append(default_wave_speed(nsm))
        pred, self.press, self.corr = out
        self._nc = int(mesh.n_inner_cells)
        return pred

    def _systems(self, nsm):
        """The three systems of the split, `[(name, NSM)]`."""
        return [("pred", nsm), ("press", self.press), ("corr", self.corr)]

    def _linsolve(self, nsm):
        """Settings of the pressure solve (GMRES), restart scaled with the problem size.
        """
        n = len(self.press.equation_to_state_index) * self._nc
        ell = elliptic_constants(n_unknowns=n, tol=self.pressure_tol,
                                 restart=self.pressure_restart).values
        tol = float(ell["c_elliptic_tol"])
        return dict(tol=tol, restart=int(ell["c_elliptic_restart"]),
                    maxiter=self.pressure_maxit, accept=max(1e-6, 10.0 * tol))

    def _degree(self, nsm):
        """The largest LSQ degree over the three systems."""
        return max(s.resolved_lsq_degree() for s in (nsm, self.press, self.corr))

    def _template(self, nsm, dimension):
        """The `ChorinTemplate` of the split."""
        newton = (None if self.pressure_solve == "linear" else
                  (sp.Float(self.pressure_newton_tol), int(self.pressure_newton_maxit)))
        return ChorinTemplate(nsm, self.press, self.corr, pressure_newton=newton,
                              projection=self.projection,
                              **self._config(nsm, dimension))

    def _print(self, nsm, dimension):
        """As `CxxBackend._print`; AMR is not supported."""
        if self._amr_loop():
            raise NotImplementedError("AMR: the hyperbolic template only")
        return super()._print(nsm, dimension)

    def _precice_march(self, template, Q, Qaux, p, time, mesh):
        """The Chorin march coupled by overlapping Schwarz: `couple['overlap']` cell rings."""
        if self.couple.get("overlap") is None:
            raise ValueError("Chorin with preCICE coupling: couple=dict(participant=, config=, "
                             "mesh=, write=, read=, tag=, overlap=layers) "
                             "(`ChorinTemplate.precice_march`)")
        if self.subcycling:
            raise ValueError("the coupled Chorin march takes one step per window")
        return template.precice_march(Q, Qaux, p, time, mesh)


def _conservative_rows(nsm):
    """Rows of the state without a nonconservative product."""
    B = nsm.sm.nonconservative_matrix
    n = len(nsm.state)
    return [i for i in range(n)
            if all(sp.sympify(B[i, j, d]) == 0 for j in range(n) for d in range(B.shape[2]))]


def _amr_procedures(template):
    """Print the template pieces an AMR level loop calls: `step`, `dt_of` (the constant
    `dt` when set, else the CFL step), the
    reconstruction maps `amr_w` / `amr_q`, `amr_clean`, the stage weights
    `rk_weights` and the refluxed rows `reflux_rows`.

    The stage weights integrate each stage's face terms over the step: SSP-RK's
    Shu-Osher weights, IMEX-RK's `wt`; the diffusive fluxes of a stage join its
    face terms, also IMEX-RK's implicit ones, so `w` must equal `wt`.
    """
    Q, Qaux, p, time, dt, mesh, W = sp.symbols("Q Qaux p time dt mesh W")
    nsm, order = template.nsm, template.order
    to_w, to_q = R.maps(nsm)
    rows = R.bed_rows(nsm) if to_w is not None else None

    def w_of(Q, Qaux, p, mesh):
        """The reconstruction variables of `Q` (the free-surface pair puts h in the
        bed slot).
        """
        W = Q if to_w is None else pointwise(to_w, Q, Qaux, p)
        if rows is not None:                    # (h, eta): h in the bed's slot
            W = put(W, (rows[0],), take(Q, (rows[1],)))
        return (W,)

    def q_of(W, Qaux, p, mesh):
        """The inverse map back to `Q` (`b = eta - h` for the free-surface pair).
        """
        if rows is not None:                    # b = eta - h
            W = put(W, (rows[0],), take(W, (rows[1],)) - take(W, (rows[0],)))
        return (W if to_q is None else pointwise(to_q, W, Qaux, p),)

    depth, tol = getattr(nsm.roles, "depth", None), getattr(nsm, "depth_round_off", None)

    def clean(Q, mesh):                         # a stage's hygiene (explicit_step)
        """A stage's round-off hygiene."""
        return (Q if depth is None or tol is None else clean_round_off(Q, (depth,), tol),)

    def dt_of(Q, Qaux, p, mesh):
        """The level's step: the constant `dt` when set, else the CFL step of its cells."""
        if template.dt_constant is not None:
            return (template.dt_constant,)
        return (template.dt(Q, Qaux, p, mesh),)

    if template.imex is Imex.ssp2:
        tab = IMEX_RK[template.imex]
        if tuple(tab.w) != tuple(tab.wt):
            raise NotImplementedError("AMR: IMEX-RK refluxing needs the implicit weights w "
                                      "equal to the explicit wt")
        beta = list(tab.wt)
    else:
        beta = []
        for a, _c in SSP_RK[template.time_order]:
            beta = [(1 - a) * b for b in beta] + [1 - a]
    procs = [
        "#define ZS_AMR_PROCEDURES 1",
        CxxPrinter().procedure("step", (Q, Qaux, p, time, dt, mesh),
                               template.step),
        CxxPrinter().procedure("dt_of", (Q, Qaux, p, mesh), dt_of),
        CxxPrinter().procedure("amr_w", (Q, Qaux, p, mesh), w_of),
        CxxPrinter().procedure("amr_q", (W, Qaux, p, mesh), q_of),
        CxxPrinter().procedure("amr_clean", (Q, mesh), clean),
        "inline std::vector<double> rk_weights() { return {"
        + ", ".join(repr(float(b)) for b in beta) + "}; }",
        "inline std::vector<int> reflux_rows() { return {"
        + ", ".join(str(i) for i in _conservative_rows(nsm)) + "}; }"]
    return "\n\n".join(procs)


def initial_state(mesh, model):
    """The initial state `Q0` (n_state, n_inner_cells) of `model` on `mesh` for a C++ backend's
    `run`: the model's initial conditions at the cell centres (zero without).

    Args:
        model: a SystemModel or NSM, or the Chorin triple `(pred, press, corr)`.
    """
    sm = model[0] if isinstance(model, (tuple, list)) else model
    sm = getattr(sm, "sm", sm)
    nc = int(mesh.n_inner_cells)
    Q = np.zeros((len(sm.state), nc))
    if sm.initial_conditions is not None:
        Q = sm.initial_conditions.apply(np.asarray(mesh.cell_centers)[:, :nc], Q)
    return np.asarray(Q, dtype=float)


def write_snapshots(path, mesh, model, snapshots):
    """Write the `snapshots` `[(time, Q)]` of a C++ backend run (`solver.snapshots`) as a zoomy
    HDF5 store at `path` (mesh, state names, one `iteration_k` per snapshot); returns `path`.
    """
    import os
    from zoomy_core.misc.io import _save_fields_to_hdf5, write_mesh_to_hdf5
    sm = model[0] if isinstance(model, (tuple, list)) else model
    path = os.path.abspath(path)
    if os.path.exists(path):
        os.remove(path)
    write_mesh_to_hdf5(path, mesh, names=[str(v) for v in sm.state])
    for k, (time, Q) in enumerate(snapshots):
        _save_fields_to_hdf5(path, k, float(time), np.asarray(Q))
    return path
