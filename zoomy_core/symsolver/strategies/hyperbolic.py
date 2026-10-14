"""The explicit hyperbolic solver, written once for every backend: a finite-volume
SSP-RK march with MOOD and IMEX options. A `HyperbolicTemplate` holds the
configuration and its methods are the graphs a backend prints.
"""
import sympy as sp

from zoomy_core.misc.options import Option, typed
from zoomy_core.symsolver.core import Cond, While
from zoomy_core.symsolver.discretization.residual import (
    explicit_step, full_aux, imex_step, implicit_step, mood_step, post_step)
from zoomy_core.symsolver.discretization.time import (
    IMEX_RK, SSP_RK, CflWaveSpeed, Imex, cfl_dt, checked, face_dt, face_rule, step_size)
from zoomy_core.symsolver.vocabulary import (
    characteristic_ghost, couple, coupled_faces, ghost_flux, interface_flux,
    interface_states, io, replace_row, select)


class CoupledFace(Option):
    """The Riemann problem of a coupled face (preCICE).
    """
    #: the same model on both sides: one Riemann problem on (own, peer) posed low -> high,
    #: which both participants solve alike (the exchanged data: the cell states)
    shared = "shared"
    #: a peer of another model (SME <-> VOF): the Riemann problem (own, ghost), the ghost
    #: taking the entering characteristics from the peer and the leaving ones from the own
    #: cell (`characteristic_ghost`); the exchanged data: the cell states and their aux
    characteristic = "characteristic"


class HyperbolicTemplate:
    """The configuration of the explicit hyperbolic solver; its methods (`step`,
    `dt`, `initial_aux`, `march`) are the graphs a backend prints.
    """

    def __init__(self, nsm, *, cfl, dimension, mood=False, dt_write=None,
                 imex=False, cfl_check_every=10, wave_speed=CflWaveSpeed.faces,
                 dt_constant=None, time_order=None, dt_checkpoint=None):
        """Set the configuration; `nsm` is the NumericalSystemModel.

        Args:
            cfl: CFL number.
            dimension: mesh dimension, enters the CFL step.
            mood: a-posteriori MOOD guard of the RK step.
            dt_write: output interval; `None`: no output.
            imex: `False` (implicit slots inside the RK step), `Imex.lie` (implicit step
                after it) or `Imex.ssp2` (IMEX-SSP2(2,2,2) stages).
            cfl_check_every: constant `dt`: CFL test every N steps (1: each, 0:
                never).
            wave_speed: CFL speeds from the first stage's Riemann problem
                (`CflWaveSpeed.faces`) or from the cell states of each face (`CflWaveSpeed.cells`).
            dt_constant: constant step, or `None` for the adaptive CFL step.
            time_order: SSP-RK order; `None`: the reconstruction order (2 for
                `Imex.ssp2`).
            dt_checkpoint: interval of `io.save_state` (landed on); `None`: at the end only.
        """
        if time_order is not None and time_order not in SSP_RK:
            raise ValueError(f"time_order {time_order}: SSP-RK of order "
                             f"{sorted(SSP_RK)} only")
        imex = typed(Imex, imex, "imex") or None
        if imex is Imex.ssp2 and time_order not in (None, 2):
            raise ValueError(f"imex 'ssp2' is of time order 2, not {time_order}")
        self.nsm = nsm
        self.cfl = cfl
        self.dimension = dimension
        self.mood = mood
        self.dt_write = dt_write
        self.dt_checkpoint = dt_checkpoint
        self.imex = imex
        self.cfl_check_every = cfl_check_every
        self.wave_speed = typed(CflWaveSpeed, wave_speed, "wave_speed")
        self.dt_constant = None if dt_constant is None else sp.Float(dt_constant)
        self.time_order = (time_order if time_order is not None else
                           2 if imex is Imex.ssp2 else self.order)

    @property
    def order(self):
        """Reconstruction order of the NSM."""
        return self.nsm.reconstruction.order

    def step(self, Q, Qaux, p, time, dt, mesh, *, coupled=None, dt_rule=None):
        """One time step `(Q, Qaux, dt)`: the (MOOD-guarded) RK step, then the
        implicit slots by Lie splitting or inside the stages of IMEX-SSP2 when
        `imex` is set.

        Args:
            coupled: `(mask, flux)`, the faces carrying a shared answer.
            dt_rule: the first stage picks `dt` from its own face wave speeds
                (`ssp2`: the cell states').
        """
        nsm, order, imex = self.nsm, self.order, self.imex
        if imex is Imex.ssp2:
            Q = imex_step(nsm, Q, Qaux, p, time, dt, mesh, order, IMEX_RK[imex],
                          self.mood, coupled, dt_rule)
        elif self.mood:
            Q = mood_step(nsm, Q, Qaux, p, time, dt, mesh, order, coupled,
                          implicit_terms=not imex, rk=self.time_order,
                          dt_rule=dt_rule)
        else:
            Q = explicit_step(nsm, Q, Qaux, p, time, dt, mesh, order,
                              coupled=coupled, implicit_terms=not imex,
                              rk=self.time_order, dt_rule=dt_rule)
        if dt_rule is not None:
            Q, dt = Q
        if imex is Imex.lie:
            Q = implicit_step(nsm, Q, Qaux, p, time + dt, dt, mesh)
        return (*post_step(nsm, Q, Qaux, p, time + dt, dt, mesh), dt)

    def dt(self, Q, Qaux, p, mesh):
        """The CFL step from the cell states of each face."""
        return cfl_dt(self.nsm, Q, Qaux, p, mesh, self.cfl, self.dimension)

    def initial_aux(self, Q, Qaux, p, mesh):
        """The aux of the initial state: pointwise formula and derivative aux."""
        return full_aux(self.nsm, Q, Qaux, p, sp.S.Zero, sp.S.One, mesh)

    def march(self, Q, Qaux, p, time, time_end, mesh):
        """March from `time` to `time_end`."""
        nsm, cfl, dimension = self.nsm, self.cfl, self.dimension
        dt_constant, cfl_check_every = self.dt_constant, self.cfl_check_every

        def step(Q, Qaux, time, cap):
            """One step of size at most `cap`, with `dt` from the configured rule.
            """
            if self.wave_speed == "faces":
                return self.step(Q, Qaux, p, time, None, mesh,
                                 dt_rule=face_rule(
                                     nsm, mesh, cfl, dimension, dt_constant,
                                     time, cap, cfl_check_every))
            dt = sp.Min(step_size(nsm, Q, Qaux, p, mesh, cfl, dimension,
                                  dt_constant, time, cfl_check_every), cap)
            return self.step(Q, Qaux, p, time, dt, mesh)

        return _time_loop(Q, Qaux, time, time_end, self.dt_write, step,
                          self.dt_checkpoint)

    def precice_march(self, Q, Qaux, p, time, mesh, *, side, subcycling=False,
                      face=CoupledFace.shared, frozen_mass_row=False):
        """The march as one preCICE participant; preCICE ends the loop and the coupled
        faces carry the interface answer (`face`).

        Args:
            side: 0 if this domain is the low side of the interface (`CoupledFace.shared`).
            subcycling: `False`: one step per window (warns above the CFL step), mass
                conserved exactly. `True`: adaptive steps inside the window, mass not
                exactly conserved.
            face: `CoupledFace.shared`: one Riemann problem on (own, peer) that both sides
                solve alike; `CoupledFace.characteristic`: the Riemann problem (own, ghost)
                towards a peer of another model, the own cells' states and aux exchanged.
            frozen_mass_row: (`characteristic`) the depth row of the coupled faces held over
                the window at its value at the window start, of the Riemann problem (own as
                the peer reads it, ghost): what the peer charges for the same window, from
                the same exchanged data.

        References:
            Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
        """
        nsm, cfl, dimension = self.nsm, self.cfl, self.dimension
        dt_constant, cfl_check_every = self.dt_constant, self.cfl_check_every
        dt_write, wave_speed = self.dt_write, self.wave_speed
        face = CoupledFace(typed(CoupledFace, face, "face"))
        characteristic = face == CoupledFace.characteristic
        if frozen_mass_row and not characteristic:
            raise ValueError("frozen_mass_row needs face=CoupledFace.characteristic")
        depth = getattr(nsm.roles, "depth", None)
        if frozen_mass_row and depth is None:
            raise ValueError("frozen_mass_row: the model declares no depth row")
        sides = nsm.call.numerical_face_sides
        aux = nsm.call.update_aux_variables or sp.S.false

        def exchanged(Q, Qaux):
            """What this participant writes: the cell states next to the coupled faces, with
            their aux towards a peer of another model (the step's aux, whose derivatives see
            the coupled faces' ghost: the column's vertical velocity)."""
            q = interface_states(Q, mesh)
            return (q, interface_states(Qaux, mesh)) if characteristic else (q,)

        def coupled_face(Q, Qaux, Wf, fresh, time):
            """`(Wc, Wf, Aw)`: the coupled faces' contribution, the window's frozen depth row,
            the aux the column is written with (`characteristic`: the step's, its derivatives
            with the ghost on the coupled faces; else `Qaux`)."""
            peer = couple.read()
            if not characteristic:
                return interface_flux(sides, aux, side, Q, Qaux, peer, p, mesh), Wf, Qaux
            own, Aown = interface_states(Q, mesh), interface_states(Qaux, mesh)
            ghost = characteristic_ghost(nsm.call.quasilinear_matrix, own, Aown, peer,
                                         p, mesh)
            Wc = ghost_flux(sides, aux, own, Aown, ghost, p, mesh)
            if frozen_mass_row:
                Wf = select(fresh, ghost_flux(sides, sp.S.false, couple.read_round_trip(),
                                              Aown, ghost, p, mesh), Wf)
                Wc = replace_row(Wc, depth, Wf)
            return Wc, Wf, full_aux(nsm, Q, Qaux, p, time, sp.S.Zero, mesh,
                                    coupled=(own, ghost))

        def body(time, Q, Qaux, going, n_write, fresh, Wf):
            """One coupling step inside the window, with its output; `(fresh, Wf)`: the
            window start flag and the frozen depth row (`frozen_mass_row`)."""
            window = couple.max_dt()
            if subcycling:                          # adaptive inside the window
                cap = window if dt_write is None else sp.Min(
                    window, n_write * dt_write - time)
                if wave_speed == "faces":
                    rule = face_rule(nsm, mesh, cfl, dimension, dt_constant,
                                     time, cap, cfl_check_every)
                else:
                    dt = sp.Min(step_size(nsm, Q, Qaux, p, mesh, cfl, dimension,
                                          dt_constant, time, cfl_check_every),
                                cap)
            elif wave_speed == "faces":             # one step per window
                rule = lambda S: checked(window, time, cfl_check_every,
                                         lambda: face_dt(nsm, S, mesh, cfl,
                                                         dimension))
            else:
                dt = checked(window, time, cfl_check_every,
                             lambda: cfl_dt(nsm, Q, Qaux, p, mesh, cfl,
                                            dimension))
            Wc, Wf, Aw = coupled_face(Q, Qaux, Wf, fresh, time)
            if wave_speed == "faces":
                Q, Qaux, dt = self.step(Q, Qaux, p, time, None, mesh,
                                        coupled=(coupled_faces(mesh), Wc),
                                        dt_rule=rule)
            else:
                Q, Qaux, dt = self.step(Q, Qaux, p, time, dt, mesh,
                                        coupled=(coupled_faces(mesh), Wc))
            couple.write(*exchanged(Q, Aw))
            going = couple.advance(dt)
            if frozen_mass_row:
                fresh = couple.window_complete()
            time = time + dt
            if dt_write is not None:
                n_write = Cond(
                    time >= (n_write - sp.Rational(1, 10**9)) * dt_write,
                    lambda: _write(time, Q, Qaux, n_write),
                    lambda: n_write)
            return time, Q, Qaux, going, n_write, fresh, Wf

        going = couple.init(*exchanged(Q, Qaux))
        if dt_write is not None:
            io.write(time, Q, Qaux)
        if frozen_mass_row:        # carry the window start flag and the frozen depth row
            own = interface_states(Q, mesh)
            Wf = ghost_flux(sides, sp.S.false, own, interface_states(Qaux, mesh), own, p,
                            mesh)                     # a placeholder: replaced at the start
            time, Q, Qaux, *_ = While(
                lambda time, Q, Qaux, going, n_write, fresh, Wf: going, body,
                (time, Q, Qaux, going, 1, sp.S.true, Wf))
            return time, Q, Qaux

        def body_live(time, Q, Qaux, going, n_write):
            """`body` without the frozen row."""
            return body(time, Q, Qaux, going, n_write, None, None)[:5]
        time, Q, Qaux, *_ = While(
            lambda time, Q, Qaux, going, n_write: going, body_live,
            (time, Q, Qaux, going, 1))
        return time, Q, Qaux


def _time_loop(Q, Qaux, time, time_end, dt_write, step, dt_checkpoint=None):
    """The time loop: `step(Q, Qaux, time, cap) -> (Q, Qaux, dt)` takes a step `dt <= cap`,
    `cap` the distance to `time_end` and to the next write / checkpoint instant; the march state
    `(time, Q, Qaux)` comes from `io.load_state` and goes to `io.save_state`.
    """
    def body(time, Q, Qaux, n_write, n_checkpoint):
        """One step landing on write / checkpoint instants, then the write / save."""
        cap = time_end - time
        if dt_write is not None:                    # land on write instants
            cap = sp.Min(cap, n_write * dt_write - time)
        if dt_checkpoint is not None:               # and on checkpoint instants
            cap = sp.Min(cap, n_checkpoint * dt_checkpoint - time)
        Q, Qaux, dt = step(Q, Qaux, time, cap)
        time = time + dt
        if dt_write is not None:
            n_write = Cond(_due(time, n_write, dt_write),
                           lambda: _write(time, Q, Qaux, n_write), lambda: n_write)
        if dt_checkpoint is not None:
            n_checkpoint = Cond(sp.And(_due(time, n_checkpoint, dt_checkpoint),
                                       time < time_end),
                                lambda: _save(time, Q, Qaux, n_checkpoint),
                                lambda: n_checkpoint)
        return time, Q, Qaux, n_write, n_checkpoint

    time, Q, Qaux = io.load_state(time, Q, Qaux)
    if dt_write is not None:
        io.write(time, Q, Qaux)
    time, Q, Qaux, _, _ = While(
        lambda time, Q, Qaux, n_write, n_checkpoint: time < time_end, body,
        (time, Q, Qaux, _next(time, dt_write), _next(time, dt_checkpoint)))
    io.save_state(time, Q, Qaux)
    return time, Q, Qaux


def _next(time, every):
    """The number of the first instant of the interval `every` after `time` (1: unused)."""
    return 1 if every is None else sp.floor(time / every + sp.Rational(1, 10**9)) + 1


def _due(time, n, every):
    """Whether `time` reached instant `n` of the interval `every`."""
    return time >= (n - sp.Rational(1, 10**9)) * every


def _write(time, Q, Qaux, n_write):
    """Write the snapshot and advance the write counter."""
    io.write(time, Q, Qaux)
    return n_write + 1


def _save(time, Q, Qaux, n_checkpoint):
    """Save the march state and advance the checkpoint counter."""
    io.save_state(time, Q, Qaux)
    return n_checkpoint + 1
