"""ChorinSplitVAMSolverJax — JAX port of the NumPy ChorinSplitVAMSolver.

Three-step Chorin projection split for the VAM(1,2,2) chain:

```
    hyperbolic (Euler) → pressure → corrector
```

wrapped, when `time_order == 2`, in the OUTER two-stage SSP-RK2 of
Escalante et al. 2024 (JCP 504:112882, "Time stepping"): the full
projection-correction cycle runs ONCE PER STAGE, the Poisson system is
re-solved in each stage (P is stage-indexed), and the step closes with
`U^{n+1} = ½ U^(0) + ½ U^(2)`.  The per-stage hyperbolic update is a single
Euler step in time (eq. 11) — second order comes from the SPATIAL
reconstruction, not from an inner RK.

Each substep is driven by its own `JaxRuntime` over the corresponding
sub-SystemModel (`sm_pred` / `sm_press` / `sm_corr`).  The
pressure step is matrix-free Newton-GMRES via
`jax.scipy.sparse.linalg.gmres`; the corrector is one
`state_update` call on `sm_corr`'s runtime.

Reuses the pure-symbolic helpers from the NumPy implementation
(`_pad_to_square`, `_substitute_dt`) — those are backend-agnostic.

Status
------
- Predictor: JAX HyperbolicSolver flux loop (symbolic Riemann via
  `JaxRuntime`).
- Pressure: matrix-free GMRES (`jax.scipy.sparse.linalg.gmres`).
- Corrector: lambdified `state_update` from `sm_corr`.
- Aux pools: per-sub-system, refreshed via LSQ stencils on the mesh
  (mirrors NumPy `update_aux_variables`).
"""
from __future__ import annotations

import time as _time
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax.scipy.sparse.linalg import gmres as jax_gmres

import param

from zoomy_core.misc.logger_config import logger
from zoomy_core.misc.misc import Zstruct
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.fvm.solver_chorin_vam_numpy import (
    ChorinSplitVAMSolver as _CoreChorin,
    _pad_to_square, _substitute_dt,
)
from zoomy_core.solver.constants import ELLIPTIC_RTOL, elliptic_constants

import zoomy_jax.fvm.ode as ode
from zoomy_jax.fvm.solver_jax import HyperbolicSolver as HyperbolicSolverJax
from zoomy_jax.fvm.solver_jax import _raise_march_error
from zoomy_jax.mesh.mesh import lsq_gradient_per_field
from zoomy_jax.transformation.jax_runtime import JaxRuntime


# ── Helpers ──────────────────────────────────────────────────────────
# (The per-field LSQ derivative kernel formerly duplicated here is now the
#  shared ``zoomy_jax.mesh.mesh.lsq_gradient_per_field`` — same contract —
#  imported above and used by both the aux refresh and the pressure matvec.)


# ── The solver ───────────────────────────────────────────────────────


def _elliptic_accept(c_elliptic_tol: float) -> float:
    """Residual above which the elliptic stage is declared FAILED.

    MIRRORS the numpy reference's gate
    (`solver_chorin_vam_numpy.py`: `max(1e-6, 10.0 * c_elliptic_tol)`).
    The solve TARGETS `c_elliptic_tol`; it is only declared BROKEN an order
    of magnitude past that, with a floor — because this operator is
    unpreconditioned with an O(1/h²) condition number, so in float64 it cannot
    reliably reach 1e-10 and a gate sitting exactly AT the target turns
    round-off into an abort.  Measured on the smooth fully-wet ladder at n=64:
    rel_resid 5.060e-10 against a target of 1e-10 — a converged solve that the
    at-the-target gate killed at step 2.

    ⚠ DIVERGENCE FLAG, deliberately surfaced rather than buried: this
    expression is a LITERAL in core as well.  It belongs in the EMITTED set
    beside `c_elliptic_tol` / `c_elliptic_restart` (mandate 6a) and this is
    the second copy of it.  `zoomy_core` is another steward's tree, so the
    fix is a core-side `c_elliptic_accept` with both backends reading it —
    filed, not silently forked."""
    return max(1e-6, 10.0 * float(c_elliptic_tol))


class EllipticSolveError(RuntimeError):
    """The elliptic (pressure) stage missed its tolerance.

    Raised — never warned.  An unconverged projection makes the corrector
    apply a pressure that does not satisfy the elliptic constraint; the
    march then produces plausible-looking numbers that are simply wrong,
    and (measured on Escalante) goes non-finite a handful of steps later.
    A solver that keeps marching on that is producing garbage silently,
    which is worse than stopping."""


class ChorinSplitVAMSolverJax(HyperbolicSolverJax):
    """JAX port of `ChorinSplitVAMSolver`.

    Parameters
    ----------
    sm_pred, sm_press, sm_corr : SystemModel
        The three sub-systems from ``split_simple(sm, pressure_vars,
        dt_symbol)``.
    reconstruction : ReconstructionSpec, optional
        Predictor reconstruction spec.
    pressure_tol : float
    pressure_maxit : int
    pressure_restart : int, optional
        GMRES restart; `None` ⇒ resolved from the block size via the emitted
        `elliptic_constants`.
    time_order : int
        Number of OUTER SSP-RK2 stages, each running a full
        hyperbolic-Euler → Poisson → correction cycle.  1 = one cycle;
        2 = the paper's two-stage scheme.
    cfl : float, optional
        Adaptive Chorin CFL (see `compute_timestep`); `None` ⇒ the
        `compute_dt` strategy.

    Driving the march
    -----------------
    The solver runs on the base `HyperbolicSolverJax` time loop:
    `Q0 = setup_simulation(mesh)`, inject the initial state into `Q0`,
    then `run_simulation(Q0, parameters=..., time0=...)` / `run_observed`
    / `solve(mesh)`.  `ad_mode` is honoured there (forward `jvp` or
    reverse `grad`).  `run_jit_steps` is the forward-only fixed
    step-count fast path.
    """

    riemann_solver = param.Selector(
        default="hr", objects=["hr", "ncp"],
        doc=("Predictor Riemann route — MIRRORS the numpy ChorinSplitVAMSolver. "
             "'hr': Audusse hydrostatic reconstruction "
             "(PositiveNonconservativeRusanov).  'ncp': plain "
             "NonconservativeRusanov with the lake-at-rest balance entry "
             "Id[h, b] = 1 — NO Audusse face reconstruction, so the bed force "
             "stays symbolically in the NCP path integral instead of splitting "
             "into g*h^2/2 flux + g*h*d_x b.  Requires h and b in the state.  "
             "Measured on the 2-layer Escalante bump: 'hr' collapses at t=4.07 "
             "while 'ncp' runs past t=9 with a flat dt."))
    pressure_tol = param.Number(
        default=ELLIPTIC_RTOL, bounds=(0, None),
        doc=("Relative tolerance for the elliptic pressure block.  Defaults to "
             "the EMITTED ``c_elliptic_tol`` "
             "(:data:`zoomy_core.solver.constants.ELLIPTIC_RTOL`) — mandate 6a: "
             "the number lives in the emitted set, not as a literal beside the "
             "``gmres`` call.  Was a hard-coded 1e-6 here, which also diverged "
             "from the numpy reference solver's default."))
    pressure_maxit = param.Integer(
        default=100, bounds=(1, None),
        doc=("Maximum OUTER iterations (RESTARTS) of the elliptic solve.  Read "
             "with :attr:`pressure_restart`: jax's ``maxiter`` counts restarts "
             "of a Krylov space of size ``restart``, so once the restarted "
             "iteration stagnates this knob buys nothing.  The lever is "
             "``restart``."))
    pressure_restart = param.Integer(
        default=None, allow_None=True, bounds=(1, None),
        doc=("GMRES restart — the Krylov-space dimension rebuilt each outer "
             "iteration, and the ACTUAL convergence lever for this block.  "
             "``None`` (default) resolves it from the block size via the "
             "emitted :func:`zoomy_core.solver.constants.elliptic_restart`, "
             "i.e. the FULL space up to a memory cap.  jax hard-coded the "
             "library default 20 here; measured, on a SMOOTH FULLY WET state "
             "(min h = 0.9, sine wave, first step) that missed tol by 1.654e-04 "
             "after restart=20 x maxiter=200 ~ 4000 matvecs.  The elliptic "
             "operator is O(1/h^2)-conditioned and this solve is "
             "unpreconditioned, so a FIXED restart degrades with every "
             "refinement — which is exactly what poisons a convergence ladder.  "
             "Pin an integer only to reproduce that degradation deliberately."))
    time_order = param.Integer(
        default=1, bounds=(1, 2),
        doc=("Number of OUTER SSP-RK2 (Shu-Osher) stages, each of which runs a "
             "FULL projection-correction cycle (hyperbolic Euler step -> "
             "Poisson solve -> correction).  1: a single cycle.  2: the scheme "
             "of Escalante et al. 2024 (JCP 504:112882) — U^(1) = S(U^(0)), "
             "U^(2) = S(U^(1)), U^{n+1} = 1/2 U^(0) + 1/2 U^(2), with P "
             "stage-indexed and the Poisson system re-solved in EACH stage.  "
             "Consumed by :meth:`chorin_cycle`; it was previously declared and "
             "read nowhere, while the RK2 sat inside the predictor instead."))
    cfl = param.Number(
        default=None, allow_None=True, bounds=(0, None),
        doc=("Adaptive Chorin CFL.  When set, :meth:`compute_timestep` (hence "
             "``run_simulation``) takes the local-CFL law of "
             ":meth:`run_jit_steps` ``(cfl=...)`` -- "
             "dt = min_f cfl*2*r_f / (dim*(2*deg+1)*|lam|_f), deg = order-1, "
             "dim from the mesh, capped by the NSM ``dt_max``; NaN wave speeds "
             "propagate, wave-free faces drop out.  ``None`` (default): the "
             "``compute_dt`` strategy (e.g. ``timestepping.constant(dt)``)."))

    #: The stage kinds this solver realises (REQ-173).  Mirrors
    #: ``ChorinSplitVAMSolver._STAGE_KINDS``; the Chorin march is exactly
    #: hyperbolic → elliptic → pointwise, one stage each.
    _STAGE_KINDS = ("hyperbolic", "elliptic", "pointwise")

    def __init__(self, sm_pred=None, sm_press=None, sm_corr=None, *,
                 stages=None, reconstruction=None, **kwargs):
        # REQ-173: accept EITHER the legacy positional triple OR a canonical
        # stage list.  The binding is delegated to core's `_bind_stages` rather
        # than reimplemented here — it binds BY KIND (never by position) and
        # validates one-stage-per-kind, and a second copy of that logic is a
        # second thing to drift.
        if stages is not None:
            if not (sm_pred is None and sm_press is None and sm_corr is None):
                raise TypeError(
                    "ChorinSplitVAMSolverJax: pass EITHER the positional "
                    "(sm_pred, sm_press, sm_corr) triple OR stages=[...], "
                    "not both.")
            sm_pred, sm_press, sm_corr = _CoreChorin._bind_stages(stages)
        elif sm_pred is None or sm_press is None or sm_corr is None:
            raise TypeError(
                "ChorinSplitVAMSolverJax: need the (sm_pred, sm_press, "
                "sm_corr) triple or stages=[...].")
        if not isinstance(sm_pred, SystemModel):
            raise TypeError("sm_pred must be a SystemModel")
        if not isinstance(sm_press, SystemModel):
            raise TypeError("sm_press must be a SystemModel")
        if not isinstance(sm_corr, SystemModel):
            raise TypeError("sm_corr must be a SystemModel")
        super().__init__(**kwargs)
        self.sm_pred = sm_pred
        self.sm_press = sm_press
        self.sm_corr = sm_corr
        self.state = list(sm_pred.state)
        self.n_state = sm_pred.n_state
        self._reconstruction_spec = reconstruction or ReconstructionSpec()
        self._dt_symbol = self._detect_dt_symbol()
        # Compiled time-loop cache, keyed by (static) n_steps.  Rebuilding the
        # jit every ``run_jit_steps`` call forced a full XLA recompile of the
        # whole scan per chunk (~320 ms/step on the 500-cell bend); caching it
        # drops steady-state to the on-device compute floor (~19 ms/step).
        self._run_step_cache: dict = {}

        for name, sm in (("press", sm_press), ("corr", sm_corr)):
            if [str(s) for s in sm.state] != [str(s) for s in sm_pred.state]:
                raise ValueError(
                    f"SM_{name}.state disagrees with SM_pred.state — the "
                    "three sub-systems must share a common state vector.")

    @param.depends("cfl", watch=True)
    def _drop_compiled_loops(self):
        """`cfl` is read while TRACING the time loop; a changed value must
        not reuse a loop compiled with the old one."""
        self._time_loop_cache = {}

    def _detect_dt_symbol(self) -> Optional[sp.Symbol]:
        """Find `dt` symbol in `sm_press` / `sm_corr` (same logic
        as NumPy ChorinSplitVAMSolver)."""
        candidates = set()
        for sm in (self.sm_press, self.sm_corr):
            for atom in sm.source.free_symbols | sm.flux.free_symbols:
                if atom in sm.parameters.values():
                    continue
                if atom in sm.state:
                    continue
                if atom in sm.aux_state:
                    continue
                if str(atom) in ("t", "dt"):
                    candidates.add(atom)
                    continue
                # Look for Δt-style symbol.
                if "Delta t" in str(atom) or "\\Delta t" in str(atom):
                    candidates.add(atom)
        if not candidates:
            return None
        if len(candidates) > 1:
            logger.warning(
                f"Multiple dt-like symbols found: {candidates}; picking one")
        return next(iter(candidates))

    # ── Setup ─────────────────────────────────────────────────────

    def setup_simulation(self, mesh):
        """Build the predictor (base solver) and the pressure / corrector
        runtimes on `mesh`.  Returns the initial state `Q` (inner cells).

        The sub-models were bound at construction, so unlike the base
        `setup_simulation(mesh, model)` there is no model argument.  The
        split sub-models carry no initial condition: the caller writes its
        initial state into the returned `Q` and hands it to
        `run_simulation` (`Qaux=None` rebuilds the predictor aux from
        it) or `run_jit_steps`.

        Also stores the run_jit_steps entry handles `_sim_Q`,
        `_sim_Qaux`, `Qaux_press`, `Qaux_corr` (refresh them after
        writing `_sim_Q` with `update_aux_variables`)."""
        t0 = _time.time()
        self._run_step_cache = {}

        # 1. Pad SM_pred to square + wrap in NSM with the pressure +
        # corrector sub-systems as ``additional_systems`` so the
        # predictor mesh stencil is sized for the pressure block's
        # second-derivative requirements.  Use Audusse HR Rusanov
        # (``PositiveNonconservativeRusanov``) for LAR balance —
        # matches the NumPy ChorinSplitVAMSolver default.
        #
        # The Audusse rescaling ``h*/h`` must apply only to momentum
        # densities, NOT to pressure modes ``P_k`` (amplitudes) or
        # bathymetry ``b`` (static).  Mirror NumPy: exclude h, b, and
        # every state index that the pressure splitter owns.
        from zoomy_core.fvm.riemann_solvers import (
            PositiveNonconservativeRusanov, NonconservativeRusanov,
        )
        sm_pred_square = _pad_to_square(self.sm_pred)
        self._sm_pred_square = sm_pred_square

        state_names = [str(s) for s in sm_pred_square.state]
        excluded = set()
        if "h" in state_names:
            excluded.add(state_names.index("h"))
        if "b" in state_names:
            excluded.add(state_names.index("b"))
        excluded.update(
            int(i) for i in self.sm_press.equation_to_state_index)
        scaled_q_indices = [
            i for i in range(sm_pred_square.n_variables)
            if i not in excluded
        ]

        if self.riemann_solver == "hr":
            riemann_cls = PositiveNonconservativeRusanov
        else:
            # "ncp": no Audusse reconstruction; the h*/h rescaling (and hence
            # scaled_q_indices) is an HR-only concept.  Id[h, b] = 1 makes the
            # h-row fluctuation dissipation cancel for dh = -db, so
            # lake-at-rest survives without face-state reconstruction.
            if "h" not in state_names or "b" not in state_names:
                raise ValueError(
                    "ChorinSplitVAMSolverJax(riemann_solver='ncp') requires "
                    f"both 'h' and 'b' in the state — found {state_names}.")
            _h_idx = state_names.index("h")
            _b_idx = state_names.index("b")

            class _NCPRusanovLARBalancedJax(NonconservativeRusanov):
                name = param.String(default="NCPRusanovLARBalancedJax")

                def get_viscosity_identity_fluctuations(self):
                    Id = super().get_viscosity_identity_fluctuations()
                    Id[_h_idx, _b_idx] = 1
                    return Id

            riemann_cls = _NCPRusanovLARBalancedJax
            scaled_q_indices = None

        nsm_pred = NumericalSystemModel.from_system_model(
            sm_pred_square,
            riemann=riemann_cls,
            reconstruction=self._reconstruction_spec,
            additional_systems=[self.sm_press, self.sm_corr],
            scaled_q_indices=scaled_q_indices,
        )

        # 2. dt-rename so the lambdify substitution is Python-safe.  Done
        # BEFORE the parent setup: the parent's initial aux update already
        # runs the Chorin refresh (``_qaux_from_q``), which reads all three
        # sub-models.
        if self._dt_symbol is not None:
            dt_safe = sp.Symbol("dt", positive=True)
            self._dt_symbol_safe = dt_safe
            self.sm_press = _substitute_dt(
                self.sm_press, self._dt_symbol, dt_safe)
            self.sm_corr = _substitute_dt(
                self.sm_corr, self._dt_symbol, dt_safe)
            for sm in (self.sm_press, self.sm_corr):
                if not sm.parameters.contains("dt"):
                    new_params = Zstruct(**sm.parameters.as_dict())
                    new_params["dt"] = dt_safe
                    sm.parameters = new_params
                    new_pvals = Zstruct(**sm.parameter_values.as_dict())
                    new_pvals["dt"] = 0.0
                    sm.parameter_values = new_pvals
        else:
            self._dt_symbol_safe = None

        # 3. JaxRuntime for pressure + corrector sub-systems.
        self.rt_press = JaxRuntime.from_nsm(
            NumericalSystemModel.from_system_model(self.sm_press))
        self.rt_corr = JaxRuntime.from_nsm(
            NumericalSystemModel.from_system_model(self.sm_corr))

        # 4. The parent JAX HyperbolicSolver setup on the predictor.
        Q, Qaux = super().setup_simulation(mesh, nsm_pred)

        # The pressure / corrector parameter vectors are BUILT from the
        # (possibly traced) predictor vector + dt, by name.  Resolved here,
        # once; ``_sub_parameters`` gathers with it inside the trace.
        self._press_param_src = self._param_gather_index(self.rt_press)
        self._corr_param_src = self._param_gather_index(self.rt_corr)

        # State-index slices.
        self._pred_state_idx = np.asarray(
            self.sm_pred.equation_to_state_index, dtype=int)
        self._press_state_idx = np.asarray(
            self.sm_press.equation_to_state_index, dtype=int)
        self._corr_state_idx = np.asarray(
            self.sm_corr.equation_to_state_index, dtype=int)

        # Pressure / corrector aux pool templates.  Every stage rebuilds both
        # pools from Q by refreshing these (all-zero) templates: the refresh
        # overwrites every registry row, and the rows outside the registries
        # are never written by anything, so they are zero in any march.
        nc = self._rt_mesh.n_inner_cells
        self._qaux_press0 = jnp.zeros((len(self.sm_press.aux_state), nc))
        self._qaux_corr0 = jnp.zeros((len(self.sm_corr.aux_state), nc))

        # Pressure aux to refresh per Krylov iteration: derivatives of
        # state-pressure indices.
        press_idx_set = set(int(i) for i in self._press_state_idx)
        self._press_aux_recompute = []
        for entry in (self.sm_press.aux_registry or []):
            if (entry.get("kind") not in ("derivative", "limited_derivative")
                    or entry.get("target_kind") != "state"
                    or entry.get("state_index") not in press_idx_set):
                continue
            if (entry.get("kind") == "limited_derivative"
                    and entry.get("limiter_scheme") is not None):
                # numpy applies `_apply_cell_limiter` INSIDE the pressure
                # matvec (solver_chorin_vam_numpy.py:1141-1144); this backend
                # does not.  Assembling a different elliptic operator than the
                # reference backend, silently, is not an option — fail here.
                raise NotImplementedError(
                    "jax Chorin: pressure aux row %d (state %d) carries "
                    "limiter_scheme %r; the jax elliptic matvec has no cell "
                    "limiter, so it would assemble a DIFFERENT operator than "
                    "the numpy reference.  Implement the limiter in "
                    "_refresh_pressure_aux before using this model."
                    % (int(entry["row"]), int(entry["state_index"]),
                       entry.get("limiter_scheme")))
            self._press_aux_recompute.append({
                "row": int(entry["row"]),
                "state_index": int(entry["state_index"]),
                "multi_index": tuple(entry["multi_index"]),
            })

        # Depth row, for the elliptic abort diagnostic (h_min).  Resolved by
        # NAME off the predictor sub-model's variable list; absent ⇒ the
        # diagnostic just omits h_min (never a hard failure, and never used
        # to clip or floor anything).
        self._h_row = None
        try:
            names = [str(k) for k in self.sm_pred.variables.keys()]
            if "h" in names:
                self._h_row = names.index("h")
        except Exception:
            self._h_row = None

        # Elliptic-stage boundary conditions (REQ-174).  The BC *vocabulary* —
        # what counts as a declared Dirichlet, how a PerFieldBoundary delegates
        # per slot — is core's, so call core's parser instead of re-deriving it:
        # one source of truth, and it follows core if the vocabulary moves.
        # (Same reuse pattern as `_bind_stages`.)  It reads only `sm_press`,
        # `_press_state_idx` and `nc` off self, plus four boundary attributes
        # off the mesh, all of which MeshJAX carries.
        # ``None`` ⇒ no pressure mode declares a Dirichlet ⇒ homogeneous
        # Neumann, bit-identical to the pre-REQ-174 path.
        self._press_dir = _CoreChorin._build_pressure_dirichlet(
            self, self._rt_mesh)
        if self._press_dir is None:
            self._press_dir_j = None
        else:
            d = self._press_dir
            self._press_dir_j = {
                "face_mask": jnp.asarray(d["face_mask"]),
                "face_value": jnp.asarray(d["face_value"]),
                "bf_cells": jnp.asarray(np.asarray(d["bf_cells"]), dtype=int),
                "cell_mask": jnp.asarray(d["cell_mask"]),
                "cell_value": jnp.asarray(d["cell_value"]),
                # Which modes carry ANY Dirichlet.  Static (host-side), so a
                # mode with none keeps the literal `u_bf=None` fast path rather
                # than a ghost array that merely reduces to it.
                "modes": tuple(bool(m) for m in
                               np.asarray(d["face_mask"]).any(axis=1)),
            }
            logger.info(
                "REQ-174: elliptic stage carries declared P Dirichlet on modes "
                "%s (%d pinned boundary faces)",
                [k for k, m in enumerate(self._press_dir_j["modes"]) if m],
                int(np.asarray(d["face_mask"]).sum()))

        # run_jit_steps entry handles (host-side; never touched by traced
        # code).
        self._sim_Q = Q
        self.update_aux_variables()

        logger.info(
            "ChorinSplitVAMSolverJax setup: pred → %s, press → %s, corr → %s "
            "in %.2fs",
            self._pred_state_idx.tolist(),
            self._press_state_idx.tolist(),
            self._corr_state_idx.tolist(),
            _time.time() - t0,
        )
        return Q

    def solve(self, mesh, write_output=True, parameters=None):
        """`setup_simulation(mesh)` then `run_simulation` from the
        state it returns.  Returns the final `(Q, Qaux)`.  The split
        sub-models carry no initial condition, so this marches the setup state;
        a driver with an initial condition writes it into the
        `setup_simulation` state and calls `run_simulation`."""
        Q = self.setup_simulation(mesh)
        return self.run_simulation(Q, write_output=write_output,
                                   parameters=parameters)

    @property
    def nc(self):
        return self._rt_mesh.n_inner_cells

    # ── Aux refresh ───────────────────────────────────────────────

    def _param_gather_index(self, rt):
        """Index array mapping `rt`'s parameter vector onto
        `concat(pred_parameters, [dt])`, by name.  A sub-system parameter
        that is neither a predictor parameter nor `dt` cannot be fed from
        the predictor vector and is refused."""
        pred = list(self._rt_model.parameter_names)
        idx = []
        for name in rt.parameter_names:
            if name == "dt":
                idx.append(len(pred))
            elif name in pred:
                idx.append(pred.index(name))
            else:
                raise ValueError(
                    f"ChorinSplitVAMSolverJax: sub-system parameter {name!r} "
                    f"is not a predictor parameter ({pred}) nor dt; the "
                    "pressure/corrector parameters are built from the "
                    "predictor vector.")
        return np.asarray(idx, dtype=int)

    @staticmethod
    def _sub_parameters(src, parameters, dt):
        """The pressure / corrector parameter vector: ``concat(parameters,
        [dt])[src]``.  Traceable in `parameters` and `dt`."""
        p = jnp.asarray(parameters)
        dt = jnp.reshape(jnp.asarray(dt, dtype=p.dtype), (1,))
        return jnp.concatenate([p, dt])[src]

    def _runtime_for_sm(self, sm):
        """The `JaxRuntime` that lambdified `sm` (REQ-151 DEFECT D).

        The pressure/corrector sub-systems get their own runtimes in
        `setup_simulation`; the predictor rides the parent `HyperbolicSolver`
        runtime (`_rt_model`, built on `SM_pred`)."""
        if sm is getattr(self, "sm_press", None):
            return getattr(self, "rt_press", None)
        if sm is getattr(self, "sm_corr", None):
            return getattr(self, "rt_corr", None)
        return getattr(self, "_rt_model", None)

    def _refresh_aux_for_sm(self, Qaux, sm, Q, parameters):
        """Refresh `sm`'s aux pool: the ALGEBRAIC `update_aux_variables`
        rule first, then every state-derivative entry — mirroring the base
        `HyperbolicSolver.update_qaux` order.  Pure function — returns a
        new Qaux array.  Safe inside `jax.jit` / `jax.lax.scan`.
        `parameters` is `sm`'s OWN parameter vector (the predictor vector
        for `sm_pred`, `_sub_parameters` for the other two).

        REQ-151 DEFECT D: this used to walk ONLY the derivative kinds, so a
        plain-Symbol aux (`hinv = KP(h)`, which every velocity scales by:
        `u = hu·hinv`) was NEVER evaluated — it sat at its init value 0 and
        silently zeroed every velocity.  That is a wrong-answer bug, not a
        crash.  `zoomy_core@676ed2d` fixed the SYMBOLIC half (`hinv` now
        survives into `SM_pred`/`SM_press` WITH its update rule); this is
        the solver-side half.  The rule is a FULL-LENGTH `(n_aux, ·)` vector
        whose non-algebraic rows pass their own aux symbol through, so applying
        it before the derivative walk is safe (the walk then refreshes the
        derivative rows).

        Walks BOTH registries: since the splitter's `_partition_pressure_aux`
        (zoomy_core@44bdfca, REQ-94) the pressure sub-model keeps only the
        LIVE pressure-derivative entries in `aux_registry` and moves the
        frozen predictor-state derivatives (`h_x`, `q_k_x`, `b_x` …) to
        `aux_input_registry` — the documented contract is that a runtime
        with a private pressure pool fills it from both, once per stage.
        Walking only `aux_registry` left every input derivative at 0, so the
        elliptic RHS was IDENTICALLY zero and P (hence r) never activated on
        flat beds (task 0039: confluence VAM inert → blow-up)."""
        # (1) ALGEBRAIC aux leg (REQ-151 DEFECT D) — e.g. the KP-desingularized
        #     ``hinv``.  Guarded on full length, mirroring the base solver:
        #     a short/prefix vector would mis-place the algebraic rows onto the
        #     leading derivative rows and then be clobbered by (2).
        rt = self._runtime_for_sm(sm)
        fn = getattr(rt, "update_aux_variables", None)
        if fn is not None:
            local = fn(Q, Qaux, parameters)
            if local is not None and jnp.shape(local)[0] == jnp.shape(Qaux)[0]:
                Qaux = local

        # (2) state-derivative entries (Chorin's historical scope).
        reg = (list(getattr(sm, "aux_registry", None) or [])
               + list(getattr(sm, "aux_input_registry", None) or []))
        return self._walk_derivative_aux(
            sm, Qaux, Q, self._rt_mesh,
            kinds=("derivative", "limited_derivative"),
            target_kinds=("state",),
            registry=reg or None)

    def update_aux_variables(self):
        """Host-side: rebuild the run_jit_steps entry pools `_sim_Qaux`,
        `Qaux_press`, `Qaux_corr` from `_sim_Q` (after a driver wrote
        its initial state there).  The traced march never reads them."""
        p = self._rt_parameters
        self._sim_Qaux = self._qaux_from_q(self._sim_Q, p)
        self.Qaux_press = self._refresh_aux_for_sm(
            self._qaux_press0, self.sm_press, self._sim_Q,
            self._sub_parameters(self._press_param_src, p, 0.0))
        self.Qaux_corr = self._refresh_aux_for_sm(
            self._qaux_corr0, self.sm_corr, self._sim_Q,
            self._sub_parameters(self._corr_param_src, p, 0.0))

    def _qaux_from_q(self, Q, parameters):
        """The predictor aux pool of `Q`: the Chorin refresh of the aux
        initial condition stored at setup.  Traceable in `Q` and
        `parameters`; used by `run_simulation(Q0, Qaux=None)`."""
        return self._refresh_aux_for_sm(
            self._rt_qaux_ic, self.sm_pred, Q, parameters)

    # ── Chorin step (pure-functional core, JIT-able) ──────────────

    def _hyperbolic_stage(self, dt, time, Q, Qaux_pred, parameters):
        """The stage's HYPERBOLIC substep: one path-conservative flux update
        advanced by a SINGLE EULER step in time, plus the algebraic source.

        This is the parent's `HyperbolicSolverJax.step` with
        `time_euler=True`: same reconstruction (order ≥ 2 stays order ≥ 2 in
        SPACE), same MOOD, same halo wrap, same RK1 source — only the time
        advance is Euler instead of the parent's internal Heun.

        WHY: Escalante et al. 2024 (JCP 504:112882, "Time stepping") puts the
        two-stage SSP-RK2 on the OUTSIDE and runs the full projection-correction
        cycle once per stage; the per-stage hyperbolic update is a single Euler
        step (eq. 11), second order coming from the SPATIAL reconstruction.
        Calling the parent `step` here instead nests the RK2 INSIDE the
        predictor — the exact inverse of the prescribed nesting — which is what
        this solver did before, and it makes the pressure projection a
        once-per-step correction of a state that already took two flux stages
        without a projection between them."""
        flux = self._rt_flux_op
        Q = self._explicit_hyperbolic_step(
            dt, time, Q, Qaux_pred, parameters, flux,
            int(self._rt_mesh.n_inner_cells), time_euler=True)
        # Source step (explicit Euler — the source is treated as O1), mirroring
        # the parent ``step``.  REQ-185: ``time`` is threaded so a
        # time-dependent source binds ``t``.
        return ode.RK1(self._rt_source_op, Q, Qaux_pred, parameters, dt, time)

    def chorin_stage(self, dt, time, Q, Qaux_pred, parameters):
        """ONE projection-correction stage: hyperbolic Euler → Poisson →
        correction.  Pure: `(Q, Qaux_pred)` in, `(Q, Qaux_pred)` out,
        `parameters` = the predictor parameter vector (the pressure and
        corrector vectors are built from it + `dt` by
        `_sub_parameters`).

        This is the paper's per-stage cycle.  `chorin_cycle` wraps
        `time_order` of these into the outer SSP-RK2.

        The predictor pool is re-derived from the INCOMING `Q` before the
        predictor runs.  That is not belt-and-braces: the predictor consumes
        `Qaux_pred` (`b_x`, `h_x`, `q_Uk_x`, `hinv` …) as its FIRST
        act, so whatever the caller hands in is what the predictor integrates
        with.  A driver that captured the setup pool once and passed the same
        handle back every step (the shape
        `thesis/cases/escalante_vam_bump/run_jax.py` had) pinned the
        predictor's derivatives at their step-0 values for the whole march.
        Measured consequence on the Escalante bump, 60 cells, CFL 0.4: the
        elliptic RHS goes inconsistent, GMRES stalls at rel_resid 7.9e-06, and
        the state is non-finite at step 7 — at EVERY CFL, because aux
        staleness grows with step COUNT, not with dt.  For a correctly-threaded
        caller the refresh is idempotent, so it changes no answer.

        The pressure and corrector pools are rebuilt from `Q` right before
        their substep (refresh of the all-zero setup templates — see
        `setup_simulation`), so they are not carried between steps."""
        p_press = self._sub_parameters(self._press_param_src, parameters, dt)
        p_corr = self._sub_parameters(self._corr_param_src, parameters, dt)
        Qaux_pred = self._refresh_aux_for_sm(
            Qaux_pred, self.sm_pred, Q, parameters)

        # 1. Hyperbolic substep — ONE Euler step (see _hyperbolic_stage).
        Q = self._hyperbolic_stage(dt, time, Q, Qaux_pred, parameters)

        # 2. Pressure step (matrix-free GMRES via JVP).
        Qaux_press = self._refresh_aux_for_sm(
            self._qaux_press0, self.sm_press, Q, p_press)
        Q = self._step_pressure_pure(Q, Qaux_press, dt, time, p_press)

        # 3. Corrector step.
        Qaux_corr = self._refresh_aux_for_sm(
            self._qaux_corr0, self.sm_corr, Q, p_corr)
        Q = self._step_corrector_pure(Q, Qaux_corr, p_corr)
        Qaux_pred = self._refresh_aux_for_sm(
            Qaux_pred, self.sm_pred, Q, parameters)
        return Q, Qaux_pred

    def chorin_cycle(self, dt, time, Q, Qaux_pred, parameters):
        """One full TIME STEP: the outer SSP-RK2 (Shu–Osher) over
        `chorin_stage`.  Pure: `(Q, Qaux_pred)` in, ``(Q,
        Qaux_pred)`` out.

        `time_order == 1`  ⇒  U^{n+1} = S(U^n), one stage.

        `time_order == 2`  ⇒  the paper's scheme (Escalante et al. 2024,
        JCP 504:112882, "Time stepping"):

        ```
            U^(1)   = S(U^(0), t)          # hyp Euler → Poisson → correct
            U^(2)   = S(U^(1), t + dt)     # the FULL cycle again; P is
                                           # stage-indexed, re-solved here
            U^{n+1} = ½ U^(0) + ½ U^(2)
        ```

        The RK2 used to sit INSIDE the predictor (the parent's
        `_explicit_hyperbolic_step`) while the projection ran once per step —
        the inverse of the prescribed nesting.

        The Shu–Osher average is taken on the STATE.  The predictor pool is
        then RE-DERIVED from the averaged state rather than averaged: the
        derivative rows are exact linear functionals of Q so averaging would
        agree, but the ALGEBRAIC rows (the KP-desingularized `hinv` above
        all) are nonlinear in Q and an averaged `hinv` is not `hinv` of the
        averaged h."""
        if int(self.time_order) == 1:
            return self.chorin_stage(dt, time, Q, Qaux_pred, parameters)

        Q0 = Q
        Q, Qaux_pred = self.chorin_stage(dt, time, Q0, Qaux_pred, parameters)
        Q, Qaux_pred = self.chorin_stage(dt, time + dt, Q, Qaux_pred,
                                         parameters)
        Q = 0.5 * (Q0 + Q)
        Qaux_pred = self._refresh_aux_for_sm(
            Qaux_pred, self.sm_pred, Q, parameters)
        return Q, Qaux_pred

    # ── base time-loop contract ───────────────────────────────────

    def step(self, dt, time, Q, Qaux, parameters=None, coupled=None):
        """One Chorin time step (`chorin_cycle`), the base `step`
        contract: `Qaux` is the predictor aux pool, `parameters` the
        predictor parameter vector (`None` ⇒ `parameters`).  Returns
        the new `Q`; `post_step` rebuilds the pool.  Pure — nothing on
        `self` changes.  Interface coupling is not implemented for the split
        solver; `coupled` must be `None`."""
        if coupled is not None:
            raise NotImplementedError(
                "ChorinSplitVAMSolverJax.step: interface coupling (coupled=...) "
                "is not implemented for the Chorin split solver.")
        parameters = self._rt_parameters if parameters is None else parameters
        Q, _ = self.chorin_cycle(dt, time, Q, Qaux, parameters)
        return Q

    def post_step(self, time, dt, Q, Qold, Qaux, parameters=None):
        """The predictor aux pool of the stepped `Q`.  Returns ``(Q,
        Qaux)``.  The predictor's `update_variables` is NOT applied: the
        Chorin cycle never applied it (the corrector owns the projected rows)."""
        parameters = self._rt_parameters if parameters is None else parameters
        return Q, self._refresh_aux_for_sm(Qaux, self.sm_pred, Q, parameters)

    def _cfl_dt(self, Q, Qaux, parameters, cfl, dt_cap):
        """The Chorin local-CFL step, the SAME law as
        `zoomy_core.fvm.timestepping.adaptive` with the mesh dimension and
        `degree = order - 1`:

        ```
            dt = min( min_f cfl·2·r_f / (dim·(2·deg+1)·|λ|_f), dt_cap )
        ```

        paired per face, so each face is limited by its OWN size and wave
        speed.  Wave-free faces (|λ| = 0) impose no limit; when every face is
        wave-free the step is `dt_cap` (REQ-190).  The division runs on a
        safe denominator (|λ| → 1 on wave-free faces) so the reverse-mode
        cotangent of the discarded branch is 0, not 0·inf = NaN.

        A NaN wave speed propagates into dt: `nan > 0` is False, so it would
        otherwise be read as wave-free, replaced by +inf, drop out of the min,
        and the step would SNAP BACK to dt_max — masking a blow-up and then
        accelerating it (observed: t frozen at 0.270 for 800 steps, then
        jumping to 0.697 in one chunk).  The NaN lets the driver's finiteness
        check fire on the step it happens."""
        deg = max(int(getattr(self.nsm.reconstruction, "order", 1)) - 1, 0)
        k = float(self._rt_mesh.dimension) * float(2 * deg + 1)
        lam = self._rt_eigenvalue_op(Q, Qaux, parameters)
        h = 2.0 * self._rt_inradius_face
        wave = lam > 0
        local = jnp.where(wave, cfl * h / (k * jnp.where(wave, lam, 1.0)),
                          jnp.inf)
        local = jnp.where(jnp.isnan(lam), jnp.nan, local)
        return jnp.minimum(jnp.min(local), dt_cap)

    def compute_timestep(self, Q, Qaux, parameters=None):
        """`cfl` set ⇒ `_cfl_dt` capped by the NSM `dt_max`;
        `cfl=None` ⇒ the `compute_dt` strategy, as in the base solver."""
        if self.cfl is None:
            return super().compute_timestep(Q, Qaux, parameters)
        parameters = self._rt_parameters if parameters is None else parameters
        cap = self.nsm.dt_max if self.nsm.dt_max is not None else jnp.inf
        return self._cfl_dt(Q, Qaux, parameters, float(self.cfl), cap)

    def run_jit_steps(self, dt, n_steps, Q, Qaux_pred=None, Qaux_press=None,
                      Qaux_corr=None, t_start=0.0, cfl=None, dt_max=None,
                      parameters=None):
        """Forward-only fast path: exactly `n_steps` steps
        (`step` → `post_step`) in one jitted `jax.lax.scan`,
        no `time_end` clamp.  Returns ``(Q, Qaux_pred, Qaux_press,
        Qaux_corr, time)``; the pressure / corrector pools are those of the
        final `Q`.  `Qaux_pred=None` rebuilds the predictor pool from
        `Q`; the pressure / corrector pools passed in only fix their shape
        (`None` ⇒ the setup templates), since every step rebuilds them.

        The compiled scan is **cached per (n_steps, adaptive)** on the solver
        and reused across calls; `dt`, `cfl`, `dt_max`, `t_start` and
        `parameters` are traced, so a chunked driver pays the compile ONCE,
        not every chunk (REQ-122).

        `cfl` given ⇒ **adaptive**: `dt` is recomputed from the state every
        step with `_cfl_dt` (the law `run_simulation` uses with the
        `cfl` param), capped by `dt_max` (default `dt`).  `cfl=None` ⇒
        the constant step `dt`.

        `ad_mode` must be `"forward"`; reverse mode is
        `run_simulation`'s."""
        self._require_forward_ad("jax.lax.scan (run_jit_steps)")
        if parameters is None:
            parameters = self._rt_parameters
        n_steps = int(n_steps)
        adaptive = cfl is not None
        key = (n_steps, adaptive)
        run = self._run_step_cache.get(key)
        if run is None:
            def _run(dt, cfl, dt_cap, Q, Qaux_pred, Qaux_press, Qaux_corr,
                     t_start, parameters):
                # ``None`` pools are rebuilt INSIDE the jit (jit retraces per
                # None-pattern): eagerly, the predictor refresh alone costs
                # ~40 ms per call on the 30-cell bump.
                if Qaux_pred is None:
                    Qaux_pred = self._qaux_from_q(Q, parameters)
                if Qaux_press is None:
                    Qaux_press = self._qaux_press0
                if Qaux_corr is None:
                    Qaux_corr = self._qaux_corr0

                def _body(carry, _k):
                    Q, Qaux_pred, t = carry
                    dt_k = (self._cfl_dt(Q, Qaux_pred, parameters, cfl, dt_cap)
                            if adaptive else dt)
                    Q_new = self.step(dt_k, t, Q, Qaux_pred, parameters)
                    Q_new, Qaux_pred = self.post_step(
                        t + dt_k, dt_k, Q_new, Q, Qaux_pred, parameters)
                    return (Q_new, Qaux_pred, t + dt_k), None

                (Q, Qaux_pred, t_final), _ = jax.lax.scan(
                    _body, (Q, Qaux_pred, t_start), None, length=n_steps)
                Qaux_press = self._refresh_aux_for_sm(
                    Qaux_press, self.sm_press, Q,
                    self._sub_parameters(self._press_param_src, parameters,
                                         dt))
                Qaux_corr = self._refresh_aux_for_sm(
                    Qaux_corr, self.sm_corr, Q,
                    self._sub_parameters(self._corr_param_src, parameters,
                                         dt))
                return Q, Qaux_pred, Qaux_press, Qaux_corr, t_final

            run = self._run_step_cache[key] = jax.jit(_run)
        dtype = Q.dtype
        out = run(jnp.asarray(dt, dtype=dtype),
                  jnp.asarray(0.0 if cfl is None else cfl, dtype=dtype),
                  jnp.asarray(dt if dt_max is None else dt_max, dtype=dtype),
                  Q, Qaux_pred, Qaux_press, Qaux_corr,
                  jnp.asarray(t_start, dtype=dtype), parameters)
        if not isinstance(out[0], jax.core.Tracer):   # never return a NaN state or time
            _raise_march_error(out[4], out[4], np.nan, n_steps, jnp.max(jnp.abs(out[0])))
        return out

    def _step_pressure_pure(self, Q, Qaux_press, dt, time, parameters):
        """Pure-functional pressure step: matrix-free GMRES on the
        elliptic block.  `parameters` is the PRESSURE sub-system's vector
        (`_sub_parameters`).  Returns `Q` with the pressure rows
        solved.

        The matvec is a JVP of the lambdified residual at `p = 0` —
        custom_linear_solve transposes the JVP as a VJP natively
        (the symbolic-linearity check inside `linear_transpose`
        can't peel apart the closure-captured frozen state from the
        linear p_vec in the scatter + lambdified-source pipeline,
        but the explicit JVP sidesteps that)."""
        rt = self.rt_press
        nc = self.nc
        e2s = self._press_state_idx
        nP = len(e2s)
        N = nP * nc
        local_of_state = {int(s): k for k, s in enumerate(e2s)}
        press_aux_specs = self._press_aux_recompute
        mesh = self._rt_mesh
        pdir = self._press_dir_j          # declared P Dirichlet BCs, or None

        def _u_bf_ghost(k, p_mat):
            """Boundary samples for mode `k`'s LSQ derivative stencil (REQ-174).

            ⚠ jax's `u_bf` is NOT numpy's `u_boundary_face`.  core's
            `compute_derivatives` takes FACE values and converts internally
            (`_resolve_u_boundary_face`: `ghost = 2·u_face − u_cell`);
            `lsq_gradient_per_field` takes the **ghost/boundary-neighbour**
            value DIRECTLY (`u_bf_delta = u_bf_i − u_i`, no conversion).  So
            the `2·u_face − u_cell` lift is ours to apply.  Passing the bare
            face value here puts a face-valued sample at the ghost position and
            silently caps the boundary gradient at 1st order — measured 4.16 vs
            0.024 in `test_mesh_derivatives_recent.py`.

            `None` ⇒ Neumann-zero (∂ₙP = 0), the standard Chorin pressure BC.
            Non-Dirichlet faces take the inner-cell value, for which
            `2·u_cell − u_cell = u_cell` reproduces that path exactly."""
            if pdir is None or not pdir["modes"][k]:
                return None
            bf_cells = pdir["bf_cells"]
            u_cell = p_mat[k, bf_cells]
            face_vals = jnp.where(pdir["face_mask"][k],
                                  pdir["face_value"][k], u_cell)
            return 2.0 * face_vals - u_cell

        def _refresh_pressure_aux(p_mat, Qaux_work):
            for spec in press_aux_specs:
                k = local_of_state[spec["state_index"]]
                grad = lsq_gradient_per_field(
                    mesh, p_mat[k, :], u_bf=_u_bf_ghost(k, p_mat),
                    multi_index=spec["multi_index"])
                Qaux_work = Qaux_work.at[spec["row"], :].set(grad)
            return Qaux_work

        def _residual(p_vec):
            p_mat = p_vec.reshape(nP, nc)
            Q_try = Q.at[e2s, :].set(p_mat)
            Qaux_work = _refresh_pressure_aux(p_mat, Qaux_press)
            R = rt.source(Q_try, Qaux_work, parameters).reshape(nP, nc)
            if pdir is not None:
                # REAL Dirichlet rows (REQ-174): replace the elliptic PDE row at
                # each pinned boundary cell with ``P_k[cell] − value = 0``.
                # Carried inside the residual so BOTH the matrix-free matvec
                # (``A·p = _residual(p) − b``) and the surfaced rel_resid see the
                # system actually being solved.  No shift/penalty/rank fix — the
                # operator is full-rank without any pin (v3); this is a
                # wrong-problem fix, not a singularity fix.
                R = jnp.where(pdir["cell_mask"],
                              p_mat - pdir["cell_value"], R)
            return R.reshape(-1)

        # Emitted elliptic constants (mandate 6a).  The restart is RESOLVED
        # FROM ``N`` — the size of THIS pressure block — so it moves with the
        # mesh instead of sitting as a literal beside the ``gmres`` call.  Same
        # resolver the numpy solver consumes, so the two backends now assemble
        # and solve the same problem with the same budget.
        ell = elliptic_constants(n_unknowns=N, tol=self.pressure_tol,
                                 restart=self.pressure_restart)
        c_elliptic_tol = ell.values["c_elliptic_tol"]
        c_elliptic_restart = ell.values["c_elliptic_restart"]
        self.last_elliptic_restart = c_elliptic_restart

        b = _residual(jnp.zeros(N))
        zero_p = jnp.zeros(N)

        def matvec(p_vec):
            _, jvp_val = jax.jvp(_residual, (zero_p,), (p_vec,))
            return jvp_val

        # NOTE ON `info`: jax's gmres CANNOT report non-convergence.  Its own
        # docstring calls `info` a "Placeholder for convergence information",
        # and it is a hard-coded 0 -- measured identical for a converged solve
        # and for one deliberately starved to maxiter=1.  So there is nothing
        # to check: the ONLY way to know whether this solve converged is to
        # measure the residual, which is what we do below.  (PETSc backends get
        # DIVERGED_ITS for free and print a warning; jax gets silence.)
        #
        # `maxiter` here is the number of RESTARTS of a size-`restart` Krylov
        # space, NOT an iteration count -- the budget is ~restart*maxiter
        # matvecs.  `restart` used to be jax's library default of 20; it is now
        # the emitted `c_elliptic_restart`, resolved from N above.
        #
        # `solve_method` stays at jax's default 'batched'.  'incremental' --
        # which checks the residual INSIDE the cycle and exits early, the way
        # scipy (the numpy reference) does -- was tried and reverted; it is a
        # serialising `while_loop` and it was NOT faster here.
        #
        # ⚠ OPEN COST QUESTION -- OBSERVED, CAUSE NOT ISOLATED.  Do not quote
        # this as a result.  The four jax Chorin test modules ran green in
        # ~170 s at the commit that introduced the emitted restart, and in
        # three later runs the SAME modules were still unfinished after 45 min.
        # The only functional delta across that boundary is the ACCEPT GATE,
        # which cannot slow a Krylov solve down -- so either the box was not
        # idle for the slow runs, or something here is not what it looks like.
        # It was NOT measured on a confirmed-idle box, which is the standing
        # requirement for any timing claim, so it is recorded as a question.
        #
        # The hypothesis worth testing first: the solve now targets the emitted
        # 1e-10 (it used to target a literal 1e-6) on an UNPRECONDITIONED
        # O(1/h^2) operator; where it cannot reach that, it burns the whole
        # `restart * maxiter` budget every step instead of stopping after 20
        # vectors.  If that is confirmed, `pressure_maxit` is still the wrong
        # lever (core measured that) -- with `restart` at the FULL space GMRES
        # has the answer after one cycle in exact arithmetic, so the remaining
        # restarts chase round-off.  The fix would be a PRECONDITIONER for this
        # block; a caller needing the old speed meanwhile sets `pressure_tol`.
        p_new, _info = jax_gmres(
            matvec, -b,
            atol=0.0, tol=c_elliptic_tol,
            restart=c_elliptic_restart,
            maxiter=self.pressure_maxit,
        )
        # REQ-173's `elliptic` stage contract: the executor must surface
        # ‖b − A x‖/‖b‖.  ⚠ BACKEND DIVERGENCE, deliberate: numpy stores it on
        # `self.last_elliptic_rel_resid`; this method is PURE and runs inside
        # jit/scan, so assigning to self here would capture a TRACER, not a
        # value -- a trap, not a diagnostic.  jax therefore surfaces the same
        # number through `jax.debug.print`, which fires from inside the fused
        # loop.  Same contract, different delivery; if REQ-173 later demands
        # the attribute specifically, it has to be threaded out through the
        # scan carry rather than assigned.
        #
        # One extra matvec (<1% of the ~10^2 the solve just spent) buys the
        # warning PETSc backends get for free.  Measured on VAM(1,3) 2-D the
        # shipped defaults converge to 2304 cells (rel resid <= 3.6e-7 vs the
        # 1e-6 tol), so this is a latent guard, not a live failure -- but an
        # unconverged pressure is otherwise indistinguishable from a good one.
        b_norm = jnp.linalg.norm(b)
        rel_resid = jnp.linalg.norm(matvec(p_new) + b) / jnp.where(
            b_norm > 0, b_norm, 1.0)

        # A missed tolerance is FATAL.  It used to `jax.debug.print` a warning
        # and march on; on the Escalante bump that produced six more
        # plausible-looking steps and then a non-finite state, with the actual
        # cause (a stale predictor aux pool) buried under a "pressure not
        # converged" line nobody could act on.  The abort is delivered through
        # `jax.debug.callback` so it fires from inside jit/scan too, where a
        # Python `raise` on a tracer is impossible.
        h_min = (jnp.min(Q[self._h_row, :self.nc])
                 if self._h_row is not None else jnp.asarray(jnp.nan))
        # Two DIFFERENT numbers, as in the numpy reference: the solve TARGETS
        # c_elliptic_tol; `tol` below is the ACCEPT gate (see _elliptic_accept).
        tol = _elliptic_accept(c_elliptic_tol)
        target = float(c_elliptic_tol)
        maxit = int(self.pressure_maxit)
        restart = int(c_elliptic_restart)
        restart_prov = ell.provenance["c_elliptic_restart"]

        def _abort(r, t_now, dt_now, hm):
            r = float(r)
            self.last_elliptic_rel_resid = r
            t_now, dt_now, hm = float(t_now), float(dt_now), float(hm)
            step_no = int(round(t_now / dt_now)) + 1 if dt_now > 0 else -1
            raise EllipticSolveError(
                "Chorin elliptic (pressure) stage did NOT converge at step "
                f"{step_no} (t={t_now:.6g}, dt={dt_now:.6g}): rel residual "
                f"{r:.3e} > accept gate {tol:.1e} (target {target:.1e}) after "
                f"the full GMRES budget "
                f"(restart={restart} x maxiter={maxit} ~ {restart * maxit} "
                f"matvecs; restart provenance: {restart_prov}); "
                f"min(h)={hm:.6g}.  The march is ABORTED — a projection that "
                "misses its tolerance leaves the corrector applying a "
                "pressure that does not satisfy the elliptic constraint, and "
                "the state goes non-finite a few steps later.  Check the aux "
                "pools handed to step() and the elliptic RHS before touching "
                "the tolerance.")

        try:
            # EAGER path: rel_resid is concrete, so raise from Python and get a
            # real EllipticSolveError with a real traceback.  Also REQ-174 —
            # the residual stays on `self` for the converged path (inside
            # jit/scan it is a tracer, and stashing a tracer on self is a trap,
            # not a diagnostic, so that path deliberately does not).
            rr = float(rel_resid)
        except Exception:
            jax.lax.cond(
                rel_resid > tol,
                lambda r, tn, dn, hm: jax.debug.callback(_abort, r, tn, dn, hm),
                lambda r, tn, dn, hm: None,
                rel_resid, jnp.asarray(time, dtype=rel_resid.dtype),
                jnp.asarray(dt, dtype=rel_resid.dtype), h_min,
            )
        else:
            self.last_elliptic_rel_resid = rr
            if rr > tol:
                _abort(rr, time, dt, h_min)
        return Q.at[e2s, :].set(p_new.reshape(nP, nc))

    def _step_corrector_pure(self, Q, Qaux_corr, parameters):
        """Pure-functional corrector: ``Q[corr_e2s] ← update(Q, Qaux_corr,
        parameters)``, `parameters` the CORRECTOR sub-system's vector
        (`_sub_parameters`, which carries dt).

        The splitter's "self-contained sub-models" refactor
        (zoomy_core@44bdfca) moved the corrector's projection formula from
        the `state_update` slot onto `update_variables` — accept either
        (older splits carry `state_update`, current ones
        `update_variables`).  Neither present is a HARD error: silently
        skipping the corrector leaves the momenta/r unprojected (the solved
        pressure is never applied — task 0039's second half)."""
        rt = self.rt_corr
        e2s = self._corr_state_idx
        fn = rt.state_update if rt.state_update is not None \
            else rt.update_variables
        if fn is None:
            raise RuntimeError(
                "Chorin corrector sub-model carries neither state_update nor "
                "update_variables — the pressure projection would silently "
                "never be applied.")
        new_vals = fn(Q, Qaux_corr, parameters)
        return Q.at[e2s, :].set(jnp.asarray(new_vals))
