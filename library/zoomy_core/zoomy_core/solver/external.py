"""`ExternalProcedure` — the third declaration kind (validation appendix D12).

`register_symbolic_function` cannot carry the solver's opaque blocks: it
evaluates the body EAGERLY at registration (`basefunction.py`) and a `None`
body yields a ZERO definition, and the `kernel_functions` externals are
expression-position scalar atoms.  The design's opaque blocks
(`prepare_mesh` / `mood_resolve` / `write_fields` / the coupling
surfaces / …) are statement-position, side-effecting and handle-typed.

So: declare the NAME + SIGNATURE in core, supply the BODY per backend, and put
every declaration on `REQUIRED_PROCEDURES` — the same contract shape
`REQUIRED_KERNELS` (`kernel_functions.py`) already uses for the numerical
UserFunctions, so a backend missing a body is a RED TEST rather than a link
error mid-build (C++) or a `NameError` at lambdify (python).
"""
from __future__ import annotations

import dataclasses

from zoomy_core.solver.ir import check_arg_keys, check_procedure_name


@dataclasses.dataclass(frozen=True)
class ExternalProcedure:
    """A solver block declared in core, bodied by each backend.

    `args` / `results` are solver argument-vocabulary slots; `doc` states
    the CONTRACT the backend body must satisfy (that text is the whole point —
    it is what a new backend implements against).
    """

    name: str
    args: tuple = ()
    results: tuple = ()
    doc: str = ""

    def __post_init__(self):
        object.__setattr__(self, "name", check_procedure_name(self.name))
        object.__setattr__(self, "args", check_arg_keys(self.name, self.args))
        object.__setattr__(self, "results", check_arg_keys(self.name, self.results))


def _decl(name, args=(), results=(), doc=""):
    return ExternalProcedure(name=name, args=args, results=results, doc=doc)


#: The opaque solver blocks — design "Core inventory (final)", section (b).
#: Every backend ships a contract test asserting its body table covers this
#: set (see ``required_procedure_gaps``).
REQUIRED_PROCEDURES = {
    p.name: p
    for p in (
        _decl("solver_prepare_mesh", doc=(
            "mesh + NSM -> MeshRT (cell_volumes, face_area, face_normal, "
            "face_cells, inradius_f, lsq_stencils, bc_face_groups, "
            "periodic_map).  Translates ONCE into the backend's containers.")),
        _decl("solver_build_operators", doc=(
            "kernels + MeshRT -> Ops (flux_face, fluct_face, bc_face, "
            "eigenvalues, reconstruct, limiter, source, update_variables, "
            "update_aux, tableau).")),
        _decl("solver_initialize_state", doc=(
            "MeshRT + IC + BC -> S(time=0, iteration=0, i_snapshot=0, Q, "
            "Qaux).")),
        _decl("solver_halo_bc",
              args=("variables", "aux_variables", "time"), doc=(
                  "Per-stage ghost/halo synchronisation (D3): amrex "
                  "UpdateState+FillPatch+FillBoundary, unstructured ghost "
                  "synthesis.  Load-bearing EVERY stage, not once per step.")),
        _decl("solver_mood_resolve",
              args=("troubled", "q0", "q_cand", "aux_variables", "parameters",
                    "time", "dt"), doc=(
                  "OUTCOME semantics only (D5): troubled cells come back "
                  "positive and finite AND mass is exactly conserved.  The "
                  "strategy is the backend's: whole-step O1 re-step (may "
                  "replay the stage loop), face-local re-solve, or "
                  "compute-both/select on GPU.")),
        _decl("solver_implicit_source",
              args=("q_cand", "aux_variables", "parameters", "dt"), doc=(
                  "a_ii*dt point-implicit source solve (local Newton or "
                  "JFNK).")),
        _decl("solver_implicit_diffusion",
              args=("q_cand", "aux_variables", "parameters", "dt"), doc=(
                  "a_ii*dt global diffusion solve; also reports rel_resid.")),
        _decl("solver_write_fields",
              args=("variables", "aux_variables", "time", "i_snapshot"), doc=(
                  "Ordered side effect (REQ-92): snapshot index is an INPUT, "
                  "the block must not re-derive it from time.")),
        _decl("solver_adapt_mesh", doc=(
            "Regrid owner (D4), between steps only — Q0 never spans a "
            "regrid.")),
        _decl("solver_reduce_min",
              args=("lam_f", "n_faces"), doc=(
                  "Global min reduction primitive: jnp.min / gMin / "
                  "ParReduce.")),
        _decl("solver_coupling_init", doc="preCICE participant setup."),
        _decl("solver_coupling_read", args=("time", "dt"), doc=(
            "Read peer data for this window (waveform-sampled at t).")),
        _decl("solver_coupling_write", args=("variables", "time"),
              doc="Write this side's interface data."),
        _decl("solver_coupling_advance", args=("dt",), doc=(
            "Advance the coupling; yields the remaining window size.")),
        _decl("solver_coupling_checkpoint", doc=(
            "Implicit-scheme checkpoint/rollback; a NO-OP in the explicit "
            "parallel scheme (v6 §2 ruling).")),
        _decl("solver_coupling_face_fixup",
              args=("flux_face", "d_plus", "d_minus", "n_faces"), doc=(
                  "7th coupling surface (D8): the per-stage face-array fixup "
                  "(applyFrozenMassRow) at the END of flux_pass, MOOD "
                  "recompute included.  STRICT no-op when coupling is "
                  "inactive.")),
        _decl("solver_coupling_finalize", doc="Tear the participant down."),

        # ── the v6 march bodies (zoomy_core.solver.march) ────────────────
        # Signatures are stated in the doc rather than in ``args``: these
        # blocks pass march-local intermediates (bf_values, Q_L/Q_R, lim_grad,
        # cell_term, the emitted constants) that are NOT march-scale argument
        # slots and must not be forced into the slot vocabulary just to be
        # declarable.  The doc text is the contract a backend implements.
        _decl("solver_dt_pass", doc=(
            "(Q, Qaux, p, c_eps_h) -> (lam_lo_f, lam_hi_f), each (n_faces,). "
            "Evaluate the eigenvalues slot at the CELL states on BOTH sides "
            "of every face and store the two signed wave-speed bounds.  v6 "
            "§1: this runs at the STEP HEAD, before any stage; the flux pass "
            "no longer returns a lambda row.  ``c_eps_h`` is the emitted "
            "wave-speed floor (REQ-82): a backend whose eigenvalue kernel "
            "already carries ``max(eps_h, h)`` (core's ``regularize_pow`` "
            "bakes it in) accepts and ignores it; one that does not MUST "
            "apply it here rather than restate the number.")),
        _decl("solver_reduce_dt", doc=(
            "(lam_lo_f, lam_hi_f, inradius_f, n_faces, c_cfl, "
            "c_cfl_dimension, c_cfl_degree_factor, c_dt_max, clamp) "
            "-> dt.  dt <= c_cfl * 2r / (c_cfl_dimension * "
            "c_cfl_degree_factor * |lam|max) per face, minimised over faces, "
            "then capped by c_dt_max and by the single ``clamp`` (t_remaining "
            "XOR dt_window — D7, never min-combined).  A wave-free face has "
            "|lam|max == 0 and a local limit of +inf: it drops OUT of the "
            "minimum instead of imposing a floor (REQ-190).  Every bound is "
            "an EMITTED named constant; no float literal may appear.")),
        _decl("solver_apply_dt_floor", doc=(
            "(dt, c_dt_floor) -> dt.  Only ever called when the ``dt_floor`` "
            "build flag is ON, which it is NOT by default: the sanctioned "
            "march has no floor, only the FATAL guard.")),
        _decl("solver_assert_dt_admissible", doc=(
            "(dt, lam_lo_f, lam_hi_f, inradius_f, time, iteration) -> dt.  "
            "FATAL when dt <= 0 or non-finite: ABORT with a diagnostic naming "
            "the offending face.  Never a silent spin, never a dt-halving "
            "retry, never a CFL reduction.")),
        _decl("solver_assert_march_progress", doc=(
            "(time, iteration, dt, t_end) -> ().  The SECOND honesty guard: a "
            "dt that collapses towards zero while staying strictly positive "
            "is invisible to the dt guard.  ABORT on a step-count bound with "
            "a diagnostic; this is a reported FINDING about the scheme.")),
        _decl("solver_stage_base", doc=(
            "(Q) -> Q0.  Snapshot the stage base: the state the residual is "
            "evaluated at, kept for the Shu-Osher average AND for the MOOD "
            "rollback.  Q0 never spans a regrid (D4).")),
        _decl("solver_clear_troubled", doc=(
            "(Q) -> troubled, an all-false (n_cells,) mask starting the "
            "step's accumulation.")),
        _decl("solver_merge_troubled", doc=(
            "(troubled, troubled_stage) -> troubled.  Elementwise OR of two "
            "cell masks.  Declared as a body because the IR carries no "
            "elementwise boolean-array operator.")),
        _decl("solver_reconstruct", doc=(
            "(Q, Qaux, bf_values, o1) -> (Q_L, Q_R, lim_grad).  Face-state "
            "reconstruction; ``lim_grad`` is the LIMITED cell gradient or a "
            "null handle at order 1.  ``o1`` truthy forces the "
            "piecewise-constant object — that is the whole-step order-1 MOOD "
            "redo, and the reason the order-1 path needs no separate code. "
            "The reconstruction object owns the limiter and any "
            "well-balanced / positivity variable change.")),
        _decl("solver_cell_ncp", doc=(
            "(Q, Qaux, p, lim_grad) -> cell_term.  Amendment 10: the "
            "cell-interior non-conservative integral, WB-critical at order "
            ">= 2.  No |cell| division — the volume factor cancels against "
            "the per-unit-volume residual.")),
        _decl("solver_no_cell_term", doc=(
            "(Q) -> cell_term, the additive identity.  The ``interior_ncp`` "
            "build flag selects between this and solver_cell_ncp, so the "
            "gather pass has ONE shape and no runtime null test.")),
        _decl("solver_flux_pass", doc=(
            "(Q, Qaux, p, t_stage, Q_L, Q_R) -> (Fface, Dp, Dm), each "
            "(n_state, n_faces) and STORED.  Each face is evaluated exactly "
            "ONCE.  The boundary Q_R is recomputed from the RECONSTRUCTED "
            "inner face state at t_stage (amendment 11); at a periodic seam "
            "it is the partner cell's state (REQ-116).  v6: NO lambda row.")),
        _decl("solver_gather_update", doc=(
            "(Q0, Fface, Dp, Dm, dt, beta, cell_term, Qaux, p, t_stage) -> "
            "(Q_cand, troubled).  Q_cand = Q0 + beta*dt*(-div(F+D) + "
            "cell_term + source), with the troubled flags written in the same "
            "pass (the fused detection).  ``troubled`` is h < c_mood_h_bound "
            "(strict 0) or any non-finite component — DETECTED, never "
            "repaired in place.")),
        _decl("solver_update_variables", doc=(
            "(Q, Qaux, p, time, dt) -> Q.  The model's per-cell "
            "update_variables slot; the IDENTITY when the model declares "
            "none, which is exactly the case for the cap-free derived SWE.")),
        _decl("solver_update_aux", doc=(
            "(Q, Qaux, p, time, dt) -> Qaux.  The algebraic aux map (e.g. the "
            "KP-desingularised hinv at c_kp_eps) followed by the LSQ-gradient "
            "rows of the aux registry.  The algebraic write is FULL-LENGTH: a "
            "prefix write silently mis-places rows and degenerates SME(>=1) "
            "to SWE.")),
    )
}


class MissingProcedureBody(LookupError):
    """A backend's body table does not cover `REQUIRED_PROCEDURES`."""


def required_procedure_gaps(bodies) -> tuple:
    """Slots of `REQUIRED_PROCEDURES` a backend's body table misses.

    `bodies` is any container supporting `in` (a dict of callables for the
    python backends, a set of emitted symbol names for the C++ ones).  Returns
    a sorted tuple — empty means the contract is met.  The backend's own
    contract test turns a non-empty result into a RED TEST.
    """
    return tuple(sorted(n for n in REQUIRED_PROCEDURES if n not in bodies))


def assert_procedure_bodies(bodies, backend: str) -> None:
    """Contract assertion for a backend body table; RAISES on any gap."""
    gaps = required_procedure_gaps(bodies)
    if gaps:
        raise MissingProcedureBody(
            f"backend {backend!r} supplies no body for {list(gaps)} — every "
            "REQUIRED_PROCEDURES entry needs one (missing body = red test, "
            "never a link error)"
        )
