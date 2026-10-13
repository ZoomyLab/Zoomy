"""`MarchSolver` — the v6 march assembled from `blocks`.

This is the REFERENCE IMPLEMENTATION of the phase-1 solver design: the
printed march that core will later emit must reproduce these semantics, so
the block structure and the signatures are the deliverable, not cleverness.

It is a **new shell over the existing kernels**.  Setup is inherited verbatim
from `zoomy_jax.fvm.solver_jax.HyperbolicSolver` (model coercion, NSM
resolution, LSQ mesh, periodic-BC remap, `JaxRuntime` construction, the
reconstruction factory, the initial aux/BC sweep).  Nothing in
`solver_jax.py` is modified; the production solver stays the production
path (standing user ruling: side-by-side, no deletion).

Shape of one step (design §3 with the v6 dt ruling):

```
    STEP HEAD   lam_lo, lam_hi = dt_pass(...)          # stored, 2 scalars/face
                dt             = reduce_dt(...)        # FROZEN for the step
                assert_dt_admissible(dt, ...)          # FATAL if dt <= 0
    STAGE LOOP  for (alpha, beta) in tableau:
                    halo_bc      -> bf_values
                    reconstruct  -> Q_L, Q_R, lim_grad
                    flux_pass    -> Fface, Dp, Dm      # STORED
                    gather_update-> Q_cand, troubled   # fused detection
                    Q = alpha*Q0 + (1-alpha)*Q_cand    # Shu-Osher
    MOOD        mood_resolve(troubled, ...)            # whole-step O1 redo
    POST        update_variables -> update_aux
```
"""

from __future__ import annotations

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import param

from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.fvm.solver2 import blocks as B
from zoomy_jax.fvm.solver2.context import Ops, build_operators, prepare_mesh
from zoomy_jax.fvm.solver2.state import MarchState, proceed, should_write


class MarchSolver(HyperbolicSolver):
    """The solver2 march.  Inherits every `param` knob and the whole setup
    path from `HyperbolicSolver`; overrides only the march itself.

    Parameters that matter here:

    * `CFL` — the user law (0.9 in 1-D, 0.45 in 2-D).  It is applied by
      `blocks.reduce_dt` and is NEVER reduced on instability: an
      instability at the law CFL is a reported finding.
    * `mood_redo` — solver2's own flag for the sanctioned whole-step
      order-1 MOOD redo.  It is deliberately NOT the inherited `mood`
      param: that one selects the production per-cell `force_o1` corrector
      and its setup guard demands `reconstruction_variables='eta'`.  The
      whole-step redo has no such requirement (amendment 5: the strategy is
      the backend's choice, only the OUTCOME is contracted), so it works with
      any reconstruction.
    """

    mood_redo = param.Boolean(default=False, doc=(
        "A-posteriori MOOD with the sanctioned whole-step order-1 redo: if "
        "any cell of the final candidate is troubled (h < 0 or non-finite), "
        "replay the WHOLE step with the piecewise-constant reconstruction. "
        "Conservative by construction (same shared face flux) and "
        "positivity-preserving by the first-order XZS lemma. Nothing is "
        "clipped or floored."))

    # ── setup ───────────────────────────────────────────────────────────
    def setup_march(self, mesh, model, *, CFL, dt_max=None,
                    dt_dimension=None):
        """Run the inherited `setup_simulation` and translate its results
        into `(MeshRT, Ops)` — the two design-§2 translation blocks.

        `dt_dimension` is the `d` of the CFL denominator
        `CFL*2r/(d(2k+1)|lam|)`.  Default = the mesh's own spatial dimension
        (the honest reading of `timestepping.adaptive`'s docstring).  It is
        exposed because the production call sites use `adaptive(CFL=...)`
        with that argument left at its default of 2 even on 1-D meshes, so a
        parity comparison must be able to say so explicitly.  It is NOT a CFL
        knob: the safety factor stays the user law.
        """
        Q, Qaux = self.setup_simulation(mesh, model)

        self._cfl = float(CFL)
        self._dt_max = self.nsm.dt_max if dt_max is None else dt_max
        self._dt_dimension = dt_dimension

        self.MeshRT = prepare_mesh(self._rt_mesh, self.nsm,
                                   mesh_np=self._rt_mesh_np)
        order = int(self.nsm.reconstruction.order)
        recon = self._build_reconstruction(
            self._rt_mesh, self._rt_model.sm, self._rt_model)
        self.Ops = build_operators(
            self._rt_model, self.MeshRT,
            reconstruct=recon, order=order,
            h_index=self._resolve_h_index(),
            aux_registry_walk=HyperbolicSolver._walk_derivative_aux,
        )
        self._build_step()
        return MarchState(0.0, 0, 0, Q, Qaux)

    def _resolve_h_index(self):
        """Row index of `h` for the PAD troubled predicate.  Explicit
        `free_surface_h_index` wins; otherwise resolve BY NAME.  Never
        positional — an unmapped row must stay `None` (non-finite detection
        only), not silently resolve to something else."""
        if self.free_surface_h_index is not None:
            return int(self.free_surface_h_index)
        names = [str(s) for s in self.nsm.state]
        return names.index("h") if "h" in names else None

    # ── the step (jitted once, closed over MeshRT/Ops) ──────────────────
    def _build_step(self):
        MeshRT, Ops_ = self.MeshRT, self.Ops
        nc = MeshRT.n_cells
        cell_centers = MeshRT.cell_centers[:, :nc]

        def _source(Q, Qaux, p, t, dt):
            if Ops_.source is None:
                return None
            return Ops_.source(Q[:, :nc], Qaux[:, :nc], p,
                               time=t, dt=dt, position=cell_centers)

        def stage_loop(Q0, Qaux, p, t, dt, o1):
            """The explicit stage loop.  `dt` is FROZEN (v6): it was computed
            at the step head and no stage recomputes it.  `o1=True` is the
            MOOD redo; identical code, piecewise-constant reconstruction."""
            Qk = Q0
            troubled = jnp.zeros((Q0.shape[1],), dtype=bool)
            for k, (alpha, beta) in enumerate(Ops_.tableau):
                t_stage = t + (0.0 if k == 0 else dt)

                Qk, Qaux_s, bf = B.halo_bc(Qk, Qaux, p, MeshRT, t_stage, Ops_)
                Q_L, Q_R, lim_grad = B.reconstruct(
                    Qk, Qaux_s, MeshRT, Ops_, bf, o1=o1)
                Fface, Dp, Dm = B.flux_pass(
                    Qk, Qaux_s, p, MeshRT, Ops_, t_stage, (Q_L, Q_R))

                # amendment 10: the cell-interior NCP integral, WB-critical at
                # order >= 2.  No |cell| division — the volume factor cancels
                # against the per-unit-volume residual.
                cell_term = None
                if lim_grad is not None and Ops_.nonconservative_matrix is not None:
                    Ball = Ops_.nonconservative_matrix(Qk, Qaux_s, p)
                    ncp = jnp.einsum("ijdc,jdc->ic", Ball, lim_grad)
                    cell_term = jnp.zeros_like(Qk).at[
                        :, :ncp.shape[1]].subtract(ncp)

                Q_cand, tr = B.gather_update(
                    Qk, Fface, Dp, Dm, MeshRT, dt, beta,
                    source=_source(Qk, Qaux_s, p, t_stage, dt),
                    cell_term=cell_term, h_index=Ops_.h_index)
                troubled = troubled | tr
                Qk = alpha * Q0 + (1.0 - alpha) * Q_cand
            return Qk, troubled

        use_mood = bool(self.mood_redo)

        @jax.jit
        def hyperbolic_step(S, dt, p):
            Q_cand, troubled = stage_loop(S.Q, S.Qaux, p, S.time, dt, False)
            if use_mood:
                Q_next = B.mood_resolve(troubled, S.Q, Q_cand, S.Qaux, p,
                                        S.time, dt, MeshRT, Ops_, stage_loop)
            else:
                Q_next = Q_cand
            t_new = S.time + dt
            Q_next = B.update_variables(Q_next, S.Qaux, p, t_new, dt,
                                        MeshRT, Ops_)
            Qaux_next = B.update_aux(Q_next, S.Qaux, p, t_new, dt,
                                     MeshRT, Ops_)
            return MarchState(t_new, S.iteration + 1, S.i_snapshot,
                              Q_next, Qaux_next), troubled

        @jax.jit
        def head(S, p, t_remaining, dt_window):
            lo, hi = B.dt_pass(S.Q, S.Qaux, p, MeshRT, Ops_)
            dt = B.reduce_dt(
                lo, hi, MeshRT.inradius_f, self._dt_max,
                CFL=self._cfl,
                dimension=(MeshRT.dimension if self._dt_dimension is None
                           else self._dt_dimension),
                t_remaining=(None if dt_window is not None else t_remaining),
                dt_window=dt_window)
            return lo, hi, dt

        self.step_head = head
        self.hyperbolic_step = hyperbolic_step
        self.stage_loop = stage_loop

    # ── the march ───────────────────────────────────────────────────────
    def march(self, S, *, time_end=None, n_steps=None, dt_window=None,
              write_interval=None, on_write=None, parameters=None,
              max_steps=1_000_000):
        """Run the march.  `proceed(S)` drives the loop; the per-step
        sequence is exactly the one in the module docstring.

        The loop is a plain python `while` so the FATAL dt guard
        (`blocks.assert_dt_admissible`) is a real abort with a real
        diagnostic.  Each block is compiled once; the host sees two dispatches
        per step (head + step).  Returns `(S, stats)`.

        `max_steps` is the SECOND honesty guard.  `assert_dt_admissible`
        catches `dt <= 0`; it does NOT catch the other zero-progress mode —
        a dt that collapses towards zero while staying strictly positive (the
        classic dry-front `u = q/h` blow-up at order >= 2 without a wet/dry
        reconstruction).  Both are "the march stopped making progress" and
        both must ABORT with a diagnostic rather than spin.
        """
        p = self._rt_parameters if parameters is None else parameters
        time_end = self.time_end if time_end is None else time_end
        stats = {"steps": 0, "dt_min": np.inf, "dt_max": 0.0,
                 "troubled_steps": 0, "wall": 0.0}

        t0 = perf_counter()
        while (proceed(S, time_end) if n_steps is None
               else S.iteration < n_steps):
            lo, hi, dt = self.step_head(
                S, p, time_end - S.time, dt_window)
            dt = B.assert_dt_admissible(dt, lo, hi, self.MeshRT.inradius_f,
                                        time=S.time, iteration=S.iteration)
            S, troubled = self.hyperbolic_step(S, dt, p)

            stats["steps"] += 1
            stats["dt_min"] = min(stats["dt_min"], dt)
            stats["dt_max"] = max(stats["dt_max"], dt)
            if bool(np.asarray(troubled).any()):
                stats["troubled_steps"] += 1

            if write_interval is not None:
                do_write, i_snap = should_write(
                    float(S.time), dt, int(S.i_snapshot), write_interval)
                if do_write and on_write is not None:
                    on_write(S)
                S = S._replace(i_snapshot=i_snap)

            if stats["steps"] >= max_steps:
                raise FloatingPointError(
                    f"march exceeded max_steps = {max_steps} at t = "
                    f"{float(S.time)} / {time_end} with dt = {dt:.6e} "
                    f"(dt_min so far {stats['dt_min']:.6e}) — the timestep "
                    f"collapsed towards zero without ever becoming "
                    f"non-positive, so the dt guard could not see it. This is "
                    f"a REPORTED FINDING about the scheme/configuration; the "
                    f"march does not silently spin and the CFL is not reduced.")
        jax.block_until_ready(S.Q)
        stats["wall"] = perf_counter() - t0
        return S, stats

    # ── convenience: setup + march ──────────────────────────────────────
    def solve_march(self, mesh, model, *, CFL, time_end=None, n_steps=None,
                    dt_dimension=None):
        S = self.setup_march(mesh, model, CFL=CFL, dt_dimension=dt_dimension)
        return self.march(S, time_end=time_end, n_steps=n_steps)


def describe_nsm(nsm) -> str:
    """The mandatory pre-march NSM operator print (user law).

    Same content as the thesis SWASHES case's `describe_nsm` — state, aux,
    parameters, flux, NCP matrix, source, eigenvalues, and the cap check
    (`update_variables` must be `None` for the capless derived SWE).
    """
    return "\n".join([
        f"state: {list(nsm.state)}",
        f"aux_state: {list(nsm.aux_state)}",
        f"parameter_values: {nsm.parameter_values}",
        f"flux:\n{nsm.flux}",
        f"hydrostatic_pressure:\n{getattr(nsm, 'hydrostatic_pressure', None)}",
        f"nonconservative_matrix:\n{nsm.nonconservative_matrix}",
        f"source:\n{nsm.source}",
        f"eigenvalues:\n{nsm.eigenvalues}",
        f"update_variables (must be None — cap-free): {nsm.update_variables}",
        f"reconstruction: order={nsm.reconstruction.order} "
        f"limiter={nsm.reconstruction.limiter}",
        f"dt_max: {nsm.dt_max}",
    ])
