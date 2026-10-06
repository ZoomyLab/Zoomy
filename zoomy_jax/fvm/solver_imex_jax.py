"""JAX IMEX solver: explicit flux + implicit diffusion + implicit source.

``IMEXSourceSolverJax`` is the jax :class:`HyperbolicSolver` with two implicit
stages appended to its explicit step.  Everything else -- the adaptive
per-face CFL, ``dt_max``, the time loop, ``post_step``, output, ``ad_mode`` --
is the base solver's.  One step is

1. the base explicit step WITHOUT the source (flux only, RK2 at order 2,
   MOOD-aware);
2. implicit Crank-Nicolson diffusion of the runtime ``diffusion_matrix``
   (when the model carries one; the base flux then leaves it out);
3. implicit Euler on the source, ``Q - Q_e - dt S(Q, aux(Q), p, t + dt) = 0``,
   solved cell-wise (``source_mode="local"``) or globally by Newton-GMRES
   (``"global"``).

Both implicit stages are Newton roots wrapped in ``jax.lax.custom_root``
(:mod:`zoomy_jax.fvm.implicit_ad`), so the solver is differentiable in
forward and reverse mode through the converged roots.
"""

import param
import jax

from zoomy_core.fvm.solver_numpy import HyperbolicSolver as HyperbolicSolverNumpy
from zoomy_jax.fvm.implicit_ad import cellwise_solve, gmres, newton_root
from zoomy_jax.fvm.jvp_jax import analytic_source_jvp_jax
from zoomy_jax.fvm.reconstruction_jax import DenseDiffusionOperatorJAX
from zoomy_jax.fvm.solver_jax import HyperbolicSolver


class IMEXSourceSolverJax(HyperbolicSolver):
    """Jax IMEX solver: the base explicit step plus implicit diffusion and an
    implicit source stage.  The knobs below are read in
    ``setup_simulation``."""

    source_mode = param.Selector(
        objects=["auto", "local", "global"], default="auto",
        doc="Implicit source solve.  ``local``: cell-wise Newton with a dense "
            "per-cell Jacobian; the aux is recomputed from the iterate through "
            "the cell-local ``update_aux_variables`` map, the derivative "
            "(LSQ-gradient) aux rows are frozen at the explicit predictor.  "
            "``global``: Newton-GMRES on the whole field with the full aux "
            "update (gradient rows included) inside the residual.  ``auto``: "
            "``global`` when the NSM carries ``source_derivative_specs``, "
            "else ``local``.")
    implicit_tol = param.Number(default=1e-8, bounds=(0, None),
        doc="Newton stops once ||R||_2 <= implicit_tol (implicit source and "
            "state-dependent implicit diffusion).")
    implicit_maxiter = param.Integer(default=6, bounds=(1, None),
        doc="Maximum Newton updates per implicit solve.")
    gmres_tol = param.Number(default=1e-7, bounds=(0, None),
        doc="Relative GMRES tolerance (global source mode, implicit "
            "diffusion, and the tangent solves of the derivatives).")
    gmres_maxiter = param.Integer(default=30, bounds=(1, None),
        doc="GMRES restarts per linear solve.")
    jv_backend = param.Selector(
        objects=["ad", "fd", "analytic"], default="ad",
        doc="Jacobian action the source NEWTON iteration uses: ``ad`` exact "
            "jax.jvp; ``fd`` one-sided finite difference with ``fd_eps``; "
            "``analytic`` the lowered symbolic source Jacobians (aux chain "
            "term by AD of the aux map).  Derivatives of the solution always "
            "use the exact AD linearization.")
    fd_eps = param.Number(default=1e-7, bounds=(0, None),
        doc="Finite-difference step for ``jv_backend='fd'``.")

    _source_in_explicit_step = False
    _diffusion_in_flux = False

    def setup_simulation(self, mesh, model):
        """Base setup, then the implicit diffusion and source operators."""
        Q, Qaux = super().setup_simulation(mesh, model)
        if getattr(self, "_bern_ode", False):
            raise ValueError(
                "IMEXSourceSolverJax: the steady-ODE well-balanced "
                "reconstruction already carries the source inside the "
                "reconstruction; an implicit source stage would step it twice.")
        self._rt_implicit_diffusion = self._build_implicit_diffusion()
        self._rt_implicit_source = self._build_implicit_source()
        return Q, Qaux

    def step(self, dt, time, Q, Qaux, parameters=None, coupled=None):
        """Explicit flux step, then implicit diffusion, then implicit source.
        Same signature and contract as the base ``step``."""
        parameters = self._rt_parameters if parameters is None else parameters
        Q = super().step(dt, time, Q, Qaux, parameters, coupled=coupled)
        if self._rt_implicit_diffusion is not None:
            Q = self._rt_implicit_diffusion(dt, time, Q, Qaux, parameters)
        return self._rt_implicit_source(dt, time, Q, Qaux, parameters)

    # ── implicit operators ──────────────────────────────────────────────

    def _resolve_source_mode(self):
        if self.source_mode != "auto":
            return self.source_mode
        return "global" if self.nsm.source_derivative_specs else "local"

    def _build_implicit_source(self):
        """``(dt, time, Q, Qaux, parameters) -> Q``: implicit Euler on the
        source at ``time + dt``,

            R(Q) = Q - Q_e - dt * S(Q, aux(Q), p, time + dt, dt, x) = 0,

        ``Q_e`` the explicit predictor.  ``aux(Q)``: ``local`` mode recomputes
        the cell-local ``update_aux_variables`` rows from the iterate on top
        of the aux of ``Q_e`` (gradient rows frozen); ``global`` mode runs the
        full ``update_qaux``."""
        mode = self._resolve_source_mode()
        jv_backend = self.jv_backend
        tol, maxiter = float(self.implicit_tol), int(self.implicit_maxiter)
        gmres_tol, gmres_maxiter = float(self.gmres_tol), int(self.gmres_maxiter)
        fd_eps = float(self.fd_eps)
        mesh, rt = self._rt_mesh, self._rt_model
        update_qaux = self.update_qaux
        uav = rt.update_aux_variables
        if jv_backend == "analytic" and (
                rt.source_jacobian_wrt_variables is None
                or (rt.n_aux and rt.source_jacobian_wrt_aux_variables is None)):
            raise ValueError(
                "jv_backend='analytic' needs the lowered source Jacobians; "
                "the SystemModel was built without them "
                "(derive_source_jacobians=False).")

        if mode == "local":
            linear_solve = cellwise_solve
        else:
            def linear_solve(matvec, b):
                return gmres(matvec, b, tol=gmres_tol, maxiter=gmres_maxiter)

        def implicit_source(dt, time, Q, Qaux, parameters):
            t_imp = time + dt
            position = mesh.cell_centers[:, :Q.shape[-1]]

            def full_aux(Qs, Qa):
                return update_qaux(Qs, Qa, Qs, Qa, mesh, rt, parameters,
                                   t_imp, dt)

            if mode == "local":
                Qaux_e = full_aux(Q, Qaux)
                if uav is None:
                    def aux_of(Qs):
                        return Qaux_e
                else:
                    def aux_of(Qs):
                        return uav(Qs, Qaux_e, parameters, time=t_imp,
                                   position=position)
            else:
                def aux_of(Qs):
                    return full_aux(Qs, Qaux)

            def residual(Qs):
                S = rt.source(Qs, aux_of(Qs), parameters, time=t_imp, dt=dt,
                              position=position)
                return Qs - Q - dt * S

            if jv_backend == "ad":
                def jvp(Qs, V):
                    return jax.jvp(residual, (Qs,), (V,))[1]
            elif jv_backend == "fd":
                def jvp(Qs, V):
                    return (residual(Qs + fd_eps * V) - residual(Qs)) / fd_eps
            else:
                def jvp(Qs, V):
                    Qa, dQa = jax.jvp(aux_of, (Qs,), (V,))
                    return V - dt * analytic_source_jvp_jax(
                        rt, Qs, Qa, V, parameters, dQaux=dQa)

            return newton_root(residual, Q, jvp, linear_solve,
                               tol=tol, maxiter=maxiter)

        return implicit_source

    def _build_implicit_diffusion(self):
        """``(dt, time, Q, Qaux, parameters) -> Q``: one Crank-Nicolson step
        of the runtime ``diffusion_matrix`` with
        :class:`DenseDiffusionOperatorJAX` (interior faces; a linear GMRES
        solve for a constant tensor, Newton-GMRES for a state-dependent one).
        ``None`` when the model carries no implicit diffusion."""
        rt = self._rt_model
        dmat = rt.diffusion_matrix
        if dmat is None:
            return None
        sm = self.nsm.sm
        _, state_dependent, _ = HyperbolicSolverNumpy._classify_diffusion(
            sm.diffusion_matrix, sm)
        op = DenseDiffusionOperatorJAX(
            self._rt_mesh_np, self._rt_mesh_np.dimension, rt.n_state,
            state_dependent=state_dependent)
        nc = op.nc
        tol, maxiter = float(self.implicit_tol), int(self.implicit_maxiter)
        gmres_tol, gmres_maxiter = float(self.gmres_tol), int(self.gmres_maxiter)

        def implicit_diffusion(dt, time, Q, Qaux, parameters):
            Qaux_in = Qaux[:, :nc]

            def A_fn(Qs):
                return dmat(Qs, Qaux_in, parameters)

            Q_new = op.implicit_solve(
                Q[:, :nc], dt, A_fn, bf_grads=None,
                tol=gmres_tol, maxiter=gmres_maxiter,
                newton_maxiter=maxiter, newton_tol=tol)
            return Q.at[:, :nc].set(Q_new)

        return implicit_diffusion
