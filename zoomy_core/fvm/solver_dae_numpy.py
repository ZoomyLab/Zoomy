"""DAE solver (numpy backend) — index-1 DAE-PDE integrator via
Ascher-Ruuth-Spiteri IMEX-Runge-Kutta time stepping.

Sits next to `HyperbolicSolver` and `IMEXSolver` in
`zoomy_core.fvm` and reuses every shared piece of the framework:

* `SystemModel` is the single symbolic source-of-truth;
  `SystemModel.from_model` auto-scans all non-state Function and
  Derivative atoms into `aux_state` + a structured
  `SystemModel.aux_registry`.
* `NumpyRuntimeModel` lambdifies the operator matrices into
  `(Q, Qaux, p) → ndarray` callables.
* `ensure_lsq_mesh` promotes the input mesh to
  `LSQMesh`, exposing `compute_derivatives` for the registry-
  driven `update_qaux` walker.
* Boundary-condition objects (`Extrapolation`, `Lambda`,
  `InflowOutflow`, …) supply face values via `face_value`.
* `Solver.update_q` / `Solver.update_qaux` (registry-aware default)
  fire after every step.

Time integration: Ascher-Ruuth-Spiteri IMEX-Runge-Kutta (ARS232 / ARS343)
on the singular DAE `M(Q)·∂_t Q = R(Q, Qaux, p)`.  Aux is updated
once per step (lagged through the Newton iteration), keeping the
Jacobian small and reusing the standard `Solver.update_qaux` hook.

Spatial scheme: Rusanov flux + non-conservative path-integral
fluctuations, mirroring `NonconservativeRusanov` but driven by
the SystemModel runtime (so derivative-aux entries like `∂_x h` flow
through as ordinary Qaux components, no special-casing).  Boundary
faces use ``BC.face_value(Q_inner, Qaux_inner, normal, d_face, time,
parameters)``.
"""
from __future__ import annotations

import os
import time as _time

import numpy as np
import param

from zoomy_core.fvm.solver_numpy import Solver
from zoomy_core.fvm.imex_ark import ars232, ars343, imex_ark_step
from zoomy_core.fvm import timestepping
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.model.boundary_conditions import boundary_face_branches
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from zoomy_core.misc.logger_config import logger
import zoomy_core.misc.io as io


_EMPTY_AUX = np.empty(0, dtype=float)


class DAESolver(Solver):
    """Index-1 DAE-PDE solver with IMEX-RK time stepping.

    Order-2 status (2026-05-14)
    ---------------------------
    At `reconstruction_order=1` this solver is the validated
    *correctness reference* for the VAM chain: lake-at-rest is
    preserved to machine precision and a perturbation propagates with
    bounded mass loss.

    At `reconstruction_order=2` the *spatial scheme* is also correct
    — lake-at-rest over a bump is well-balanced to ~1e-14 (the η = h+b
    `SurfaceReconstruction` + the cell-interior non-conservative
    integral telescope exactly), and the slope limiter is frozen
    through the Newton iteration so `f_I` is a smooth function of the
    stage unknown.  But the *time integration* does not converge: the
    monolithic IMEX-ARK stage Jacobian is ill-conditioned (cond ~1e7),
    concentrated in the algebraic pressure-constraint rows.  At that
    conditioning the finite-difference Jacobian (step ~1e-7) is
    unreliable, so the stage Newton degrades from quadratic to slow
    *linear* convergence and does not reach `newton_tol` within
    `newton_maxit`.

    This is structural, not a bug: the monolithic DAE couples a
    well-conditioned hyperbolic evolution block and an ill-conditioned
    elliptic pressure-constraint block into a single FD Jacobian.  The
    fix is the Chorin / projection split — an *explicit* hyperbolic
    predictor (where the order-2 reconstruction lives, no Jacobian
    needed) plus a *separate linear* elliptic pressure solve (which can
    be preconditioned properly).  The split solver is the supported
    home for order 2; this class stays the order-1 reference.
    """

    time_end = param.Number(default=0.1, bounds=(0, None),
                            doc="Simulation end time")
    method = param.Selector(default="ars343",
                            objects=["ars232", "ars343"],
                            doc="IMEX-RK tableau name")
    compute_dt = param.Parameter(
        default=None,
        doc="Adaptive-timestep callable (see "
            "``zoomy_core.fvm.timestepping``); defaults to "
            "``adaptive(CFL=0.3)``")
    newton_tol = param.Number(default=1e-9, bounds=(0, None),
                              doc="Per-stage Newton residual tolerance")
    newton_maxit = param.Integer(default=30, bounds=(1, None),
                                 doc="Maximum Newton iterations per stage")
    h_index = param.Integer(default=0, bounds=(0, None),
                            doc="State index of the water-depth-like "
                                "variable used for max-eigenvalue / "
                                "CFL estimation (default 0 = ``h``)")
    nc_integration_order = param.Integer(default=3, bounds=(1, 8),
                            doc="Gauss-Legendre quadrature order for "
                                "the non-conservative path integral")
    reconstruction_order = param.Integer(default=1, bounds=(1, 2),
                            doc="Spatial reconstruction order.  "
                                "1 (default) = piecewise-constant "
                                "cell-centre states; 2 = LSQ-MUSCL "
                                "slope-limited face reconstruction with "
                                "the surface-elevation (η = h+b) trick "
                                "for well-balancing.\n\n"
                                "ORDER 2 IS NOT PRODUCTION-READY with "
                                "this monolithic implicit DAE solver — "
                                "see the class docstring 'Order-2 "
                                "status' note.  At order 2 the spatial "
                                "scheme is correct (well-balancing is "
                                "exact to machine precision via the "
                                "η = h+b SurfaceReconstruction + the "
                                "cell-interior non-conservative "
                                "integral, and the limiter is frozen "
                                "through the Newton iteration so f_I is "
                                "smooth), but the IMEX-ARK stage Newton "
                                "stalls: the monolithic stage Jacobian "
                                "is ill-conditioned (~1e7), so the "
                                "finite-difference Jacobian is "
                                "unreliable and Newton degrades to slow "
                                "linear convergence (see the note).  "
                                "Order 1 is the supported default; "
                                "order 2 is the explicit-predictor "
                                "path under the Chorin-split solver.")
    limiter = param.Selector(default="venkatakrishnan",
                            objects=["venkatakrishnan", "barth_jespersen",
                                     "minmod", "van_albada"],
                            doc="Slope limiter for the order-2 MUSCL "
                                "reconstruction.  ``van_albada`` is the "
                                "smooth, strictly-bounded variant (no "
                                "min(1,·) clamp).")
    well_balanced = param.Boolean(default=True,
                            doc="If True, apply Audusse-Bristeau-Klein "
                                "hydrostatic reconstruction (h_L*, h_R* "
                                "= max(0, h + b − b*)) before the "
                                "Rusanov flux + add the (P_raw − P_star) "
                                "pressure-jump terms to the NC "
                                "fluctuations.  Required for "
                                "lake-at-rest preservation over "
                                "varying bathymetry.  ``h`` and ``b`` "
                                "are auto-located in state / Qaux via "
                                ":class:`FieldHandle`-style lookup.")
    jacobian_mode = param.Selector(default="sparse_fd",
                            objects=["sparse_fd", "dense_fd"],
                            doc="Newton Jacobian assembly strategy.  "
                                "``sparse_fd`` uses graph colouring of "
                                "the cell-dependency stencil for "
                                "``O(n_colors × n_state)`` f_I evals "
                                "per Jacobian; ``dense_fd`` is the "
                                "naive ``O((nc·n_state)²)`` path (only "
                                "viable for tiny grids).")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.compute_dt is None:
            self.compute_dt = timestepping.adaptive(CFL=0.3)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_simulation(self, mesh, model, *, write_output=False):
        """Build operators + state once, including BC plumbing,
        output infrastructure, and the registry-driven Qaux walker."""
        # Normalise the input through the SAME front door as
        # ``Solver.setup_simulation``: Model / SystemModel / NSM → NSM, then
        # work on ``nsm.sm``.  This must be the NSM view and not a bare
        # ``_coerce_to_system_model`` result, because ``derive()`` REGISTERS
        # AUX (the KP-desingularised ``hinv``): a bare SystemModel is one aux
        # row shorter than the kernels ``NumpyRuntimeModel.from_system_model``
        # compiles below (it promotes to an NSM internally), so every Qaux
        # buffer allocated from ``len(sm.aux_state)`` was mis-sized and the
        # first ``mass_matrix`` call raised ``IndexError: index N is out of
        # bounds``.  One promotion, one aux width, one source of truth.
        nsm, _source_model = self._coerce_to_nsm(model)
        self.nsm = nsm
        model = nsm.sm
        mesh = ensure_lsq_mesh(mesh, model)
        self._sim_mesh = mesh
        self._sim_model = model

        sm = model
        self.sm = sm
        # Expose the SystemModel on the model too so
        # ``Solver._sm_from_solver_or_model`` and any user-provided
        # ``update_qaux`` override can pick it up generically.
        if not hasattr(model, "_chain_systemmodel"):
            model._chain_systemmodel = sm

        self.rt = NumpyRuntimeModel.from_system_model(sm)
        self.tab = ars232() if self.method == "ars232" else ars343()

        # Spatial reconstruction.  Order 2 → LSQ-MUSCL slope-limited
        # face states; order 1 → piecewise-constant cell-centre.
        from zoomy_core.fvm.reconstruction import (
            ConstantReconstructionV2, LSQMUSCLReconstruction,
            SurfaceReconstruction,
        )
        if self.reconstruction_order >= 2:
            self._reconstruct = LSQMUSCLReconstruction(
                mesh, mesh.dimension, limiter=self.limiter,
            )
        else:
            self._reconstruct = ConstantReconstructionV2(
                mesh, mesh.dimension,
            )

        nc = mesh.n_inner_cells
        n_state = sm.n_state
        dim = mesh.dimension
        self.nc = nc
        self.n_state = n_state
        self.n_aux_total = len(sm.aux_state)
        self.dim = dim

        # DAE partition: rows with all-zero mass-matrix row are
        # algebraic; rest are evolution.
        Q_probe = np.zeros(n_state)
        Q_probe[0] = 1.0
        for k in range(1, n_state):
            Q_probe[k] = 0.1 * (k + 1)
        Qaux_probe = np.zeros(self.n_aux_total)
        p_arr = self._parameters_array()
        M_probe = np.asarray(
            self.rt.mass_matrix(Q_probe, Qaux_probe, p_arr), dtype=float,
        )
        row_norms = np.linalg.norm(M_probe, axis=1)
        self.dyn_mask_cell = row_norms > 1e-12
        self.evol_idx = np.where(self.dyn_mask_cell)[0]
        self.alg_idx = np.where(~self.dyn_mask_cell)[0]
        self.dyn_mask = np.tile(self.dyn_mask_cell, nc)

        # Cache scalars used per-cell inside the Rusanov flux loop.
        if sm.parameters.contains("g"):
            self._g_cached = float(sm.parameter_values.g)
        else:
            self._g_cached = 9.81
        self._u_idx_cached = None
        for i, s in enumerate(sm.state):
            name = str(s)
            if name.startswith("U_") or name == "u":
                self._u_idx_cached = i
                break

        # Field handles for hydrostatic reconstruction (``h``, ``b``).
        # Searched in state then aux_state — works regardless of where
        # bathymetry lives (Q-state for legacy SWE, Qaux for chain DAE).
        self._h_loc = self._find_field_location("h")
        self._b_loc = self._find_field_location("b")

        # Well-balanced reconstruction wrapper.  When ``h`` is a state
        # variable and ``b`` is located (state or Qaux),
        # ``SurfaceReconstruction`` reconstructs the surface elevation
        # η = h + b (constant at lake-at-rest → the limiter sees a flat
        # field) and recovers h = η − b per side.  The h/b indices come
        # from the single field resolution above — the wrapper never
        # searches.
        self._surface_recon = None
        if (self.well_balanced and self._h_loc is not None
                and self._h_loc[0] == "state"
                and self._b_loc is not None):
            self._surface_recon = SurfaceReconstruction(
                self._reconstruct,
                h_index=self._h_loc[1],
                b_index=self._b_loc[1],
                b_in_state=(self._b_loc[0] == "state"),
            )
        # The reconstruction object that exposes ``_limited_grad`` after
        # each call — consumed by the cell-interior non-conservative
        # integral in ``_residual``.
        self._active_recon = (self._surface_recon
                              if self._surface_recon is not None
                              else self._reconstruct)
        # Frozen limiter coefficients — set per step / per projection.
        self._frozen_phi = None

        # Symbolic Riemann numerics → numpy runtime.  The face-loop
        # below consumes the *printed* PositiveNonconservativeRusanov
        # (hydrostatic reconstruction + Rusanov flux + NC path-integral
        # fluctuations + pressure-jump) — no hand-rolled Riemann logic.
        # ``scaled_q_indices=[]``: the chain state is primitive
        # (h, U_k, W_k, P_k), so HR does not depth-rescale any rows.
        from zoomy_core.fvm.riemann_solvers import (
            PositiveNonconservativeRusanov,
        )
        from zoomy_core.transformation.to_numpy import NumpyRuntimeSymbolic
        numerics = PositiveNonconservativeRusanov(
            sm, scaled_q_indices=[],
            integration_order=self.nc_integration_order,
        )
        # ``local_max_abs_eigenvalue`` is a real expression — for this
        # hand-built chain model (no closed-form spectrum) the symbolic
        # Numerics uses the Gelfand bound ``wave_speed_bound`` (an
        # upper bound) on the quasilinear matrix, so the flux lowers with no
        # opaque wave-speed kernel.
        numerics_module = dict(NumpyRuntimeModel.module)
        self.numerics_rt = NumpyRuntimeSymbolic(
            numerics, module=numerics_module,
        )

        # IC from model.  Same contract as ``Solver.initialize``: the IC
        # object writes the FULL (n_state, nc) / (n_aux, nc) array.  A failure
        # here is a real defect (wrong IC width, bad user function) and is
        # raised — swallowing it left Q/Qaux at zero, which for a prescribed
        # bathymetry/curvature aux is a silently different problem.
        Q = np.zeros((n_state, nc))
        if (hasattr(model, "initial_conditions")
                and model.initial_conditions is not None):
            Q = np.asarray(
                model.initial_conditions.apply(
                    mesh.cell_centers[:, :nc], Q,
                ),
                dtype=float,
            )

        # Aux IC.  Function-aux and derivative-aux rows are overwritten by
        # ``update_qaux`` right after, so the IC only has to be sane there.
        Qaux = np.zeros((self.n_aux_total, nc))
        if (hasattr(model, "aux_initial_conditions")
                and model.aux_initial_conditions is not None):
            Qaux = np.asarray(
                model.aux_initial_conditions.apply(
                    mesh.cell_centers[:, :nc], Qaux,
                ),
                dtype=float,
            )
        self._sim_Qaux = Qaux
        self._sim_parameters = p_arr

        # BC plumbing — uses Solver._setup_bc-style logic.
        self._setup_bc(mesh, model)

        # First Qaux fill: walk the registry once before the manifold
        # projection (so function-aux + derivative-aux entries have
        # sensible values at t = 0).
        #
        # ``self.rt``, NOT ``model``: ``Solver._apply_local_aux_formula`` needs a
        # LOWERED ``update_aux_variables`` and skips a non-callable one, and on a
        # SystemModel/NSM that slot is a symbolic ZArray.  The base
        # ``Solver.setup_simulation`` rebinds ``_sim_model`` to the runtime for
        # exactly this reason (``solver_numpy.py`` create_runtime → _sim_model);
        # this solver keeps the symbolic model there and the runtime in
        # ``self.rt``, so the local-aux leg has to be handed the runtime
        # explicitly.  Without it the KP ``hinv`` never refreshes: it stays at
        # its initial-condition value while ``h`` evolves, and every source term
        # carrying ``1/h`` is evaluated at the wrong depth.
        self._sim_Qaux = self.update_qaux(
            Q, self._sim_Qaux, Q, self._sim_Qaux,
            mesh, self.rt, p_arr, 0.0, 0.0,
        )

        # Jacobian colouring (sparse FD) — needed by
        # project_to_manifold below.
        if self.jacobian_mode == "sparse_fd":
            self._build_jacobian_coloring()

        # Project initial Q to the algebraic constraint manifold.
        Q = self.project_to_manifold(Q, time=0.0)

        # Mesh-CFL inradius (smallest min-inradius across face
        # neighbours).
        inner_face_mask = ((mesh.face_cells[0] < nc)
                           & (mesh.face_cells[1] < nc))
        if inner_face_mask.any():
            inradii = np.minimum(
                mesh.cell_inradius[mesh.face_cells[0, inner_face_mask]],
                mesh.cell_inradius[mesh.face_cells[1, inner_face_mask]],
            )
            self._sim_min_inradius = float(inradii.min())
        else:
            self._sim_min_inradius = float(mesh.cell_inradius[:nc].min())
        self._sim_max_eigenvalue = self._build_max_eigenvalue_callable()

        # Output setup (parallel to IMEXSolver.setup_simulation).
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory,
                f"{self.settings.output.filename}.h5",
            )
            io.init_output_directory(
                self.settings.output.directory,
                self.settings.output.clean_directory,
            )
            mesh.write_to_hdf5(output_hdf5_path)
            io.save_settings(self.settings)
            self._sim_save_fields = io.get_save_fields(
                output_hdf5_path, write_all=False,
            )
            self._output_hdf5_path = output_hdf5_path
        else:
            self._sim_save_fields = (
                lambda time, next_write_at, i_snap, Q, Qaux: i_snap
            )
            self._output_hdf5_path = None

        self._sim_Q = Q
        self._sim_time = 0.0
        return Q

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _setup_bc(self, mesh, model):
        """Cache per-boundary-face BC indices.

        The runtime exposes the indexed BC kernel as
        ``self.rt.boundary_conditions(bc_idx, time, position, distance,
        Q_cell, Qaux_cell, parameters, normal) → q_face``.  Per face the
        only state needed is its kernel branch `bc_idx`, matched to the
        face's tag BY NAME (`boundary_face_branches`); no Python
        BC-object list."""
        n_bf = mesh.n_boundary_faces
        self._n_bf = n_bf
        self._bf_cells = mesh.boundary_face_cells
        self._bf_fidx = mesh.boundary_face_face_indices
        self._bc_indices = boundary_face_branches(
            mesh, getattr(model, "_bc_source", None))
        if n_bf > 0:
            dim = mesh.dimension
            self._d_face = np.array([
                np.linalg.norm(
                    mesh.face_centers[self._bf_fidx[i], :dim]
                    - mesh.cell_centers[:dim, self._bf_cells[i]]
                ) for i in range(n_bf)
            ])
        else:
            self._d_face = np.array([])
        self._bf_face_to_idx = {
            int(self._bf_fidx[i]): i for i in range(n_bf)
        }

    # ------------------------------------------------------------------
    # Parameters helper
    # ------------------------------------------------------------------

    def _parameters_array(self):
        return np.array(
            [float(v) for v in self.sm.parameter_values.values()],
            dtype=float,
        )

    def _find_field_location(self, name):
        """Return `("state", idx)` if `name` is a state variable
        of `self.sm`, `("aux", idx)` if an aux variable, else
        `None`.  Numpy analogue of `Numerics.find_field`.
        """
        state_names = [str(s) for s in self.sm.state]
        if name in state_names:
            return ("state", state_names.index(name))
        aux_names = [str(s) for s in self.sm.aux_state]
        if name in aux_names:
            return ("aux", aux_names.index(name))
        return None

    # ------------------------------------------------------------------
    # Reconstruction support: boundary-face values + frozen limiter
    # ------------------------------------------------------------------

    def _boundary_face_values(self, Q, Qaux, time, p):
        """BC-provided cell-centre values at every boundary face,
        shape `(n_state, n_boundary_faces)` — feed the reconstruction
        for limiter bounds and the boundary `Q_R` placeholder."""
        mesh = self._sim_mesh
        dim = self.dim
        bc_fn = self.rt.boundary_conditions
        bf_Q = np.zeros((self.n_state, self._n_bf))
        for i_bf in range(self._n_bf):
            c_in = int(self._bf_cells[i_bf])
            f_idx = int(self._bf_fidx[i_bf])
            normal = mesh.face_normals[:dim, f_idx]
            position = mesh.face_centers[f_idx, :]
            bf_Q[:, i_bf] = bc_fn(
                self._bc_indices[i_bf], time, position,
                self._d_face[i_bf], Q[:, c_in], Qaux[:, c_in], p, normal,
            )
        return bf_Q

    def _compute_frozen_phi(self, Q, Qaux, time):
        """Slope-limiter coefficients computed once from a lagged state
        and held fixed through the implicit-Newton iteration.

        The LSQ gradient is linear in `Q` (smooth); the limiter φ is
        the non-smooth part — its kinks (neighbour min/max, sign
        branch, per-cell face-min) otherwise poison the
        finite-difference Jacobian.  Freezing φ — exactly as `Qaux`
        is lagged — makes `f_I` a smooth function of the Newton
        unknown.  Returns `None` at first order (no limiter)."""
        if self.reconstruction_order < 2:
            return None
        p = self._parameters_array()
        bf_Q = self._boundary_face_values(Q, Qaux, time, p)
        if self._surface_recon is not None:
            return self._surface_recon.compute_phi(Q, Qaux, bf_Q)
        return self._reconstruct.compute_phi(Q, bf_Q)

    # ------------------------------------------------------------------
    # Residual assembly (Rusanov flux + NC fluctuations + source)
    # ------------------------------------------------------------------

    def _residual(self, Q, Qaux, time):
        """Per-cell residual `R = S − ∇·F − ∇·P − B·∇Q` via Rusanov
        flux + linear-path-integral NC fluctuations.  Shape
        `(n_state, nc)`.

        Aux is passed in (does NOT change inside f_I): this is the
        lagged-aux Newton convention so the Jacobian doesn't have to
        track LSQ-derivative sensitivities.  Qaux is refreshed once
        per timestep by `update_qaux`.
        """
        p = self.sm_parameters_arr if hasattr(self, "sm_parameters_arr") \
            else self._parameters_array()
        nc = self.nc
        n_state = self.n_state
        dim = self.dim
        mesh = self._sim_mesh

        # Cell-centre source.
        Sv = np.empty((n_state, nc))
        for c in range(nc):
            Sv[:, c] = np.asarray(
                self.rt.source(Q[:, c], Qaux[:, c], p), dtype=float,
            ).ravel()

        # Boundary-face cell-centre BC values (feed the reconstruction
        # for limiter bounds + Q_R placeholder).
        n_bf = self._n_bf
        bf_Q = self._boundary_face_values(Q, Qaux, time, p)

        # Spatial reconstruction: per-face left / right state.  Order 1
        # → cell-centre constant; order 2 → LSQ-MUSCL slope-limited.
        # Q_L_all, Q_R_all are (n_state, n_faces).
        #
        # When well-balanced, ``self._surface_recon`` reconstructs the
        # surface elevation η = h + b (constant at lake-at-rest → the
        # limiter sees a flat field) together with b, recovers
        # h = η − b on each side, and returns the per-side reconstructed
        # bathymetry b_L_all / b_R_all (None otherwise) for the per-face
        # Qaux below.  The L/R bathymetry is kept per-side (no
        # averaging) — exact order-≥2 telescoping additionally needs the
        # cell-interior non-conservative integral.
        # Frozen limiter coefficients (computed once per step from the
        # lagged state) keep the reconstruction — and hence ``f_I`` —
        # smooth in the Newton unknown.
        frozen_phi = getattr(self, "_frozen_phi", None)
        if self._surface_recon is not None:
            Q_L_all, Q_R_all, b_L_all, b_R_all = self._surface_recon(
                Q, Qaux, bf_Q, phi=frozen_phi,
            )
        else:
            Q_L_all, Q_R_all = self._reconstruct(Q, bf_Q, phi=frozen_phi)
            b_L_all = b_R_all = None

        # Override Q_R at boundary faces with the BC applied to the
        # reconstructed interior face state.
        bc_fn = self.rt.boundary_conditions
        for i_bf in range(n_bf):
            c_in = int(self._bf_cells[i_bf])
            f_idx = int(self._bf_fidx[i_bf])
            normal = mesh.face_normals[:dim, f_idx]
            position = mesh.face_centers[f_idx, :]
            Q_R_all[:, f_idx] = bc_fn(
                self._bc_indices[i_bf], time, position,
                self._d_face[i_bf], Q_L_all[:, f_idx], Qaux[:, c_in],
                p, normal,
            )

        # Face loop: printed PositiveNonconservativeRusanov numerics.
        cv = mesh.cell_volumes
        R = np.copy(Sv)

        # Cell-interior non-conservative integral (path-conservative,
        # order ≥ 2):  ∫_cell B(Q_i(x))·∂_x Q_i(x) dx ≈ B(Q_i)·s_i
        # with the limited reconstruction slope s_i (the |cell| factor
        # cancels the per-unit-volume residual normalisation).  The
        # face fluctuations below carry only the inter-cell jump part;
        # this term is the intra-cell smooth part that the order-1
        # scheme omits exactly (zero slope ⇒ this loop is skipped).
        # ``B`` is the printed Model operator and the slope comes from
        # the reconstruction — a quadrature of existing building blocks.
        grad = getattr(self._active_recon, "_limited_grad", None)
        if grad is not None and grad.any():
            for c in range(nc):
                B_c = np.asarray(
                    self.rt.nonconservative_matrix(
                        Q[:, c], Qaux[:, c], p),
                    dtype=float,
                )
                if B_c.ndim == 2 and dim == 1:
                    B_c = B_c[..., None]
                for d_ax in range(dim):
                    R[:, c] -= B_c[:, :, d_ax] @ grad[:, d_ax, c]

        # Per-face flux + fluctuations from the printed symbolic
        # numerics: hydrostatic reconstruction, Rusanov conservative
        # flux, NC path-integral fluctuations and the pressure-jump all
        # live inside PositiveNonconservativeRusanov — the solver only
        # passes reconstructed face states and scatters into cells.
        nflux = self.numerics_rt.numerical_flux
        nfluct = self.numerics_rt.numerical_fluctuations
        for f in range(mesh.n_faces):
            cA = int(mesh.face_cells[0, f])
            cB = int(mesh.face_cells[1, f])
            normal = mesh.face_normals[:dim, f]
            area = float(mesh.face_volumes[f])

            # Reconstructed face states.  Qaux stays cell-centred except
            # for the bathymetry row, which carries the per-side
            # surface-reconstructed b_L / b_R when SurfaceReconstruction
            # is active (so the printed hydrostatic reconstruction sees
            # a consistent ``h_face + b_face``).
            Q_L = Q_L_all[:, f]
            Q_R = Q_R_all[:, f]
            if f in self._bf_face_to_idx:
                i_bf = self._bf_face_to_idx[f]
                c_in = int(self._bf_cells[i_bf])
                Qaux_L = Qaux[:, c_in]
                Qaux_R = Qaux[:, c_in]
                inner_is_A = (c_in == cA)
            else:
                Qaux_L = Qaux[:, cA]
                Qaux_R = Qaux[:, cB]
                inner_is_A = True
            if (b_L_all is not None and self._b_loc is not None
                    and self._b_loc[0] == "aux"):
                b_row = self._b_loc[1]
                Qaux_L = Qaux_L.copy()
                Qaux_R = Qaux_R.copy()
                Qaux_L[b_row] = b_L_all[f]
                Qaux_R[b_row] = b_R_all[f]

            # ``numerical_flux`` returns the flux alone — n_state rows.
            # This solver's dt comes from its own eigenvalue bound.
            num_flux = np.asarray(
                nflux(Q_L, Q_R, Qaux_L, Qaux_R, p, normal),
                dtype=float,
            ).ravel()
            fluct = np.asarray(
                nfluct(Q_L, Q_R, Qaux_L, Qaux_R, p, normal),
                dtype=float,
            )
            Dp = fluct[0].ravel()
            Dm = fluct[1].ravel()

            if f in self._bf_face_to_idx:
                c_in = cA if inner_is_A else cB
                sgn = +1.0 if inner_is_A else -1.0
                R[:, c_in] -= sgn * (num_flux + Dm) * area / cv[c_in]
            else:
                R[:, cA] -= (num_flux + Dm) * area / cv[cA]
                R[:, cB] += (num_flux - Dp) * area / cv[cB]

        return R

    def _build_max_eigenvalue_callable(self):
        """`(Q, Qaux, parameters) → max|λ|` callable for adaptive dt:
        `|U⃗₀| + √(g·h_max)` Saint-Venant upper bound, with all
        leading-order horizontal velocity components included.
        """
        sm = self.sm
        h_idx = self.h_index
        velocity_indices = [
            i for i, s in enumerate(sm.state)
            if str(s) in {"U_0", "V_0", "u", "v"}
        ]
        g_val = self._g_cached

        def max_eig(Q, Qaux, parameters):
            h = Q[h_idx, :]
            c = np.sqrt(g_val * np.maximum(h, 1e-6))
            if velocity_indices:
                u_sq = np.zeros_like(h)
                for i in velocity_indices:
                    u_sq = u_sq + Q[i, :] ** 2
                return np.sqrt(u_sq) + c
            return c

        return max_eig

    # ------------------------------------------------------------------
    # IMEX RHS interface
    # ------------------------------------------------------------------

    def f_E(self, t, Y):
        return np.zeros_like(Y)

    def _Y_to_Q(self, Y):
        return Y.reshape(self.nc, self.n_state).T

    def _Q_to_Y(self, Q):
        return Q.T.ravel()

    def f_I(self, t, Y):
        """Implicit RHS: per-cell `M_evol⁻¹·R_evol` on evolution
        rows; constraint residual on algebraic rows.

        Aux is held fixed at `self._sim_Qaux` (refreshed once per
        timestep) — the lagged-aux Newton convention.
        """
        Q = self._Y_to_Q(Y)
        Qaux = self._sim_Qaux
        p = self._parameters_array()
        R = self._residual(Q, Qaux, t)
        out = np.zeros_like(R)
        out[self.alg_idx, :] = R[self.alg_idx, :]
        for c in range(self.nc):
            M_c = np.asarray(
                self.rt.mass_matrix(Q[:, c], Qaux[:, c], p),
                dtype=float,
            )
            M_evol = M_c[np.ix_(self.evol_idx, self.evol_idx)]
            out[self.evol_idx, c] = np.linalg.solve(
                M_evol, R[self.evol_idx, c],
            )
        return self._Q_to_Y(out)

    # ------------------------------------------------------------------
    # Sparse FD Jacobian via graph colouring
    # ------------------------------------------------------------------

    def J_I(self, t, Y):
        if self.jacobian_mode == "dense_fd":
            return self._dense_J_I(t, Y)
        return self._sparse_J_I(t, Y)

    def _dense_J_I(self, t, Y):
        n = len(Y)
        f0 = self.f_I(t, Y)
        J = np.zeros((n, n))
        eps = 1e-7
        for k in range(n):
            step = eps * max(1.0, abs(Y[k]))
            Yp = Y.copy(); Yp[k] += step
            J[:, k] = (self.f_I(t, Yp) - f0) / step
        return J

    def _sparse_J_I(self, t, Y):
        n = len(Y)
        n_state = self.n_state
        f0 = self.f_I(t, Y)
        J = np.zeros((n, n))
        eps = 1e-7
        for color, cells in enumerate(self._color_cells):
            for i_state in range(n_state):
                Yp = Y.copy()
                steps = {}
                for c in cells:
                    k = c * n_state + i_state
                    step = eps * max(1.0, abs(Y[k]))
                    Yp[k] += step
                    steps[c] = step
                df = self.f_I(t, Yp) - f0
                for c in cells:
                    k = c * n_state + i_state
                    step = steps[c]
                    for c_t in self._seen_by[c]:
                        base_r = c_t * n_state
                        J[base_r:base_r + n_state, k] = (
                            df[base_r:base_r + n_state] / step
                        )
        return J

    def _build_jacobian_coloring(self):
        """Greedy graph-colouring of the cell-dependency stencil
        (LSQ neighbours + face neighbours).  Cells of the same colour
        share no dependency cone — they can be perturbed
        simultaneously in FD Jacobian assembly.

        The cone must span the RECONSTRUCTION stencil, not merely the face
        stencil.  At order 2 the face value on the far side of a neighbour
        carries that neighbour's own LSQ gradient, so the residual of cell
        `c` reaches `c +- 2`.  Colouring the one-ring cone regardless of
        order left, measured against a dense FD assembly on this system,
        14.4 percent of the true Jacobian OUTSIDE the assumed pattern where
        it was never written, and let two cells two apart share a colour so
        that the entries which were written are contaminated.  An inexact
        Jacobian is exactly what degraded the stage Newton from quadratic to
        linear convergence: order 2 cost 9.1x the order-1 time per step at
        identical dt, 760 residual evaluations for a single step.
        """
        mesh = self._sim_mesh
        nc = self.nc
        adj = [{c} for c in range(nc)]
        if (hasattr(mesh, "_lsq_neighbors")
                and mesh._lsq_neighbors is not None):
            for c in range(nc):
                for nbr in mesh._lsq_neighbors[c]:
                    nbr_i = int(nbr)
                    if 0 <= nbr_i < nc:
                        adj[c].add(nbr_i)
        for f in range(mesh.n_faces):
            cA = int(mesh.face_cells[0, f])
            cB = int(mesh.face_cells[1, f])
            if 0 <= cA < nc and 0 <= cB < nc:
                adj[cA].add(cB)
                adj[cB].add(cA)
        if int(self.reconstruction_order) >= 2:
            deps = [set().union(*(adj[d] for d in adj[c])) for c in range(nc)]
        else:
            deps = adj
        seen_by = [set() for _ in range(nc)]
        for c in range(nc):
            for d in deps[c]:
                seen_by[d].add(c)
        self._seen_by = [sorted(s) for s in seen_by]
        colors = [-1] * nc
        cells_with_color = []
        for c in range(nc):
            used = set()
            for c_prev in range(c):
                if colors[c_prev] < 0:
                    continue
                if seen_by[c] & seen_by[c_prev]:
                    used.add(colors[c_prev])
            k = 0
            while k in used:
                k += 1
            colors[c] = k
            while len(cells_with_color) <= k:
                cells_with_color.append([])
            cells_with_color[k].append(c)
        self._colors = colors
        self._color_cells = cells_with_color

    # ------------------------------------------------------------------
    # Manifold projection
    # ------------------------------------------------------------------

    def project_to_manifold(self, Q, *, time=0.0, tol=1e-10, maxit=20):
        """Newton-project `Q` onto the algebraic constraint manifold
        `f_I[alg] = 0` by adjusting algebraic state entries.

        Aborts (returns the original `Q`) if the Newton residual
        does not decrease or if any iterate becomes non-finite — the
        caller can then proceed with the un-projected IC and let the
        IMEX-ARK Newton (which uses a more conservative update) do
        the projection.
        """
        # Freeze the limiter for the projection Newton too (same reason
        # as in ``step``: a smooth ``f_I`` keeps the FD Jacobian sane).
        self._frozen_phi = self._compute_frozen_phi(
            Q, self._sim_Qaux, time)
        Y0 = self._Q_to_Y(Q)
        Y = Y0.copy()
        nc = self.nc
        n_state = self.n_state
        alg_idx = self.alg_idx
        if len(alg_idx) == 0:
            return Q
        alg_rows = np.concatenate([
            c * n_state + alg_idx for c in range(nc)
        ])
        prev_norm = np.inf
        for _ in range(maxit):
            f = self.f_I(time, Y)
            if not np.all(np.isfinite(f)):
                return Q
            f_alg = f[alg_rows]
            norm = float(np.linalg.norm(f_alg))
            if norm < tol:
                return self._Y_to_Q(Y)
            # Divergence guard: residual must drop monotonically once
            # within an order of magnitude of the initial norm; let it
            # increase early but bail if it explodes.
            if norm > 1e3 * prev_norm and prev_norm != np.inf:
                return Q
            prev_norm = norm
            J = self.J_I(time, Y)
            sub = J[np.ix_(alg_rows, alg_rows)]
            try:
                delta = np.linalg.solve(sub, -f_alg)
            except np.linalg.LinAlgError:
                return Q
            if not np.all(np.isfinite(delta)):
                return Q
            Y[alg_rows] += delta
        return self._Y_to_Q(Y)

    # ------------------------------------------------------------------
    # Step / run loop
    # ------------------------------------------------------------------

    def step(self, dt):
        """One IMEX-ARK step on the stored state.

        Convention: refresh `self._sim_Qaux` via `update_qaux`
        once per step (lagged through the Newton iteration), apply
        `update_q` to the new state after.  The slope-limiter
        coefficients are frozen the same way — computed once from the
        lagged state so `f_I` stays smooth in the Newton unknown.
        """
        Q_old = self._sim_Q.copy()
        Qaux_old = self._sim_Qaux.copy()
        self._frozen_phi = self._compute_frozen_phi(
            Q_old, Qaux_old, self._sim_time)
        Y = self._Q_to_Y(Q_old)
        Y_new = imex_ark_step(
            self._sim_time, Y, dt, self.tab,
            self.f_E, self.f_I, self.J_I, self.dyn_mask,
            newton_tol=self.newton_tol, newton_maxit=self.newton_maxit,
        )
        Q_new = self._Y_to_Q(Y_new)
        # update_q + update_qaux post-step.  Some chain Models don't
        # implement the per-cell ``update_variables(Q_cell, aux, p)``
        # signature that SWE-style models expose, so we guard with
        # try/except — the default for those models is "no clamp".
        try:
            Q_new = self.update_q(Q_new, Qaux_old, self._sim_mesh,
                                  self._sim_model, self._sim_parameters, dt)
        except TypeError:
            pass
        # ``self.rt`` (see the note in setup_simulation): the local-aux leg needs
        # the LOWERED ``update_aux_variables``, so the KP ``hinv`` tracks ``h``.
        Qaux_new = self.update_qaux(
            Q_new, Qaux_old, Q_old, Qaux_old,
            self._sim_mesh, self.rt, self._sim_parameters,
            self._sim_time + dt, dt,
        )
        self._sim_Q = Q_new
        self._sim_Qaux = Qaux_new

    def run_simulation(self):
        t_start = _time.time()
        time_now = 0.0
        next_write_at = 0.0
        i_snapshot = 0.0
        n_snap = max(self.settings.output.snapshots - 1, 1)
        dt_snapshot = self.time_end / n_snap
        iteration = 0
        parameters = self._sim_parameters

        # Initial snapshot.
        i_snapshot = self._sim_save_fields(
            time_now, next_write_at, i_snapshot,
            self._sim_Q, self._sim_Qaux,
        )
        next_write_at += dt_snapshot

        while time_now < self.time_end:
            dt = self.compute_dt(
                self._sim_Q, self._sim_Qaux, parameters,
                self._sim_min_inradius, self._sim_max_eigenvalue,
            )
            dt = float(min(dt, self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0.0:
                break

            self._sim_time = time_now
            self.step(dt)
            time_now += dt
            iteration += 1

            if time_now >= next_write_at:
                i_snapshot = self._sim_save_fields(
                    time_now, next_write_at, i_snapshot,
                    self._sim_Q, self._sim_Qaux,
                )
                next_write_at += dt_snapshot

            if iteration % 10 == 0:
                logger.info(
                    f"[DAESolver] iter {iteration}  t={time_now:.6f}  "
                    f"dt={dt:.6f}  next-snap={next_write_at:.4f}"
                )

        self._sim_time = time_now
        elapsed = _time.time() - t_start
        logger.info(
            f"[DAESolver] finished in {elapsed:.2f}s "
            f"({iteration} steps)"
        )
        return self._sim_Q, self._sim_Qaux

    def solve(self, mesh, model, *, write_output=True):
        self.setup_simulation(mesh, model, write_output=write_output)
        return self.run_simulation()

    # ------------------------------------------------------------------
    # VTK / Paraview post-processing
    # ------------------------------------------------------------------

    def export_vtk(self, *, field_names=None, aux_field_names=None,
                   filename="dae_output", skip_aux=False):
        """Post-process HDF5 snapshots to a VTK time series for
        Paraview.  Requires `write_output=True` at setup time."""
        if self._output_hdf5_path is None:
            raise RuntimeError(
                "export_vtk requires write_output=True at setup time"
            )
        if field_names is None:
            field_names = [str(s) for s in self.sm.state]
        if aux_field_names is None:
            aux_field_names = [str(s) for s in self.sm.aux_state]
        try:
            from zoomy_prepost import hdf5_to_vtk
        except ImportError as e:
            raise ImportError(
                "export_vtk needs zoomy_prepost (the h5 -> vtk writer). It is "
                "a sibling install in every zoomy environment — install it "
                "(pip install -e library/zoomy_prepost)."
            ) from e
        from zoomy_core.misc import misc
        abs_h5 = os.path.join(misc.get_main_directory(), self._output_hdf5_path)
        hdf5_to_vtk(
            abs_h5,
            os.path.dirname(abs_h5),
            basename=filename,
            include_aux=not skip_aux,
            field_names=field_names,
            aux_field_names=aux_field_names,
        )
