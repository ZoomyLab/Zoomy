"""Pressure-splitting solver: IMEX + Poisson constraint (numpy backend).

Extends `IMEXSolver` with a pressure Poisson correction step (Chorin
projection). Uses face-based divergence and gradient, and GMRES for the
Poisson solve.

Solver hierarchy:
    SplittingSolver(IMEXSolver)
      -> step(dt):
            [explicit flux] -> viscous diffusion -> pressure_correction -> update_state
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from zoomy_core.fvm.solver_imex_numpy import IMEXSolver, IMEXStats
from zoomy_core.fvm.solver_numpy import _build_free_surface_numerics, _EMPTY_AUX
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.mesh.lsq_reconstruction import find_derivative_indices
import zoomy_core.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.misc.io as io
from zoomy_core.misc.logger_config import logger


import param


class SplittingSolver(IMEXSolver):
    """IMEX solver with pressure Poisson correction (Chorin projection).

    The model state Q holds velocity components.  Pressure is a separate
    field solved at each timestep via the incompressibility constraint.

    Subclasses for free-surface flow override `_velocity_indices` to
    point at the actual velocity slots (e.g. `[2, 3, ...]` past the
    bathymetry/depth pair) — by default we assume INS-style layout
    where the velocity components are at indices `0..dim-1`.
    """

    pressure_gmres_tol = param.Number(default=1e-6, doc="GMRES tolerance for Poisson")
    pressure_gmres_maxiter = param.Integer(default=200, doc="Max GMRES iterations for Poisson")
    viscosity = param.Number(default=0.01, doc="Kinematic viscosity v")

    def _velocity_indices(self, model):
        """State indices subject to viscous diffusion + pressure projection.

        Default (INS-style state `[u, v[, w]]`): `[0, ..., dim-1]`.
        Subclasses override for layouts like `[b, h, hu, hv, ...]`.
        """
        return list(range(model.dimension))

    def setup_simulation(self, mesh, model, write_output=False):
        """Build all operators including Poisson infrastructure.

        `model` may be a `Model`, a `SystemModel`, or an
        already-built `NumericalSystemModel`.  It is coerced to an NSM
        (via `HyperbolicSolver._coerce_to_nsm`) and stored on
        `self.nsm` — the base `get_flux_operator` /
        `get_compute_max_abs_eigenvalue` read `self.nsm.reconstruction` and
        `self.nsm.eigenvalue_eps`, so WITHOUT this assignment the solver
        raised ``AttributeError: 'SplittingSolver' object has no attribute
        'nsm'`` for every model (REQ-148).
        """
        t0 = time.time()
        # Coerce Model / SystemModel / NSM to an NSM and seed ``self.nsm`` — the
        # single wiring the class was missing (mirrors HyperbolicSolver).
        nsm, source_model = self._coerce_to_nsm(model)
        self.nsm = nsm
        # REQ-190: fill the adaptive dt cap from the standard NSM ``dt_max``
        # (explicit cap wins; wave-free domains step at dt_max, not a floor).
        timestepping.apply_default_dt_max(self.compute_dt, nsm.dt_max)
        mesh = ensure_lsq_mesh(mesh, nsm)
        # Periodic-BC topology resolution needs the ``BoundaryConditions``
        # object — the Model carries it directly; a declarative SystemModel
        # retains it as ``_bc_source``.
        bcs_obj = (source_model.boundary_conditions
                   if source_model is not None
                   else getattr(nsm.sm, "_bc_source", None))
        if hasattr(mesh, "resolve_periodic_bcs") and bcs_obj is not None:
            mesh.resolve_periodic_bcs(bcs_obj)
        # The self-contained SystemModel every downstream method consumes.
        model = nsm.sm
        self.sm = model
        dim = model.dimension
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model_rt = self.create_runtime(Q, Qaux, mesh, model)

        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells
        n_vars = Q.shape[0]

        # Store simulation state
        self._sim_mesh = mesh
        self._sim_model = model_rt
        self._sim_parameters = parameters
        self._sim_Q = Q
        self._sim_Qaux = Qaux
        self._sim_time = 0.0
        self._sim_dim = dim
        self._sim_nc = nc
        self._sim_n_cells = n_cells
        self._sim_n_vars = n_vars
        self._sim_nu = self.viscosity

        # Pressure field
        self._sim_pressure = np.zeros(n_cells, dtype=float)

        # Precompute face connectivity
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        self._sim_iA = iA
        self._sim_iB = iB
        self._sim_face_normals = mesh.face_normals[:dim, :]
        self._sim_face_volumes = mesh.face_volumes
        self._sim_cell_vol = mesh.cell_volumes

        # Build sparse Laplacian matrix for Poisson
        self._sim_poisson_solver = self._build_poisson_solver(mesh, dim, nc, n_cells, iA, iB)

        # Build operators from parent hierarchy (ghost-cell-free).
        # get_flux_operator populates self._bc_objects, _bf_cells,
        # _bf_fidx, _d_face, _n_bf — used by the viscous and pressure
        # substeps to evaluate BCs inline at boundary faces.
        self._sim_compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model_rt)
        self._sim_flux_operator = self.get_flux_operator(mesh, model_rt)

        # Precompute interior face index set (for split substep loops)
        bf_face_set = set(mesh.boundary_face_face_indices)
        self._sim_interior_faces = np.array(
            [f for f in range(mesh.n_faces) if f not in bf_face_set], dtype=int,
        )

        # Velocity component slots in the state vector (subclass-overridable)
        self._sim_vel_idx = self._velocity_indices(model)

        # Per-face inradius for the LOCAL CFL (shared base helper — no global
        # ``.min()`` collapse: each face pairs its own size with its own λ).
        self._sim_face_inradius = self._compute_face_inradius(mesh)

        # Output setup
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            self._sim_save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            self._sim_save_fields = lambda time, time_stamp, i_snapshot, Q, Qaux: i_snapshot

    def _build_poisson_solver(self, mesh, dim, nc, n_cells, iA, iB):
        """Build sparse Laplacian and return a Poisson solve function."""
        face_volumes = mesh.face_volumes
        cell_vol = mesh.cell_volumes

        L = lil_matrix((nc, nc), dtype=float)
        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            dist = np.linalg.norm(
                mesh.cell_centers[:dim, b] - mesh.cell_centers[:dim, a]
            )
            if dist < 1e-14:
                continue
            coeff = face_volumes[f] / dist
            if a < nc and b < nc:
                L[a, b] += coeff / cell_vol[a]
                L[a, a] -= coeff / cell_vol[a]
                L[b, a] += coeff / cell_vol[b]
                L[b, b] -= coeff / cell_vol[b]
            elif a < nc:
                pass  # b is ghost: Neumann p[b]=p[a]
            elif b < nc:
                pass

        # Pin one pressure DOF to remove null space
        L[0, :] = 0
        L[0, 0] = 1.0
        L_csr = csr_matrix(L)

        tol = self.pressure_gmres_tol
        maxiter = self.pressure_gmres_maxiter

        def solve_poisson(rhs):
            rhs_inner = rhs[:nc].copy()
            rhs_inner[0] = 0.0  # pinned DOF
            result = np.zeros(n_cells)

            def matvec(x):
                return L_csr @ x

            A_op = LinearOperator((nc, nc), matvec=matvec, dtype=float)
            sol, info = gmres(A_op, rhs_inner, atol=0.0,
                              rtol=tol, maxiter=maxiter)
            result[:nc] = sol
            # Ghost cells: Neumann
            for i_bf in range(mesh.n_boundary_faces):
                ghost = mesh.boundary_face_ghosts[i_bf]
                inner = mesh.boundary_face_cells[i_bf]
                result[ghost] = result[inner]
            return result, info

        return solve_poisson

    def step(self, dt):
        """One splitting timestep: flux -> viscous diffusion -> pressure correction.

        Ghost-cell-free: BCs are evaluated inline inside each substep at
        boundary faces (via `self._bc_objects[i_bf].face_value(...)`).
        No separate ghost-cell fill between substeps.
        """
        Q = self._sim_Q
        Qaux = self._sim_Qaux
        parameters = self._sim_parameters
        time_now = self._sim_time

        # Step 1: Explicit convective flux (BCs inline in flux_operator).
        # RK1 step matches HyperbolicSolver.step's inline pattern; the new
        # flux_operator takes (dt, time, Q, Qaux, parameters, dQ).
        dQ = np.zeros_like(Q)
        dQ = self._sim_flux_operator(dt, time_now, Q, Qaux, parameters, dQ)
        Q_star = Q + dt * dQ

        # Step 2: Explicit viscous diffusion (BC eval inline at boundary faces)
        Q_star = self._apply_viscous_diffusion(
            Q_star, Qaux, parameters, time_now, dt,
        )

        # Step 3: Pressure correction (Poisson solve + velocity update;
        # u-divergence uses BC at boundary faces, p has Neumann).
        Qnew, p = self._pressure_correction(
            Q_star, Qaux, parameters, time_now, dt,
        )

        # Unified post-step pair (update_q THEN update_qaux), mirroring
        # HyperbolicSolver/IMEX: previously Splitting committed only _sim_Q and
        # _sim_pressure, leaving Qaux FROZEN at the IC — wrong for any
        # free-surface model (FSFSplittingSolver) whose flux/source read aux
        # (hinv / derivative-aux).  ``self.update_qaux`` resolves through the
        # composing DerivativeAwareSolverMixin to the canonical aux_registry
        # walk (local update_aux_variables prefix + LSQ-gradient rows).
        mesh, model = self._sim_mesh, self._sim_model
        Qnew = self.update_q(Qnew, Qaux, mesh, model, parameters, dt)
        Qauxnew = self.update_qaux(
            Qnew, Qaux, Q, Qaux, mesh, model, parameters, time_now + dt, dt)

        # Commit new state
        self._sim_Q = Qnew
        self._sim_Qaux = Qauxnew
        self._sim_pressure = p

    def _apply_viscous_diffusion(self, Q_star, Qaux, parameters, time_now, dt):
        """Explicit viscous diffusion for velocity components, ghost-cell-free.

        Interior faces use both adjacent inner cells; boundary faces evaluate
        the BC inline (`bc.face_value(...)`) to obtain `u_face`, then form
        the gradient against the inner cell center over distance `d_face`.
        """
        dim = self._sim_dim
        n_vars = self._sim_n_vars
        nc = self._sim_nc
        nu = self._sim_nu
        iA = self._sim_iA
        iB = self._sim_iB
        mesh = self._sim_mesh
        face_volumes = self._sim_face_volumes
        cell_vol = self._sim_cell_vol
        face_normals = mesh.face_normals[:dim, :]
        interior_faces = self._sim_interior_faces
        bc_fn = self._bc_fn
        bc_indices = self._bc_indices
        face_centers = mesh.face_centers
        bf_cells = self._bf_cells
        bf_fidx = self._bf_fidx
        d_face = self._d_face
        n_bf = self._n_bf
        vel_idx = self._sim_vel_idx
        has_aux = Qaux.shape[0] > 0

        for k in vel_idx:
            u_d = Q_star[k, :]
            lap = np.zeros(nc)

            # Interior faces — both cells valid
            for f in interior_faces:
                a, b = iA[f], iB[f]
                diff = u_d[b] - u_d[a]
                dist = np.linalg.norm(
                    mesh.cell_centers[:dim, b] - mesh.cell_centers[:dim, a]
                )
                if dist > 1e-14:
                    lap_contrib = diff / dist * face_volumes[f]
                    lap[a] += lap_contrib / cell_vol[a]
                    lap[b] -= lap_contrib / cell_vol[b]

            # Boundary faces — evaluate BC inline at each face via the
            # indexed kernel.
            for i_bf in range(n_bf):
                f = bf_fidx[i_bf]
                inner = bf_cells[i_bf]
                q_inner = Q_star[:, inner]
                qaux_inner = Qaux[:, inner] if has_aux else _EMPTY_AUX
                normal = face_normals[:, f]
                position = face_centers[f, :]
                q_face = bc_fn(
                    bc_indices[i_bf], time_now, position, d_face[i_bf],
                    q_inner, qaux_inner, parameters, normal,
                )
                u_face = q_face[k]
                diff = u_face - u_d[inner]
                if d_face[i_bf] > 1e-14:
                    lap_contrib = diff / d_face[i_bf] * face_volumes[f]
                    lap[inner] += lap_contrib / cell_vol[inner]

            Q_star[k, :] += dt * nu * lap
        return Q_star

    def _pressure_correction(self, Q_star, Qaux, parameters, time_now, dt):
        """Compute divergence, solve Poisson, correct velocity. Ghost-cell-free.

        Divergence of `u*`: interior faces use the average of the two cell
        values; boundary faces evaluate the BC inline to obtain `u_face`.
        Pressure has Neumann BCs at boundaries (`p_face = p_inner`) — this
        matches the Poisson Laplacian assembly in `_build_poisson_solver`,
        which skips boundary-face entries.
        """
        dim = self._sim_dim
        nc = self._sim_nc
        iA = self._sim_iA
        iB = self._sim_iB
        face_normals = self._sim_face_normals
        face_volumes = self._sim_face_volumes
        cell_vol = self._sim_cell_vol
        interior_faces = self._sim_interior_faces
        bc_fn = self._bc_fn
        bc_indices = self._bc_indices
        mesh = self._sim_mesh
        face_centers = mesh.face_centers
        bf_cells = self._bf_cells
        bf_fidx = self._bf_fidx
        d_face = self._d_face
        n_bf = self._n_bf
        vel_idx = self._sim_vel_idx
        has_aux = Qaux.shape[0] > 0

        # ── Compute divergence of u* ─────────────────────────────────
        # The velocity components live at state slots ``vel_idx`` —
        # ``[0..dim-1]`` for INS, ``[2..2+dim-1]`` for free-surface.
        div_u = np.zeros(nc)
        for f in interior_faces:
            a, b = iA[f], iB[f]
            n = face_normals[:, f]
            u_face = 0.5 * (Q_star[vel_idx, a] + Q_star[vel_idx, b])
            flux_div = float(np.dot(u_face, n))
            div_u[a] += flux_div * face_volumes[f] / cell_vol[a]
            div_u[b] -= flux_div * face_volumes[f] / cell_vol[b]
        for i_bf in range(n_bf):
            f = bf_fidx[i_bf]
            inner = bf_cells[i_bf]
            q_inner = Q_star[:, inner]
            qaux_inner = Qaux[:, inner] if has_aux else _EMPTY_AUX
            n = face_normals[:, f]
            position = face_centers[f, :]
            q_face = bc_fn(
                bc_indices[i_bf], time_now, position, d_face[i_bf],
                q_inner, qaux_inner, parameters, n,
            )
            u_face = np.asarray(q_face)[vel_idx]
            flux_div = float(np.dot(u_face, n))
            div_u[inner] += flux_div * face_volumes[f] / cell_vol[inner]

        # ── Poisson solve ────────────────────────────────────────────
        rhs = div_u / dt
        p, _ = self._sim_poisson_solver(rhs)

        # ── grad(p): Neumann at boundaries (p_face = p_inner) ────────
        grad_p = np.zeros((dim, nc))
        for f in interior_faces:
            a, b = iA[f], iB[f]
            n = face_normals[:, f]
            p_face = 0.5 * (p[a] + p[b])
            for d in range(dim):
                contrib = p_face * n[d] * face_volumes[f]
                grad_p[d, a] += contrib / cell_vol[a]
                grad_p[d, b] -= contrib / cell_vol[b]
        for i_bf in range(n_bf):
            f = bf_fidx[i_bf]
            inner = bf_cells[i_bf]
            n = face_normals[:, f]
            p_face = p[inner]  # Neumann
            for d in range(dim):
                contrib = p_face * n[d] * face_volumes[f]
                grad_p[d, inner] += contrib / cell_vol[inner]

        Qnew = Q_star.copy()
        for d_local, k in enumerate(vel_idx):
            Qnew[k, :] = Q_star[k, :] - dt * grad_p[d_local]
        return Qnew, p

    def run_simulation(self):
        """Time loop for splitting solver."""
        time_now = 0.0
        iteration = 0
        nc = self._sim_nc

        while time_now < self.time_end:
            dt = self.compute_dt(
                self._sim_Q, self._sim_Qaux, self._sim_parameters,
                self._sim_face_inradius, self._sim_compute_max_abs_eigenvalue,
            )
            dt = min(float(dt), float(self.time_end - time_now))
            if not np.isfinite(dt) or dt <= 0:
                break

            self._sim_time = time_now
            self.step(dt)

            time_now += dt
            iteration += 1

            if iteration % 10 == 0:
                div_max = 0.0  # logged for diagnostics
                logger.info(
                    f"split it={iteration}, t={time_now:.4f}, dt={dt:.2e}"
                )

        self._sim_time = time_now
        logger.info(f"Splitting solver finished in {iteration} iterations, t={time_now:.4f}")
        return self._sim_Q, self._sim_pressure

    def solve(self, mesh, model, write_output=False):
        """Convenience: setup_simulation + run_simulation."""
        self.setup_simulation(mesh, model, write_output=write_output)
        return self.run_simulation()


class FSFSplittingSolver(SplittingSolver):
    """Splitting solver for free-surface flows (SWE, SME, VAM).

    Combines:
    - Positive Rusanov for explicit flux (requires h/b)
    - Implicit source stepping
    - Pressure Poisson correction

    Requires model variables 'b' and 'h' at state indices 0 and 1.
    Velocity components (subject to viscous diffusion + Chorin pressure
    projection) start at index 2 and span `dim` slots.
    """

    def _build_numerics(self, symbolic_model):
        return _build_free_surface_numerics(symbolic_model)

    def _velocity_indices(self, model):
        # Free-surface state layout: [b, h, hu_0, hv_0, ..., (extra moments)]
        # Chorin projection acts on the horizontal momentum components,
        # which sit at indices 2 .. 2+dim-1.
        return list(range(2, 2 + model.dimension))
