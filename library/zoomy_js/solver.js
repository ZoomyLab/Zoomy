/**
 * zoomy_js/solver.js — Generic 2D structured-grid finite-volume solver.
 *
 * Mirrors zoomy_core.fvm.solver_numpy.HyperbolicSolver.
 *
 * The solver owns the state arrays and the time-stepping loop. All
 * physics is delegated to an injected Model2D (see model.js) whose
 * kernels are generated from a zoomy_core SystemModel — the solver
 * itself is entirely physics-agnostic.
 *
 * Scheme: structured minmod-MUSCL reconstruction → injected numerical
 * (Riemann) flux → Heun / SSPRK2 time integration. This is the
 * "hand-written model-agnostic glue" of the Zoomy codegen pipeline:
 * the building blocks (flux, numericalFlux, boundaryConditions) are
 * generated; the assembly here is written once and reused for any
 * hyperbolic model.
 *
 * Boundary handling is *dispatch only*: the solver decides which
 * boundary tag applies to which ghost cell / wall face, and the
 * generated `boundaryConditions` kernel computes the actual state.
 */

import { cellMaxWaveSpeed } from "./model.js";

/** Minmod limiter — the TVD slope-limiting building block. */
function minmod(a, b) {
  if (a * b <= 0) return 0;
  return Math.abs(a) < Math.abs(b) ? a : b;
}

/**
 * Explicit 2D structured-grid FVM solver (MUSCL + Riemann flux + Heun).
 *
 * Grid: (nx+2) × (ny+2) cells, one ghost layer per side. State is
 * stored struct-of-arrays: `nVars` separate Float64Arrays of length NN.
 */
export class HyperbolicSolver2D {
  /**
   * @param {object} opts
   * @param {number}   opts.nx     interior cells in x
   * @param {number}   opts.ny     interior cells in y
   * @param {number}   opts.dx     cell spacing (uniform)
   * @param {Model2D}  opts.model  physics model (generated kernels)
   * @param {object}   opts.bc     boundary dispatch — see below
   * @param {number}   [opts.cfl=0.4]    CFL number (≤0.5 for MUSCL+Heun TVD)
   * @param {number}   [opts.dtMin=1e-4]
   * @param {number}   [opts.dtMax=0.5]
   *
   * `opts.bc` describes which boundary tag applies where:
   *   {
   *     wall:   <bcIdx>,              // internal obstacle cells
   *     edges:  { left, right, bottom, top }
   *   }
   * Each edge entry is either a bcIdx (number) or a callback
   * `(coord, time) => bcIdx` so an edge can mix tags (e.g. an inflow
   * opening carved into an otherwise-wall edge).
   */
  constructor({ nx, ny, dx, model, bc, cfl = 0.4, dtMin = 1e-4, dtMax = 0.5 }) {
    this.nx = nx;
    this.ny = ny;
    this.Nx = nx + 2;
    this.Ny = ny + 2;
    this.NN = this.Nx * this.Ny;
    this.dx = dx;
    this.model = model;
    this.nVars = model.nVars;
    this.nAux = model.nAux;
    this.bc = bc;
    this.cfl = cfl;
    this.dtMin = dtMin;
    this.dtMax = dtMax;

    this.Q = null;     // current state: nVars × Float64Array(NN)
    this.Qaux = null;  // static aux state: nAux × Float64Array(NN)
    this.wall = null;  // Uint8Array(NN): 1 = obstacle cell
  }

  /** Allocate state + scratch. Initial conditions are set by the caller
   *  by writing into `this.Q` (and `this.Qaux`) afterwards. */
  initialize() {
    const { nVars, nAux, NN } = this;
    const mk = (n) => Array.from({ length: n }, () => new Float64Array(NN));
    this.Q = mk(nVars);
    this.Qaux = mk(nAux);
    this.wall = new Uint8Array(NN);

    // Heun stage buffers + RHS + minmod slopes.
    this._Q1 = mk(nVars);
    this._Q2 = mk(nVars);
    this._R = mk(nVars);
    this._sx = mk(nVars);
    this._sy = mk(nVars);
    this._fx = mk(nVars);
    this._fy = mk(nVars);

    // Per-face reconstruction scratch (reused every face).
    this._qL = new Float64Array(nVars);
    this._qR = new Float64Array(nVars);
    this._auxL = new Float64Array(nAux);
    this._auxR = new Float64Array(nAux);
    this._X = new Float64Array(3);
    this._n = new Float64Array(this.model.dimension || 2);
  }

  /** Flat index from (j, i). */
  idx(j, i) { return j * this.Nx + i; }

  // ── Boundary dispatch ──────────────────────────────────────────────

  /** Resolve an edge spec (bcIdx or callback) to a concrete tag index. */
  _resolveEdge(spec, coord, time) {
    return typeof spec === "function" ? spec(coord, time) : spec;
  }

  /**
   * Fill the one-cell ghost frame of `Qarr` from the generated
   * `boundaryConditions` kernel. Order mirrors the reference worker:
   * left/right columns first, then bottom/top rows (so corners take
   * the row value).
   */
  applyBoundaryConditions(Qarr, time) {
    const { Nx, Ny, nx, ny, nVars, nAux, dx, model, bc } = this;
    const { params } = model;
    const X = this._X, n = this._n;
    const qIn = this._qL, auxIn = this._auxL;

    const fill = (cGhost, cIn, bcIdx, nx0, ny0) => {
      for (let v = 0; v < nVars; v++) qIn[v] = Qarr[v][cIn];
      for (let a = 0; a < nAux; a++) auxIn[a] = this.Qaux[a][cIn];
      n.fill(0);
      n[0] = nx0;
      if (n.length > 1) n[1] = ny0;
      X[0] = (cIn % Nx) * dx;
      X[1] = ((cIn / Nx) | 0) * dx;
      const ghost = model.boundaryConditions(
        bcIdx, time, X, dx, qIn, auxIn, params, n
      );
      for (let v = 0; v < nVars; v++) Qarr[v][cGhost] = ghost[v];
    };

    for (let j = 1; j <= ny; j++) {
      fill(j * Nx + 0, j * Nx + 1,
           this._resolveEdge(bc.edges.left, j, time), -1, 0);
      fill(j * Nx + (Nx - 1), j * Nx + (Nx - 2),
           this._resolveEdge(bc.edges.right, j, time), 1, 0);
    }
    for (let i = 0; i < Nx; i++) {
      const ci = Math.min(Math.max(i, 1), nx);  // clamp corners to interior col
      fill(0 * Nx + i, 1 * Nx + ci,
           this._resolveEdge(bc.edges.bottom, i, time), 0, -1);
      fill((Ny - 1) * Nx + i, (Ny - 2) * Nx + ci,
           this._resolveEdge(bc.edges.top, i, time), 0, 1);
    }
  }

  // ── Reconstruction ─────────────────────────────────────────────────

  /** Per-cell minmod slopes in x and y; zeroed adjacent to walls so the
   *  scheme drops to first order at obstacle boundaries. */
  _computeSlopes(Qarr) {
    const { Nx, Ny, nVars, wall, _sx: sx, _sy: sy } = this;
    for (let j = 1; j < Ny - 1; j++) {
      for (let i = 1; i < Nx - 1; i++) {
        const c = j * Nx + i;
        if (wall[c]) {
          for (let v = 0; v < nVars; v++) { sx[v][c] = 0; sy[v][c] = 0; }
          continue;
        }
        const e = c + 1, w = c - 1, nn = c + Nx, ss = c - Nx;
        const xOk = !wall[e] && !wall[w];
        const yOk = !wall[nn] && !wall[ss];
        for (let v = 0; v < nVars; v++) {
          const q = Qarr[v];
          sx[v][c] = xOk ? minmod(q[c] - q[w], q[e] - q[c]) : 0;
          sy[v][c] = yOk ? minmod(q[c] - q[ss], q[nn] - q[c]) : 0;
        }
      }
    }
  }

  /** Reconstruct the two face states for the face between `cL` and `cR`
   *  along slope arrays `slope`, applying wall reflection (via the
   *  generated BC kernel) and the optional positivity clamp. Results go
   *  into `this._qL` / `this._qR`. Returns false if the face is fully
   *  inside a wall (flux is zero there). */
  _reconstructFace(Qarr, slope, cL, cR, axisNormal, time) {
    const { nVars, nAux, wall, model, dx } = this;
    const qL = this._qL, qR = this._qR;
    const auxL = this._auxL, auxR = this._auxR;
    const wL = wall[cL], wR = wall[cR];
    if (wL && wR) return false;

    for (let a = 0; a < nAux; a++) {
      auxL[a] = this.Qaux[a][cL];
      auxR[a] = this.Qaux[a][cR];
    }

    if (wL) {
      // Wall on the left: reflect the reconstructed fluid (right) state.
      for (let v = 0; v < nVars; v++)
        qR[v] = Qarr[v][cR] - 0.5 * slope[v][cR];
      for (let a = 0; a < nAux; a++) auxL[a] = auxR[a];
      const reflected = this._wallState(qR, auxR, axisNormal, time);
      for (let v = 0; v < nVars; v++) qL[v] = reflected[v];
    } else if (wR) {
      for (let v = 0; v < nVars; v++)
        qL[v] = Qarr[v][cL] + 0.5 * slope[v][cL];
      for (let a = 0; a < nAux; a++) auxR[a] = auxL[a];
      const reflected = this._wallState(qL, auxL, axisNormal, time);
      for (let v = 0; v < nVars; v++) qR[v] = reflected[v];
    } else {
      for (let v = 0; v < nVars; v++) {
        qL[v] = Qarr[v][cL] + 0.5 * slope[v][cL];
        qR[v] = Qarr[v][cR] - 0.5 * slope[v][cR];
      }
    }
    if (model.positivityFix) {
      model.positivityFix(qL);
      model.positivityFix(qR);
    }
    return true;
  }

  /** Wall-side state for an interior obstacle face — the generated
   *  `boundaryConditions` kernel evaluated with the solver's wall tag. */
  _wallState(qInner, auxInner, axisNormal, time) {
    const n = this._n;
    n.fill(0);
    for (let d = 0; d < n.length; d++) n[d] = axisNormal[d] || 0;
    this._X[0] = 0; this._X[1] = 0;
    return this.model.boundaryConditions(
      this.bc.wall, time, this._X, this.dx,
      qInner, auxInner, this.model.params, n
    );
  }

  /** RHS = −div F into `Rarr`, via MUSCL reconstruction + the injected
   *  numerical flux at every interior face. */
  _computeRHS(Qarr, time, Rarr) {
    const { Nx, ny, nx, nVars, wall, model, dx,
            _sx: sx, _sy: sy, _fx: fx, _fy: fy } = this;
    const { params } = model;
    const dim = this._n.length;
    const nX = new Float64Array(dim); nX[0] = 1;
    const nY = new Float64Array(dim); if (dim > 1) nY[1] = 1;

    // X-direction faces: face east of cell cL stored at fx[v][cL].
    for (let j = 1; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        const cL = j * Nx + i, cR = cL + 1;
        if (!this._reconstructFace(Qarr, sx, cL, cR, nX, time)) {
          for (let v = 0; v < nVars; v++) fx[v][cL] = 0;
          continue;
        }
        const f = model.numericalFlux(
          this._qL, this._qR, this._auxL, this._auxR, params, nX
        );
        for (let v = 0; v < nVars; v++) fx[v][cL] = f[v];
      }
    }

    // Y-direction faces: face north of cell cL stored at fy[v][cL].
    for (let j = 0; j <= ny; j++) {
      for (let i = 1; i <= nx; i++) {
        const cL = j * Nx + i, cR = cL + Nx;
        if (!this._reconstructFace(Qarr, sy, cL, cR, nY, time)) {
          for (let v = 0; v < nVars; v++) fy[v][cL] = 0;
          continue;
        }
        const f = model.numericalFlux(
          this._qL, this._qR, this._auxL, this._auxR, params, nY
        );
        for (let v = 0; v < nVars; v++) fy[v][cL] = f[v];
      }
    }

    const invDx = 1 / dx;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1; i <= nx; i++) {
        const c = j * Nx + i;
        if (wall[c]) {
          for (let v = 0; v < nVars; v++) Rarr[v][c] = 0;
          continue;
        }
        for (let v = 0; v < nVars; v++) {
          Rarr[v][c] =
            (fx[v][c - 1] - fx[v][c] + fy[v][c - Nx] - fy[v][c]) * invDx;
        }
      }
    }
  }

  // ── Time stepping ──────────────────────────────────────────────────

  /** Optional positivity / wet-dry clamp over every cell. */
  _positivityFixAll(Qarr) {
    const { model, nVars, NN } = this;
    if (!model.positivityFix) return;
    const q = this._qL;
    for (let c = 0; c < NN; c++) {
      for (let v = 0; v < nVars; v++) q[v] = Qarr[v][c];
      model.positivityFix(q);
      for (let v = 0; v < nVars; v++) Qarr[v][c] = q[v];
    }
  }

  /**
   * One Heun (SSPRK2) time step:
   *   Q*    = Q + dt·L(Q)
   *   Q^n+1 = ½(Q + Q* + dt·L(Q*))
   *
   * @param {number} time  current simulation time (for time-dependent BCs)
   * @returns {number} dt used
   */
  step(time) {
    const { Nx, Ny, nx, ny, nVars, NN, dx, cfl, dtMin, dtMax, model } = this;
    const Q = this.Q, Q1 = this._Q1, Q2 = this._Q2, R = this._R;

    this._positivityFixAll(Q);
    this.applyBoundaryConditions(Q, time);

    // CFL-limited time step from the per-cell signal speed.
    let sMax = 1e-10;
    const qc = this._qR, auxc = this._auxR;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1; i <= nx; i++) {
        const c = j * Nx + i;
        if (this.wall[c]) continue;
        for (let v = 0; v < nVars; v++) qc[v] = Q[v][c];
        for (let a = 0; a < this.nAux; a++) auxc[a] = this.Qaux[a][c];
        const s = cellMaxWaveSpeed(model, qc, auxc);
        if (s > sMax) sMax = s;
      }
    }
    const dt = Math.max(dtMin, Math.min(cfl * dx / sMax, dtMax));

    // Stage 1: Q1 = Q + dt·L(Q)
    this._computeSlopes(Q);
    this._computeRHS(Q, time, R);
    for (let v = 0; v < nVars; v++)
      for (let c = 0; c < NN; c++)
        Q1[v][c] = Q[v][c] + dt * R[v][c];
    this._positivityFixAll(Q1);
    this.applyBoundaryConditions(Q1, time + dt);

    // Stage 2: Q2 = ½(Q + Q1 + dt·L(Q1))
    this._computeSlopes(Q1);
    this._computeRHS(Q1, time + dt, R);
    for (let v = 0; v < nVars; v++)
      for (let c = 0; c < NN; c++)
        Q2[v][c] = 0.5 * (Q[v][c] + Q1[v][c] + dt * R[v][c]);
    this._positivityFixAll(Q2);

    // Q ← Q2 (swap buffers)
    this.Q = Q2;
    this._Q2 = Q;
    this.applyBoundaryConditions(this.Q, time + dt);
    return dt;
  }

  /**
   * Advance within a wall-clock budget.
   * @param {number} budget  max seconds
   * @param {number} time    current simulation time
   * @returns {{ dtAcc:number, nSteps:number }}
   */
  tick(budget, time) {
    const deadline = performance.now() + budget * 1000;
    let dtAcc = 0, nSteps = 0;
    while (performance.now() < deadline) {
      const dt = this.step(time + dtAcc);
      dtAcc += dt;
      nSteps++;
    }
    return { dtAcc, nSteps };
  }
}

