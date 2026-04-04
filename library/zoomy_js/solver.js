/**
 * zoomy_js/solver.js — Generic 2D structured-grid finite volume solver.
 *
 * Mirrors the design of zoomy_core.fvm.solver_numpy.HyperbolicSolver:
 *   initialize()  → allocate state arrays
 *   step()        → one explicit FVM time step (returns dt)
 *   tick(budget)  → run steps within a wall-clock budget
 *
 * The solver operates on a (nx+2) × (ny+2) grid (1 ghost cell per side)
 * with nVars state variables stored as separate Float32Arrays (struct-of-arrays).
 *
 * Model-specific physics are injected via override methods:
 *   computeFluxes(vars, i, j, out)   — physical flux at cell (i,j)
 *   maxCellWaveSpeed(vars, k)        — max |λ| for CFL
 *   applyBC(vars)                    — set ghost cells
 *   positivityFix(vars)              — e.g. wet/dry
 *   wallReflect(vars, ki, kj, out)   — obstacle reflection
 *   isWall(k)                        — true if cell k is an obstacle
 */

export class HyperbolicSolver2D {
  /**
   * @param {object} opts
   * @param {number} opts.nx       Interior cells in x
   * @param {number} opts.ny       Interior cells in y
   * @param {number} opts.dx       Cell spacing
   * @param {number} opts.nVars    Number of conserved variables (default 3)
   * @param {number} opts.cfl      CFL number (default 0.45)
   * @param {number} opts.dtMin    Minimum allowed dt (default 1e-4)
   * @param {number} opts.dtMax    Maximum allowed dt (default 0.5)
   */
  constructor({ nx, ny, dx, nVars = 3, cfl = 0.45, dtMin = 1e-4, dtMax = 0.5 }) {
    this.nx = nx;
    this.ny = ny;
    this.N = nx + 2;          // grid width including ghosts
    this.NN = this.N * this.N;
    this.dx = dx;
    this.nVars = nVars;
    this.cfl = cfl;
    this.dtMin = dtMin;
    this.dtMax = dtMax;

    // State: two sets for double-buffering (current + next)
    // Each variable is a separate Float32Array of length NN
    this.vars = null;   // current: array of nVars Float32Array(NN)
    this.vars2 = null;  // next buffer
    this.wall = null;   // Float32Array(NN) obstacle mask
  }

  /** Allocate state arrays and set initial conditions. */
  initialize() {
    const { nVars, NN } = this;
    this.vars = Array.from({ length: nVars }, () => new Float32Array(NN));
    this.vars2 = Array.from({ length: nVars }, () => new Float32Array(NN));
    this.wall = new Float32Array(NN);
    this.initConditions(this.vars);
    this.applyBC(this.vars);
  }

  /** Override: set initial conditions on vars. */
  initConditions(vars) {
    // default: do nothing
  }

  // ─── Override points (model-specific) ──────────────

  /**
   * Compute x-flux and y-flux for a single cell.
   * @param {Float32Array[]} vars — state arrays
   * @param {number} k — flat index into arrays
   * @param {Float32Array} outFx — write nVars x-flux components
   * @param {Float32Array} outFy — write nVars y-flux components
   */
  computeFluxes(vars, k, outFx, outFy) {
    throw new Error("computeFluxes must be overridden");
  }

  /**
   * Max wave speed at cell k (for CFL).
   * @returns {number}
   */
  maxCellWaveSpeed(vars, k) {
    throw new Error("maxCellWaveSpeed must be overridden");
  }

  /** Apply boundary conditions (set ghost cells). */
  applyBC(vars) {
    // default: do nothing
  }

  /** Positivity / wet-dry fix. */
  positivityFix(vars) {
    // default: do nothing
  }

  /** True if cell k is a wall/obstacle. */
  isWall(k) {
    return this.wall[k] > 0;
  }

  /**
   * Write reflected state for a neighbour that is a wall.
   * @param {Float32Array[]} vars
   * @param {number} ki — interior cell index
   * @param {number} kj — neighbour (wall) cell index
   * @param {number[]} out — nVars-length scratch for reflected state values
   */
  wallReflect(vars, ki, kj, out) {
    // default: copy interior state with negated momentum
    out[0] = vars[0][ki];
    for (let v = 1; v < this.nVars; v++) out[v] = -vars[v][ki];
  }

  // ─── Core solver ──────────────────────────────────

  /**
   * One FVM time step (Lax-Friedrichs). Modifies state in-place.
   * @returns {number} dt used
   */
  step() {
    const { N, nx, ny, nVars, vars, vars2, dx, cfl, dtMin, dtMax } = this;

    this.positivityFix(vars);

    // Pass 1: global max wave speed
    let sMax = 1e-10;
    for (let k = 0; k < this.NN; k++) {
      const s = this.maxCellWaveSpeed(vars, k);
      if (s > sMax) sMax = s;
    }
    const dt = Math.max(dtMin, Math.min(cfl * dx / sMax, dtMax));
    const r = dt / dx;

    // Scratch arrays for per-cell flux (avoid allocation in loop)
    const fxi = new Float32Array(nVars), fyi = new Float32Array(nVars);
    const fxn = new Float32Array(nVars), fyn = new Float32Array(nVars);
    const fxs = new Float32Array(nVars), fys = new Float32Array(nVars);
    const fxe = new Float32Array(nVars), fye = new Float32Array(nVars);
    const fxw = new Float32Array(nVars), fyw = new Float32Array(nVars);
    const refl = new Array(nVars);

    // Temporary cell-value holders for neighbours (after wall reflection)
    const qi = new Array(nVars), qn = new Array(nVars), qs = new Array(nVars);
    const qe = new Array(nVars), qw = new Array(nVars);

    // Pass 2: fused reconstruction + flux + LF + update
    for (let j = 1; j <= ny; j++) {
      for (let i = 1; i <= nx; i++) {
        const c  = j * N + i;
        const cn = (j + 1) * N + i;
        const cs = (j - 1) * N + i;
        const ce = j * N + (i + 1);
        const cw = j * N + (i - 1);

        // Read centre cell
        for (let v = 0; v < nVars; v++) qi[v] = vars[v][c];

        // Read neighbours with wall reflection
        const neighbours = [[cn, qn], [cs, qs], [ce, qe], [cw, qw]];
        for (const [kn, qout] of neighbours) {
          if (this.isWall(kn)) {
            this.wallReflect(vars, c, kn, refl);
            for (let v = 0; v < nVars; v++) qout[v] = refl[v];
          } else {
            for (let v = 0; v < nVars; v++) qout[v] = vars[v][kn];
          }
        }

        // Compute fluxes at centre and 4 neighbours
        this._fluxFromValues(qi, fxi, fyi);
        this._fluxFromValues(qn, fxn, fyn);
        this._fluxFromValues(qs, fxs, fys);
        this._fluxFromValues(qe, fxe, fye);
        this._fluxFromValues(qw, fxw, fyw);

        // Lax-Friedrichs at 4 interfaces + update
        for (let v = 0; v < nVars; v++) {
          const Fw = 0.5 * (fxw[v] + fxi[v] - sMax * (qi[v] - qw[v]));
          const Fe = 0.5 * (fxi[v] + fxe[v] - sMax * (qe[v] - qi[v]));
          const Gs = 0.5 * (fys[v] + fyi[v] - sMax * (qi[v] - qs[v]));
          const Gn = 0.5 * (fyi[v] + fyn[v] - sMax * (qn[v] - qi[v]));
          vars2[v][c] = qi[v] + r * (Fw - Fe + Gs - Gn);
        }
      }
    }

    // Swap buffers
    const tmp = this.vars;
    this.vars = this.vars2;
    this.vars2 = tmp;

    this.positivityFix(this.vars);
    this.applyBC(this.vars);
    return dt;
  }

  /**
   * Run solver steps within a wall-clock budget (seconds).
   * @param {number} budget — max seconds to spend
   * @returns {{ dtAcc: number, nSteps: number }}
   */
  tick(budget) {
    const deadline = performance.now() + budget * 1000;
    let dtAcc = 0, nSteps = 0;
    while (performance.now() < deadline) {
      dtAcc += this.step();
      nSteps++;
    }
    return { dtAcc, nSteps };
  }

  // ─── Internal helpers ─────────────────────────────

  /** Compute flux from value array (not index). Override for perf if needed. */
  _fluxFromValues(q, outFx, outFy) {
    // Default: delegate to computeFluxes via a temporary index scheme.
    // Subclasses should override this for the fused hot path.
    throw new Error("_fluxFromValues must be overridden");
  }

  /** Flat index from (j, i). */
  idx(j, i) { return j * this.N + i; }
}
