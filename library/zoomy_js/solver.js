/**
 * zoomy_js/solver.js — Generic 2D structured-grid finite volume solver.
 *
 * Mirrors zoomy_core.fvm.solver_numpy.HyperbolicSolver.
 *
 * The solver owns state arrays and the time-stepping loop.
 * All physics are delegated to an injected **model** object (see model.js).
 * The numerical flux scheme is also injected.
 *
 * No model-specific override points live here — the solver is physics-agnostic.
 */

/**
 * Explicit 2D structured-grid FVM solver (Godunov-type).
 *
 * Operates on a (nx+2) × (ny+2) grid with 1 ghost cell per side.
 * State is stored as nVars separate Float32Arrays (struct-of-arrays).
 */
export class HyperbolicSolver2D {
  /**
   * @param {object} opts
   * @param {number}  opts.nx     Interior cells in x
   * @param {number}  opts.ny     Interior cells in y
   * @param {number}  opts.dx     Cell spacing (uniform)
   * @param {Model2D} opts.model  Physics model (flux, BC, wavespeed, …)
   * @param {number}  opts.cfl    CFL number (default 0.45)
   * @param {number}  opts.dtMin  Minimum allowed dt (default 1e-4)
   * @param {number}  opts.dtMax  Maximum allowed dt (default 0.5)
   */
  constructor({ nx, ny, dx, model, cfl = 0.45, dtMin = 1e-4, dtMax = 0.5 }) {
    this.nx = nx;
    this.ny = ny;
    this.N = nx + 2;
    this.NN = this.N * this.N;
    this.dx = dx;
    this.model = model;
    this.nVars = model.nVars;
    this.cfl = cfl;
    this.dtMin = dtMin;
    this.dtMax = dtMax;

    this.vars = null;   // current state: nVars × Float32Array(NN)
    this.vars2 = null;  // double-buffer
    this.wall = null;   // Float32Array(NN) obstacle mask
  }

  /** Allocate state arrays and apply initial + boundary conditions. */
  initialize() {
    const { nVars, NN, model } = this;
    this.vars = Array.from({ length: nVars }, () => new Float32Array(NN));
    this.vars2 = Array.from({ length: nVars }, () => new Float32Array(NN));
    this.wall = new Float32Array(NN);
    model.initConditions(this.vars);
    model.applyBC(this.vars, this.wall);
  }

  /**
   * One explicit FVM time step (Lax-Friedrichs, constant reconstruction).
   *
   * Data flow (matches zoomy_core HyperbolicSolver.solve loop body):
   *   1. positivityFix
   *   2. global max wave speed
   *   3. fused stencil + flux + LF update over interior cells
   *   4. positivityFix + applyBC
   *
   * @returns {number} dt used
   */
  step() {
    const { N, nx, ny, nVars, vars, vars2, dx, cfl, dtMin, dtMax, model, wall } = this;

    model.positivityFix(vars);

    // ── Pass 1: global max wave speed ──
    let sMax = 1e-10;
    for (let k = 0; k < this.NN; k++) {
      const s = model.maxWaveSpeed(vars, k);
      if (s > sMax) sMax = s;
    }
    const dt = Math.max(dtMin, Math.min(cfl * dx / sMax, dtMax));
    const r = dt / dx;

    // ── Scratch (allocated once, reused every step) ──
    if (!this._qi) {
      const n = nVars;
      this._qi = new Array(n); this._qn = new Array(n); this._qs = new Array(n);
      this._qe = new Array(n); this._qw = new Array(n); this._refl = new Array(n);
      this._fxi = new Float32Array(n); this._fyi = new Float32Array(n);
      this._fxn = new Float32Array(n); this._fyn = new Float32Array(n);
      this._fxs = new Float32Array(n); this._fys = new Float32Array(n);
      this._fxe = new Float32Array(n); this._fye = new Float32Array(n);
      this._fxw = new Float32Array(n); this._fyw = new Float32Array(n);
    }
    const { _qi: qi, _qn: qn, _qs: qs, _qe: qe, _qw: qw, _refl: refl } = this;
    const { _fxi: fxi, _fyi: fyi, _fxn: fxn, _fyn: fyn, _fxs: fxs, _fys: fys,
            _fxe: fxe, _fye: fye, _fxw: fxw, _fyw: fyw } = this;

    // ── Pass 2: fused reconstruction + flux + LF + update ──
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
        this._readNeighbour(vars, wall, c, cn, qn, refl);
        this._readNeighbour(vars, wall, c, cs, qs, refl);
        this._readNeighbour(vars, wall, c, ce, qe, refl);
        this._readNeighbour(vars, wall, c, cw, qw, refl);

        // Compute physical fluxes
        model.flux(qi, fxi, fyi);
        model.flux(qn, fxn, fyn);
        model.flux(qs, fxs, fys);
        model.flux(qe, fxe, fye);
        model.flux(qw, fxw, fyw);

        // Lax-Friedrichs at 4 interfaces → update
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

    model.positivityFix(this.vars);
    model.applyBC(this.vars, this.wall);
    return dt;
  }

  /**
   * Run solver steps within a wall-clock budget.
   * @param {number} budget — max seconds
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

  /** Flat index from (j, i). */
  idx(j, i) { return j * this.N + i; }

  // ── Internal ──

  /** Read a neighbour cell, applying wall reflection if needed. */
  _readNeighbour(vars, wall, ki, kn, qout, refl) {
    if (wall[kn] > 0) {
      this.model.wallReflect(vars, ki, kn, refl);
      for (let v = 0; v < this.nVars; v++) qout[v] = refl[v];
    } else {
      for (let v = 0; v < this.nVars; v++) qout[v] = vars[v][kn];
    }
  }
}
