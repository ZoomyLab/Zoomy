/**
 * zoomy_js/swe.js — 2D Shallow Water Equations model for the FVM solver.
 *
 * Mirrors zoomy_core.model.models.shallow_water.ShallowWaterEquations:
 *   flux:  F = [hu, hu²+½gh², huv]  G = [hv, huv, hv²+½gh²]
 *   wave speed: max(|u|+c, |v|+c)  where c = sqrt(g*h)
 *   wet/dry: zero momentum where h ≤ WET_TOL
 *
 * Also provides:
 *   - SWESolver class (extends HyperbolicSolver2D with SWE physics)
 *   - createSWEGame() factory for the irrigation game
 *   - render() for uint8 flow image
 */

import { HyperbolicSolver2D } from "./solver.js";

// ── Physical constants ──
const G = 9.81;
const HALF_G = 0.5 * G;
const WET_TOL = 1e-6;

/**
 * SWE solver for 2D structured grids.
 *
 * State variables (nVars = 3):
 *   vars[0] = h   (water height)
 *   vars[1] = hu  (x-momentum)
 *   vars[2] = hv  (y-momentum)
 *
 * The obstacle mask is stored in this.wall (inherited from HyperbolicSolver2D).
 */
export class SWESolver extends HyperbolicSolver2D {
  constructor(opts) {
    super({ ...opts, nVars: 3 });
    this.hInit = opts.hInit || 0.01;
  }

  initConditions(vars) {
    vars[0].fill(this.hInit);
  }

  positivityFix(vars) {
    const [h, hu, hv] = vars;
    for (let k = 0; k < this.NN; k++) {
      if (h[k] <= WET_TOL) { h[k] = 0; hu[k] = 0; hv[k] = 0; }
    }
  }

  maxCellWaveSpeed(vars, k) {
    const h = vars[0][k];
    if (h <= 0) return 0;
    const u = vars[1][k] / h, v = vars[2][k] / h;
    const c = Math.sqrt(G * h);
    return Math.max(Math.abs(u) + c, Math.abs(v) + c);
  }

  wallReflect(vars, ki, kj, out) {
    out[0] = vars[0][ki];
    out[1] = -vars[1][ki];
    out[2] = -vars[2][ki];
  }

  /** Compute SWE flux from value array [h, hu, hv]. */
  _fluxFromValues(q, outFx, outFy) {
    const h = q[0];
    const sh = h > 0 ? h : 1;
    const u = h > 0 ? q[1] / sh : 0;
    const v = h > 0 ? q[2] / sh : 0;
    const hh = HALF_G * h * h;
    outFx[0] = h * u;
    outFx[1] = h * u * u + hh;
    outFx[2] = h * u * v;
    outFy[0] = h * v;
    outFy[1] = h * u * v;
    outFy[2] = h * v * v + hh;
  }
}


// ═════════════════════════════════════════════════════
// SWE Irrigation Game
// ═════════════════════════════════════════════════════

/**
 * Create an SWE game instance with boundary conditions and rendering.
 *
 * @param {object} cfg
 * @param {number} cfg.scale       Grid scale factor (default 5 → 300×300)
 * @param {number} cfg.endTime     Game duration in seconds (default 60)
 * @param {number} cfg.budget      Solver budget per tick in seconds (default 0.08)
 * @param {number[][]} cfg.oIn     Inflow openings  [[a,b], ...]
 * @param {number[][]} cfg.oOut    East outflow openings
 * @param {number[][]} cfg.oTop    North outflow openings
 * @param {number[][]} cfg.oBot    South outflow openings
 * @param {number} cfg.hInflow     Inflow water height
 * @param {number} cfg.qInflow     Inflow momentum
 * @returns {SWEGame}
 */
export function createSWEGame(cfg = {}) {
  return new SWEGame(cfg);
}

export class SWEGame {
  constructor(cfg = {}) {
    const scale = cfg.scale || 5;
    this.nx = 60 * scale;
    this.ny = 60 * scale;
    this.ng = 5;  // ghost cells for display image
    this.imgW = this.nx + 2 * this.ng;
    this.imgH = this.ny + 2 * this.ng;
    this.endTime = cfg.endTime || 60;
    this.budget = cfg.budget || 0.08;
    this.hInflow = cfg.hInflow || 0.1;
    this.qInflow = cfg.qInflow || 0.01;

    // Boundary openings (in interior cell coords)
    this.oIn  = cfg.oIn  || [[scale * 35, scale * 45]];
    this.oOut = cfg.oOut || [[scale * 20, scale * 30], [scale * 45, scale * 55]];
    this.oTop = cfg.oTop || [[scale * 15, scale * 25], [scale * 40, scale * 45]];
    this.oBot = cfg.oBot || [[scale * 30, scale * 40]];
    this.nGauges = this.oTop.length + this.oOut.length + this.oBot.length;

    // Wall segments (complement of openings)
    this.wIn  = wallSegs(this.oIn,  this.nx);
    this.wOut = wallSegs(this.oOut, this.nx);
    this.wTop = wallSegs(this.oTop, this.nx);
    this.wBot = wallSegs(this.oBot, this.nx);

    // Create solver
    const N = this.nx + 2;
    this.solver = new SWESolver({
      nx: this.nx, ny: this.ny,
      dx: 20.0 / N,
      cfl: 0.45,
    });

    // Bind SWE boundary conditions to the solver
    const self = this;
    this.solver.applyBC = (vars) => self._applyBC(vars);

    // Raster (display-sized obstacle mask)
    this.raster = new Uint8Array(this.imgW * this.imgH);

    // Game state
    this.gameTime = 0;
    this.running = false;
    this.finished = false;
    this.outflow = new Float64Array(this.nGauges);
  }

  /** Initialize / reset the game. */
  init() {
    this.solver.initialize();
    this.raster.fill(0);
    this.gameTime = 0;
    this.running = false;
    this.finished = false;
    this.outflow.fill(0);
  }

  /** Start simulation with a raster obstacle mask (Uint8Array, imgW×imgH). */
  start(rasterBuf) {
    this.raster = new Uint8Array(rasterBuf);
    this._stampWalls();
    this.running = true;
  }

  /**
   * Advance simulation within budget. Returns game state.
   * @returns {{ image: Uint8Array, time: number, outflow: number[], finished: boolean, nSteps: number }}
   */
  tick() {
    let nSteps = 0;
    if (this.running) {
      const t0 = performance.now();
      let dtAcc = 0;
      while ((performance.now() - t0) < this.budget * 1000) {
        const dt = this.solver.step();
        this.gameTime += dt;
        dtAcc += dt;
        nSteps++;
        if (this.gameTime >= this.endTime) {
          this.running = false;
          this.finished = true;
          this.gameTime = this.endTime;
          break;
        }
      }
      this._accumulateOutflow(dtAcc);
      this._stampWalls();
    }
    return {
      image: this.render(),
      time: this.gameTime,
      outflow: Array.from(this.outflow),
      finished: this.finished,
      nSteps,
    };
  }

  /** Render the flow field as a Uint8Array(imgW × imgH). 0=wall, 1-255=water. */
  render() {
    const { nx, ny, ng, imgW, imgH, raster, solver } = this;
    const N = solver.N;
    const h = solver.vars[0];
    const img = new Uint8Array(imgW * imgH);

    // Interior: scale h to 1-255
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const rj = j + ng, ri = i + ng;
        if (raster[rj * imgW + ri] > 0) continue;
        const v = Math.min(h[(j + 1) * N + (i + 1)] * 1275, 254);
        img[rj * imgW + ri] = (v | 0) + 1;
      }
    }

    // Boundary walls
    for (const [a, b] of this.wIn)  for (let j = a; j < b; j++) for (let i = 0; i < ng; i++) img[(j+ng)*imgW+i] = 0;
    for (const [a, b] of this.wOut) for (let j = a; j < b; j++) for (let i = imgW-ng; i < imgW; i++) img[(j+ng)*imgW+i] = 0;
    for (const [a, b] of this.wTop) for (let j = imgH-ng; j < imgH; j++) for (let i = a+ng; i < b+ng; i++) img[j*imgW+i] = 0;
    for (const [a, b] of this.wBot) for (let j = 0; j < ng; j++) for (let i = a+ng; i < b+ng; i++) img[j*imgW+i] = 0;

    // Port openings: fill with adjacent interior flow
    for (const [a, b] of this.oIn)
      for (let j = a; j < b; j++) { const v = img[(j+ng)*imgW+ng]; for (let i = 0; i < ng; i++) img[(j+ng)*imgW+i] = v; }
    for (const [a, b] of this.oOut)
      for (let j = a; j < b; j++) { const v = img[(j+ng)*imgW+(imgW-ng-1)]; for (let i = imgW-ng; i < imgW; i++) img[(j+ng)*imgW+i] = v; }
    for (const [a, b] of this.oTop)
      for (let i = a; i < b; i++) { const v = img[(imgH-ng-1)*imgW+(i+ng)]; for (let j = imgH-ng; j < imgH; j++) img[j*imgW+(i+ng)] = v; }
    for (const [a, b] of this.oBot)
      for (let i = a; i < b; i++) { const v = img[ng*imgW+(i+ng)]; for (let j = 0; j < ng; j++) img[j*imgW+(i+ng)] = v; }

    // Raster obstacles in ghost region
    for (let k = 0; k < imgW * imgH; k++) if (raster[k] > 0) img[k] = 0;

    return img;
  }

  // ─── Internal ──────────────────────────────────────

  /** Copy raster obstacle mask into solver wall field. */
  _stampWalls() {
    const { nx, ny, ng, imgW, solver } = this;
    const N = solver.N;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++)
        solver.wall[(j + 1) * N + (i + 1)] = this.raster[(j + ng) * imgW + (i + ng)] > 0 ? 1 : 0;
  }

  /** Apply SWE boundary conditions with inflow/outflow openings. */
  _applyBC(vars) {
    const [h, hu, hv] = vars;
    const N = this.solver.N;
    const { hInflow, qInflow, oIn, oOut, oTop, oBot } = this;

    // East/West boundaries (loop over rows)
    for (let j = 0; j < N; j++) {
      const jN = j * N;
      // East: reflective
      h[jN + N - 1]  = h[jN + N - 2];
      hu[jN + N - 1] = -hu[jN + N - 2];
      hv[jN + N - 1] = hv[jN + N - 2];
      // West: reflective
      h[jN]  = h[jN + 1];
      hu[jN] = -hu[jN + 1];
      hv[jN] = hv[jN + 1];
    }
    for (const [a, b] of oOut)
      for (let j = a; j < b; j++) hu[j * N + N - 1] = Math.max(hu[j * N + N - 2], 0);
    for (const [a, b] of oIn) {
      for (let j = a; j < b; j++) {
        const jN = j * N;
        h[jN] = Math.max(h[jN + 1], hInflow);
        hu[jN] = hu[jN + 1] >= 0 ? qInflow : hu[jN + 1];
      }
    }

    // North/South boundaries (loop over columns)
    const jn = (N - 1) * N, j2 = (N - 2) * N;
    for (let i = 0; i < N; i++) {
      h[jn + i]  = h[j2 + i];
      hu[jn + i] = hu[j2 + i];
      hv[jn + i] = -hv[j2 + i];
      h[i]       = h[N + i];
      hu[i]      = hu[N + i];
      hv[i]      = -hv[N + i];
    }
    for (const [a, b] of oTop)
      for (let i = a; i < b; i++) hv[jn + i] = Math.max(hv[j2 + i], 0);
    for (const [a, b] of oBot)
      for (let i = a; i < b; i++) hv[i] = Math.min(hv[N + i], 0);
  }

  /** Accumulate outflow at gauge openings. */
  _accumulateOutflow(dtAcc) {
    const { oTop, oOut, oBot, solver } = this;
    const [, hu, hv] = solver.vars;
    const N = solver.N;
    let ig = 0;

    for (const [a, b] of oTop) {
      let s = 0;
      for (let i = a; i < b; i++) s += hv[(N - 2) * N + i];
      this.outflow[ig++] += s * dtAcc;
    }
    for (let oi = 0; oi < oOut.length; oi++) {
      const [a, b] = oOut[oi];
      let s = 0;
      for (let j = a; j < b; j++) s += hu[j * N + (N - 2)];
      this.outflow[3 - oi] += s * dtAcc;
    }
    for (const [a, b] of oBot) {
      let s = 0;
      for (let i = a; i < b; i++) s += -hv[1 * N + i];
      this.outflow[ig++] += s * dtAcc;
    }
  }
}


// ── Helpers ──

function wallSegs(openings, nx) {
  const w = [];
  let p = 0;
  for (const [a, b] of openings) { w.push([p, a]); p = b; }
  w.push([p, nx]);
  return w;
}
