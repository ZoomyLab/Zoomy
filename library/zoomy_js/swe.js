/**
 * zoomy_js/swe.js — 2D Shallow Water Equations model + irrigation game.
 *
 * Implements the Model2D interface (see model.js) for the SWE:
 *   Q = [h, hu, hv]   (water height, x-momentum, y-momentum)
 *   F = [hu, hu²+½gh², huv]   G = [hv, huv, hv²+½gh²]
 *
 * The SWE model object is created by createSWEModel(). It is passed
 * to HyperbolicSolver2D as the `model` parameter — no subclassing.
 *
 * SWEGame wraps the solver + model with game-specific state
 * (outflow tracking, timer, scoring, rendering).
 */

import { HyperbolicSolver2D } from "./solver.js";

// ── Physical constants ──
const G = 9.81;
const HALF_G = 0.5 * G;
const WET_TOL = 1e-6;

// ═════════════════════════════════════════════════════
// SWE Model (implements Model2D interface)
// ═════════════════════════════════════════════════════

/**
 * Create a 2D SWE model object.
 *
 * @param {object} opts
 * @param {number}     opts.hInit    Initial water height (default 0.01)
 * @param {number}     opts.g        Gravity (default 9.81)
 * @param {function}   opts.applyBC  Boundary condition function (vars, wall) → void
 * @returns {Model2D}
 */
export function createSWEModel({ hInit = 0.01, g = G, applyBC } = {}) {
  const halfG = 0.5 * g;

  return {
    nVars: 3,

    initConditions(vars) {
      vars[0].fill(hInit);  // h
      // hu, hv default to 0
    },

    /** SWE flux: F_x = [hu, hu²+½gh², huv], F_y = [hv, huv, hv²+½gh²]. */
    flux(q, outFx, outFy) {
      const h = q[0];
      const sh = h > 0 ? h : 1;
      const u = h > 0 ? q[1] / sh : 0;
      const v = h > 0 ? q[2] / sh : 0;
      const hh = halfG * h * h;
      outFx[0] = h * u;      outFx[1] = h * u * u + hh;  outFx[2] = h * u * v;
      outFy[0] = h * v;      outFy[1] = h * u * v;       outFy[2] = h * v * v + hh;
    },

    maxWaveSpeed(vars, k) {
      const h = vars[0][k];
      if (h <= 0) return 0;
      const u = vars[1][k] / h, v = vars[2][k] / h;
      const c = Math.sqrt(g * h);
      return Math.max(Math.abs(u) + c, Math.abs(v) + c);
    },

    positivityFix(vars) {
      const [h, hu, hv] = vars;
      for (let k = 0, n = h.length; k < n; k++) {
        if (h[k] <= WET_TOL) { h[k] = 0; hu[k] = 0; hv[k] = 0; }
      }
    },

    isWall(wall, k) { return wall[k] > 0; },

    wallReflect(vars, ki, kj, out) {
      out[0] = vars[0][ki];
      out[1] = -vars[1][ki];
      out[2] = -vars[2][ki];
    },

    applyBC: applyBC || function() {},
  };
}


// ═════════════════════════════════════════════════════
// SWE Irrigation Game
// ═════════════════════════════════════════════════════

/**
 * Factory: create a fully configured SWE game instance.
 * @param {object} cfg  (see SWEGame constructor)
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
    this.ng = 5;
    this.imgW = this.nx + 2 * this.ng;
    this.imgH = this.ny + 2 * this.ng;
    this.endTime = cfg.endTime || 60;
    this.budget = cfg.budget || 0.08;
    this.hInflow = cfg.hInflow || 0.1;
    this.qInflow = cfg.qInflow || 0.01;

    this.oIn  = cfg.oIn  || [[scale * 35, scale * 45]];
    this.oOut = cfg.oOut || [[scale * 20, scale * 30], [scale * 45, scale * 55]];
    this.oTop = cfg.oTop || [[scale * 15, scale * 25], [scale * 40, scale * 45]];
    this.oBot = cfg.oBot || [[scale * 30, scale * 40]];
    this.nGauges = this.oTop.length + this.oOut.length + this.oBot.length;

    this.wIn  = _wallSegs(this.oIn,  this.nx);
    this.wOut = _wallSegs(this.oOut, this.nx);
    this.wTop = _wallSegs(this.oTop, this.nx);
    this.wBot = _wallSegs(this.oBot, this.nx);

    // Create model with game-specific BC
    const self = this;
    this.model = createSWEModel({
      applyBC: (vars, wall) => self._applyBC(vars),
    });

    // Create solver (physics-agnostic, model injected)
    const N = this.nx + 2;
    this.solver = new HyperbolicSolver2D({
      nx: this.nx, ny: this.ny,
      dx: 20.0 / N,
      model: this.model,
      cfl: 0.45,
    });

    this.raster = new Uint8Array(this.imgW * this.imgH);
    this.gameTime = 0;
    this.running = false;
    this.finished = false;
    this.outflow = new Float64Array(this.nGauges);
  }

  init() {
    this.solver.initialize();
    this.raster.fill(0);
    this.gameTime = 0;
    this.running = false;
    this.finished = false;
    this.outflow.fill(0);
  }

  start(rasterBuf) {
    this.raster = new Uint8Array(rasterBuf);
    this._stampWalls();
    this.running = true;
  }

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

  render() {
    const { nx, ny, ng, imgW, imgH, raster, solver } = this;
    const N = solver.N, h = solver.vars[0];
    const img = new Uint8Array(imgW * imgH);

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const rj = j + ng, ri = i + ng;
        if (raster[rj * imgW + ri] > 0) continue;
        const v = Math.min(h[(j + 1) * N + (i + 1)] * 1275, 254);
        img[rj * imgW + ri] = (v | 0) + 1;
      }
    }

    for (const [a, b] of this.wIn)  for (let j = a; j < b; j++) for (let i = 0; i < ng; i++) img[(j+ng)*imgW+i] = 0;
    for (const [a, b] of this.wOut) for (let j = a; j < b; j++) for (let i = imgW-ng; i < imgW; i++) img[(j+ng)*imgW+i] = 0;
    for (const [a, b] of this.wTop) for (let j = imgH-ng; j < imgH; j++) for (let i = a+ng; i < b+ng; i++) img[j*imgW+i] = 0;
    for (const [a, b] of this.wBot) for (let j = 0; j < ng; j++) for (let i = a+ng; i < b+ng; i++) img[j*imgW+i] = 0;

    for (const [a, b] of this.oIn)
      for (let j = a; j < b; j++) { const v = img[(j+ng)*imgW+ng]; for (let i = 0; i < ng; i++) img[(j+ng)*imgW+i] = v; }
    for (const [a, b] of this.oOut)
      for (let j = a; j < b; j++) { const v = img[(j+ng)*imgW+(imgW-ng-1)]; for (let i = imgW-ng; i < imgW; i++) img[(j+ng)*imgW+i] = v; }
    for (const [a, b] of this.oTop)
      for (let i = a; i < b; i++) { const v = img[(imgH-ng-1)*imgW+(i+ng)]; for (let j = imgH-ng; j < imgH; j++) img[j*imgW+(i+ng)] = v; }
    for (const [a, b] of this.oBot)
      for (let i = a; i < b; i++) { const v = img[ng*imgW+(i+ng)]; for (let j = 0; j < ng; j++) img[j*imgW+(i+ng)] = v; }

    for (let k = 0; k < imgW * imgH; k++) if (raster[k] > 0) img[k] = 0;
    return img;
  }

  // ─── Internal ──────────────────────────────────────

  _stampWalls() {
    const { nx, ny, ng, imgW, solver } = this;
    const N = solver.N;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++)
        solver.wall[(j + 1) * N + (i + 1)] = this.raster[(j + ng) * imgW + (i + ng)] > 0 ? 1 : 0;
  }

  _applyBC(vars) {
    const [h, hu, hv] = vars;
    const N = this.solver.N;
    const { hInflow, qInflow, oIn, oOut, oTop, oBot } = this;

    for (let j = 0; j < N; j++) {
      const jN = j * N;
      h[jN + N - 1] = h[jN + N - 2]; hu[jN + N - 1] = -hu[jN + N - 2]; hv[jN + N - 1] = hv[jN + N - 2];
      h[jN] = h[jN + 1]; hu[jN] = -hu[jN + 1]; hv[jN] = hv[jN + 1];
    }
    for (const [a, b] of oOut) for (let j = a; j < b; j++) hu[j * N + N - 1] = Math.max(hu[j * N + N - 2], 0);
    for (const [a, b] of oIn) for (let j = a; j < b; j++) {
      const jN = j * N;
      h[jN] = Math.max(h[jN + 1], hInflow);
      hu[jN] = hu[jN + 1] >= 0 ? qInflow : hu[jN + 1];
    }

    const jn = (N - 1) * N, j2 = (N - 2) * N;
    for (let i = 0; i < N; i++) {
      h[jn+i] = h[j2+i]; hu[jn+i] = hu[j2+i]; hv[jn+i] = -hv[j2+i];
      h[i] = h[N+i]; hu[i] = hu[N+i]; hv[i] = -hv[N+i];
    }
    for (const [a, b] of oTop) for (let i = a; i < b; i++) hv[jn + i] = Math.max(hv[j2 + i], 0);
    for (const [a, b] of oBot) for (let i = a; i < b; i++) hv[i] = Math.min(hv[N + i], 0);
  }

  _accumulateOutflow(dtAcc) {
    const { oTop, oOut, oBot, solver } = this;
    const [, hu, hv] = solver.vars;
    const N = solver.N;
    let ig = 0;
    for (const [a, b] of oTop) { let s = 0; for (let i = a; i < b; i++) s += hv[(N-2)*N+i]; this.outflow[ig++] += s * dtAcc; }
    for (let oi = 0; oi < oOut.length; oi++) { const [a,b] = oOut[oi]; let s = 0; for (let j = a; j < b; j++) s += hu[j*N+(N-2)]; this.outflow[3-oi] += s * dtAcc; }
    for (const [a, b] of oBot) { let s = 0; for (let i = a; i < b; i++) s += -hv[1*N+i]; this.outflow[ig++] += s * dtAcc; }
  }
}

function _wallSegs(openings, nx) {
  const w = []; let p = 0;
  for (const [a, b] of openings) { w.push([p, a]); p = b; }
  w.push([p, nx]); return w;
}
