/**
 * apps/swe-game/worker.js — CPU simulation worker (module worker).
 *
 * Replaces the old inlined MUSCL/HLLC/Heun worker: the numerics core
 * is now the generic HyperbolicSolver2D from library/zoomy_js driven
 * by the codegen'd SWE kernels (see generate.py / swe2d_model.js).
 * Everything game-specific — the opening layout, the raster image, the
 * outflow gauges, the colour scaling, the message protocol — stays
 * here. The main thread talks to this worker exactly as before.
 */

import { HyperbolicSolver2D } from "../../library/zoomy_js/solver.js";
import { createSWE2DModel } from "./swe2d_model.js";

// ── Game constants ──────────────────────────────────────────────────
const WET = 1e-6;
const END_TIME = 60.0;
const BUDGET = 0.08;          // wall-clock budget per frame (s)
const NG = 5;                 // raster ghost border (visual only)

// Openings as fractions of NX so they scale with resolution.
const OPF = {
  IN:  [[0.5833, 0.7500]],
  OUT: [[0.3333, 0.5000], [0.7500, 0.9167]],
  TOP: [[0.2500, 0.4167], [0.6667, 0.7500]],
  BOT: [[0.5000, 0.6667]],
};

// ── Worker state ────────────────────────────────────────────────────
let NX, IW, IH, N, NN, DX;
let O_IN, O_OUT, O_TOP, O_BOT;
let W_IN, W_OUT, W_TOP, W_BOT;
let solver, swe;
let raster;
let gameTime = 0, running = false, finished = false;
let outflow = [0, 0, 0, 0, 0];
let MODE = "h";                       // "h" or "vmag"
let SCALE_HMAX = 0.25, SCALE_VMAX = 1.5;

// ── Helpers ─────────────────────────────────────────────────────────
function wallSegs(op, nx) {
  const w = []; let p = 0;
  for (const [a, b] of op) { w.push([p, a]); p = b; }
  w.push([p, nx]);
  return w;
}
function scaleFrac(arr, nx) {
  return arr.map(([a, b]) => [Math.round(a * nx), Math.round(b * nx)]);
}
function inSeg(x, segs) {
  for (const [a, b] of segs) if (x >= a && x < b) return true;
  return false;
}

// ── Setup / lifecycle ───────────────────────────────────────────────
function setup(nx) {
  NX = nx;
  IW = NX + 2 * NG; IH = IW;
  N = NX + 2; NN = N * N;
  DX = 20.0 / N;
  O_IN  = scaleFrac(OPF.IN,  NX);
  O_OUT = scaleFrac(OPF.OUT, NX);
  O_TOP = scaleFrac(OPF.TOP, NX);
  O_BOT = scaleFrac(OPF.BOT, NX);
  W_IN  = wallSegs(O_IN,  NX);
  W_OUT = wallSegs(O_OUT, NX);
  W_TOP = wallSegs(O_TOP, NX);
  W_BOT = wallSegs(O_BOT, NX);

  swe = createSWE2DModel();
  const BC = swe.BC;
  // Boundary dispatch: openings carry inflow / zero-Neumann outflow,
  // everything else (and the internal sketch mask) is a reflective
  // wall. The generated boundaryConditions kernel computes the state.
  solver = new HyperbolicSolver2D({
    nx: NX, ny: NX, dx: DX,
    model: swe.model,
    bc: {
      wall: BC.wall,
      edges: {
        left:   (j) => (inSeg(j, O_IN)  ? BC.inflow  : BC.wall),
        right:  (j) => (inSeg(j, O_OUT) ? BC.outflow : BC.wall),
        bottom: (i) => (inSeg(i, O_BOT) ? BC.outflow : BC.wall),
        top:    (i) => (inSeg(i, O_TOP) ? BC.outflow : BC.wall),
      },
    },
    cfl: 0.4,
  });
  solver.initialize();
  init();
}

function init() {
  solver.Q[0].fill(0.01);             // initial still water depth
  solver.Q[1].fill(0);
  solver.Q[2].fill(0);
  solver.wall.fill(0);
  raster = new Uint8Array(IW * IH);
  gameTime = 0;
  running = false;
  finished = false;
  outflow = [0, 0, 0, 0, 0];
}

/** Stamp the visual raster's clay pixels into the solver's wall mask. */
function stampWalls() {
  for (let j = 0; j < NX; j++)
    for (let i = 0; i < NX; i++)
      solver.wall[(j + 1) * N + (i + 1)] =
        raster[(j + NG) * IW + (i + NG)] > 0 ? 1 : 0;
}

function startSim(rasterBuf) {
  raster = new Uint8Array(rasterBuf);
  stampWalls();
  running = true;
}

// ── Rendering ───────────────────────────────────────────────────────
function render() {
  const hScale = 254 / Math.max(0.05, SCALE_HMAX);
  const vScale = 254 / Math.max(0.10, SCALE_VMAX);
  const img = new Uint8Array(IW * IH);
  const h = solver.Q[0], hu = solver.Q[1], hv = solver.Q[2];
  for (let j = 0; j < NX; j++) {
    for (let i = 0; i < NX; i++) {
      const k = (j + 1) * N + (i + 1);
      const rj = j + NG, ri = i + NG;
      if (raster[rj * IW + ri] > 0) continue;
      const hk = h[k];
      let v;
      if (MODE === "vmag") {
        if (hk <= WET) { v = 0; }
        else {
          const u_ = hu[k] / hk, w_ = hv[k] / hk;
          v = Math.min(254, Math.sqrt(u_ * u_ + w_ * w_) * vScale);
        }
      } else {
        v = Math.min(hk * hScale, 254);
      }
      img[rj * IW + ri] = (v | 0) + 1;
    }
  }
  for (const [a, b] of W_IN)  for (let j = a; j < b; j++) for (let i = 0; i < NG; i++) img[(j + NG) * IW + i] = 0;
  for (const [a, b] of W_OUT) for (let j = a; j < b; j++) for (let i = IW - NG; i < IW; i++) img[(j + NG) * IW + i] = 0;
  for (const [a, b] of W_TOP) for (let j = IH - NG; j < IH; j++) for (let i = a + NG; i < b + NG; i++) img[j * IW + i] = 0;
  for (const [a, b] of W_BOT) for (let j = 0; j < NG; j++) for (let i = a + NG; i < b + NG; i++) img[j * IW + i] = 0;
  for (const [a, b] of O_IN)
    for (let j = a; j < b; j++) { const v = img[(j + NG) * IW + NG]; for (let i = 0; i < NG; i++) img[(j + NG) * IW + i] = v; }
  for (const [a, b] of O_OUT)
    for (let j = a; j < b; j++) { const v = img[(j + NG) * IW + (IW - NG - 1)]; for (let i = IW - NG; i < IW; i++) img[(j + NG) * IW + i] = v; }
  for (const [a, b] of O_TOP)
    for (let i = a; i < b; i++) { const v = img[(IH - NG - 1) * IW + (i + NG)]; for (let j = IH - NG; j < IH; j++) img[j * IW + (i + NG)] = v; }
  for (const [a, b] of O_BOT)
    for (let i = a; i < b; i++) { const v = img[NG * IW + (i + NG)]; for (let j = 0; j < NG; j++) img[j * IW + (i + NG)] = v; }
  for (let k = 0; k < IW * IH; k++) if (raster[k] > 0) img[k] = 0;
  return img;
}

// ── Step loop ───────────────────────────────────────────────────────
function tick() {
  let ns = 0;
  if (running) {
    const t0 = performance.now();
    let dtacc = 0;
    while ((performance.now() - t0) < BUDGET * 1000) {
      const dt = solver.step(gameTime);
      gameTime += dt;
      dtacc += dt;
      ns++;
      if (gameTime >= END_TIME) {
        running = false; finished = true; gameTime = END_TIME;
        break;
      }
    }
    // Outflow accumulation through the five garden gauges. Sum of h·u
    // over the opening cells × DX (m/cell) × dt = accumulated volume,
    // resolution-independent.
    const hu = solver.Q[1], hv = solver.Q[2];
    {
      const [a, b] = O_BOT[0];
      let s = 0;
      for (let i = a; i < b; i++) s += -hv[1 * N + i];
      outflow[0] += s * DX * dtacc;
    }
    for (let oi = 0; oi < O_OUT.length; oi++) {
      const [a, b] = O_OUT[oi];
      let s = 0;
      for (let j = a; j < b; j++) s += hu[j * N + (N - 2)];
      outflow[1 + oi] += s * DX * dtacc;
    }
    for (let oi = 0; oi < O_TOP.length; oi++) {
      const [a, b] = O_TOP[oi];
      let s = 0;
      for (let i = a; i < b; i++) s += hv[(N - 2) * N + i];
      outflow[oi === 1 ? 3 : 4] += s * DX * dtacc;
    }
    stampWalls();
  }
  const img = render();
  // On the final frame, report the interior max depth / speed so the
  // main thread can refine its per-level colour-scale database.
  let maxH = 0, maxV = 0;
  if (finished) {
    const h = solver.Q[0], hu = solver.Q[1], hv = solver.Q[2];
    for (let j = 1; j < N - 1; j++)
      for (let i = 1; i < N - 1; i++) {
        const k = j * N + i;
        if (solver.wall[k]) continue;
        const hk = h[k];
        if (hk > maxH) maxH = hk;
        if (hk > WET) {
          const u = hu[k] / hk, vv = hv[k] / hk;
          const vm = Math.sqrt(u * u + vv * vv);
          if (vm > maxV) maxV = vm;
        }
      }
  }
  self.postMessage(
    {
      type: "frame", image: img.buffer,
      time: gameTime, of: outflow.slice(), fin: finished, ns,
      iw: IW, ih: IH, maxH, maxV,
    },
    [img.buffer]
  );
}

// ── Message protocol (unchanged from the legacy worker) ─────────────
self.onmessage = function (e) {
  const m = e.data;
  if (m.type === "config") {
    setup(m.nx);
    tick();
    self.postMessage({ type: "ready", iw: IW, ih: IH, nx: NX });
    return;
  }
  if (NX === undefined) return;          // not configured yet

  if (m.type === "tick") {
    tick();
  } else if (m.type === "start") {
    startSim(m.raster);
    tick();
  } else if (m.type === "reset") {
    init();
    tick();
  } else if (m.type === "setwalls") {
    raster = new Uint8Array(m.raster);
    stampWalls();
    const img = render();
    self.postMessage(
      {
        type: "frame", image: img.buffer,
        time: gameTime, of: outflow.slice(), fin: finished, ns: 0,
        iw: IW, ih: IH, maxH: 0, maxV: 0,
      },
      [img.buffer]
    );
  } else if (m.type === "setmode") {
    MODE = m.mode;
    const img = render();
    self.postMessage(
      {
        type: "frame", image: img.buffer,
        time: gameTime, of: outflow.slice(), fin: finished, ns: 0,
        iw: IW, ih: IH,
      },
      [img.buffer]
    );
  } else if (m.type === "setscales") {
    if (m.hMax > 0) SCALE_HMAX = m.hMax;
    if (m.vMax > 0) SCALE_VMAX = m.vMax;
  } else if (m.type === "setqcurve") {
    if (Array.isArray(m.curve) && m.curve.length >= 1) {
      swe.setQCurve(m.curve.map((p) => [p.t, p.q]));
    }
  }
};
