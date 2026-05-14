/**
 * End-to-end pipeline test for the SWE game's CPU path.
 *
 * Exercises the whole codegen → solver chain on the *real* game model:
 * the kernels emitted by generate.py (SWE + HLLC + parametric inflow
 * boundary), wrapped by swe2d_model.mjs, driven by the generic
 * HyperbolicSolver2D from library/zoomy_js.
 *
 * Run:  node --test test/   (from apps/swe-game)
 */

import test from "node:test";
import assert from "node:assert/strict";

import { HyperbolicSolver2D } from "../../../library/zoomy_js/solver.js";
import { createSWE2DModel } from "../swe2d_model.mjs";

if (typeof performance === "undefined") {
  globalThis.performance = { now: () => Number(process.hrtime.bigint()) / 1e6 };
}

/**
 * Build a solver on an nx×nx grid. `inflowRows` is the [a, b) interior
 * row range on the left edge that gets the inflow tag; everything else
 * on every edge is wall.
 */
function makeSolver({ nx = 24, inflowRows = null } = {}) {
  const { model, params, setQCurve, BC } = createSWE2DModel();
  const left =
    inflowRows == null
      ? BC.wall
      : (j) => (j >= inflowRows[0] && j < inflowRows[1] ? BC.inflow : BC.wall);
  const solver = new HyperbolicSolver2D({
    nx,
    ny: nx,
    dx: 20.0 / (nx + 2),
    model,
    bc: {
      wall: BC.wall,
      edges: { left, right: BC.wall, bottom: BC.wall, top: BC.wall },
    },
    cfl: 0.4,
  });
  solver.initialize();
  return { solver, params, setQCurve, BC };
}

function interiorMass(solver) {
  const { Nx, nx, ny } = solver;
  const h = solver.Q[0];
  let m = 0;
  for (let j = 1; j <= ny; j++)
    for (let i = 1; i <= nx; i++) m += h[j * Nx + i];
  return m;
}

function allFinite(solver) {
  for (const arr of solver.Q)
    for (let k = 0; k < arr.length; k++)
      if (!Number.isFinite(arr[k])) return false;
  return true;
}

test("generated kernels load and expose the Model2D contract", () => {
  const { model } = createSWE2DModel();
  assert.equal(model.nVars, 3);
  assert.equal(model.dimension, 2);
  for (const fn of ["flux", "numericalFlux", "eigenvalues", "boundaryConditions"])
    assert.equal(typeof model[fn], "function", `missing ${fn}`);
  // flux of a still pond: F_x = [hu, hu²/h+½gh², huv/h] = [0, ½gh², 0]
  const F = model.flux(Float64Array.of(2.0, 0, 0), new Float64Array(0), model.params);
  assert.ok(Math.abs(F[0]) < 1e-12 && Math.abs(F[1]) < 1e-12);
  assert.ok(F[2] > 0, "expected nonzero hydrostatic x-momentum flux");
});

test("closed domain with q=0 inflow stays at rest", () => {
  const { solver, setQCurve } = makeSolver({ inflowRows: [8, 16] });
  setQCurve([[0, 0], [60, 0]]); // no inflow
  solver.Q[0].fill(0.1); // uniform depth = h_in

  let t = 0;
  for (let s = 0; s < 50; s++) t += solver.step(t);

  assert.ok(allFinite(solver));
  const { Nx, nx, ny } = solver;
  let maxMom = 0;
  for (let j = 1; j <= ny; j++)
    for (let i = 1; i <= nx; i++) {
      const c = j * Nx + i;
      maxMom = Math.max(maxMom, Math.abs(solver.Q[1][c]), Math.abs(solver.Q[2][c]));
    }
  assert.ok(maxMom < 1e-6, `q=0 inflow should not stir the pond (got ${maxMom})`);
});

test("active inflow pushes water into the domain", () => {
  const { solver, setQCurve } = makeSolver({ inflowRows: [8, 16] });
  setQCurve([[0, 1], [60, 1]]); // full inflow
  solver.Q[0].fill(0.05); // start below h_in so the inflow blend pulls it up

  const m0 = interiorMass(solver);
  let t = 0;
  for (let s = 0; s < 200; s++) t += solver.step(t);

  assert.ok(allFinite(solver), "solver produced non-finite values");
  const m1 = interiorMass(solver);
  assert.ok(m1 > m0 * 1.02,
    `inflow should raise interior mass (${m0} -> ${m1})`);
});

test("Q(t) curve modulates the inflow strength", () => {
  const run = (curve) => {
    const { solver, setQCurve } = makeSolver({ inflowRows: [8, 16] });
    setQCurve(curve);
    solver.Q[0].fill(0.05);
    const m0 = interiorMass(solver);
    let t = 0;
    for (let s = 0; s < 150; s++) t += solver.step(t);
    return interiorMass(solver) - m0;
  };
  const gainFull = run([[0, 1], [60, 1]]);
  const gainHalf = run([[0, 0.5], [60, 0.5]]);
  assert.ok(gainFull > gainHalf,
    `stronger Q(t) must inflow more water (full ${gainFull} vs half ${gainHalf})`);
  assert.ok(gainHalf > 0, "half-strength inflow should still add water");
});
