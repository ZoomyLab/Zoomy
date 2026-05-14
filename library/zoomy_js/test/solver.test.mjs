/**
 * Node tests for the generic HyperbolicSolver2D (solver.js).
 *
 * The solver is exercised through the *real* codegen kernels (the
 * MiniSWE + HLLC + Wall fixture from gen_fixtures.py), so these tests
 * cover the whole "generated building blocks → hand-written glue"
 * contract: MUSCL reconstruction, the injected numerical flux, Heun
 * time-stepping, and boundary dispatch through the generated BC kernel.
 *
 * Run:  node --test test/   (from library/zoomy_js)
 */

import test from "node:test";
import assert from "node:assert/strict";

import { HyperbolicSolver2D } from "../solver.js";
import { swe2dModel, BC_WALL } from "./swe2d_model.mjs";

// performance.now() — present in modern Node, but guard just in case.
if (typeof performance === "undefined") {
  globalThis.performance = { now: () => Number(process.hrtime.bigint()) / 1e6 };
}

/** A closed (all-wall) solver on an nx×nx grid. */
function makeClosedSolver(nx = 20) {
  const solver = new HyperbolicSolver2D({
    nx,
    ny: nx,
    dx: 1.0 / (nx + 2),
    model: swe2dModel,
    bc: {
      wall: BC_WALL,
      edges: { left: BC_WALL, right: BC_WALL, bottom: BC_WALL, top: BC_WALL },
    },
    cfl: 0.4,
  });
  solver.initialize();
  return solver;
}

/** Sum of h over interior cells. */
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
    for (let k = 0; k < arr.length; k++) if (!Number.isFinite(arr[k])) return false;
  return true;
}

test("lake at rest stays at rest", () => {
  const solver = makeClosedSolver(20);
  solver.Q[0].fill(1.0); // uniform depth, zero momentum

  let t = 0;
  for (let s = 0; s < 60; s++) t += solver.step(t);

  const { Nx, nx, ny } = solver;
  let maxMom = 0,
    maxDh = 0;
  for (let j = 1; j <= ny; j++)
    for (let i = 1; i <= nx; i++) {
      const c = j * Nx + i;
      maxMom = Math.max(maxMom, Math.abs(solver.Q[1][c]), Math.abs(solver.Q[2][c]));
      maxDh = Math.max(maxDh, Math.abs(solver.Q[0][c] - 1.0));
    }
  assert.ok(maxMom < 1e-9, `spurious momentum ${maxMom}`);
  assert.ok(maxDh < 1e-9, `depth drifted ${maxDh}`);
});

test("mass is conserved on a closed domain", () => {
  const solver = makeClosedSolver(24);
  // Smooth gaussian bump — stays wet, so no positivity clamping.
  const { Nx, nx, ny } = solver;
  const h = solver.Q[0];
  for (let j = 1; j <= ny; j++)
    for (let i = 1; i <= nx; i++) {
      const x = (i - nx / 2) / nx,
        y = (j - ny / 2) / ny;
      h[j * Nx + i] = 1.0 + 0.3 * Math.exp(-40 * (x * x + y * y));
    }

  const m0 = interiorMass(solver);
  let t = 0;
  for (let s = 0; s < 120; s++) t += solver.step(t);

  assert.ok(allFinite(solver), "solver produced non-finite values");
  const m1 = interiorMass(solver);
  // Interior face fluxes telescope exactly — the FV glue is conservative
  // by construction. The only mass exchange is across the ghost-cell
  // reflective walls, which are not perfectly conservative; that error
  // is small and *bounded* (it oscillates as waves slosh, it does not
  // accumulate), so a loose ceiling is the meaningful check here.
  const relErr = Math.abs(m1 - m0) / m0;
  assert.ok(relErr < 5e-3, `mass leak too large: ${relErr} (${m0} -> ${m1})`);
});

test("an x-symmetric dam break stays x-symmetric", () => {
  const solver = makeClosedSolver(24);
  const { Nx, nx, ny } = solver;
  const h = solver.Q[0];
  // Tall column down the middle, symmetric about the vertical centreline.
  for (let j = 1; j <= ny; j++)
    for (let i = 1; i <= nx; i++) {
      const d = Math.abs(i - (nx + 1) / 2);
      h[j * Nx + i] = d < nx / 6 ? 1.5 : 0.6;
    }

  let t = 0;
  for (let s = 0; s < 60; s++) t += solver.step(t);
  assert.ok(allFinite(solver), "solver produced non-finite values");

  // Mirror i -> nx+1-i must match for every variable (hu flips sign).
  let maxAsym = 0;
  for (let j = 1; j <= ny; j++)
    for (let i = 1; i <= nx; i++) {
      const c = j * Nx + i;
      const cM = j * Nx + (nx + 1 - i);
      maxAsym = Math.max(
        maxAsym,
        Math.abs(solver.Q[0][c] - solver.Q[0][cM]),
        Math.abs(solver.Q[1][c] + solver.Q[1][cM]),
        Math.abs(solver.Q[2][c] - solver.Q[2][cM])
      );
    }
  assert.ok(maxAsym < 1e-9, `symmetry broken by ${maxAsym}`);
});

test("internal wall mask reflects flow without NaNs", () => {
  const solver = makeClosedSolver(24);
  const { Nx, nx, ny } = solver;
  solver.Q[0].fill(1.0);
  // A solid block obstacle in the interior.
  for (let j = 8; j <= 14; j++)
    for (let i = 8; i <= 14; i++) solver.wall[j * Nx + i] = 1;
  // Give the flow a kick.
  for (let j = 1; j <= ny; j++)
    for (let i = 1; i <= 4; i++) solver.Q[1][j * Nx + i] = 0.5;

  let t = 0;
  for (let s = 0; s < 80; s++) t += solver.step(t);

  assert.ok(allFinite(solver), "solver produced non-finite values");
  // Wall cells must carry no flow.
  for (let j = 8; j <= 14; j++)
    for (let i = 8; i <= 14; i++) {
      const c = j * Nx + i;
      assert.equal(solver.Q[1][c], 0, "momentum leaked into a wall cell");
    }
});
