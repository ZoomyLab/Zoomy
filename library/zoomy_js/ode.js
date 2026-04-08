/**
 * zoomy_js/ode.js — Time integration schemes.
 *
 * Mirrors zoomy_core.fvm.ode (RK1, RK2) and the implicit local Newton
 * from zoomy_core.fvm.solver_imex_numpy.IMEXSourceSolver._implicit_source_local.
 *
 * Operator signature (matches zoomy_core):
 *   operator(dt, vars, dVars) → dVars
 *
 * For the structured 2D grid, vars and dVars are arrays of Float32Array.
 */

import { solveSmallLinear } from "./model.js";

/**
 * Forward Euler (RK1): Q_new = Q + dt * operator(Q).
 *
 * @param {function} operator  (dt, vars, dVars) → dVars
 * @param {Float32Array[]} vars  Current state (nVars × NN)
 * @param {number} dt
 * @returns {Float32Array[]} Updated state (new arrays)
 */
export function RK1(operator, vars, dt) {
  const nVars = vars.length, NN = vars[0].length;
  const dVars = Array.from({ length: nVars }, () => new Float32Array(NN));
  operator(dt, vars, dVars);
  const out = Array.from({ length: nVars }, () => new Float32Array(NN));
  for (let v = 0; v < nVars; v++)
    for (let k = 0; k < NN; k++)
      out[v][k] = vars[v][k] + dt * dVars[v][k];
  return out;
}

/**
 * Heun's method (RK2): Q_new = Q + 0.5*dt*(k1 + k2).
 *
 * @param {function} operator
 * @param {Float32Array[]} vars
 * @param {number} dt
 * @returns {Float32Array[]}
 */
export function RK2(operator, vars, dt) {
  const nVars = vars.length, NN = vars[0].length;
  const k1 = Array.from({ length: nVars }, () => new Float32Array(NN));
  operator(dt, vars, k1);

  // Q* = Q + dt * k1
  const qStar = Array.from({ length: nVars }, () => new Float32Array(NN));
  for (let v = 0; v < nVars; v++)
    for (let k = 0; k < NN; k++)
      qStar[v][k] = vars[v][k] + dt * k1[v][k];

  const k2 = Array.from({ length: nVars }, () => new Float32Array(NN));
  operator(dt, qStar, k2);

  const out = Array.from({ length: nVars }, () => new Float32Array(NN));
  for (let v = 0; v < nVars; v++)
    for (let k = 0; k < NN; k++)
      out[v][k] = vars[v][k] + 0.5 * dt * (k1[v][k] + k2[v][k]);
  return out;
}

/**
 * Local implicit Newton solve for source terms (cell-wise).
 *
 * Mirrors IMEXSourceSolver._implicit_source_local:
 *   For each cell, solve:  (I - dt * J) * d = -(Q - Qexp - dt * S)
 *   where J = dS/dQ (source Jacobian), iterating until convergence.
 *
 * @param {Model2D} model   Must implement source(vars, k, outS) and sourceJacobian(vars, k, outJ)
 * @param {Float32Array[]} varsExp  Explicit-step result (nVars × NN)
 * @param {number} dt
 * @param {object} opts
 * @param {number} opts.tol       Convergence tolerance (default 1e-8)
 * @param {number} opts.maxIter   Max Newton iterations (default 6)
 * @param {number} opts.NN        Number of cells
 * @returns {Float32Array[]} Implicitly-updated state
 */
export function implicitLocalNewton(model, varsExp, dt, { tol = 1e-8, maxIter = 6, NN }) {
  const nv = model.nVars;
  // Copy varsExp → Q (working copy)
  const Q = varsExp.map(v => new Float32Array(v));

  // Scratch per cell
  const S = new Float64Array(nv);
  const J = new Float64Array(nv * nv);
  const A = new Float64Array(nv * nv);  // I - dt*J
  const r = new Float64Array(nv);       // residual

  for (let iter = 0; iter < maxIter; iter++) {
    let maxCorr = 0;
    for (let k = 0; k < NN; k++) {
      // Evaluate source and Jacobian at current Q
      model.source(Q, k, S);
      model.sourceJacobian(Q, k, J);

      // Build system: A = I - dt*J,  r = -(Q - Qexp - dt*S)
      for (let i = 0; i < nv; i++) {
        r[i] = -(Q[i][k] - varsExp[i][k] - dt * S[i]);
        for (let j = 0; j < nv; j++) {
          A[i * nv + j] = (i === j ? 1 : 0) - dt * J[i * nv + j];
        }
      }

      // Solve A * d = r  (result in r)
      solveSmallLinear(nv, A, r);

      // Update Q and track convergence
      for (let i = 0; i < nv; i++) {
        Q[i][k] += r[i];
        const abs = Math.abs(r[i]);
        if (abs > maxCorr) maxCorr = abs;
      }
    }
    if (maxCorr < tol) break;
  }
  return Q;
}
