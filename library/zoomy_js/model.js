/**
 * zoomy_js/model.js — Model interface for 2D structured-grid FVM solvers.
 *
 * Mirrors zoomy_core.model.basemodel.Model.
 *
 * A model is a plain object implementing some or all of these methods.
 * The solver calls them via the model reference — no base class needed.
 *
 * @typedef {Object} Model2D
 *
 * Required (for any solver):
 * @property {number} nVars — number of conserved variables
 * @property {function(Float32Array[], number, Float32Array, Float32Array): void}
 *   flux(vars, k, outFx, outFy) — physical flux F_x, F_y at cell k
 * @property {function(Float32Array[], number): number}
 *   maxWaveSpeed(vars, k) — max |eigenvalue| at cell k (for CFL)
 * @property {function(Float32Array[], Float32Array): void}
 *   applyBC(vars, wall) — set ghost cells (boundary conditions)
 * @property {function(Float32Array[]): void}
 *   positivityFix(vars) — positivity / wet-dry fix
 * @property {function(Float32Array, number): boolean}
 *   isWall(wall, k) — true if cell k is an obstacle
 * @property {function(Float32Array[], number, number, number[]): void}
 *   wallReflect(vars, ki, kj, out) — reflected state at obstacle neighbour
 * @property {function(Float32Array[]): void}
 *   initConditions(vars) — set initial conditions
 *
 * Optional (for IMEX source splitting):
 * @property {function(Float32Array[], number, Float32Array): void}
 *   source(vars, k, out) — source term S(Q) at cell k (nVars components)
 * @property {function(Float32Array[], number, Float32Array): void}
 *   sourceJacobian(vars, k, outJ) — dS/dQ Jacobian at cell k (nVars×nVars flat)
 *
 * Optional (for non-conservative / path-conservative fluxes):
 * @property {function(Float32Array[], number, number[], Float32Array): void}
 *   quasilinearMatrix(vars, k, normal, outA) — A(Q)·n at cell k (nVars×nVars flat)
 */

/**
 * Helper: solve a small dense linear system A*x = b in-place.
 * A is nVars×nVars stored row-major flat, b is nVars. Result written to b.
 * Uses partial pivoting Gaussian elimination.
 *
 * @param {number} n — system size
 * @param {Float64Array} A — n×n matrix (row-major, MODIFIED)
 * @param {Float64Array} b — RHS vector (MODIFIED to contain solution)
 */
export function solveSmallLinear(n, A, b) {
  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxVal = Math.abs(A[col * n + col]), maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(A[row * n + col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxRow !== col) {
      for (let j = col; j < n; j++) {
        const tmp = A[col * n + j]; A[col * n + j] = A[maxRow * n + j]; A[maxRow * n + j] = tmp;
      }
      const tmp = b[col]; b[col] = b[maxRow]; b[maxRow] = tmp;
    }
    // Eliminate
    const pivot = A[col * n + col];
    if (Math.abs(pivot) < 1e-30) continue;
    for (let row = col + 1; row < n; row++) {
      const factor = A[row * n + col] / pivot;
      for (let j = col + 1; j < n; j++) A[row * n + j] -= factor * A[col * n + j];
      b[row] -= factor * b[col];
    }
  }
  // Back-substitute
  for (let row = n - 1; row >= 0; row--) {
    for (let j = row + 1; j < n; j++) b[row] -= A[row * n + j] * b[j];
    b[row] /= A[row * n + row];
  }
}
