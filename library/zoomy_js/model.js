/**
 * zoomy_js/model.js — Model2D interface for the generic FVM solver.
 *
 * A Model2D bundles the kernels generated from a zoomy_core SystemModel
 * (via zoomy_core.transformation.to_js — JsModel / JsNumerics) with the
 * metadata the solver needs. The solver is physics-agnostic: it only
 * ever calls the functions on this object.
 *
 * Kernel signatures match the JsModel / JsNumerics codegen output:
 *
 * @typedef {Object} Model2D
 * @property {number} nVars       conserved variables
 * @property {number} nAux        auxiliary variables (static, e.g. bathymetry)
 * @property {number} dimension   spatial dimension (2 for the grid solver)
 * @property {Float64Array} params  model parameter vector `p`
 *
 * @property {(Q:Float64Array, Qaux:Float64Array, p:Float64Array) => Float64Array} flux
 *   Physical flux tensor, row-major (nVars, dimension).
 * @property {(Qm:Float64Array, Qp:Float64Array, Qauxm:Float64Array,
 *             Qauxp:Float64Array, p:Float64Array, n:Float64Array) => Float64Array} numericalFlux
 *   Numerical (Riemann) flux across a face with unit normal `n`; length nVars.
 * @property {(Q:Float64Array, Qaux:Float64Array, p:Float64Array,
 *             n:Float64Array) => Float64Array} eigenvalues
 *   Eigenvalues of the normal-projected system; length nVars.
 * @property {(bcIdx:number, time:number, X:Float64Array, dX:number,
 *             Q:Float64Array, Qaux:Float64Array, p:Float64Array,
 *             n:Float64Array) => Float64Array} boundaryConditions
 *   Boundary-side state for tag `bcIdx`; length nVars.
 * @property {(q:Float64Array) => void} [positivityFix]
 *   Optional in-place positivity / wet-dry clamp on a single cell state.
 *   The one model-specific hook the codegen pipeline does not (yet)
 *   produce; supply it from the app for free-surface models.
 */

/**
 * Max |eigenvalue| at a cell over the axis-aligned normals — the per-cell
 * signal speed used for the CFL time-step estimate.
 *
 * @param {Model2D} model
 * @param {Float64Array} q     cell state, length nVars
 * @param {Float64Array} qaux  cell aux state, length nAux
 * @returns {number}
 */
export function cellMaxWaveSpeed(model, q, qaux) {
  const { params, dimension } = model;
  const n = new Float64Array(dimension);
  let s = 0;
  for (let d = 0; d < dimension; d++) {
    n.fill(0);
    n[d] = 1;
    const ev = model.eigenvalues(q, qaux, params, n);
    for (let v = 0; v < ev.length; v++) {
      const a = Math.abs(ev[v]);
      if (a > s) s = a;
    }
  }
  return s;
}
