/**
 * zoomy_js/model.js — Model2D interface for the generic FVM solver.
 *
 * A Model2D bundles the kernels generated from a zoomy_core SystemModel
 * (via zoomy_core.transformation.to_js — JsModel / JsNumerics) with the
 * metadata the solver needs. The solver is physics-agnostic: it only
 * ever calls the functions on this object.
 *
 * The generated kernels use the **out-parameter** convention: each
 * takes a trailing `res` array the caller owns and writes into it,
 * rather than allocating and returning a Float64Array. That keeps the
 * solver's hot loop allocation-free.
 *
 * @typedef {Object} Model2D
 * @property {number} nVars       conserved variables
 * @property {number} nAux        auxiliary variables (static, e.g. bathymetry)
 * @property {number} dimension   spatial dimension (2 for the grid solver)
 * @property {Float64Array} params  model parameter vector `p`
 *
 * @property {(Q:Float64Array, Qaux:Float64Array, p:Float64Array,
 *             res:Float64Array) => void} flux
 *   Physical flux tensor into `res`, row-major (nVars, dimension).
 * @property {(Qm:Float64Array, Qp:Float64Array, Qauxm:Float64Array,
 *             Qauxp:Float64Array, p:Float64Array, n:Float64Array,
 *             res:Float64Array) => void} numericalFlux
 *   Numerical (Riemann) flux across a face with unit normal `n` into
 *   `res` (length nVars).
 * @property {(Q:Float64Array, Qaux:Float64Array, p:Float64Array,
 *             n:Float64Array, res:Float64Array) => void} eigenvalues
 *   Eigenvalues of the normal-projected system into `res` (length nVars).
 * @property {(bcIdx:number, time:number, X:Float64Array, dX:number,
 *             Q:Float64Array, Qaux:Float64Array, p:Float64Array,
 *             n:Float64Array, res:Float64Array) => void} boundaryConditions
 *   Boundary-side state for tag `bcIdx` into `res` (length nVars).
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
 * @param {Float64Array} q      cell state, length nVars
 * @param {Float64Array} qaux   cell aux state, length nAux
 * @param {Float64Array} nScratch  reusable normal buffer, length dimension
 * @param {Float64Array} evScratch reusable eigenvalue buffer, length nVars
 * @returns {number}
 */
export function cellMaxWaveSpeed(model, q, qaux, nScratch, evScratch) {
  const { params, dimension } = model;
  let s = 0;
  for (let d = 0; d < dimension; d++) {
    nScratch.fill(0);
    nScratch[d] = 1;
    model.eigenvalues(q, qaux, params, nScratch, evScratch);
    for (let v = 0; v < evScratch.length; v++) {
      const a = Math.abs(evScratch[v]);
      if (a > s) s = a;
    }
  }
  return s;
}
