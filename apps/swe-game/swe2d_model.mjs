/**
 * apps/swe-game/swe2d_model.mjs — app-side Model2D adapter.
 *
 * The thin adapter between the codegen pipeline and the generic
 * solver: it maps the snake_case kernels emitted by generate.py onto
 * the camelCase Model2D interface that library/zoomy_js expects,
 * supplies the one model-specific hook codegen does not emit
 * (positivityFix), and owns the runtime Q(t) → parameter conversion.
 *
 * The Q(t) editor curve is *not* baked into the kernels — the model
 * carries a fixed 20-point timeline as parameters; `setQCurve` samples
 * the live editor curve at those fixed times and writes the values
 * into the parameter array. Editing the curve = rewriting that slice.
 */

import * as K from "./generated/swe2d.js";
import META from "./generated/swe2d_meta.json" with { type: "json" };

const WET = 1e-6;

/** Boundary-tag → index map (BoundaryConditions sorts tags). */
export const BC = Object.fromEntries(
  META.boundaryTags.map((tag, i) => [tag, i])
);

/** Piecewise-linear evaluation of an editor curve [[t,v],…] at time t. */
function lerpCurve(points, t) {
  if (t <= points[0][0]) return points[0][1];
  const last = points[points.length - 1];
  if (t >= last[0]) return last[1];
  for (let i = 0; i < points.length - 1; i++) {
    const [t0, v0] = points[i];
    const [t1, v1] = points[i + 1];
    if (t >= t0 && t <= t1) {
      const u = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
      return v0 * (1 - u) + v1 * u;
    }
  }
  return last[1];
}

/**
 * Build the SWE Model2D plus its parameter array and the runtime
 * helpers the game needs.
 *
 * @returns {{
 *   model: import("../../library/zoomy_js/model.js").Model2D,
 *   params: Float64Array,
 *   setQCurve: (points: number[][]) => void,
 *   BC: Record<string, number>,
 * }}
 */
export function createSWE2DModel() {
  const params = Float64Array.from(META.parameterDefaults);

  const model = {
    nVars: META.nVars,
    nAux: META.nAux,
    dimension: META.dimension,
    params,
    flux: K.flux,
    numericalFlux: K.numerical_flux,
    eigenvalues: K.eigenvalues,
    // Per-tag boundary kernels, indexed by tag (BoundaryConditions
    // sorts tags) — bcIdx 0/1/2 ↦ bc_inflow / bc_outflow / bc_wall.
    // No symbolic Piecewise: the solver indexes this array directly.
    boundaryConditions: META.boundaryTags.map((tag) => K["bc_" + tag]),
    /** Wet/dry clamp — the one model-specific hook codegen doesn't emit. */
    positivityFix(q) {
      if (q[0] <= WET) {
        q[0] = 0;
        q[1] = 0;
        q[2] = 0;
      }
    },
  };

  /**
   * Write the live Q(t) editor curve into the timeline parameter
   * block. The 20 timeline *times* stay at their fixed defaults; only
   * the *values* are updated — the editor curve is sampled at each
   * fixed time. This sidesteps duplicate-time degeneracies in the
   * symbolic interpolation and means editing the curve only rewrites
   * a contiguous 20-float slice of `params`.
   *
   * @param {number[][]} editorPoints  sorted [[t,v],…]; ≥1 point.
   */
  function setQCurve(editorPoints) {
    const { nPoints, timeStart, valueStart } = META.timelineBlock;
    const pts =
      editorPoints && editorPoints.length >= 1
        ? editorPoints
        : [[0, 1], [params[timeStart + nPoints - 1], 1]];
    const curve = pts.length >= 2 ? pts : [pts[0], pts[0]];
    for (let i = 0; i < nPoints; i++) {
      params[valueStart + i] = lerpCurve(curve, params[timeStart + i]);
    }
  }

  return { model, params, setQCurve, BC };
}
