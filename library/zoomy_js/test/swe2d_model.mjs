/**
 * Test fixture: a Model2D built from the real zoomy_core codegen output
 * (see gen_fixtures.py). This is the thin app-side adapter that Phase C's
 * generate.py produces — it maps the generated snake_case kernels onto
 * the camelCase Model2D interface and supplies the one model-specific
 * hook the codegen pipeline does not emit (positivityFix).
 */

import * as K from "./fixtures/swe2d.generated.js";

const G = 9.81;
const WET = 1e-6;

/** Boundary tag index — the model was generated with a single "wall"
 *  tag, which BoundaryConditions sorts to index 0. */
export const BC_WALL = 0;

/** @type {import("../model.js").Model2D} */
export const swe2dModel = {
  nVars: 3,
  nAux: 0,
  dimension: 2,
  params: Float64Array.of(G),

  flux: K.flux,
  numericalFlux: K.numerical_flux,
  eigenvalues: K.eigenvalues,
  // Per-tag BC kernels indexed by tag — this fixture has a single
  // "wall" tag, so index 0 = bc_wall.
  boundaryConditions: [K.bc_wall],

  /** Wet/dry clamp — drop a near-dry cell to a true dry state. */
  positivityFix(q) {
    if (q[0] <= WET) {
      q[0] = 0;
      q[1] = 0;
      q[2] = 0;
    }
  },
};
