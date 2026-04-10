/**
 * zoomy_js/ncflux.js — Non-conservative (path-conservative) flux schemes.
 *
 * Mirrors zoomy_core.fvm.nonconservative_flux.Rusanov.
 *
 * For models with non-conservative products B(Q)·∂Q/∂x, the numerical flux
 * uses path integration of the quasilinear matrix A(Q) from Q_i to Q_j
 * via Legendre-Gauss quadrature.
 *
 * Fluctuations:
 *   Dp = 0.5 * (A_int + sMax·I) * (Q_j - Q_i)   (outgoing from j)
 *   Dm = 0.5 * (A_int - sMax·I) * (Q_j - Q_i)   (incoming to i)
 */

/**
 * Legendre-Gauss quadrature nodes and weights on [0, 1].
 * Pre-computed for orders 1–4.
 */
const GAUSS = {
  1: { x: [0.5], w: [1.0] },
  2: { x: [0.2113248654, 0.7886751346], w: [0.5, 0.5] },
  3: { x: [0.1127016654, 0.5, 0.8872983346], w: [0.2777777778, 0.4444444444, 0.2777777778] },
  4: { x: [0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558],
       w: [0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226] },
};

/**
 * Rusanov-type non-conservative flux with path integration.
 */
export class RusanovNC {
  /**
   * @param {object} opts
   * @param {number} opts.quadOrder  Gauss quadrature order (1–4, default 3)
   */
  constructor({ quadOrder = 3 } = {}) {
    const q = GAUSS[quadOrder];
    if (!q) throw new Error(`Unsupported quadOrder ${quadOrder}. Use 1–4.`);
    this.quadX = q.x;
    this.quadW = q.w;
  }

  /**
   * Compute fluctuations (Dp, Dm) for one face.
   *
   * Requires model.quasilinearMatrix(qPath, normal, outA) to fill outA
   * with the normal-projected quasilinear matrix A(Q)·n (nVars×nVars, row-major flat).
   *
   * @param {Model2D} model
   * @param {number[]} qi     State at cell i (nVars)
   * @param {number[]} qj     State at cell j (nVars)
   * @param {number[]} normal Face normal [nx, ny]
   * @param {number}   sMax   Global max wave speed
   * @param {Float64Array} outDp  Fluctuation Dp (nVars, written)
   * @param {Float64Array} outDm  Fluctuation Dm (nVars, written)
   */
  fluctuations(model, qi, qj, normal, sMax, outDp, outDm) {
    const nv = model.nVars;
    const nn = nv * nv;

    // Scratch: path-integrated A matrix (nv×nv) and per-point A
    if (!this._Aint) {
      this._Aint = new Float64Array(nn);
      this._Apt  = new Float64Array(nn);
      this._qPath = new Float64Array(nv);
    }
    const Aint = this._Aint, Apt = this._Apt, qPath = this._qPath;

    // dQ = qj - qi
    Aint.fill(0);
    for (let p = 0; p < this.quadX.length; p++) {
      const xi = this.quadX[p], wt = this.quadW[p];
      // q along path: qi + xi*(qj - qi)
      for (let v = 0; v < nv; v++) qPath[v] = qi[v] + xi * (qj[v] - qi[v]);
      model.quasilinearMatrix(qPath, normal, Apt);
      for (let m = 0; m < nn; m++) Aint[m] += wt * Apt[m];
    }

    // Dp[v] = 0.5 * sum_u (Aint[v,u] + sMax*delta(v,u)) * dQ[u]
    // Dm[v] = 0.5 * sum_u (Aint[v,u] - sMax*delta(v,u)) * dQ[u]
    for (let v = 0; v < nv; v++) {
      let dp = 0, dm = 0;
      for (let u = 0; u < nv; u++) {
        const dQ = qj[u] - qi[u];
        const a = Aint[v * nv + u];
        const eye = (v === u) ? sMax : 0;
        dp += (a + eye) * dQ;
        dm += (a - eye) * dQ;
      }
      outDp[v] = 0.5 * dp;
      outDm[v] = 0.5 * dm;
    }
  }
}
