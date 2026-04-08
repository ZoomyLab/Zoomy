/**
 * zoomy_js/flux.js — Conservative numerical flux schemes.
 *
 * Mirrors zoomy_core.fvm.flux (LaxFriedrichs, Rusanov).
 *
 * These are exported as classes with static methods for documentation
 * and testing. In the solver's hot loop the LF formula is inlined directly.
 */

/** Global Lax-Friedrichs flux. */
export class LaxFriedrichs {
  /**
   * F_num = 0.5 * (F_L + F_R - s * (Q_R - Q_L))
   *
   * @param {number} fL  Physical flux, left state
   * @param {number} fR  Physical flux, right state
   * @param {number} qL  Conserved variable, left state
   * @param {number} qR  Conserved variable, right state
   * @param {number} s   Global max wave speed
   * @returns {number}
   */
  static numericalFlux(fL, fR, qL, qR, s) {
    return 0.5 * (fL + fR - s * (qR - qL));
  }
}

/** Local Lax-Friedrichs (Rusanov) flux. */
export class Rusanov {
  /**
   * F_num = 0.5 * (F_L + F_R - max(|λ_L|, |λ_R|) * (Q_R - Q_L))
   *
   * @param {number} fL  Physical flux, left state
   * @param {number} fR  Physical flux, right state
   * @param {number} qL  Conserved variable, left state
   * @param {number} qR  Conserved variable, right state
   * @param {number} sL  Max |eigenvalue|, left
   * @param {number} sR  Max |eigenvalue|, right
   * @returns {number}
   */
  static numericalFlux(fL, fR, qL, qR, sL, sR) {
    const s = Math.max(sL, sR);
    return 0.5 * (fL + fR - s * (qR - qL));
  }
}
