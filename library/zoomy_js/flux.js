/**
 * zoomy_js/flux.js — Numerical flux schemes for the FVM solver.
 *
 * Mirrors zoomy_core.fvm.flux (LaxFriedrichs, Rusanov).
 *
 * In the hot solver loop these are inlined for performance. This module
 * exports them as standalone functions for testing and documentation.
 */

/**
 * Lax-Friedrichs (global) numerical flux.
 *
 *   F_num = 0.5 * (F_L + F_R - s * (Q_R - Q_L))
 *
 * @param {number} fL  — physical flux at left state
 * @param {number} fR  — physical flux at right state
 * @param {number} qL  — conserved variable at left state
 * @param {number} qR  — conserved variable at right state
 * @param {number} s   — global max wave speed (dissipation coefficient)
 * @returns {number} numerical flux
 */
export function laxFriedrichs(fL, fR, qL, qR, s) {
  return 0.5 * (fL + fR - s * (qR - qL));
}

/**
 * Rusanov (local Lax-Friedrichs) numerical flux.
 *
 *   F_num = 0.5 * (F_L + F_R - max(|λ_L|, |λ_R|) * (Q_R - Q_L))
 *
 * @param {number} fL   — physical flux at left state
 * @param {number} fR   — physical flux at right state
 * @param {number} qL   — conserved variable at left state
 * @param {number} qR   — conserved variable at right state
 * @param {number} sL   — max |eigenvalue| at left state
 * @param {number} sR   — max |eigenvalue| at right state
 * @returns {number} numerical flux
 */
export function rusanov(fL, fR, qL, qR, sL, sR) {
  const s = Math.max(sL, sR);
  return 0.5 * (fL + fR - s * (qR - qL));
}
