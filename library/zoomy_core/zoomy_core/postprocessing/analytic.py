"""Analytic reference solutions used in comparison plots and gifs.

Moved here from `library/zoomy_foam/tools/compare_stoker.py` so case
scripts can import them without path hacks (the foam tool re-exports).
"""
from __future__ import annotations

import numpy as np

__all__ = ["stoker"]


def stoker(x: np.ndarray, t: float, h_L: float, h_R: float, x0: float,
           g: float = 9.81):
    """Stoker (1957) wet-wet shallow-water dam-break solution.

    h_L > h_R > 0, u_L = u_R = 0, flat bed.  Closed-form rarefaction
    (left) + contact + shock (right).  Returns `(h, u)` at points
    `x`, time `t > 0`.
    """
    from scipy.optimize import brentq

    c_L = np.sqrt(g * h_L)

    # Middle state h_M from the shock+rarefaction matching condition.
    # Shock branch: u_M(h_M) = (h_M - h_R) * sqrt(g/2 * (h_M+h_R)/(h_M*h_R))
    # Rarefaction (left) branch: u_M(h_M) = 2*(c_L - sqrt(g*h_M))
    def _residual(h_M):
        c_M = np.sqrt(g * h_M)
        u_raref = 2.0 * (c_L - c_M)
        u_shock = (h_M - h_R) * np.sqrt(0.5 * g * (h_M + h_R) / (h_M * h_R))
        return u_raref - u_shock

    # h_M lies in (h_R, h_L).
    h_M = brentq(_residual, h_R + 1e-12, h_L - 1e-12, xtol=1e-14)
    c_M = np.sqrt(g * h_M)
    u_M = 2.0 * (c_L - c_M)

    # Wave speeds (in lab frame; u_L = 0 so absolute = relative-to-L)
    s_HL = -c_L                      # left rarefaction head
    s_TL = u_M - c_M                 # left rarefaction tail
    s_S = u_M * h_M / (h_M - h_R)    # shock

    h = np.empty_like(x, dtype=float)
    u = np.empty_like(x, dtype=float)
    xi = (x - x0) / max(t, 1e-12)

    upstream = xi < s_HL
    h[upstream] = h_L
    u[upstream] = 0.0

    fan = (xi >= s_HL) & (xi < s_TL)
    u[fan] = 2.0 / 3.0 * (c_L + xi[fan])
    c_fan = (2.0 * c_L - xi[fan]) / 3.0
    h[fan] = c_fan ** 2 / g

    middle = (xi >= s_TL) & (xi < s_S)
    h[middle] = h_M
    u[middle] = u_M

    downstream = xi >= s_S
    h[downstream] = h_R
    u[downstream] = 0.0

    return h, u
