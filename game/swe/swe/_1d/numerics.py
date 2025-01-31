import jax.numpy as np

def lax_friedrichs(Ql, Qr, Fl, Fr, max_speed):
    dissipation = max_speed * (Qr - Ql)
    return 0.5 * (Fl + Fr - dissipation)

def roe(Ql, Qr, Fl, Fr, sign) :
    Qm = 1/2 * (Ql + Qr)
    h = Qm[0]
    u =  np.where(h > 0, Qm[1] / h, 0)
    b = Qm[2]
    g = 9.81
    qr = Qr.copy()
    ql = Ql.copy()
    qr = qr.at[2].set(np.where(Qr[2] > Ql[0] + Ql[2], Ql[0] + Ql[2], Qr[2]))
    ql = ql.at[2].set(np.where(Ql[2] > Qr[0] + Qr[2], Qr[0] + Qr[2], Ql[2]))
    c = np.sqrt(g * h)
    zeros = np.zeros_like(h)
    ones = np.ones_like(h)
    eps = np.ones_like(h) * (10**-10)
    A = np.array([[zeros, ones , zeros], [-u**2 + g * h, 2 * u, g * h], [zeros, zeros, zeros]])
    Aabs = np.array([[((c - u)*abs(c + u) + (c + u)*abs(c - u))/(2*c + eps), (-abs(c - u) + abs(c + u))/(2*c + eps), c*((c - u)*abs(c + u) + (c + u)*abs(c - u))/(2*(c**2 - u**2) + eps)], [(c - u)*(c + u)*(-abs(c - u) + abs(c + u))/(2*c + eps), ((c - u)*abs(c - u) + (c + u)*abs(c + u))/(2*c + eps), c*(-abs(c - u) + abs(c + u)) /2], [zeros, zeros, zeros]])
    return 0.5 * np.einsum('ij..., j...->i...', A + sign * Aabs, (qr-ql))
