"""MalpassetSWE's spectrum at an Audusse star state bounds the star state's velocity.

The hydrostatic reconstruction keeps the cell velocity, `q* = q h*/h`; the Rusanov / CFL face
speed is the spectrum at `(h*, q*)`.  With the KP `hinv(h*)` of the physical operators that
speed fell far below `q*/h*` once `h* < wet_dry_eps` and a dry neighbour drained negative
(Malpasset thesis case, order 1, h_min -8.9e-3 m): the state below is that face.
"""
import sympy as sp

from zoomy_core.model.models import MalpassetSWE


def test_star_speed_bounds_star_velocity():
    m = MalpassetSWE(n=0.033, nu=1.0, eps=1e-2, h_friction_floor=0.5, ev_gate=False)
    v, a, p, n = m.variables, m.aux_variables, m.parameters, m.normal
    h, b, u = 2.602, 39.44, 4.32            # wet cell; dry neighbour's bed 42.04
    h_star = h + b - 42.04                  # 1.65e-3 < wet_dry_eps
    q_star = u * h * h_star / h
    kp = sp.sqrt(2) * h_star / sp.sqrt(h_star ** 4 + 1e-2 ** 4)
    subs = {v.b: 42.04, v.h: h_star, v.hu: q_star, v.hv: 0.0, a.hinv: kp,
            p.g: 9.81, p.wet_dry_eps: 1e-2, n.n0: 1.0, n.n1: 0.0}
    speed = max(abs(float(sp.N(e.subs(subs)))) for e in m.eigenvalues())
    assert speed >= u
