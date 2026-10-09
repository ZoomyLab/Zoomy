"""The solvers — plain Python functions over the vocabulary, written ONCE."""
import sympy as sp
from symsolve import While, Cond, Root, LinSolve, unpack
from vocabulary import (ghost, set_ghost, edge, reconstruct, face_map, div,
                        cell_map, reduce_max, reduce_any, select, comp,
                        assemble, lap, kernels as K, io, couple)


def rhs(Q, p, order, peer=None):                  # L(Q) = -div F*
    Qg = ghost(Q)
    if peer is not None:                          # coupled boundary
        Qg = set_ghost(Qg, *peer)
    QL, QR = unpack(reconstruct(Qg, order), 2)
    return -div(face_map(K.rusanov, QL, QR, p.g), p.dx)


def step(Q, dt, p, order=2, peer=None):           # SSP-RK + MOOD + IMEX
    Q0 = Q
    for a in ((0, sp.S.Half) if order == 2 else (0,)):
        Q = a * Q0 + (1 - a) * (Q + dt * rhs(Q, p, order, peer))
    if order == 2:
        bad = cell_map(K.troubled, Q, p.hb)
        Q = Cond(reduce_any(bad),
                 lambda: select(bad, Q0 + dt * rhs(Q0, p, 1, peer), Q),
                 lambda: Q)
    h, hu = comp(Q, 0), comp(Q, 1)
    hu = Root(lambda x: x - hu + dt * p.cf * x * abs(x) / h**2, hu)
    hu = LinSolve(lambda x: x - dt * p.nu * lap(x, p.dx), hu)
    return assemble(h, hu)


def cfl_dt(Q, p, fixed_dt):
    if fixed_dt is not None:
        return fixed_dt
    return p.cfl * p.dx / reduce_max(cell_map(K.wavespeed, Q, p.g))


def hyperbolic(Q, t, T, p, order=2, fixed_dt=None):
    def body(t, Q):
        dt = sp.Min(cfl_dt(Q, p, fixed_dt), T - t)
        Q = step(Q, dt, p, order)
        io.write(t + dt, Q)
        return t + dt, Q
    return While(lambda t, Q: t < T, body, (t, Q))


def hyperbolic_precice(Q, t, p, side, order=1, fixed_dt=None):
    def body(t, Q, go):
        dt = sp.Min(cfl_dt(Q, p, fixed_dt), couple.max_dt())
        peer = couple.read(dt)
        Q = step(Q, dt, p, order, peer=(side, peer))
        couple.write(edge(Q, side))
        go = couple.advance(dt)
        io.write(t + dt, Q)
        return t + dt, Q, go
    go = couple.init(edge(Q, side))
    return While(lambda t, Q, go: go, body, (t, Q, go))[:2]
