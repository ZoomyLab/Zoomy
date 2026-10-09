import sys
import sympy as sp
from zoomy_core.symsolver import chorin as CH
from zoomy_core.symsolver.core import LinSolve
from zoomy_core.symsolver.hyperbolic import full_aux
from zoomy_core.symsolver.vocabulary import cells, derivative_aux, put, zeros
v = sys.argv[1]
if v == "noghost":
    def pressure(press, Q, p, time, dt, mesh):
        rows = tuple(press.equation_to_state_index)
        Qaux = full_aux(press, Q, zeros(len(press.aux_state), mesh), p, time, dt, mesh)
        def R(P):
            Y = put(Q, rows, P)
            return cells(press.call.source, Y, derivative_aux(press.call.aux_derivatives, Y, Qaux, mesh), p, time, dt, mesh)
        b = R(sp.S.Zero)
        return put(Q, rows, LinSolve(lambda P: R(P) - b, -b))
    CH.pressure = pressure
if v == "nopressure":
    CH.pressure = lambda press, Q, p, time, dt, mesh: Q
if v == "nocorrect":
    CH.correct = lambda corr, Q, p, time, dt, mesh: Q
sys.argv = ["x"]
import profile_chorin
