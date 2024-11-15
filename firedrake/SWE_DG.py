from firedrake import *
from copy import deepcopy

from mpi4py import MPI
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 100
mesh = UnitSquareMesh(N, N, quadrilateral=True)

V = FunctionSpace(mesh, "DG", 0)
Vout = FunctionSpace(mesh, "DG", 0)
W = VectorFunctionSpace(mesh, "DG", 0)
Wout = VectorFunctionSpace(mesh, "DG", 0)

VW = V * W
WV = W * V


x, y = SpatialCoordinate(mesh)

x0 = 0.3
y0 = 0.5
x1 = 0.7
y1 = 0.5

width = 0.05
height = conditional(Or(And(And(x < x0+width, x > x0-width), And(y < y0+width, y > y0-width) ), And(And(x < x1+width, x > x1-width), And(y < y1+width, y > y1-width) )), 8.0, 1.0)
velocity = conditional(Or(And(And(x < x0+width, x > x0-width), And(y < y0+width, y > y0-width) ), And(And(x < x1+width, x > x1-width), And(y < y1+width, y > y1-width) )), as_vector((0.0, 0.0)) , as_vector((0., 0.)))

IC = Function(VW, name="IC")
IC.sub(0).assign(Function(VW.sub(0)).interpolate(height))
IC.sub(1).assign(Function(VW.sub(1)).interpolate(height * velocity))

Q = Function(VW, name="Q")
Q_ = Function(VW, name="Qold")
Q.assign(IC)
Q_.assign(IC)

h, hu = split(Q)
h_, hu_ = split(Q_)

v, w = TestFunctions(VW)
v_, w_ = TrialFunctions(VW)
a = inner(w, w_) * dx + inner(v, v_) * dx

T = 0.01
CFL = 0.45
incircle = (1./N)
g = 9.81

n = FacetNormal(mesh)

I = Identity(2)
P = lambda h, hu: sqrt(0.5 * g * h * h) * I
ev = lambda h, hu, n: abs(dot(hu, n)/h) + sqrt(g * h)

h_D = h
hu_D = hu - 2 * dot(hu, n) * n
hun_D = dot(hu_D, n)

flux_h = lambda h, hu, n : dot(hu, n)
flux_hu = lambda h, hu, n: dot(outer(hu, hu) / h + P(h, hu), n)

p = '+'
m = '-'
F_H_l = (hu(p))
F_H_r = (hu(m))
F_H_n = 0.5*dot((F_H_l + F_H_r), n(m))
F_dis_H = -0.5  * avg(ev(h, hu, n)) * (h(p)-h(m))
F_H = (v(p)-v(m)) * (F_H_n + F_dis_H) * dS

F_HU_l = outer(hu(p), hu(p)) / h(p) + 0.5 * g * h(p) * h(p) * I
F_HU_r = outer(hu(m), hu(m)) / h(m) + 0.5 * g * h(m) * h(m) * I
F_HU_n = 0.5*dot((F_HU_l + F_HU_r), n(m))
F_dis_HU = -0.5 * avg(ev(h, hu, n)) * (hu(p)-hu(m))
F_HU = dot((w(p)-w(m)), (F_HU_n + F_dis_HU)) * dS

h_g = h
hu_n = dot(hu, n) * n
hu_t = hu - hu_n
hu_g = -hu_n + hu_t

BC_H_l = hu
BC_H_r = hu_g
BC_H_n = 0.5*dot((BC_H_l + BC_H_r), n)
BC_dis_H = -0.5 * ev(h, hu, n) * (h_g - h)
BC_H = -v*(BC_H_n + BC_dis_H) * ds

BC_HU_l = outer(hu, hu) / h + 0.5 * g * h * h * I
BC_HU_r = outer(hu_g, hu_g) / h_g + 0.5 * g * h_g * h_g * I
BC_HU_n = 0.5*dot((BC_HU_l + BC_HU_r), n)
BC_dis_HU = -0.5 * ev(h, hu, n) * (hu_g - hu)
BC_HU = -dot(w, (BC_HU_n + BC_dis_HU)) * ds


t = 0.0
print("t=", t)
step = 0
output_freq = 1

outfile = VTKFile("swe.pvd")
outfile.write(project(h, V, name="H"), project(hu, W, name="HU"), time=t)

h0 = deepcopy(h)


ev_cell = lambda h, hu: abs(hu[0]) / h + sqrt(g * h)
ev_max = max(project(ev_cell(h, hu), V).vector())
dt = CFL * incircle / ev_max
L = dt * (F_H + F_HU + BC_H + BC_HU)

params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
dQ = Function(VW)
prob = LinearVariationalProblem(a, L, dQ)
solv = LinearVariationalSolver(prob, solver_parameters=params)
while t < T - 0.5 * dt:
    ev_max = max(project(ev_cell(h, hu), V).vector())
    dt = CFL * incircle / ev_max
    # solve(F == 0, Q)
    solv.solve()
    Q.assign(Q + dQ)
    Q_.assign(Q)

    step += 1
    t += dt

    if step % output_freq == 0:
        max_h = project(h, V).vector().max()
        error = assemble((abs(h-h0))*dx)
        outfile.write(project(h, V, name="H"), project(hu, W, name="HU"), time=t)
        print(f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L1={error}, h_max={max_h}")
