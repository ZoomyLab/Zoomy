from firedrake import *
from copy import deepcopy

from mpi4py import MPI
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 40
mesh = UnitSquareMesh(N, N, quadrilateral=True)

V = FunctionSpace(mesh, "DG", 1)
Vout = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "DG", 1)
Wout = VectorFunctionSpace(mesh, "DG", 1)

VW = V * W
WV = W * V


x, y = SpatialCoordinate(mesh)

x0 = 0.3
y0 = 0.5
x1 = 0.7
y1 = 0.5

width = 0.05
height = conditional(Or(And(And(x < x0+width, x > x0-width), And(y < y0+width, y > y0-width) ), And(And(x < x1+width, x > x1-width), And(y < y1+width, y > y1-width) )), 8.0, 5.0)
velocity = conditional(Or(And(And(x < x0+width, x > x0-width), And(y < y0+width, y > y0-width) ), And(And(x < x1+width, x > x1-width), And(y < y1+width, y > y1-width) )), as_vector((0.0, 0.0)) , as_vector((0., 0.)))

IC = Function(VW, name="IC")
h0 = Function(VW.sub(0)).interpolate(height)
hu0 = Function(VW.sub(1)).interpolate(height * velocity)
IC.sub(0).assign(h0)
IC.sub(1).assign(hu0)

Q = Function(VW, name="Q")
Q_ = Function(VW, name="Qold")
Q.assign(IC)
Q_.assign(IC)

h, hu = split(Q)
h_, hu_ = split(Q_)

v, w = TestFunctions(VW)
v_, w_ = TrialFunctions(VW)
a = inner(w, w_) * dx + inner(v, v_) * dx

T = 1.0
CFL = 1. / (2 + 2 + 1)
incircle = (1./N)
g = 9.81

n = FacetNormal(mesh)

I = Identity(2)
P = lambda h, hu: sqrt(0.5 * g * h * h) * I
ev = lambda h, hu, n: abs(dot(hu, n)/h) + sqrt(g * h)
f_h = lambda h, hu : hu
f_hu = lambda h, hu: outer(hu, hu) / h + P(h, hu)

h_D = h
hu_D = hu - 2 * dot(hu, n) * n
hun_D = dot(hu_D, n)


p = '+'
m = '-'

h_g = h
hu_n = dot(hu, n) * n
hu_t = hu - hu_n
hu_g = -hu_n + hu_t
n_g = -n

DG_H = inner(grad(v), f_h(h, hu)) * dx
DG_HU = inner(grad(w), f_hu(h, hu)) * dx

F_H = (v(p)-v(m)) * (dot(0.5*(f_h(h(p), hu(p)) + f_h(h(m), hu(m))), n(m)) - 0.5*avg(ev(h, hu, n))*(h(p) - h(m))) * dS
F_HU = dot((w(p)-w(m)), (dot(0.5*(f_hu(h(p), hu(p)) + f_hu(h(m), hu(m))), n(m)) - 0.5*avg(ev(h, hu, n))*(hu(p) - hu(m)))) * dS

BC_H = -v * (dot(0.5*(f_h(h, hu) + f_h(h_g, hu_g)), n) - 0.5*0.5*(ev(h, hu, n)+ ev(h_g, hu_g, n_g))*(h_g - h)) * ds
BC_HU = -dot(w, dot(0.5*(f_hu(h, hu) + f_hu(h_g, hu_g)), n) - 0.5*0.5*(ev(h, hu, n)+ ev(h_g, hu_g, n_g))*(hu_g - hu)) * ds

t = 0.0
print("t=", t)
step = 0
output_freq = 1

outfile = VTKFile("swe.pvd")
outfile.write(project(h, Vout, name="H"), project(hu, Wout, name="HU"), time=t)


ev_cell = lambda h, hu: abs(hu[0]) / h + sqrt(g * h)
ev_max = max(project(ev_cell(h, hu), V).vector())
dt = CFL * incircle / ev_max
L = dt * (DG_H + DG_HU + F_H + F_HU + BC_H + BC_HU)

Q1 = Function(VW)
Q2 = Function(VW)
h1, hu1 = split(Q1)
h2, hu2 = split(Q2)
L1 = replace(L, {h: h1, hu: hu1}); 
L2 = replace(L, {h: h2, hu: hu2}); 

#params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
params = {'ksp_type': 'cg'}
dQ = Function(VW)
# prob = LinearVariationalProblem(a, L, dQ)
prob = LinearVariationalProblem(a, L, dQ)
solv = LinearVariationalSolver(prob, solver_parameters=params)
prob1 = LinearVariationalProblem(a, L1, dQ)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dQ)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
limiterH = VertexBasedLimiter(V)
#limiterH.compute_bounds(Q.sub(1))
#limiterH.apply(Q.sub(0))
#limiterH.apply(Q_.sub(0))
#limiterH.apply_limiter(Q.sub(0))
#limiterH.apply_limiter(Q_.sub(0))
#print(np.max(Q.sub(0).dat.data_ro))
#print(np.min(Q.sub(0).dat.data_ro))
#print('---------------')

while t < T - 0.5 * dt:
    ev_max = max(project(ev_cell(h, hu), V).vector())
    dt = CFL * incircle / ev_max

    # solve(F == 0, Q)

    #limiterH.compute_bounds(Q.sub(0))
    solv.solve()
    Q.assign(Q_ + dQ)
    #limiterH.apply_limiter(Q.sub(0))
    #limiterH = VertexBasedLimiter(V)
    limiterH.apply(Q.sub(0))
    tmp_func = V.get_work_function()
    #print(W.value_size)
    for i in range(W.value_size):
        d = Q.sub(1).dat.data_with_halos
        dd = tmp_func.dat.data_with_halos
        #print(d.shape, dd.shape)
        tmp_func.dat.data_with_halos[:] = d[:, i]
        limiterH.apply(tmp_func)
        Q.sub(1).dat.data_with_halos[:, i] = tmp_func.dat.data_with_halos


    #limiterH.apply(Q.sub(1))
    #print(np.max(Q.sub(0).dat.data_ro))
    #print(np.min(Q.sub(0).dat.data_ro))
    #limiterH.apply(dQ.sub(0))
    #Q.assign(Q)
    Q_.assign(Q)
    #limiterH.apply(Q_.sub(0))
    #limiterH.apply_limiter(Q_.sub(0))
    #print(np.max(Q.sub(0).dat.data_ro))
    #print(np.min(Q.sub(0).dat.data_ro))
    #print('---------------')

    # solv.solve()
    # Q1.assign(Q_ + dQ)
    # solv1.solve()
    # Q2.assign(0.5 * Q_ + 0.5 * (Q1+ dQ))
    # Q.assign(Q2)
    # Q_.assign(Q2)

    #solv.solve()
    #Q.assign(Q_ + dQ)
    #solv1.solve()
    #Q1.assign(0.75 * Q_ + 0.25 * (Q+ dQ))
    #solv2.solve()
    #Q2.assign(1./3. * Q_ + 2./3. * (Q1+ dQ))
    #Q.assign(Q2)
    #Q_.assign(Q2)

    step += 1
    t += dt

    if step % output_freq == 0:
        max_h = project(h, V).vector().max()
        error = assemble((abs(h-h0))*dx)
        L2_err = sqrt(assemble((h - h0)*(h - h0)*dx))
        L2_init = sqrt(assemble(h0*h0*dx))
        error = L2_err / L2_init
        outfile.write(project(h, Vout, name="H"), project(hu, Wout, name="HU"), time=t)
        print(f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L2/|h_0|_L2={error}, h_max={max_h}")
