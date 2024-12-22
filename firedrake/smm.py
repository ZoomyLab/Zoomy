from firedrake import *
from copy import deepcopy

from mpi4py import MPI
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 100
mesh = UnitIntervalMesh(N)

# Vl = FunctionSpace(mesh, "DG", 1)
V = FunctionSpace(mesh, "DG", 0)
Vout = FunctionSpace(mesh, "DG", 0)
W = VectorFunctionSpace(mesh, "DG", 0)
# Wl = VectorFunctionSpace(mesh, "DG", 0)
Wout = VectorFunctionSpace(mesh, "DG", 0)

VW = V * W
WV = W * V


x = SpatialCoordinate(mesh)


IC = Function(VW, name="IC")
# ht = Function(Vl).interpolate(height)
# ht.assign(ht)
h0 = Function(VW.sub(0)).interpolate(1.)
hu0 = Function(VW.sub(1)).interpolate(as_vector([1.0]))
IC.sub(0).assign(h0)
IC.sub(1).assign(hu0)

Q = Function(VW, name="Q")
Q_= Function(VW, name="Qold")
Q.assign(IC)
Q_.assign(IC)

h, hu = split(Q)
h_, hu_ = split(Q_)

v, w = TestFunctions(VW)
v_, w_ = TrialFunctions(VW)

# h = Function(V).interpolate(1.)
# hu = Function(W).interpolate(as_vector([1.0]))
# v = TestFunction(V)
# v_ = TrialFunction(V)
# w = TestFunction(W)
# w_ = TrialFunction(W)
# h = Function(V).interpolate(h0)
# hu = Function(W).interpolate(hu0)

a = inner(w, w_) * dx + inner(v, v_) * dx
# a = inner(v, v_) * dx


T = 3.0
CFL = 1.0 / (2 + 2 + 1)
incircle = 1.0 / N
g = 9.81

n = FacetNormal(mesh)

I = Identity(1)
P = lambda h, hu: sqrt(0.5 * g * h * h) 
ev = lambda q, n: abs((q.sub(1)/q.sub(0)* n)) + sqrt(g * q.sub(0))
ev_n = lambda q, pm, n: abs((dot(q.sub(1)(pm), n(pm))/q.sub(0)(pm))) + sqrt(g * q.sub(0)(pm))

f_h = lambda h, hu: hu
f_hu = lambda h, hu: (hu* hu) / h + P(h, hu)

f_q = lambda q, pm: [q.sub(1)(pm), outer(q.sub(1)(pm), q.sub(1)(pm)) / q.sub(0)(pm) + 0.5 * g * q.sub(0)(pm)**2 * I]

q = [v, w]

p = "+"
m = "-"

i = 0 
# scalar
F_Q = ((q[i](p) - q[i](m)) * (
        dot(0.5 * (f_q(Q, p)[i] + f_q(Q, m)[i]), n(m)) 
        - 0.5 * (0.5*(ev_n(Q, p, n) + ev_n(Q, m, n))) * (Q.sub(i)(p) - Q.sub(i)(m))
        )) * dS
# F_Q = ((v(p) - v(m)) * (dot(0.5 * (hu(p) + hu(m)), n(m)) - 0.5 * (0.5* (h(p)-h(m))))) * dS

# vector
i = 1
F_Q += dot((q[i](p) - q[i](m)),(
    dot((0.5 * (f_q(Q, p)[i] + f_q(Q, m)[i])), n(m)) 
    - 0.5 * (0.5 * (ev_n(Q, p, n) + ev_n(Q, m, n)) * (Q.sub(i)(p) - Q.sub(i)(m)))
    )) * dS


# h_D = h
# hu_D = hu - 2 * dot(hu, n) * n
# hun_D = dot(hu_D, n)




# h_g = h
# hu_n = dot(hu, n) * n
# hu_t = hu - hu_n
# hu_g = -hu_n + hu_t
# n_g = -n

# DG_H = inner(grad(v), f_h(h, hu)) * dx
# DG_HU = inner(grad(w), f_hu(h, hu)) * dx

# F_H = (
#     (v(p) - v(m))
#     * (
#         (0.5 * (f_h(h(p), hu(p)) + f_h(h(m), hu(m)))* n(m))
#         - 0.5 * avg(ev(h, hu, n)) * (h(p) - h(m))
#     )
#     * dS
# )

# F_HU = (
#     (
#         (w(p) - w(m))*
#         (
#             (0.5 * (f_hu(h(p), hu(p)) + f_hu(h(m), hu(m)))* n(m))
#             - 0.5 * avg(ev(h, hu, n)) * (hu(p) - hu(m))
#         ),
#     )
#     * dS
# )

# BC_H = (
#     -v
#     * (
#         dot(0.5 * (f_h(h, hu) + f_h(h_g, hu_g)), n)
#         - 0.5 * 0.5 * (ev(h, hu, n) + ev(h_g, hu_g, n_g)) * (h_g - h)
#     )
#     * ds
# )
# BC_HU = (
#     -dot(
#         w,
#         dot(0.5 * (f_hu(h, hu) + f_hu(h_g, hu_g)), n)
#         - 0.5 * 0.5 * (ev(h, hu, n) + ev(h_g, hu_g, n_g)) * (hu_g - hu),
#     )
#     * ds
# )

t = 0.0
print("t=", t)
step = 0
output_freq = 1

# outfile = VTKFile("swe.pvd")
# outfile.write(project(h, Vout, name="H"), project(hu, Wout, name="HU"), time=t)

# ev_cell = lambda h, hu: abs(hu[0]) / h + sqrt(g * h)
# ev_max = max(project(ev_cell(h, hu), V).vector())
ev_max = 1.
# dt = CFL * incircle / ev_max
dt = 0.00001
# L = dt * (DG_H + DG_HU + F_H + F_HU + BC_H + BC_HU)
L = dt * (F_Q)


# Q1 = Function(VW)
# Q2 = Function(VW)
# h1, hu1 = split(Q1)
# h2, hu2 = split(Q2)
# L1 = replace(L, {h: h1, hu: hu1})
# L2 = replace(L, {h: h2, hu: hu2})
# params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
params = {"ksp_type": "cg"}
dQ = Function(VW)
# dQ = Function(V)
# prob = LinearVariationalProblem(a, L, dQ)
prob = LinearVariationalProblem(a, L, dQ)
solv = LinearVariationalSolver(prob, solver_parameters=params)
# prob1 = LinearVariationalProblem(a, L1, dQ)
# solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
# prob2 = LinearVariationalProblem(a, L2, dQ)
# solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
# limiterH = VertexBasedLimiter(V)
# limiterH.compute_bounds(Q.sub(1))
# limiterH.apply(Q.sub(0))
# limiterH.apply(Q_.sub(0))
# limiterH.apply_limiter(Q.sub(0))
# limiterH.apply_limiter(Q_.sub(0))
# print(np.max(Q.sub(0).dat.data_ro))
# print(np.min(Q.sub(0).dat.data_ro))
# print('---------------')
# tmp_func = V.get_work_function()


T = 3 * dt


# def apply_limiter_P1(field):
#     field_P1 = project(field, Vl)
#     limiter.apply(field_P1)
#     field = project
#     return field_P1


# def apply_limiter(Q):
#     q_limit = project(Q.sub(0), Vl)
#     limiterH.apply(Q.sub(0))
#     for i in range(W.value_size):
#         tmp_func.dat.data_with_halos[:] = Q.sub(1).dat.data_with_halos[:, i]
#         limiterH.apply(tmp_func)
#         Q.sub(1).dat.data_with_halos[:, i] = tmp_func.dat.data_with_halos


while t < T - 0.5 * dt:
    # ev_max = max(project(ev_cell(h, hu), V).vector())
    # dt = CFL * incircle / ev_max

    # solve(F == 0, Q)

    # limiterH.compute_bounds(Q.sub(0))
    #solv.solve()
    #Q.assign(Q_ + dQ)
    #apply_limiter(Q)
    #Q_.assign(Q)

    # h, hu = split(Q)
    # h_, hu_ = split(Q_)

    # print(f'h: {Q.sub(0).dat.data_with_halos.shape}')
    # ht.project(h)
    # print(f'ht: {ht.dat.data_with_halos.shape}')
    # limiterH.apply_limiter(Q.sub(0))
    # limiterH = VertexBasedLimiter(V)
    # print(W.value_size)

    # limiterH.apply(Q.sub(1))
    # print(np.max(Q.sub(0).dat.data_ro))
    # print(np.min(Q.sub(0).dat.data_ro))
    # limiterH.apply(dQ.sub(0))
    # Q.assign(Q)
    # limiterH.apply(Q_.sub(0))
    # limiterH.apply_limiter(Q_.sub(0))
    # print(np.max(Q.sub(0).dat.data_ro))
    # print(np.min(Q.sub(0).dat.data_ro))
    # print('---------------')

    solv.solve()
    Q.assign(Q + dQ)
    # Q1.assign(Q_ + dQ)
    # apply_limiter(Q1)
    # solv1.solve()
    # Q2.assign(0.5 * Q_ + 0.5 * (Q1+ dQ))
    # apply_limiter(Q2)
    # Q.assign(Q2)
    # Q_.assign(Q2)

    # solv.solve()
    # Q.assign(Q_ + dQ)
    # solv1.solve()
    # Q1.assign(0.75 * Q_ + 0.25 * (Q+ dQ))
    # solv2.solve()
    # Q2.assign(1./3. * Q_ + 2./3. * (Q1+ dQ))
    # Q.assign(Q2)
    # Q_.assign(Q2)

    step += 1
    t += dt

    # if step % output_freq == 0:
    #     max_h = project(h, V).vector().max()
    #     error = assemble((abs(h - h0)) * dx)
    #     L2_err = sqrt(assemble((h - h0) * (h - h0) * dx))
    #     L2_init = sqrt(assemble(h0 * h0 * dx))
    #     error = L2_err / L2_init
    #     outfile.write(project(h, Vout, name="H"), project(hu, Wout, name="HU"), time=t)
    #     print(
    #         f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L2/|h_0|_L2={error}, h_max={max_h}"
    #     )
    # limiterH.apply_limiter(Q.sub(0))
    # limiterH = VertexBasedLimiter(V)
    # print(W.value_size)

    # limiterH.apply(Q.sub(1))
    # print(np.max(Q.sub(0).dat.data_ro))
    # print(np.min(Q.sub(0).dat.data_ro))
    # limiterH.apply(dQ.sub(0))
    # Q.assign(Q)
    # limiterH.apply(Q_.sub(0))
    # limiterH.apply_limiter(Q_.sub(0))
    # print(np.max(Q.sub(0).dat.data_ro))
    # print(np.min(Q.sub(0).dat.data_ro))
    # print('---------------')

    # solv.solve()
    # Q1.assign(Q_ + dQ)
    # apply_limiter(Q1)
    # solv1.solve()
    # Q2.assign(0.5 * Q_ + 0.5 * (Q1+ dQ))
    # apply_limiter(Q2)
    # Q.assign(Q2)
    # Q_.assign(Q2)

    # solv.solve()
    # Q.assign(Q_ + dQ)
    # solv1.solve()
    # Q1.assign(0.75 * Q_ + 0.25 * (Q+ dQ))
    # solv2.solve()
    # Q2.assign(1./3. * Q_ + 2./3. * (Q1+ dQ))
    # Q.assign(Q2)
    # Q_.assign(Q2)

    # step += 1
    # t += dt

    # if step % output_freq == 0:
    #     max_h = project(h, V).vector().max()
    #     error = assemble((abs(h - h0)) * dx)
    #     L2_err = sqrt(assemble((h - h0) * (h - h0) * dx))
    #     L2_init = sqrt(assemble(h0 * h0 * dx))
    #     error = L2_err / L2_init
    #     outfile.write(project(h, Vout, name="H"), project(hu, Wout, name="HU"), time=t)
    #     print(
    #         f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L2/|h_0|_L2={error}, h_max={max_h}"
    #     )
    # limiter = VertexBasedLimiter(V)
    # print(W.value_size)

    # limiter.apply(Q.sub(1))
    # print(np.max(Q.sub(0).dat.data_ro))
    # print(np.min(Q.sub(0).dat.data_ro))
    # limiter.apply(dQ.sub(0))
    # Q.assign(Q)
    # limiter.apply(Q_.sub(0))
    # limiter.apply_limiter(Q_.sub(0))
    # print(np.max(Q.sub(0).dat.data_ro))
    # print(np.min(Q.sub(0).dat.data_ro))
    # print('---------------')

    # solv.solve()
    # Q1.assign(Q_ + dQ)
    # apply_limiter(Q1)
    # solv1.solve()
    # Q2.assign(0.5 * Q_ + 0.5 * (Q1+ dQ))
    # apply_limiter(Q2)
    # Q.assign(Q2)
    # Q_.assign(Q2)

    # solv.solve()
    # Q.assign(Q_ + dQ)
    # solv1.solve()
    # Q1.assign(0.75 * Q_ + 0.25 * (Q+ dQ))
    # solv2.solve()
    # Q2.assign(1./3. * Q_ + 2./3. * (Q1+ dQ))
    # Q.assign(Q2)
    # Q_.assign(Q2)

    # step += 1
    # t += dt

    # if step % output_freq == 0:
    #     max_h = project(h, V).vector().max()
    #     error = assemble((abs(h - h0)) * dx)
    #     L2_err = sqrt(assemble((h - h0) * (h - h0) * dx))
    #     L2_init = sqrt(assemble(h0 * h0 * dx))
    #     error = L2_err / L2_init
    #     outfile.write(project(h, Vout, name="H"), project(hu, Wout, name="HU"), time=t)
    #     print(
    #         f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L2/|h_0|_L2={error}, h_max={max_h}"
    #     )
    # limiter = VertexBasedLimiter(V)
    # print(W.value_size)

    # limiter.apply(Q.sub(1))
    # print(np.max(Q.sub(0).dat.data_ro))
    # print(np.min(Q.sub(0).dat.data_ro))
    # limiter.apply(dQ.sub(0))
    # Q.assign(Q)
    # limiter.apply(Q_.sub(0))
    # limiter.apply_limiter(Q_.sub(0))
    # print(np.max(Q.sub(0).dat.data_ro))
    # print(np.min(Q.sub(0).dat.data_ro))
    # print('---------------')

    # solv.solve()
    # Q1.assign(Q_ + dQ)
    # apply_limiter(Q1)
    # solv1.solve()
    # Q2.assign(0.5 * Q_ + 0.5 * (Q1+ dQ))
    # apply_limiter(Q2)
    # Q.assign(Q2)
    # Q_.assign(Q2)

    # solv.solve()
    # Q.assign(Q_ + dQ)
    # solv1.solve()
    # Q1.assign(0.75 * Q_ + 0.25 * (Q+ dQ))
    # solv2.solve()
    # Q2.assign(1./3. * Q_ + 2./3. * (Q1+ dQ))
    # Q.assign(Q2)
    # # Q_.assign(Q2)

    # step += 1
    # t += dt

    # if step % output_freq == 0:
    #     max_h = project(h, V).vector().max()
    #     error = assemble((abs(h - h0)) * dx)
    #     L2_err = sqrt(assemble((h - h0) * (h - h0) * dx))
    #     L2_init = sqrt(assemble(h0 * h0 * dx))
    #     error = L2_err / L2_init
    #     outfile.write(project(h, Vout, name="H"), project(hu, Wout, name="HU"), time=t)
    #     print(
    #         f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L2/|h_0|_L2={error}, h_max={max_h}"
    #     )
