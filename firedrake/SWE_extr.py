from firedrake import *
from firedrake.__future__ import interpolate
from copy import deepcopy

from mpi4py import MPI
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 40
Nz = 40
#m = UnitSquareMesh(N, N, quadrilateral=True)
mm = UnitIntervalMesh(N)
m = ExtrudedMesh(mm, layers=Nz)

dim = 2


# We create a 1D mesh immersed 2D from (0, 0) to (1, 1) which we call "line".
# Note that it only has 1 cell
cells = np.asarray([[0, 1]])
vertex_coords = np.asarray([[0.0, 0.5], [1.0, 0.5]])
#plex = mesh.plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
plex = mesh.plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
line = mesh.Mesh(plex, dim=2)
Vline = FunctionSpace(line, "DG", 1)

Vl = FunctionSpace(m, "DG", 1)
V = FunctionSpace(m, "DG", 1)
Vout = FunctionSpace(m, "DG", 1)
W = VectorFunctionSpace(m, "DG", 1)
Wl = VectorFunctionSpace(m, "DG", 1)
Wout = VectorFunctionSpace(m, "DG", 1)

VW = V * W
WV = W * V


if dim == 2:
    x, y = SpatialCoordinate(m)
elif dim == 3:
    x, y, z = SpatialCoordinate(m)
else:
    assert(False)

x0 = 0.3
y0 = 0.5
x1 = 0.7
y1 = 0.5

width = 0.05
height = conditional(
    Or(
        And(And(x < x0 + width, x > x0 - width), And(y < y0 + width, y > y0 - width)),
        And(And(x < x1 + width, x > x1 - width), And(y < y1 + width, y > y1 - width)),
    ),
    8.0,
    5.0,
)
velocity = conditional(
    Or(
        And(And(x < x0 + width, x > x0 - width), And(y < y0 + width, y > y0 - width)),
        And(And(x < x1 + width, x > x1 - width), And(y < y1 + width, y > y1 - width)),
    ),
    as_vector((0.0, 0.0, 0.0)[:dim]),
    as_vector((0.0, 0.0, 0.0)[:dim]),
)

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

T = 3.0
CFL = 1.0 / (2 + 2 + 1)
incircle = 1.0 / N
g = 9.81

n = FacetNormal(m)

I = Identity(dim)
P = lambda h, hu: sqrt(0.5 * g * h * h) * I
ev = lambda h, hu, n: abs(dot(hu, n) / h) + sqrt(g * h)
f_h = lambda h, hu: hu
f_hu = lambda h, hu: outer(hu, hu) / h + P(h, hu)

h_D = h
hu_D = hu - 2 * dot(hu, n) * n
hun_D = dot(hu_D, n)


p = "+"
m = "-"

h_g = h
hu_n = dot(hu, n) * n
hu_t = hu - hu_n
hu_g = -hu_n + hu_t
n_g = -n

DG_H = inner(grad(v), f_h(h, hu)) * dx
DG_HU = inner(grad(w), f_hu(h, hu)) * dx

F_H = (
    (v(p) - v(m))
    * (
        dot(0.5 * (f_h(h(p), hu(p)) + f_h(h(m), hu(m))), n(m))
        - 0.5 * avg(ev(h, hu, n)) * (h(p) - h(m))
    )
    * (dS_v + dS_h)
    #* (dS)
    )

F_HU = (
    dot(
        (w(p) - w(m)),
        (
            dot(0.5 * (f_hu(h(p), hu(p)) + f_hu(h(m), hu(m))), n(m))
            - 0.5 * avg(ev(h, hu, n)) * (hu(p) - hu(m))
        ),
    )
    * (dS_v + dS_h)
    #* (dS)
    )

BC_H = (
    -v
    * (
        dot(0.5 * (f_h(h, hu) + f_h(h_g, hu_g)), n)
        - 0.5 * 0.5 * (ev(h, hu, n) + ev(h_g, hu_g, n_g)) * (h_g - h)
    )
    * ds_v
    #* ds
    )

BC_HU = (
    -dot(
        w,
        dot(0.5 * (f_hu(h, hu) + f_hu(h_g, hu_g)), n)
        - 0.5 * 0.5 * (ev(h, hu, n) + ev(h_g, hu_g, n_g)) * (hu_g - hu),
    )
    * (ds_v + ds_t + ds_b)
    #* (ds)
    )

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
L1 = replace(L, {h: h1, hu: hu1})
L2 = replace(L, {h: h2, hu: hu2})
# params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
params = {"ksp_type": "cg"}
dQ = Function(VW)
prob = LinearVariationalProblem(a, L, dQ)
solv = LinearVariationalSolver(prob, solver_parameters=params)
prob1 = LinearVariationalProblem(a, L1, dQ)
solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
prob2 = LinearVariationalProblem(a, L2, dQ)
solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
limiterH = VertexBasedLimiter(V)

tmp_func = V.get_work_function()
T = 100 * dt

def apply_limiter(Q):
    q_limit = project(Q.sub(0), Vl)
    limiterH.apply(Q.sub(0))
    for i in range(W.value_size):
        tmp_func.dat.data_with_halos[:] = Q.sub(1).dat.data_with_halos[:, i]
        limiterH.apply(tmp_func)
        Q.sub(1).dat.data_with_halos[:, i] = tmp_func.dat.data_with_halos

def vector_to_scalar(field, dim):
    tmp_func.dat.data_with_halos[:] = field.dat.data_with_halos[:, dim]
    return tmp_func

def compute_z_integral(Q):
    #f_line = assemble(interpolate(Q.sub(0), Vline))
    f_line = assemble(interpolate(vector_to_scalar(Q.sub(1), 0), Vline))
    print(assemble(f_line**2 * dx))





while t < T - 0.5 * dt:
    ev_max = max(project(ev_cell(h, hu), V).vector())
    dt = CFL * incircle / ev_max

    compute_z_integral(Q)

    solv.solve()
    Q1.assign(Q_ + dQ)
    apply_limiter(Q1)
    solv1.solve()
    Q2.assign(0.5 * Q_ + 0.5 * (Q1+ dQ))
    apply_limiter(Q2)
    Q.assign(Q2)
    Q_.assign(Q2)

    step += 1
    t += dt

    if step % output_freq == 0:
        max_h = project(h, V).vector().max()
        error = assemble((abs(h - h0)) * dx)
        L2_err = sqrt(assemble((h - h0) * (h - h0) * dx))
        L2_init = sqrt(assemble(h0 * h0 * dx))
        error = L2_err / L2_init
        outfile.write(project(h, Vout, name="H"), project(hu, Wout, name="HU"), time=t)
        print(
            f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L2/|h_0|_L2={error}, h_max={max_h}"
        )

