from firedrake import *
from time import time
from firedrake.__future__ import interpolate
import numpy as np

# Create the base mesh and extrude it
N = 10
base_mesh = UnitSquareMesh(N,N , quadrilateral=True)
Nz = 3

n_dof_base = 4
mesh = ExtrudedMesh(base_mesh, layers=Nz)

# Define the function space and the function f
DIM_H = 1
DIM_V = 1
horiz_elt = FiniteElement("DG", quadrilateral, DIM_H)
vert_elt = FiniteElement("DG", interval, DIM_V)
elt = TensorProductElement(horiz_elt, vert_elt)

W = VectorFunctionSpace(mesh, elt)
Wh = VectorFunctionSpace(base_mesh, horiz_elt)
V = FunctionSpace(mesh, elt)
Vh = FunctionSpace(base_mesh, horiz_elt)
Wout = VectorFunctionSpace(mesh, "CG", 1) 
Vout = FunctionSpace(mesh, "CG", 1) 
Vhout = FunctionSpace(base_mesh, "CG", 1)
Whout = VectorFunctionSpace(base_mesh, "CG", 1)

x, y, z = SpatialCoordinate(mesh)
xh, yh = SpatialCoordinate(base_mesh)

dof_points = Function(W).interpolate(as_vector([x,y,z]))

HU = Function(W).interpolate(as_vector([0., 0., 0.]))  # Example definition for f

h0 = conditional(And(And(x < 0.66, x > 0.33), And(y < 0.66, y > 0.33)), 2., 1.)
hh0 = conditional(And(And(xh < 0.66, xh > 0.33), And(yh < 0.66, yh > 0.33)), 2., 1.)
hh = Function(Vh).interpolate(hh0)
h = Function(V).interpolate(h0)
Um = Function(Wh).interpolate(as_vector([0., 0.]))


outfile = VTKFile("out.pvd")
outfile2d = VTKFile("out2d.pvd")

#MPI
num_layers = Nz
num_cells_base = base_mesh.num_cells()
num_dofs_per_cell = 4
num_cells_extruded = num_cells_base * num_layers

t = 0.
outfile.write(project(HU, Wout, name="HU"), time=t)
outfile2d.write(project(hh, Vhout, name="h"), project(Um, Whout, name='Um'), time=t)

v = TestFunction(Vh)
w = TestFunction(W)
v_ = TrialFunction(Vh)
w_ = TrialFunction(W)

# Setup
T = 0.01
CFL = 1./3.
incircle = (1./N)
g = 9.81
nu = 0.001

n = FacetNormal(mesh)
nh = FacetNormal(base_mesh)


p = '+'
m = '-'

ev = lambda h, U, n: abs(dot(U, n)) + sqrt(g * h)
evh = lambda h, U, n: abs(dot(U, n)) + sqrt(g * h)

quad_degree_h = 2
quad_degree_hu = 2

a_mass = inner(v, v_) * dx(degree=quad_degree_h)
DG_H = inner(grad(v), hh*Um) * dx(degree=quad_degree_h)

#flux
F_H_l = ((hh * Um)(p))
F_H_r = ((hh * Um)(m))
F_H_n = 0.5*dot((F_H_l + F_H_r), nh(m))
F_dis_H = -0.5  * avg(evh(hh, Um, nh)) * (hh(p)-hh(m))
F_H = (v(p)-v(m)) * (F_H_n + F_dis_H) * dS(degree=quad_degree_h)

#Ghost cells
hh_g = hh
um_n = dot(Um, nh) * nh
um_t = Um - um_n
Um_g = -um_n + um_t

#BC 
BC_H_l = hh * Um
BC_H_r = hh_g * Um_g
BC_H_n = 0.5*dot((BC_H_l + BC_H_r), nh)
BC_dis_H = -0.5 * evh(hh, Um, nh) * (hh_g - hh)
BC_H = -v*(BC_H_n + BC_dis_H) * ds(degree=quad_degree_h)

dt = 0.0001

L_H = dt * (DG_H + F_H+ BC_H)
#params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
params = {"ksp_type": "cg"}
dH = Function(Vh)

prob_H = LinearVariationalProblem(a_mass, L_H, dH)
solv_H = LinearVariationalSolver(prob_H, solver_parameters=params)

T = 0.001
step = 0
output_freq = 2


while t < T - 0.5 * dt:
    start = time()

    solv_H.solve()
    hh.assign(hh + dH)

    step += 1
    t += dt

    if step % output_freq == 0:
        outfile.write(project(HU, Wout, name="HU"), time=t)
        outfile2d.write(project(hh, Vhout, name="h"), project(Um, Whout, name='Um'), time=t)
        print(
            f"t={t}, dt={dt}, time for step={time()-start}"
        )
        
