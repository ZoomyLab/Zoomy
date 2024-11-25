from firedrake import *
from time import time
from firedrake.__future__ import interpolate
import numpy as np

from depth_integrator import DepthIntegrator

# Create the base mesh and extrude it
N = 10
base_mesh = UnitSquareMesh(N,N , quadrilateral=True)
Nz = 3

n_dof_base = 4
mesh = ExtrudedMesh(base_mesh, layers=Nz)
#print(mesh.coordinates.dat.data)

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

#Ulayer = Function(Vh)
x, y, z = SpatialCoordinate(mesh)
xh, yh = SpatialCoordinate(base_mesh)

dof_points = Function(W).interpolate(as_vector([x,y,z]))


# Define the output function g
HU = Function(W).interpolate(as_vector([0., 0., 0.]))  # Example definition for f

h0 = conditional(And(And(x < 0.66, x > 0.33), And(y < 0.66, y > 0.33)), 2., 1.)
hh0 = conditional(And(And(xh < 0.66, xh > 0.33), And(yh < 0.66, yh > 0.33)), 2., 1.)
hh = Function(Vh).interpolate(hh0)

h = Function(V).interpolate(h0)
Um = Function(Wh).interpolate(as_vector([0., 0.]))
phim = Function(Vh).interpolate(0.)
U = Function(W).interpolate(as_vector([0., 0., 0.]))
hphi = Function(V).interpolate(0.)
omega = Function(V).interpolate(0.)

outfile = VTKFile("out.pvd")
outfile2d = VTKFile("out2d.pvd")


num_layers = Nz
num_cells_base = base_mesh.num_cells()  # Number of cells in the base mesh
num_dofs_per_cell = Vh.finat_element.space_dimension()  # DOFs per cell
num_cells_extruded = num_cells_base * num_layers
DI = DepthIntegrator(num_layers, num_cells_base, num_dofs_per_cell, DIM_V)

t = 0.
hphi = assemble(interpolate((U.sub(0).dx(0) + U.sub(1).dx(1)), V))
U.project(HU/h)
DI.integrate(h, U, hphi, omega, hh, Um, phim, dof_points)
outfile.write(project(U, Wout, name="U"), project(omega, Vout, name='omega'), time=t)
outfile2d.write(project(hh, Vhout, name="h"), project(Um, Whout, name='Um'), project(phim, Vhout, name="phi_mean"), time=t)

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



# lambdas
P_hyd = lambda h, U: as_tensor([[0.5*g*h**2, 0., 0.], [0., 0.5*g*h**2, 0.], [0., 0., 0.]])
#convection = lambda h, U: as_tensor([[h*U.sub(0)**2, h*U.sub(0)*U.sub(1), 0. ], [h*U.sub(0)*U.sub(1), h*U.sub(1)**2, 0.], [0., 0., 0.]])
convection = lambda h, U: as_tensor([[h*U[0]**2, h*U[0]*U[1], h*U[0]*omega ], [h*U[0]*U[1], h*U[1]**2, h*U[1]*omega], [0., 0., 0.]])
#stress = lambda h, U: as_tensor([[0., 0., nu/h*U[0].dx(2)], [0., 0., nu/h*U[1].dx(2)], [0., 0., 0.]])
stress = lambda h, U: as_tensor([[0., 0., 0], [0., 0., 0], [0., 0., 0.]])

ev = lambda h, U, n: abs(dot(U, n)) + sqrt(g * h)
evh = lambda h, U, n: abs(dot(U, n)) + sqrt(g * h)



p = '+'
m = '-'

quad_degree_h = 2
quad_degree_hu = 2

# MASS BALANCE
# time / mass matrices
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



#MOMENTUM BALANCE
# time / mass matrices
a_mom = inner(w, w_) * dx
DG_HU = inner(grad(w), convection(h, U) + P_hyd(h, U)) * dx(degree=quad_degree_hu)

# flux
F_HU_l = convection(h, U)(p) + P_hyd(h, U)(p) + stress(h, U)(p)
F_HU_r = convection(h, U)(m) + P_hyd(h, U)(m) + stress(h, U)(m)
F_HU_n = 0.5*dot((F_HU_l + F_HU_r), n(m))
F_dis_HU = -0.5 * avg(ev(h, U, n)) * (h(p) * U(p)-h(m) * U(m))
F_HU = dot((w(p)-w(m)), (F_HU_n + F_dis_HU)) * (dS_h(degree=quad_degree_hu) + dS_v(degree=quad_degree_hu))

#Ghost cells
h_g = h
U_n = dot(HU/h, n) * n
U_t = HU/h - U_n
U_g = -U_n + U_t

# BC
BC_HU_l = convection(h, U) + P_hyd(h, U) + stress(h, U)
BC_HU_r = convection(h_g, U_g) + P_hyd(h_g, U_g) + + stress(h_g, U_g)
BC_HU_n = 0.5*dot((BC_HU_l + BC_HU_r), n)
BC_dis_HU = -0.5 * ev(h, U, n) * (h * U_g - HU)
BC_HU = -dot(w, (BC_HU_n + BC_dis_HU)) * (ds_v(degree=quad_degree_hu) + ds_b(degree=quad_degree_hu) + ds_t(degree=quad_degree_hu))




#ev_cell = lambda h, U: sqrt(U.sub(0)**2 + U.sub(1)**2 + U.sub(2)**2) + sqrt(g * h)
ev_cell = lambda h, U:  norm(U) + sqrt(g * h)
ev_max = lambda h, U: max(project(ev_cell(h, U), V).vector())

#dt = CFL * incircle / ev_max(h, U)
dt = 0.0001

L_H = dt * (DG_H + F_H + BC_H)
L_HU = dt * (DG_HU + F_HU + BC_HU)

#params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
params = {"ksp_type": "cg"}
dH = Function(Vh)
dHU = Function(W)
prob_H = LinearVariationalProblem(a_mass, L_H, dH)
solv_H = LinearVariationalSolver(prob_H, solver_parameters=params)
prob_HU = LinearVariationalProblem(a_mom, L_HU, dHU)
solv_HU = LinearVariationalSolver(prob_HU, solver_parameters=params)



# LIMITER
limiterH = VertexBasedLimiter(Vh)
limiterHU = VertexBasedLimiter(V)
lim_func_HU = V.get_work_function()

def apply_limiter_H(field):
    limiterH.apply(field)

def apply_limiter_HU(field):
    for i in range(W.value_size):
        lim_func_HU.dat.data_with_halos[:] = field.dat.data_with_halos[:, i]
        limiterHU.apply(lim_func_HU)
        field.dat.data_with_halos[:, i] = lim_func_HU.dat.data_with_halos


def apply_limiter(H, HU):
    apply_limiter_H(H)
    apply_limiter_HU(HU)


T = 0.001
step = 0
output_freq = 1


while t < T - 0.5 * dt:
    start = time()
    dt = CFL * incircle / ev_max(h, U)

    U.project(HU/h)
    hphi = assemble(interpolate((U.sub(0).dx(0) + U.sub(1).dx(1)), V))
    DI.integrate(h, U, hphi, omega, hh, Um, phim, dof_points)

    solv_H.solve()
    hh.assign(hh + dH)
    solv_HU.solve()
    HU.assign(HU + dHU)
    #apply_limiter(hh, HU)


    step += 1
    t += dt
    
    print(
        f"t={t}, dt={dt}, ev_max={ev_max(h, U)}, time for step={time()-start}"
    )

        
