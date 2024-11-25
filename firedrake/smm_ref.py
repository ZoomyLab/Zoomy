from firedrake import *
from time import time
from firedrake.__future__ import interpolate
import numpy as np

from depth_integrator import DepthIntegrator

# Create the base mesh and extrude it
base_mesh = UnitSquareMesh(100,100 , quadrilateral=True)
Nz = 30

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

#Ulayer = Function(Vh)
x, y, z = SpatialCoordinate(mesh)
xh, yh = SpatialCoordinate(base_mesh)

dof_points = Function(W).interpolate(as_vector([x,y,z]))

#print(f'DG_base dim {DG_base.dim()}')
#print(f'DG dim {DG.dim()}')


#df = assemble(interpolate(f.dx(0), DG))
#dff = assemble(interpolate(ff.dx(0), DG_base))

# Define the output function g
U = Function(W).interpolate(as_vector([x, y, 1.]))  # Example definition for f
h = Function(V).interpolate(1.)
hh = Function(Vh).interpolate(1.)
Um = Function(Wh).interpolate(as_vector([xh, yh]))  # Example definition for f
phim = Function(Vh).interpolate(0)

hphi = Function(V).interpolate(0)
psi = Function(V).interpolate(0)

outfile = VTKFile("out.pvd")
outfile2d = VTKFile("out2d.pvd")
_time = 0
outfile.write(project(U, W, name="U"), project(psi, V, name='psi'), time=_time)
outfile2d.write(project(hh, Vh, name="h"), project(Um, Wh, name='Um'), project(phim, Vh, name="phi_mean"), time=_time)

num_layers = Nz
num_cells_base = base_mesh.num_cells()  # Number of cells in the base mesh
num_dofs_per_cell = Vh.finat_element.space_dimension()  # DOFs per cell
num_cells_extruded = num_cells_base * num_layers

hphi = assemble(interpolate((U.sub(0).dx(0) + U.sub(1).dx(1)), V))

DI = DepthIntegrator(num_layers, num_cells_base, num_dofs_per_cell, DIM_V)

start = time()
DI.integrate(h, U, hphi, psi, hh, Um, phim, dof_points)
print(f'time for depth integration: {time()-start}')



v = TestFunction(Vh)
w = TestFunction(W)
v_ = TrialFunction(Vh)
w_ = TrialFunction(W)


#a_mass = inner(v, v_) * dx 
#a_mom = inner(w, w_) * dx
#
#
##CONTINUE HERE
#T = 0.01
#CFL = 0.45
#incircle = (1./N)
#g = 9.81
#
#n = FacetNormal(mesh)
#
#P_hyd = lambda h, U: as_tensor([[0.5*g*h**2, 0., 0.], [0., 0.5*g*h**2, 0.], [0., 0., 0.]])
#convection = lambda h, U: as_tensor([[h*U.sub(0)**2, h*U.sub(0)*U.sub(1), 0. ], [h*U.sub(0)*U.sub(1), h*U.sub(1)**2, 0.], [0., 0., 0.]])
#stress = lambda h, U: as_tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
#
#ev = lambda h, U, n: abs(dot(U, n)) + sqrt(g * h)
#
#h_D = h
#u_D = U - 2 * dot(U, n) * n
#un_D = dot(u_D, n)
#
#p = '+'
#m = '-'
#F_H_l = ((h * Um)(p))
#F_H_r = ((h * Um)(m))
#F_H_n = 0.5*dot((F_H_l + F_H_r), n(m))
#F_dis_H = -0.5  * avg(ev(h, hu, n)) * (h(p)-h(m))
#F_H = (v(p)-v(m)) * (F_H_n + F_dis_H) * dS
#
##TODO Continue
#F_HU_l = outer(hu(p), hu(p)) / h(p) + 0.5 * g * h(p) * h(p) * I
#F_HU_r = outer(hu(m), hu(m)) / h(m) + 0.5 * g * h(m) * h(m) * I
#F_HU_n = 0.5*dot((F_HU_l + F_HU_r), n(m))
#F_dis_HU = -0.5 * avg(ev(h, hu, n)) * (hu(p)-hu(m))
#F_HU = dot((w(p)-w(m)), (F_HU_n + F_dis_HU)) * dS
#
#h_g = h
#hu_n = dot(hu, n) * n
#hu_t = hu - hu_n
#hu_g = -hu_n + hu_t
#
#BC_H_l = hu
#BC_H_r = hu_g
#BC_H_n = 0.5*dot((BC_H_l + BC_H_r), n)
#BC_dis_H = -0.5 * ev(h, hu, n) * (h_g - h)
#BC_H = -v*(BC_H_n + BC_dis_H) * ds
#
#BC_HU_l = outer(hu, hu) / h + 0.5 * g * h * h * I
#BC_HU_r = outer(hu_g, hu_g) / h_g + 0.5 * g * h_g * h_g * I
#BC_HU_n = 0.5*dot((BC_HU_l + BC_HU_r), n)
#BC_dis_HU = -0.5 * ev(h, hu, n) * (hu_g - hu)
#BC_HU = -dot(w, (BC_HU_n + BC_dis_HU)) * ds



_time = 1
outfile.write(project(U, W, name="U"), project(psi, V, name='psi'), time=_time)
outfile2d.write(project(hh, Vh, name="h"), project(Um, Wh, name='Um'), project(phim, Vh, name="phi_mean"), time=_time)
