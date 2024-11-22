from firedrake import *
from time import time
from firedrake.__future__ import interpolate
import numpy as np

# Create the base mesh and extrude it
base_mesh = UnitSquareMesh(3,3 , quadrilateral=True)
n_dof_base = 4

Nz = 2
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
h = Function(Vh).interpolate(1.)
Um = Function(Wh).interpolate(as_vector([0, 0]))  # Example definition for f
phim = Function(Vh).interpolate(0)

hphi = Function(V).interpolate(0)
psi = Function(V).interpolate(0)


# Loop through layers
start = time()

outfile = VTKFile("out.pvd")
outfile2d = VTKFile("out2d.pvd")
_time = 0
outfile.write(project(U, W, name="U"), project(psi, V, name='psi'), time=_time)
outfile2d.write(project(h, Vh, name="h"), project(Um, Wh, name='Um'), time=_time)

def base_reshape(field):
    return field.dat.data_with_halos[:].reshape((n_dof_base, -1))

def extr_reshape(field):
    return field.dat.data_with_halos[:].reshape((n_dof_base, Nz,-1)).reshape((n_dof_base, Nz, -1, DIM_V+1))

hphi = assemble(interpolate((U.sub(0).dx(0) + U.sub(1).dx(1)), V))

def depth_integration(h, U, Um, hphi, phim, psi):
    """
    #    #(elem, dof_h)
    #    #print(d_2d[0])
    #    #(elem, layer, dof_h, dof_v)
    #    #print(d_slice[0,2, :, 0])
    """
    for layer in range(Nz):  # Loop through layers except the top one
        if layer == 0:
            z_low = extr_reshape(dof_points.sub(2))[:, layer, :, 0]
            z_high = extr_reshape(dof_points.sub(2))[:, layer, :, 1]
            z_prev = np.zeros_like(z_low)
            h_re = base_reshape(h)
            phi_low = extr_reshape(hphi)[:, layer, :, 0] / h_re
            phi_high = extr_reshape(hphi)[:, layer, :, 1] / h_re
            psi_pre = np.zeros_like(phi_low)
            u_low = extr_reshape(U.sub(0))[:, layer, :, 0]
            u_high = extr_reshape(U.sub(0))[:, layer, :, 1]
            u_pre = np.zeros_like(u_low)
            base_reshape(Um.sub(0))[:] = u_high
        else:
            z_prev = extr_reshape(dof_points.sub(2))[:, layer-1, :, 1]
            z_low = extr_reshape(dof_points.sub(2))[:, layer, :, 0]
            z_high = extr_reshape(dof_points.sub(2))[:, layer, :, 1]
            h_re = base_reshape(h)
            phi_low = extr_reshape(hphi)[:, layer, :, 0] / h_re
            phi_high = extr_reshape(hphi)[:, layer, :, 1] / h_re
            psi_pre = extr_reshape(psi)[:, layer-1, :, 1]
            u_low = extr_reshape(U.sub(0))[:, layer, :, 0] 
            u_high = extr_reshape(U.sub(0))[:, layer, :, 1]
            u_pre = base_reshape(Um.sub(0))[:]
    
        #print(layer)
        #print(base_reshape(Um.sub(0)).shape)
        #print(extr_reshape(psi)[:, layer, :, 0].shape)
        #print(z_prev.shape)
        #print(z_low.shape)
        #print(z_high.shape)
        #print(u_pre.shape)
        #print(u_low.shape)
        #print(u_high.shape)
        #print(psi_pre.shape)
        #print(phi_low.shape)
        #print(phi_high.shape)


        #base_reshape(Um.sub(0))[:] = u_pre + (z_low-z_prev) * u_low + (z_high-z_low) * u_high
        extr_reshape(psi)[:, layer, :, 0] = psi_pre + (z_low-z_prev) * phi_low
        extr_reshape(psi)[:, layer, :, 1] = psi_pre + (z_low-z_prev) * phi_low + (z_high-z_low) * phi_high


depth_integration(h, U, Um, hphi, phim, psi)
    


print(f'time: {time()-start}')

_time = 1
outfile.write(project(U, W, name="U"), project(psi, V, name='psi'), time=_time)
outfile2d.write(project(h, Vh, name="h"), project(Um, Wh, name='Um'), time=_time)
