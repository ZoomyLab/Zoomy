from firedrake import *
from time import time
from firedrake.__future__ import interpolate
import numpy as np

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


# Loop through layers

outfile = VTKFile("out.pvd")
outfile2d = VTKFile("out2d.pvd")
_time = 0
outfile.write(project(U, W, name="U"), project(psi, V, name='psi'), time=_time)
outfile2d.write(project(hh, Vh, name="h"), project(Um, Wh, name='Um'), project(phim, Vh, name="phi_mean") time=_time)

num_layers = Nz
num_cells_base = base_mesh.num_cells()  # Number of cells in the base mesh
num_dofs_per_cell = Vh.finat_element.space_dimension()  # DOFs per cell
num_cells_extruded = num_cells_base * num_layers
def base_reshape(field):
    return field.dat.data[:].reshape((-1, num_dofs_per_cell))

def extr_reshape(field):
    return field.dat.data[:].reshape((-1, num_layers, num_dofs_per_cell, DIM_V+1))

hphi = assemble(interpolate((U.sub(0).dx(0) + U.sub(1).dx(1)), V))

def depth_integration(h, U, hphi, psi, hh, Um, phim):
    """
    We perform the midpoint rule for integration along the extrusion direction. 
    As the DG-1 element has two dof in z-direction (legendre-integration points inside the cells (at z_low, z_high), we need to compute the exact integration points. The midpoints of the fields are already the location of the dof. 
    """
    for layer in range(Nz):  # Loop through layers except the top one
        if layer == 0:
            z_low = extr_reshape(dof_points.sub(2))[:, layer, :, 0]
            z_high = extr_reshape(dof_points.sub(2))[:, layer, :, 1]
            z_prev = np.zeros_like(z_low)
            z_next = extr_reshape(dof_points.sub(2))[:, layer+1, :, 0]
            h_re = base_reshape(h)
            phi_low = extr_reshape(hphi)[:, layer, :, 0] / h_re
            phi_high = extr_reshape(hphi)[:, layer, :, 1] / h_re
            phi_pre = np.zeros_like(phi_low)
            psi_pre = np.zeros_like(phi_low)
            u_low = extr_reshape(U.sub(0))[:, layer, :, 0]
            u_high = extr_reshape(U.sub(0))[:, layer, :, 1]
            u_pre = np.zeros_like(u_low)
            v_low = extr_reshape(U.sub(1))[:, layer, :, 0]
            v_high = extr_reshape(U.sub(1))[:, layer, :, 1]
            v_pre = np.zeros_like(u_low)
            z_start = z_prev
            z_mid = 0.5 * (z_low + z_high) 
            z_end = 0.5 * (z_high + z_next)
        elif layer == Nz-1:
            z_prev = extr_reshape(dof_points.sub(2))[:, layer-1, :, 1]
            z_low = extr_reshape(dof_points.sub(2))[:, layer, :, 0]
            z_high = extr_reshape(dof_points.sub(2))[:, layer, :, 1]
            z_next = np.ones_like(z_low)
            h_re = base_reshape(h)
            phi_low = extr_reshape(hphi)[:, layer, :, 0] / h_re
            phi_high = extr_reshape(hphi)[:, layer, :, 1] / h_re
            phi_pre = base_reshape(phim)[:]
            psi_pre = extr_reshape(psi)[:, layer-1, :, 1]
            u_low = extr_reshape(U.sub(0))[:, layer, :, 0] 
            u_high = extr_reshape(U.sub(0))[:, layer, :, 1]
            u_pre = base_reshape(Um.sub(0))[:]
            v_low = extr_reshape(U.sub(1))[:, layer, :, 0] 
            v_high = extr_reshape(U.sub(1))[:, layer, :, 1]
            v_pre = base_reshape(Um.sub(1))[:]
            z_start = 0.5 * (z_prev + z_low)
            z_mid = 0.5 * (z_low + z_high) 
            z_end = z_next
        else:
            z_prev = extr_reshape(dof_points.sub(2))[:, layer-1, :, 1]
            z_low = extr_reshape(dof_points.sub(2))[:, layer, :, 0]
            z_high = extr_reshape(dof_points.sub(2))[:, layer, :, 1]
            z_next = extr_reshape(dof_points.sub(2))[:, layer+1, :, 0]
            h_re = base_reshape(h)
            phi_low = extr_reshape(hphi)[:, layer, :, 0] / h_re
            phi_high = extr_reshape(hphi)[:, layer, :, 1] / h_re
            phi_pre = base_reshape(phim)[:]
            psi_pre = extr_reshape(psi)[:, layer-1, :, 1]
            u_low = extr_reshape(U.sub(0))[:, layer, :, 0] 
            u_high = extr_reshape(U.sub(0))[:, layer, :, 1]
            u_pre = base_reshape(Um.sub(0))[:]
            v_low = extr_reshape(U.sub(1))[:, layer, :, 0] 
            v_high = extr_reshape(U.sub(1))[:, layer, :, 1]
            v_pre = base_reshape(Um.sub(1))[:]
            z_start = 0.5 * (z_prev + z_low)
            z_mid = 0.5 * (z_low + z_high) 
            z_end = 0.5 * (z_high + z_next)

        dz_low =  z_mid - z_start
        dz_high =  z_end - z_mid
    
        
        base_reshape(phim)[:] = phi_pre + dz_low * phi_low + dz_high * phi_high
        base_reshape(Um.sub(0))[:] = u_pre + dz_low * u_low + dz_high * u_high
        base_reshape(Um.sub(1))[:] = v_pre + dz_low * v_low + dz_high * v_high
        extr_reshape(psi)[:, layer, :, 0] = psi_pre + dz_low * phi_low
        extr_reshape(psi)[:, layer, :, 1] = psi_pre + dz_low * phi_low + dz_high * phi_high
        h_reshaped = base_reshape(hh)[:]
        extr_reshape(h)[:, layer, :, 0] =  h_reshaped
        extr_reshape(h)[:, layer, :, 1] = h_reshaped

    for layer in range(Nz):  # Loop through layers except the top one
        extr_reshape(psi)[:, layer, :, 0] = base_reshape(phim)[:] - extr_reshape(psi)[:, layer, :, 0]
        extr_reshape(psi)[:, layer, :, 1] = base_reshape(phim)[:] - extr_reshape(psi)[:, layer, :, 1]


start = time()
depth_integration(h, U, hphi, psi, hh, Um, phim)
print(f'time for depth integration: {time()-start}')





_time = 1
outfile.write(project(U, W, name="U"), project(psi, V, name='psi'), time=_time)
outfile2d.write(project(hh, Vh, name="h"), project(Um, Wh, name='Um'), time=_time)
