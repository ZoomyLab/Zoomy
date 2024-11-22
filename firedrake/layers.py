from firedrake import *
from time import time
from firedrake.__future__ import interpolate

# Create the base mesh and extrude it
base_mesh = UnitSquareMesh(2, 2, quadrilateral=True)
Nz = 3
mesh = ExtrudedMesh(base_mesh, layers=Nz)
#print(mesh.coordinates.dat.data)

# Define the function space and the function f
DIM_H = 1
DIM_V = 1
horiz_elt = FiniteElement("DG", quadrilateral, DIM_H)
vert_elt = FiniteElement("DG", interval, DIM_V)
elt = TensorProductElement(horiz_elt, vert_elt)
DG = VectorFunctionSpace(mesh, elt)

DG_base = VectorFunctionSpace(base_mesh, "DG", DIM_H)
#print(DG.dim())
f = Function(DG)
ff = Function(DG_base)
x, y, z = SpatialCoordinate(mesh)
xb, yb = SpatialCoordinate(base_mesh)
# sub is for mixed spaces!! Use vector instead!!
dof_points = Function(DG).interpolate(as_vector([x,y,z]))
f.interpolate(as_vector([x, y, 1.]))  # Example definition for f
ff.interpolate(as_vector([xb, yb]))

print(f'DG_base dim {DG_base.dim()}')
print(f'DG dim {DG.dim()}')


#df = assemble(interpolate(f.dx(0), DG))
#dff = assemble(interpolate(ff.dx(0), DG_base))

# Define the output function g
g = Function(DG).interpolate(f)


# Loop through layers
start = time()

n_dof_base = 4

outfile = VTKFile("out.pvd")
outfile.write(project(f, DG, name="Q"), time=0)

def base_reshape(field):
    return field.dat.data_with_halos[:].reshape((n_dof_base, -1))

def extr_reshape(field):
    return field.dat.data_with_halos[:].reshape((n_dof_base, Nz,-1)).reshape((n_dof_base, Nz, -1, DIM_V+1))

for layer in range(Nz):  # Loop through layers except the top one

    if layer == 0:
        z_low = extr_reshape(dof_points.sub(2))[:, layer, :, 0]
        z_high = extr_reshape(dof_points.sub(2))[:, layer, :, 1]
        z_prev = np.zeros_like(z_low)
        current_layer_low = extr_reshape(f.sub(2))[:, layer, :, 0]
        current_layer_high = extr_reshape(f.sub(2))[:, layer, :, 1]
        previous_layer = np.ones_like(current_layer_low)
    else:
        current_layer_low = extr_reshape(f.sub(2))[:, layer, :, 0]
        current_layer_high = extr_reshape(f.sub(2))[:, layer, :, 1]
        previous_layer = extr_reshape(g.sub(2))[:, layer - 1, :, 1]
        z_prev = extr_reshape(dof_points.sub(2))[:, layer-1, :, 1]
        z_low = extr_reshape(dof_points.sub(2))[:, layer, :, 0]
        z_high = extr_reshape(dof_points.sub(2))[:, layer, :, 1]
        #print(z_prev)
        #print(z_low)
        #print(z_high)

    extr_reshape(g.sub(2))[:, layer, :, 0] = previous_layer + (z_low-z_prev) * current_layer_low
    extr_reshape(g.sub(2))[:, layer, :, 1] = previous_layer + (z_low-z_prev) * current_layer_low + (z_high-z_low) * current_layer_high


    #if layer == 1:
    #    #d_2d = (ff.dat.data_ro[:])
    #    #d_slice = (f.dat.data_ro[:])

    #    d_2d = (ff.dat.data_ro[:].reshape((n_dof_base, -1)))
    #    d_slice = (f.dat.data_ro[:].reshape((n_dof_base, Nz,-1)).reshape((n_dof_base, Nz, -1, DIM_V+1)))

    #    #(elem, dof_h)
    #    #print(d_2d[0])
    #    #(elem, layer, dof_h, dof_v)
    #    #print(d_slice[0,2, :, 0])

    

        


print(f'time: {time()-start}')

outfile.write(project(g, DG, name="Q"), time=1)
