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
DG = FunctionSpace(mesh, elt)

DG_base = FunctionSpace(base_mesh, "DG", DIM_H)
print(DG.dim())
f = Function(DG)
ff = Function(DG_base)
x, y, z = SpatialCoordinate(mesh)
xb, yb = SpatialCoordinate(base_mesh)
f.interpolate(x + 100 * y +000*z )  # Example definition for f
ff.interpolate(xb + 100 * yb)

df = assemble(interpolate(f.dx(0), DG))
dff = assemble(interpolate(ff.dx(0), DG_base))

# Define the output function g
g = Function(DG)

#print(Nz, mesh.layers)
#print(ff.dat.data.shape)
#print(f.dat.data.shape)
#print(DG.dim())
#print(DG_base.dim())

# Loop through layers
start = time()

offset = (1+DIM_V) * Nz
for layer in range(Nz):  # Loop through layers except the top one
    # Extract f on the current and previous layers
    current_layer = Function(DG_base)
    previous_layer = Function(DG_base)

    # Restrict to the current and previous layers
    if layer == 0 or layer == Nz:
        current_layer.interpolate(0)
        previous_layer.interpolate(0)
    #else:
    #    current_layer.dat.data[:] = f.dat.data_ro[layer::offset]
    #    previous_layer.dat.data[:] = f.dat.ata_ro[layer - 1::offset]
    #1
    ne_base = 4
    ne_per_column = ne_base * Nz
    n_nodes = 4
    if layer == 1:
        ne_base = 4
        #d_2d = (ff.dat.data_ro[:])
        #d_slice = (f.dat.data_ro[:])
        d_2d = (ff.dat.data_ro[:].reshape((ne_base, -1)))
        #d_slice = (f.dat.data_ro[:].reshape((ne_per_column,-1)).reshape((ne_per_column, -1, DIM_V+1)))
        d_slice = (f.dat.data_ro[:].reshape((ne_base, Nz,-1)).reshape((ne_base, Nz, -1, DIM_V+1)))

        #(elem, dof_h)
        print(d_2d[0])
        #(elem, layer, dof_h, dof_v)
        print(d_slice[0,2, :, 0])

        #print(d_2d[0])
        #print(d_slice[0, :, 0])

        #print(d_slice)
        #print(d_slice - d_slice2)
        #print(d_2d-d_slice)
        #print(d_2d-d_slice2)
        #print(f.dat.data_ro[::2][:int(ff.dat.data_ro.shape[0]/1)])
    #g.dat.data[layer::offset] = current_layer.dat.data[:] + previous_layer.dat.data[:]

print(f'time: {time()-start}')
#print("Constructed g:", g.dat.data)

