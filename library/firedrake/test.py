from firedrake import *

# Step 1: Create the base and extruded meshes
base_mesh = UnitSquareMesh(2, 3, quadrilateral=True)  # 2x2 base mesh with quadrilaterals
num_layers = 2
extruded_mesh = ExtrudedMesh(base_mesh, layers=num_layers)

# Step 2: Define DG-1 function spaces
base_V = FunctionSpace(base_mesh, "DG", 1)
extruded_V = FunctionSpace(extruded_mesh, "DG", 1)

# Step 3: Create functions and assign unique values
base_function = Function(base_V)
extruded_function = Function(extruded_V)


# Assign values to DOFs
base_function.dat.data[:] = range(base_V.dim())
extruded_function.dat.data[:] = range(extruded_V.dim())

x, y, z = SpatialCoordinate(extruded_mesh)
xh, yh = SpatialCoordinate(base_mesh)
#base_function.interpolate(10*yh+000*yh)
#extruded_function.interpolate(10*y+1000*z)
base_function.interpolate(10*xh)
extruded_function.interpolate(10*x)


# Step 4: Determine dimensions
num_cells_base = base_mesh.num_cells()  # Number of cells in the base mesh
num_dofs_per_cell = base_V.finat_element.space_dimension()  # DOFs per cell
num_cells_extruded = num_cells_base * num_layers

# Step 5: Reshape data
# Reshape base mesh data (cells x DOFs per cell)
#base_data_reshaped = base_function.dat.data[:].reshape((num_cells_base, num_dofs_per_cell))
#base_data_reshaped = base_function.dat.data[:].reshape((num_dofs_per_cell, -1))
base_data_reshaped = base_function.dat.data[:].reshape((-1, num_dofs_per_cell))

# Reshape extruded mesh data (layers x cells x DOFs per cell)
#return field.dat.data_with_halos[:].reshape((n_dof_base, Nz,-1)).reshape((n_dof_base, Nz, -1, DIM_V+1))
#extruded_data_reshaped = extruded_function.dat.data[:].reshape(
#    (num_layers,  num_cells_base, num_dofs_per_cell, 2)
#)
#extruded_data_reshaped = extruded_function.dat.data[:].reshape(
    #(num_dofs_per_cell, num_layers,  -1)).reshape((num_dofs_per_cell, num_layers, -1, 2 ))
extruded_data_reshaped = extruded_function.dat.data[:].reshape((-1, num_layers, num_dofs_per_cell, 2 ))


# Step 6: Access and compare data
print("Base mesh data (reshaped):")
print(base_data_reshaped.shape)
print(base_data_reshaped[:, :])

print("\nExtruded mesh data (reshaped):")
print(extruded_data_reshaped.shape)
print(extruded_data_reshaped[:, 0, :, 1])



# Example: Compare data for layer 1 to the base mesh
#print("\nLayer 1 compared to base mesh:")
#print(extruded_data_reshaped[0, 0] - base_data_reshaped)  # Subtract base mesh data from layer 1

