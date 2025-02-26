from firedrake import *

# Step 1: Create a 2D base mesh and extrude it
base_mesh = UnitSquareMesh(2, 2)
extruded_mesh = ExtrudedMesh(base_mesh, layers=2)

# Step 2: Define function spaces
base_V = FunctionSpace(base_mesh, "DG", 1)
extruded_V = FunctionSpace(extruded_mesh, "DG", 1)

# Step 3: Create functions for the base and extruded meshes
base_function = Function(base_V)
extruded_function = Function(extruded_V)

# Step 4: Assign unique values to the base mesh DOFs
base_function.dat.data[:] = range(base_V.dim())

# Step 5: Assign values layer by layer
num_cells_base = base_mesh.num_cells()  # Number of cells in the base mesh
num_dofs_per_cell = base_V.finat_element.space_dimension()  # DOFs per cell

for layer in range(2):  # Loop through extruded layers
    for cell in range(num_cells_base):  # Loop through cells in the base mesh
        # Compute the extruded cell index
        extruded_cell_index = cell + layer * num_cells_base
        # Assign DOFs from base mesh to the corresponding layer
        start = cell * num_dofs_per_cell
        end = (cell + 1) * num_dofs_per_cell
        extruded_function.dat.data[start + layer * base_V.dim():end + layer * base_V.dim()] = \
            base_function.dat.data[start:end]

# Step 6: Print and verify
print("Base mesh DOFs:", base_function.dat.data[:])
print("Extruded mesh DOFs:", extruded_function.dat.data[:])

