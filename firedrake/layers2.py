from firedrake import *

# Create the base mesh and extrude it
base_mesh = UnitSquareMesh(4, 4)
mesh = ExtrudedMesh(base_mesh, layers=5, layer_height=0.1)

# Define the function space and the function f
DG = FunctionSpace(mesh, "DG", 1)
f = Function(DG)
x, y, z = SpatialCoordinate(mesh)
f.interpolate(x + y + z)  # Example definition for f

# Define the output function g
g = Function(DG)

# Loop through layers (excluding the topmost layer)
for layer in range(1, mesh.layers):  # Start from 1 because we need to access both the current and previous layers
    # Create a submesh for the current layer
    layer_submesh = MeshView(mesh, slices=[slice(layer - 1, layer)])  # Create a submesh for the current layer
    
    # Restrict f to the current layer
    f_layer = Function(DG)
    f_layer.assign(f)
    f_layer.restrict(layer_submesh)  # Restrict f to the current layer
    
    # Extract the data for the current layer
    print(f"Data for layer {layer}: {f_layer.dat.data}")
    
    # Compute g for this layer (this is just an example, you can do your actual computation)
    g.dat.data[layer::mesh.layers] = f_layer.dat.data[:]  # This is just an example, you can adjust your logic

print("Constructed g:", g.dat.data)

