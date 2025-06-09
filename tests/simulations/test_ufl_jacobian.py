from firedrake import *
from ufl import diff
from ufl.algorithms import expand_derivatives


# Create a mesh (e.g., 1D interval for simplicity)
mesh = IntervalMesh(1, 1.0)

# Define function space (e.g., continuous Galerkin of degree 1)
V = FunctionSpace(mesh, "CG", 1)

# Define the function u in V
u = Function(V, name="u")

# Define the flux vector f
f = as_vector([u**2, sin(u), abs(u)])

# Compute the Jacobian of f with respect to u
J_f = expand_derivatives(diff(f, u))

# Print the symbolic Jacobian
print("Symbolic Jacobian J_f:")
print(J_f)
