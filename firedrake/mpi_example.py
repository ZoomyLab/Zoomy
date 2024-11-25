from firedrake import *

# Mesh definition
mesh = UnitSquareMesh(10, 10)

# Function space
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define bilinear and linear forms
a = dot(grad(u), grad(v)) * dx  # Diffusion term
L = Constant(1.0) * v * dx  # Source term

# Apply Dirichlet boundary conditions
bc = DirichletBC(V, 0.0, "on_boundary")

# Solve the problem
u_sol = Function(V)
solve(a == L, u_sol, bcs=bc)

# Output the solution
File("solution.pvd").write(u_sol)

# Print solution norm
print("Solution norm:", norm(u_sol))
