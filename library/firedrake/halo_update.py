from firedrake import *

# Define a simple mesh and function space
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)

# Initialize a function
u = Function(V)
u.interpolate(sin(pi*x)*sin(pi*x))

# Time-stepping loop
dt = 0.01
T = 1.0
t = 0

while t < T:
    # Access data and modify locally
    u.dat.data[:] += dt * u.dat.data[:]

    # Synchronize halo regions before proceeding
    #u.dat.scatter()
    #u.dat.global_to_local()
    u.assign(u)

    # Update time
    t += dt

