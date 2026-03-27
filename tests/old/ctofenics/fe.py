from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem
import ufl
import ctypes

# Load the shared library
lib = ctypes.CDLL("./test.so")

# Define argument and return types
lib.myfunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
                       np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
lib.myfunc.restype = None

# Define a Python wrapper for the C function
def myfunc(q):
    q = np.array(q, dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    lib.myfunc(q, out)
    return out[0]

# Create a simple mesh and function space
mesh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
V = fem.functionspace(mesh, ("CG", 1))

# Expression callable
class MyExpression:
    def __call__(self, x):
        q = [x[0], x[1]]
        return [myfunc(q)]

ufl_elem = V.ufl_element()
X = V.element.basix_element.interpolation_points()

# Create Expression
expr = fem.Expression(MyExpression(), ufl_elem, X)


# Interpolate expression to function space
f = fem.Function(V)
f.interpolate(expr)

# Print the function values at mesh vertices
print(f.x.array[:])

