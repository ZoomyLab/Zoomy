from firedrake import *
from firedrake.__future__ import interpolate
#from ufl import conditional, And

# Create a mesh
Nx = 10
Nz = 10
_mesh = UnitSquareMesh(Nx, Nz)

_dx = 1./Nx
_dz = 1./Nz

# Define the spatial coordinates
x, y = SpatialCoordinate(_mesh)


V = FunctionSpace(_mesh, "DG", 1)

q_f = x
q = Function(V).interpolate(q_f)


def compute_integral(q, x0):
    y0 = 1.0
    w_y = conditional(y <= y0, 1., 0.)
    w_x = conditional(And(x>x0-_dx/2, x<x0+_dx/2), 1., 0.)
    integral = assemble(q * w_x * w_y / _dx * dx)
    return integral

#_w_y = lambda _x, _y: conditional(y <= _y, 1., 0.)
#_w_x = lambda _x, _y: conditional(And(x>_x-_dx/2, x<_x+_dx/2), 1., 0.)
#psi = lambda _x, _y:  _w_x(_x, _y) * _w_y(_x, _y) /_dx * dx
#
#integral = assemble(psi(0.5, 0.5))
#integral = Function(V).interpolate(psi(x, y))

#print(f"Integral {integral}")

def create_line(a, b, N):
    dim = len(a)
    cells = np.asarray([[i, i + 1] for i in range(N)])
    vertex_coords = np.linspace(a, b, N + 1)
    topo_dim = 1
    plex = mesh.plex_from_cell_list(topo_dim, cells, vertex_coords, comm=_mesh.comm)
    line = mesh.Mesh(plex, dim=dim)
    return line, vertex_coords

def get_line(_x):
    _line, _vertex_coords = create_line([_x, 0.], [_x, 1.], 10)
    return _line

from time import time

Vline = []
q_line = []
q_int = []
q_int2 = []
M = 100
_dxx = 1./M

start = time()
for i in range(M):
    Vline.append(FunctionSpace(get_line(i * _dxx), "DG", 0))
print(f'setup Vline {time()-start}')

start = time()
for i in range(M):
    q_line.append(assemble(interpolate(q, Vline[i])))
print(f'interpolate Vline {time()-start}')

start = time()
for i in range(M):
    q_int.append(assemble(q_line[i] * dx))
print(f'intergrate Vline {time()-start}')

start = time()
for i in range(M):
    q_int2.append(compute_integral(q, i*_dxx))
print(f'interpolate Vline {time()-start}')

