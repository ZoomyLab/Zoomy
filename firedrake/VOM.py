from firedrake import *
from firedrake.__future__ import interpolate
from copy import deepcopy

from mpi4py import MPI
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 10
Nz = 1
mm = UnitIntervalMesh(N)
m = ExtrudedMesh(mm, layers=Nz)
dim = 2

def compute_cell_centers(_mesh):
    x = SpatialCoordinate(_mesh)
    DG0 = VectorFunctionSpace(_mesh, "DG", 0)
    cell_centers = Function(DG0)

    # Project the spatial coordinates into the DG0 space (averaging per cell)
    cell_centers.interpolate(as_vector(x))
    return cell_centers.dat.data

def create_line(a, b, N):
    dim = len(a)
    cells = np.asarray([[i, i + 1] for i in range(N)])
    vertex_coords = np.linspace(a, b, N + 1)
    topo_dim = 1
    plex = mesh.plex_from_cell_list(topo_dim, cells, vertex_coords, comm=m.comm)
    line = mesh.Mesh(plex, dim=dim)
    return line, vertex_coords

line, vertex_coords = create_line([0.0, 0.5], [1.0, 0.5], N)
Vline = FunctionSpace(line, "DG", 0)

V = FunctionSpace(m, "DG", 0)
V1 = FunctionSpace(m, "DG", 1)
Vout1 = FunctionSpace(m, "CG", 1)
zeta, _ = SpatialCoordinate(line)
zeta_coords = line.coordinates.dat.data

cell_centers = compute_cell_centers(m)
vertex_coords = m.coordinates.dat.data
m_vom = VertexOnlyMesh(m, cell_centers)
VOM = FunctionSpace(m_vom, "DG", 0)

x, y = SpatialCoordinate(m)
q_f = x  + y

q = Function(V, name='q').interpolate(q_f)

#q.assign(q)
v = TestFunction(V)

phi = ( q.dx(0) + q.dx(1) ) 

q_line = assemble(interpolate(q, Vline))
phi_line = assemble(interpolate(phi, Vline))


_q_avg = assemble(q_line * dx)
_phi_avg = assemble(phi_line * dx)

dz = 1. / (N+1)
#integration_bound = np.arange(0. + dz/2, 1, dz)
integration_bound = np.linspace(0, 1, N+1)


_psi = np.zeros(integration_bound.shape[0], dtype=float)
for i in range(_psi.shape[0]):
    w = conditional(zeta <= integration_bound[i], 1., 0.)
    #_psi[i] = assemble((_phi_avg - phi_line) * w * dx)
    _psi[i] = assemble((_q_avg - q_line) * w * dx)

outfile = VTKFile("swe.pvd")
q_vom = Function(VOM).interpolate(q)
print(q_vom.dat.data_ro.shape)
print(q.dat.data_ro.shape)
#print(q_vom.dat.data)
#print(q.dat.data)
q_vom_v = Function(V)
q_vom_v.dat.data_with_halos[:] = q_vom.dat.data_with_halos[:]
q_v1 = Function(V1).interpolate(q_vom_v)
print(q_v1.dat.data.shape)
#q_vom_v = Function(V).interpolate(q_vom)
#qnew = Function(V).interpolate(q_vom)
#qnew = Function(V)
#q_vom_ip = Interpolator(TestFunction(V1),  V)

#qnew = project(q_vom, V)
#print(q_vom.dat.data_ro)
#q_vom_v = Function(V)
#project(q_vom, V)
outfile.write(project(q, V, name="q"), project(q_vom_v, V, name='q_vom'), project(q_v1, Vout1, name='q_v1'), time=0)
print(vertex_coords.shape)
#print(q_vom.dat.data_ro.shape)
#print(_psi.shape)


#print(f'h_avg: {_q_avg}')
#print(f'phi_avg: {_phi_avg}')
#print(f'psi: {_psi}')
#
#print(vertex_coords)
#print(_psi)

