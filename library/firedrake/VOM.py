from firedrake import *
from firedrake.__future__ import interpolate
from copy import deepcopy

from mpi4py import MPI
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 10
Nz = 3
mm = UnitIntervalMesh(N)
m = ExtrudedMesh(mm, layers=Nz)
sampling_factor = 1
mm_sampled = UnitIntervalMesh(sampling_factor*N)
m_sampled = ExtrudedMesh(mm_sampled, layers=sampling_factor*Nz)
dim = 2

def compute_cell_centers(_mesh):
    x = SpatialCoordinate(_mesh)
    DG0 = VectorFunctionSpace(_mesh, "DG", 0)
    cell_centers = Function(DG0)

    # Project the spatial coordinates into the DG0 space (averaging per cell)
    cell_centers.interpolate(as_vector(x))
    return cell_centers.dat.data

def compute_vertex_coords(_mesh):
    vertex_coords = _mesh.coordinates.dat.data_with_halos
    return vertex_coords

def create_line(a, b, N):
    dim = len(a)
    cells = np.asarray([[i, i + 1] for i in range(N)])
    vertex_coords = np.linspace(a, b, N + 1)
    topo_dim = 1
    plex = mesh.plex_from_cell_list(topo_dim, cells, vertex_coords, comm=m.comm)
    line = mesh.Mesh(plex, dim=dim)
    return line, vertex_coords

def create_line_with_subsamples(a, b, N):
    dim = len(a)
    cells = np.asarray([[i, i + 1] for i in range(N)])
    cells_sub = np.asarray([[i, i + 1] for i in range(3*N)])
    vertex_coords = np.linspace(a, b, N + 1)
    vc = vertex_coords
    _dx = vc[1]-vc[0]
    #TODO does not work this way
    vertex_coords_sub = np.array([vc[i]-0.99*_dx, vc[i], vc[i+1]-eps] for i in range(vertex_coords.shape[0]))
    topo_dim = 1
    plex = mesh.plex_from_cell_list(topo_dim, cells, vertex_coords, comm=m.comm)
    line = mesh.Mesh(plex, dim=dim)
    return line, vertex_coords

line, vertex_coords = create_line([0.0, 0.5], [1.0, 0.5], sampling_factor*N)
Vline = FunctionSpace(line, "DG", 0)

V = FunctionSpace(m, "DG", 1)
Vout1 = FunctionSpace(m, "DG", 1)
zeta, _ = SpatialCoordinate(line)
zeta_coords = line.coordinates.dat.data

cell_centers = compute_cell_centers(m)
cell_centers_sampled = compute_cell_centers(m_sampled)
vertex_coords = compute_vertex_coords(m)
vertex_coords_sampled = compute_vertex_coords(m_sampled)

m_vom = VertexOnlyMesh(m_sampled, cell_centers_sampled)
#m_vom = VertexOnlyMesh(m_sampled, vertex_coords_sampled)

VOM = FunctionSpace(m_vom, "DG", 0)
V0_vom = FunctionSpace(m_sampled, "DG", 0)

x, y = SpatialCoordinate(m)
q_f = x  + y

q = Function(V, name='q').interpolate(q_f)

phi = ( q.dx(0) + q.dx(1) ) 

q_line = assemble(interpolate(q, Vline))
phi_line = assemble(interpolate(phi, Vline))


_q_avg = assemble(q_line * dx)
_phi_avg = assemble(phi_line * dx)

dz = 1. / (N+1)
#integration_bound = np.arange(0. + dz/2, 1, dz)
integration_bound = np.linspace(0, 1, 2*Nz+1)


_psi = np.zeros(integration_bound.shape[0], dtype=float)
for i in range(_psi.shape[0]):
    w = conditional(zeta <= integration_bound[i], 1., 0.)
    #_psi[i] = assemble((_phi_avg - phi_line) * w * dx)
    _psi[i] = assemble((_q_avg - q_line) * w * dx)

outfile = VTKFile("swe.pvd")

# project q onto VOM space
q0 = Function(V0_vom).interpolate(q)
q_vom = Function(VOM).interpolate(q0)

# now I can e.g. extract data, integrate and do whatever with q_vom
#TODO do integration here. 

# project back to V0 and then to V
q_vom_v = Function(V0_vom)
q_vom_v.dat.data_with_halos[:] = q_vom.dat.data_with_halos[:]
q_v1 = Function(V).interpolate(q_vom_v)

print(f'max q: {q.dat.data_with_halos.max()}')
print(f'max q_v1: {q_v1.dat.data_with_halos.max()}')

outfile.write(project(q, V, name="q"), project(q_v1, Vout1, name='q_v1'), time=0)
