from firedrake import *
from mpi4py import MPI  # Import MPI explicitlyfrom mpi4py import MPI  # Import MPI explicitla
from time import time
from firedrake.__future__ import interpolate
import numpy as np

from depth_integrator_new import DepthIntegrator


# Create the base mesh and extrude it
N = 10
#base_mesh = Mesh('./../meshes/quad_2d/mesh_fine.msh')
base_mesh = UnitSquareMesh(N,N , quadrilateral=True)
Nz = 5

n_dof_base = 4
_mesh = ExtrudedMesh(base_mesh, layers=Nz)
#print(mesh.coordinates.dat.data)

# Define the function space and the function f
DIM_H = 1
DIM_V = 1
horiz_elt = FiniteElement("DG", quadrilateral, DIM_H)
vert_elt = FiniteElement("DG", interval, DIM_V)
elt = TensorProductElement(horiz_elt, vert_elt)

W = VectorFunctionSpace(_mesh, elt)
V = FunctionSpace(_mesh, elt)
Wout = VectorFunctionSpace(_mesh, "DG", 0) 
Vout = FunctionSpace(_mesh, "DG", 0) 

V_rank = FunctionSpace(_mesh, "DG", 1)
rank = MPI.COMM_WORLD.rank
rank_field = Function(V_rank)
rank_field.interpolate(Constant(rank))

x, y, z = SpatialCoordinate(_mesh)

dof_points = Function(W).interpolate(as_vector([x,y,z]))

# Define the output function g
HU = Function(W).interpolate(as_vector([0., 0., 0.]))


h0 = conditional(And(And(x <= 0.33, x >= -0.33), And(y <= 0.33, y >= -0.33)), 2., 1.)
H = Function(V).interpolate(h0)

Hb = Function(V).interpolate(0.)


HUm = Function(W).interpolate(as_vector([0., 0., 0.]))
omega = Function(V).interpolate(0.)

phi_symbolic = HU.sub(0).dx(0) + HU.sub(1).dx(1)
phi = Function(V).interpolate(phi_symbolic)
dxh = Function(V).interpolate(H.dx(0))
dyh =  Function(V).interpolate(H.dx(1))
dxhb = Function(V).interpolate(Hb.dx(0))
dyhb = Function(V).interpolate(Hb.dx(1))

outfile = VTKFile("out.pvd")

num_layers = Nz
num_cells_base = base_mesh.num_cells()
#num_dofs_per_cell = Vh.finat_element.space_dimension()
num_dofs_per_cell = 4
num_cells_extruded = num_cells_base * num_layers
DI = DepthIntegrator(num_layers, num_dofs_per_cell, DIM_V)

t = 0.
HU, HUm, omega = DI.integrate(H, HU, Hb, HUm, omega, dof_points,  phi, dxh, dyh, dxhb, dyhb)
#HU.assign(HU)
#HUm.assign(HU)
omega.assign(omega)
outfile.write(project(H, Vout, name="H"),  project(HU, Wout, name="HU"), project(HUm, Wout, name="HU_mean"), project(omega, Vout, name='omega'), project(rank_field, V_rank, name='rank'), time=t)

v = TestFunction(V)
v_ = TrialFunction(V)
w = TestFunction(W)
w_ = TrialFunction(W)

# Setup
CFL = 1./5. * 0.5
incircle = (1./N)
g = 9.81
nu = 0.0

n = FacetNormal(_mesh)

def get_max_abs_ev(H, HU):
    return 10;
    # Compute the local maximum
    sqrt_g_H = sqrt(g*H)
    eigenvalues = Function(W).interpolate(abs(HU)/H + as_vector([sqrt_g_H, sqrt_g_H, 0.]))
    local_max = eigenvalues.dat.data_ro.max()
    
    # Perform a global reduction to get the global maximum
    comm = _mesh.comm
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    return global_max


# lambdas
P_hyd = as_tensor([[0.5*g*H**2, 0., 0.], [0., 0.5*g*H**2, 0.], [0., 0., 0.]])
convection = as_tensor([[HU[0]**2/H, HU[0]*HU[1]/H, HU[0]*omega ], [HU[0]*HU[1]/H, HU[1]**2/H, HU[1]*omega], [0., 0., 0.]])
#stress = as_tensor([[0., 0., 0], [0., 0., 0], [0., 0., 0.]])
stress = as_tensor([[0., 0., -nu/H*(HU[0]/H).dx(2)], [0., 0., -nu/H*(HU[1]/H).dx(2)], [0., 0., 0.]])

ev_n = abs(dot(HU/H, n)) + sqrt(g * H)
ev = abs(norm(HU/H)) + sqrt(g*H)



p = '+'
m = '-'

quad_degree_h = 2
quad_degree_hu = 2

# MASS BALANCE
# time / mass matrices
a_mass = inner(v, v_) * dx(degree=quad_degree_h)
DG_H = inner(grad(v), HUm) * dx(degree=quad_degree_h)


f_H = HUm

#flux
F_H_l = f_H(p)
F_H_r = f_H(m)
F_H_n = 0.5*dot((F_H_l + F_H_r), n(m))
F_dis_H = -0.5  * avg(ev_n) * (H(p)-H(m))
F_H = (v(p)-v(m)) * (F_H_n + F_dis_H) * (dS_h(degree=quad_degree_h) + dS_v(degree=quad_degree_h))


def wall_H():
    #Ghost cells
    H_g =H 
    HUm_n = dot(HUm, n) * n
    HUm_t = HUm - HUm_n
    HUm_g = -HUm_n + HUm_t
    BC_H_l = HUm 
    BC_H_r = HUm_g
    BC_H_n = 0.5*dot((BC_H_l + BC_H_r), n)
    BC_dis_H = -0.5 * ev_n * (H_g - H)
    return BC_H_n + BC_dis_H

def inflow_H():
    #Ghost cells
    H_BC = Function(V).interpolate(2.)
    #H_g = Function(V).interpolate(conditional(And(y > -0.3, y < 0.3), H_BC, H))
    H_g = H_BC
    HU_BC = Function(W).interpolate(as_vector([0.1, 0., 0.]))
    #HUm_g = Function(W).interpolate(conditional(And(y > -0.3, y < 0.3), HUm, HU_BC))
    HUm_g = HU_BC
    BC_H_l = HUm 
    BC_H_r = HUm_g
    BC_H_n = 0.5*dot((BC_H_l + BC_H_r), n)
    BC_dis_H = -0.5 * ev_n * (H_g - H)
    return BC_H_n + BC_dis_H


#BC 
#BC_H = -v*(wall_H()) * (ds_v(degree=quad_degree_h) + ds_t(degree=quad_degree_h) + ds_b(degree=quad_degree_h))
#BC_H  = -v*(wall_H()) * (ds_v(1000, degree=quad_degree_h))
#BC_H += -v*(wall_H()) * (ds_v(1001, degree=quad_degree_h))
#BC_H += -v*(wall_H()) * (ds_v(1002, degree=quad_degree_h))
#BC_H += -v*(wall_H()) * (ds_v(1003, degree=quad_degree_h))
#BC_H += -v*(wall_H()) * (ds_t(degree=quad_degree_h) + ds_b(degree=quad_degree_h))



##MOMENTUM BALANCE
# time / mass matrices
f_HU = convection + P_hyd + stress
a_mom = inner(w, w_) * dx
DG_HU = inner(grad(w), f_HU) * dx

# flux

F_HU_l = f_HU(p)
F_HU_r = f_HU(m)
F_HU_n = 0.5*dot((F_HU_l + F_HU_r), n(m))
F_dis_HU = -0.5 * avg(ev_n) * (HU(p)-HU(m))
F_HU = dot((w(p)-w(m)), (F_HU_n + F_dis_HU)) * (dS_h(degree=quad_degree_hu) + dS_v(degree=quad_degree_hu))


# BC
def wall_HU():
    BC_HU_l = f_HU
    
    HU_n = dot(HU, n) * n
    HU_t = HU - HU_n
    HU_g = -HU_n + HU_t
    P_hyd_g = as_tensor([[0.5*g*H**2, 0., 0.], [0., 0.5*g*H**2, 0.], [0., 0., 0.]])
    convection_g = as_tensor([[HU[0]**2/H, HU[0]*HU[1]/H, HU[0]*omega ], [HU[0]*HU[1]/H, HU[1]**2/H, HU[1]*omega], [0., 0., 0.]])
    #stress_g = as_tensor([[0., 0., 0], [0., 0., 0], [0., 0., 0.]])
    stress_g = as_tensor([[0., 0., -nu/H*(HU_g/H)[0].dx(2)], [0., 0., -nu/H*(HU_g/H)[1].dx(2)], [0., 0., 0.]])
    f_HU_g = convection_g + P_hyd_g + stress_g
    BC_HU_r = f_HU_g
    
    BC_HU_n = 0.5*dot((BC_HU_l + BC_HU_r), n)
    BC_dis_HU = -0.5 * ev_n * (HU_g - HU)
    return BC_HU_n + BC_dis_HU

def inflow_HU():
    BC_HU_l = f_HU

    H_BC = Function(V).interpolate(2.)
    #H_g = Function(V).interpolate(conditional(And(y > -0.3, y < 0.3), H_BC, H))
    H_g = H_BC

    
    HU_n = dot(HU, n) * n
    HU_t = HU - HU_n
    HU_wall = -HU_n + HU_t

    #HU_in = HU 
    HU_in = Function(W).interpolate(as_vector([0.1, 0., 0.]))
    #HU_g = Function(W).interpolate(conditional(And(y > -0.3, y < 0.3), HU_in, HU_walll))
    #HU_g = conditional(-2* dot(HU, n) * n + HU)
    #HU_g = conditional(And(y > -0.3, y < 0.3), HU_in, HU_wall)
    HU_g = HU_in
    #HU_g =HU_wall

    P_hyd_g = as_tensor([[0.5*g*H_g**2, 0., 0.], [0., 0.5*g*H_g**2, 0.], [0., 0., 0.]])
    convection_g = as_tensor([[HU_g[0]**2/H_g, HU_g[0]*HU_g[1]/H_g, HU_g[0]*omega ], [HU_g[0]*HU_g[1]/H_g, HU_g[1]**2/H_g, HU_g[1]*omega], [0., 0., 0.]])
    #stress_g = as_tensor([[0., 0., 0], [0., 0., 0], [0., 0., 0.]])
    stress_g = as_tensor([[0., 0., -nu/H_g*(HU_g/H_g)[0].dx(2)], [0., 0., -nu/H_g*(HU_g/H_g)[1].dx(2)], [0., 0., 0.]])
    f_HU_g = convection_g + P_hyd_g + stress_g
    BC_HU_r = f_HU_g
    
    BC_HU_n = 0.5*dot((BC_HU_l + BC_HU_r), n)
    BC_dis_HU = -0.5 * ev_n * (HU_g - HU)
    return BC_HU_n + BC_dis_HU

#BC_HU = -dot(w, (wall_HU())) * (ds_v(degree=quad_degree_hu) + ds_b(degree=quad_degree_hu) + ds_t(degree=quad_degree_hu))
#BC_HU  = -dot(w, (wall_HU())) * (ds_v(1000, degree=quad_degree_hu))
#BC_HU += -dot(w, (wall_HU())) * (ds_v(1001, degree=quad_degree_hu))
#BC_HU += -dot(w, (wall_HU())) * (ds_v(1002, degree=quad_degree_hu))
#BC_HU += -dot(w, (wall_HU())) * (ds_v(1003, degree=quad_degree_hu))
#BC_HU += -dot(w, (wall_HU())) * (ds_b(degree=quad_degree_hu) + ds_t(degree=quad_degree_hu))
                                        





#dt = CFL * incircle / get_max_abs_ev(H, HU)
dt = 0.001

# L_H = dt * (DG_H + F_H + BC_H)
# L_HU = dt * (DG_HU  + F_HU + BC_HU)

L_H = dt * (DG_H + F_H)
L_HU = dt * (DG_HU  + F_HU)

#hh1 = Function(Vh)
#hh2 = Function(Vh)
#HU1 = Function(W)
#HU2 = Function(W)
#L_H_1 = replace(L, {h: h1, hu: hu1})
#L2 = replace(L, {h: h2, hu: hu2})
#params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
#params = {"ksp_type": "cg"}
params = {"ksp_type": "cg", "pc_type": "jacobi"}
#params = {
#    "snes_type": "newtonls",
#    "ksp_type": "gmres",
#    "pc_type": "asm",  # Additive Schwarz method
#    "sub_pc_type": "ilu"
#}


dH = Function(V)
dHU = Function(W)
prob_H = LinearVariationalProblem(a_mass, L_H, dH)
solv_H = LinearVariationalSolver(prob_H, solver_parameters=params)
prob_HU = LinearVariationalProblem(a_mom, L_HU, dHU)
solv_HU = LinearVariationalSolver(prob_HU, solver_parameters=params)

ksp_H = solv_H.snes.ksp
ksp_HU = solv_HU.snes.ksp

#prob1 = LinearVariationalProblem(a, L1, dQ)
#solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
#prob2 = LinearVariationalProblem(a, L2, dQ)
#solv2 = LinearVariationalSolver(prob2, solver_parameters=params)


# LIMITER
limiter = VertexBasedLimiter(V)
lim_func = V.get_work_function()

def apply_limiter_H(field):
    limiter.apply(field)

def apply_limiter_HU(field):
    for i in range(W.value_size):
        lim_func.dat.data_with_halos[:] = field.dat.data_with_halos[:, i]
        #lim_func.dat.data[:] = field.dat.data_ro[:, i]
        limiter.apply(lim_func)
        #field.dat.data_with_halos[:, i] = lim_func.dat.data_ro[:]
        field.dat.data_with_halos[:, i] = lim_func.dat.data_with_halos[:]
        #field.dat.data[:, i] = lim_func.dat.data_ro


def apply_limiter(H, HU):
    apply_limiter_H(H)
    apply_limiter_HU(HU)


T = 3.
step = 0
output_freq = 1


start0 = time()
while t < T - 0.5 * dt:
    start = time()
    #dt = CFL * incircle / get_max_abs_ev(H, HU)

    solv_H.solve()
    solv_HU.solve()

    H.assign(H + dH)
    HU.assign(HU+dHU)
    apply_limiter(H, HU)
    #DI.integrate(H, HU, Hb, HUm, omega, dof_points, phi, dxh, dyh, dxhb, dyhb)

    phi_symbolic = HU.sub(0).dx(0) + HU.sub(1).dx(1)
    phi = Function(V).interpolate(phi_symbolic)
    dxh = Function(V).interpolate(H.dx(0))
    dyh =  Function(V).interpolate(H.dx(1))
    dxhb = Function(V).interpolate(Hb.dx(0))
    dyhb = Function(V).interpolate(Hb.dx(1))
    HU, HUm, omega = DI.integrate(H, HU, Hb, HUm, omega, dof_points,  phi, dxh, dyh, dxhb, dyhb)
    #HU.assign(HU)
    #HUm.assign(HUm)
    #HUm.assign(HU.copy())
    #HUm.assign(Function(W).interpolate(HU))
    #omega.assign(omega)
    #HU.assign(HU)
    #HUm.assign(HUm)
    #HUm.assign(Function(W).interpolate(HU))
    #omega.assign(omega)

    # Update field
    #phi.assign(Function(V).interpolate(phi_symbolic))
    #dxh.assign(Function(V).interpolate(H.dx(0)))
    #dyh.assign(Function(V).interpolate(H.dx(1)))
    #dxhb.assign(Function(V).interpolate(Hb.dx(0)))
    #dyhb.assign(Function(V).interpolate(Hb.dx(1)))


    step += 1
    t += dt

    if step % output_freq == 0:
        #max_h = project(h, V).vector().max()
        #error = assemble((abs(h - h0)) * dx)
        #L2_err = sqrt(assemble((h - h0) * (h - h0) * dx))
        #L2_init = sqrt(assemble(h0 * h0 * dx))
        #error = L2_err / L2_init
        #outfile.write(project(U, Wout, name="U"), project(omega, Vout, name='omega'), time=t)
        #outfile2d.write(project(hh, Vhout, name="h"), project(Um, Whout, name='Um'), project(phim, Vhout, name="phi_mean"), time=t)
        outfile.write(project(H, Vout, name="H"),  project(HU, Wout, name="HU"), project(HUm, Wout, name="HU_mean"), project(omega, Vout, name='omega'), project(rank_field, V_rank, name='rank'), time=t)
        if _mesh.comm.rank == 0:
            print(
                f"global time: {time()-start0} \t solution time:{t} \t  dt:{dt} \t step time:{time()-start}"
            )
        
        #print(
        #    f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L2/|h_0|_L2={error}, h_max={max_h}"
        #)
