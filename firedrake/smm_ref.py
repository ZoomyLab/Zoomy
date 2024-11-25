from firedrake import *
from mpi4py import MPI  # Import MPI explicitlyfrom mpi4py import MPI  # Import MPI explicitla
from time import time
from firedrake.__future__ import interpolate
import numpy as np

from depth_integrator_new import DepthIntegrator


# Create the base mesh and extrude it
N = 20
base_mesh = UnitSquareMesh(N,N , quadrilateral=True)
Nz = 10

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
Wout = VectorFunctionSpace(_mesh, "CG", 1) 
Vout = FunctionSpace(_mesh, "CG", 1) 

x, y, z = SpatialCoordinate(_mesh)

dof_points = Function(W).interpolate(as_vector([x,y,z]))

# Define the output function g
HU = Function(W).interpolate(as_vector([0., 0., 0.]))

h0 = conditional(And(And(x < 0.66, x > 0.33), And(y < 0.66, y > 0.33)), 2., 1.)
H = Function(V).interpolate(h0)

Hb = Function(V).interpolate(0.)

HUm = Function(W).interpolate(as_vector([0., 0., 0.]))
omega = Function(V).interpolate(0.)

outfile = VTKFile("out.pvd")

num_layers = Nz
num_cells_base = base_mesh.num_cells()
#num_dofs_per_cell = Vh.finat_element.space_dimension()
num_dofs_per_cell = 4
num_cells_extruded = num_cells_base * num_layers
DI = DepthIntegrator(num_layers, num_cells_base, num_dofs_per_cell, DIM_V)

t = 0.
DI.integrate(H, HU, Hb, HUm, omega, dof_points)
outfile.write(project(H, Vout, name="H"),  project(HU, Wout, name="HU"), project(omega, Vout, name='omega'), time=t)

v = TestFunction(V)
w = TestFunction(W)
v_ = TrialFunction(V)
w_ = TrialFunction(W)

# Setup
T = 0.1
CFL = 1./5. * 0.1
incircle = (1./N)
g = 9.81
nu = 0.0

n = FacetNormal(_mesh)

def get_max_abs_ev(H, HU):
    # Compute the local maximum
    sqrt_g_H = sqrt(g*H)
    eigenvalues = Function(W).interpolate(abs(HU)/H + as_vector([sqrt_g_H, sqrt_g_H, 0.]))
    #local_max = as_vector([abs(HU.sub(0)/H) + sqrt(g * H), abs(HU.sub(1)/H) + sqrt(g*H)]).dat.data_ro.max()
    local_max = eigenvalues.dat.data_ro.max()
    
    # Perform a global reduction to get the global maximum
    comm = _mesh.comm
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    #if _mesh.comm.rank == 0:
    #    print(f"Global maximum: {global_max}")
    return global_max


# lambdas
P_hyd = as_tensor([[0.5*g*H**2, 0., 0.], [0., 0.5*g*H**2, 0.], [0., 0., 0.]])
convection = as_tensor([[HU[0]**2/H, HU[0]*HU[1]/H, HU[0]*omega ], [HU[0]*HU[1]/H, HU[1]**2/H, HU[1]*omega], [0., 0., 0.]])
#stress = as_tensor([[0., 0., 0], [0., 0., 0], [0., 0., 0.]])
stress = as_tensor([[0., 0., -nu/H*(HU[0]/H).dx(2)], [0., 0., -nu/H*(HU[1]/H).dx(2)], [0., 0., 0.]])

ev_n = abs(dot(HU/H, n)) + sqrt(g * H)
ev = abs(norm(HU/H)) + sqrt(g*H)
#evh = lambda h, U, n: abs(dot(U, n)) + sqrt(g * h)



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

#Ghost cells
H_g =H 
HUm_n = dot(HUm, n) * n
HUm_t = HUm - HUm_n
HUm_g = -HUm_n + HUm_t

#BC 
BC_H_l = HUm 
BC_H_r = HUm_g
BC_H_n = 0.5*dot((BC_H_l + BC_H_r), n)
BC_dis_H = -0.5 * ev_n * (H_g - H)
BC_H = -v*(BC_H_n + BC_dis_H) * (ds_v(degree=quad_degree_h) + ds_t(degree=quad_degree_h) + ds_b(degree=quad_degree_h))



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
BC_HU = -dot(w, (BC_HU_n + BC_dis_HU)) * (ds_v(degree=quad_degree_hu) + ds_b(degree=quad_degree_hu) + ds_t(degree=quad_degree_hu))
                                        




#ev_cell = lambda h, U: sqrt(U.sub(0)**2 + U.sub(1)**2 + U.sub(2)**2) + sqrt(g * h)
#ev_cell = lambda h, U:  norm(U) + sqrt(g * h)
#ev_max = lambda h, U: max(project(ev_cell(h, U), V).vector())

dt = CFL * incircle / get_max_abs_ev(H, HU)
#dt = 0.0001

L_H = dt * (DG_H + F_H + BC_H)
L_HU = dt * (DG_HU  + F_HU + BC_HU)

#hh1 = Function(Vh)
#hh2 = Function(Vh)
#HU1 = Function(W)
#HU2 = Function(W)
#L_H_1 = replace(L, {h: h1, hu: hu1})
#L2 = replace(L, {h: h2, hu: hu2})
#params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
#params = {"ksp_type": "cg"}
params = {
    "snes_type": "newtonls",
    "ksp_type": "gmres",
    "pc_type": "asm",  # Additive Schwarz method
    "sub_pc_type": "ilu"
}
dH = Function(V)
dHU = Function(W)
prob_H = LinearVariationalProblem(a_mass, L_H, dH)
solv_H = LinearVariationalSolver(prob_H, solver_parameters=params)
prob_HU = LinearVariationalProblem(a_mom, L_HU, dHU)
solv_HU = LinearVariationalSolver(prob_HU, solver_parameters=params)

#prob1 = LinearVariationalProblem(a, L1, dQ)
#solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
#prob2 = LinearVariationalProblem(a, L2, dQ)
#solv2 = LinearVariationalSolver(prob2, solver_parameters=params)


# LIMITER
limiter = VertexBasedLimiter(V)
lim_func = V.get_work_function()

def apply_limiter_H(field):
    #field_P1 = project(field, Vh)
    limiter.apply(field)
    #field = project
    #return field_P1

def apply_limiter_HU(field):
    for i in range(W.value_size):
        #lim_func_HU.dat.data_with_halos[:] = field.dat.data_with_halos[:, i]
        lim_func.dat.data[:] = field.dat.data[:, i]
        limiter.apply(lim_func)
        field.dat.data[:, i] = lim_func.dat.data


def apply_limiter(H, HU):
    apply_limiter_H(H)
    apply_limiter_HU(HU)


T = 0.2
step = 0
output_freq = 1


start0 = time()
while t < T - 0.5 * dt:
    start = time()
    #dt = CFL * incircle / ev_max(h, U)
    dt = CFL * incircle / get_max_abs_ev(H, HU)

    #U.project(HU/h)
    #phi = assemble(interpolate((U.sub(0).dx(0) + U.sub(1).dx(1)), V))
    #DI.integrate(h, U, phi, omega, hh, Um, phim, dof_points)
    #print(f'integration time = {time() - start}')

    solv_H.solve()
    H.assign(H + dH)
    solv_HU.solve()
    HU.assign(HU + dHU)
    apply_limiter(H, HU)
    DI.integrate(H, HU,Hb, HUm, omega, dof_points)

    #solv1.solve()
    #Q2.assign(0.5 * Q_ + 0.5 * (Q1+ dQ))
    #apply_limiter(Q2)
    #Q.assign(Q2)
    #Q_.assign(Q2)

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
        outfile.write(project(H, Vout, name="H"),  project(HU, Wout, name="HU"), project(omega, Vout, name='omega'), time=t)
        if _mesh.comm.rank == 0:
            print(
                f"global time: {time()-start0} \t solution time:{t} \t  dt:{dt} \t step time:{time()-start}"
            )
        
        #print(
        #    f"t={t}, dt={dt}, ev_max={ev_max}, |h-h_0|_L2/|h_0|_L2={error}, h_max={max_h}"
        #)
