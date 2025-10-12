import os
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import fem
import basix
import tqdm
from petsc4py import PETSc
import ufl
from  dolfinx.fem import petsc
import dolfinx
from dolfinx import mesh as mesh

from attrs import define, field

import numpy.typing as npt

from library.python.mesh.mesh import Mesh
import library.transformation.to_ufl as trafo
from library.dolfinx.mesh import load_mesh, evaluate_on_all_facets_midpoint


@define(frozen=True, slots=True, kw_only=True)            
class Solver():


    def create_function_space(self, domain):
        elem_Q = basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(3,))
        space_Q = fem.functionspace(domain, elem_Q)
        elem_Qaux= basix.ufl.element("DG", domain.topology.cell_name(), 0, shape=(2,))
        space_Qaux = fem.functionspace(domain, elem_Qaux)
        return space_Q, space_Qaux
    
    
    
    def numerical_flux(self, model, Ql, Qr, Qauxl, Qauxr, parameters, n, domain):
    
        I = ufl.as_tensor([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        return ufl.dot(0.5 *(model.flux(Ql, Qauxl, parameters)+ model.flux(Qr, Qauxr, parameters)), n)- 0.5 * 0.5*(max_abs_eigenvalue(model, Ql, Qauxl, n, domain) + max_abs_eigenvalue(model, Qr, Qauxr, n, domain) )* I * (Qr- Ql)
    
    def extract_scalar_fields(self, Q):
        n_dofs = Q.function_space.num_sub_spaces
        out = []
        for i in range(n_dofs):
            qi = Q.sub(i).collapse()
            # qi.x.array[qi.x.array < 1e-12] = 0.
            qi.name = f"q_{i}"
            out.append(qi)
        return out
    
    def _max_abs_eigenvalue(self, model, Q, Qaux, n, domain):
        
        eigenvalues = model.eigenvalues(Q, Qaux, model.parameters, n)
        evs = evaluate_on_all_facets_midpoint(eigenvalues, domain)[1]
        return np.max(abs(evs))
    
    def max_abs_eigenvalue(self, model, Q, Qaux, n, domain):
        ev = model.eigenvalues(Q, Qaux, model.parameters, n)
        max_ev = abs(ev[0, 0])
        for i in range(1, model.n_variables):
            max_ev = ufl.conditional(ev[1, 0] > max_ev, ev[1, 0], max_ev)
        return max_ev
    
    def compute_time_step_size(self, model, Q, Qaux, n, reference_cell_diameter, domain, CFL=0.45):
        
        n1 = ufl.as_vector((1, 0))
        n2 = ufl.as_vector((0, 1))
        evs_m = _max_abs_eigenvalue(model, Q, Qaux, n1, domain)
        evs_p = _max_abs_eigenvalue(model, Q, Qaux, n2, domain)
    
        local_max_eigenvalue = max(evs_m, evs_p)
    
        # Global maximum reduction across all ranks
        global_max_eigenvalue = MPI.COMM_WORLD.allreduce( local_max_eigenvalue, op=MPI.MAX)
        
        dt = CFL * reference_cell_diameter / global_max_eigenvalue
        
        if np.isnan(dt) or np.isinf(dt) or dt < 10**(-6):
            dt = 10**(-6)
    
        return dt
    
    
    # %%
    def update_qaux(self, Q, Qaux, Qold, Qauxold, model, parameters, time, dt):
        return Qaux
    
    
    # %%
    def weak_form_swe(self, model, functionspace, q_n, q_np, qaux_n, qaux_np, parameters, t, x, dt, domain, cell_tags, facet_tags, unique_facet_tags, facet_boundary_function_id):
        # facet normals
        # domain = functionspace.mesh
        n = ufl.FacetNormal(domain)
    
        # our integration measures over the inner boundaries, the domain boundaries and the whole domain. 
        # Note that we separate the domain boundaries in order to potentially apply different boundary conditions
        # on each side
        dS = ufl.Measure("dS", domain=domain)
        # facet_tags = generate_facets_tags(domain, P0, P1)
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        dx = ufl.dx
    
    
        # implicit/explicit switch
        q = q_n
        qaux = qaux_n
        # q_ghost = ufl.Function(functionspace)
    
    
        # We would like to have gradients of the bottom topography. However, DG0 is constant in each cell, resulting in zero gradients.
        # we help ourselves by projecting DG0 to a CG1 (linear continuous functions) space, where the gradients do exist.
        # note that this is a 'cheap trick'. In reality, the computation of the bottom topography gradient is critical and deserves
        # more attention.
        elem_CG1 = basix.ufl.element("CG", domain.topology.cell_name(), 1)
        space_CG1 = fem.functionspace(domain, elem_CG1)
    
    
    
        test_q = ufl.TestFunction(functionspace)
        trial_q = ufl.TrialFunction(functionspace)
        
        # weak formulation
        weak_form =  ufl.dot(test_q, (trial_q-q)/dt) * dx
        weak_form += ufl.dot((test_q("+") - test_q("-")), 
                             numerical_flux(model, q("+"), q("-"), qaux("+"), qaux("-"), parameters, n("+"), domain)) * dS
        # weak_form += ufl.dot((test_q), numerical_flux(q, q_extrapolation, n)) * (ds(1) + ds(2) + ds(3) + ds(4))
        for i, tag in enumerate(unique_facet_tags):
            # q_ghost.interpolate(boundary_functions[i](q))
            #TODO dX is wrong
            dX = x[0]
            weak_form += ufl.dot((test_q), numerical_flux(model, q, model.bcs(t, x, dX, q, qaux, parameters, n)[i,:], qaux, qaux,parameters, n, domain)) * ds(tag)
            # weak_form += ufl.dot((test_q), numerical_flux(q, q, n)) * ds(tag)
    
        #################TODO#################################
        weak_form += 0
        ######################################################
    
    
        weak_form_lhs = fem.form(ufl.lhs(weak_form))
        weak_form_rhs = fem.form(ufl.rhs(weak_form))
    
        return weak_form_lhs, weak_form_rhs
    
    
    def prepare_solver(self, weak_form_lhs, weak_form_rhs):
        A = petsc.create_matrix(weak_form_lhs)
        b = petsc.create_vector(weak_form_rhs)
    
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.BCGS)
        preconditioner = solver.getPC()
        preconditioner.setType(PETSc.PC.Type.JACOBI)
    
        return solver, A, b
    
    
    def solve_time_loop(self, name: str, path_to_mesh: str, model, weak_form_function,  initial_condition, end_time, output_path, CFL):
        
        domain, cell_tags, facet_tags, unique_facet_tags, facet_boundary_function_id, min_inradius = load_mesh(path_to_mesh)
        
        model = trafo.FenicsXRuntimeModel.from_model(domain, model)
        
        ### Parameters
        gx = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(self, 0))
        gy = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(self, 0))
        gz = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(self, 9.81))
        friction_coeff = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(self, 0.))
        
        reference_edge_length = min_inradius
        
        functionspace, functionspace_qaux = create_function_space(domain)
        parameters = model.parameters
        
        
        
        t = fem.Constant(domain, dolfinx.default_scalar_type(self, 0.0))    
        
        # dt will be overwritten
        dt = fem.Constant(domain, dolfinx.default_scalar_type(self, 0.1))    
    
        x = ufl.SpatialCoordinate(domain)
        x = ufl.as_vector((x[0], x[1], 0)) 
            
        q_n = fem.Function(functionspace, name=r'$q^n$')
        q_np1 = fem.Function(functionspace, name=r'$q^{n+1}$')
        
        qaux_n = fem.Function(functionspace_qaux, name=r'$q_{aux}$')
        qaux_np1 = fem.Function(functionspace_qaux, name=r'$q_{aux}^{n+1}$')
        
        q_n.interpolate(initial_condition)
        q_np1.interpolate(initial_condition)
        
        normals = ufl.FacetNormal(domain)
    
    
        update_qaux(q_n, qaux_n, q_np1, qaux_np1, model, model.parameters, t, dt)
        
        weak_form_lhs, weak_form_rhs = weak_form_function(model, functionspace, q_n, q_np1, qaux_n, qaux_np1, parameters, t, x, dt, domain, cell_tags, facet_tags, unique_facet_tags, facet_boundary_function_id)
        
        solver, A, b = prepare_solver(weak_form_lhs, weak_form_rhs)    
        A = petsc.create_matrix(weak_form_lhs)
        b = petsc.create_vector(weak_form_rhs)
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.BCGS)
        preconditioner = solver.getPC()
        preconditioner.setType(PETSc.PC.Type.JACOBI)
    
        num_timesteps = int(end_time/dt.value)
        
        # VTK writer
        os.makedirs(output_path, exist_ok=True)
        vtk_file_abs_path_name = os.path.join(output_path, f"{name}.pvd")
        vtk_writer = dolfinx.io.VTKFile(
            domain.comm, vtk_file_abs_path_name, "w+"
        )
        
    
        vtk_writer.write_function(extract_scalar_fields(q_n), t=0.0)
        n_snapshots = 50
        dt_snapshot = end_time / n_snapshots
        next_snapshot_time = dt_snapshot
        
    
        progress = tqdm.tqdm(desc="Setup " + name + ", solving PDE", total=end_time)
    
        max_timesteps = 10000
        i=0
        while t.value < end_time and i < max_timesteps:
            q_n.interpolate(q_np1)
            
            # time step size estimation
            dt.value  = compute_time_step_size(model, q_np1, qaux_np1, normals, reference_edge_length, domain, CFL=CFL)
            progress.update(dt.value)
    
    
            A.zeroEntries()
            petsc.assemble_matrix(A,weak_form_lhs)
            A.assemble()
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b,weak_form_rhs)
    
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            solver.solve(b, q_np1.x.petsc_vec)
    
            q_np1.x.scatter_forward()
            t.value += dt.value
            i += 1
            
            
            if t.value > next_snapshot_time:
                vtk_writer.write_function(extract_scalar_fields(q_np1), t=t.value)
                next_snapshot_time += dt_snapshot
    
        progress.close()
        return q_np1
