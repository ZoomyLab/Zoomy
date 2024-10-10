import numpy as np
from types import SimpleNamespace
import pyprog
from attr import define, field
from typing import Callable, Optional, Type
from copy import deepcopy
from time import time as gettime
import os
import shutil

# import logging
# logging.basicConfig(level=logging.DEBUG)

from library.model.model import *
from library.model.models.base import register_parameter_defaults
from library.model.models.shallow_moments import reconstruct_uvw
import library.pysolver.reconstruction as recon
import library.pysolver.flux as flux
import library.pysolver.nonconservative_flux as nonconservative_flux
import library.pysolver.ader_flux as ader_flux
import library.pysolver.timestepping as timestepping
import library.misc.io as io
import library.mesh.fvm_mesh as fvm_mesh
from library.pysolver.ode import *
from library.solver import python_c_interface as c_interface
from library.solver import modify_sympy_c_code as modify_sympy_c_code


@define(slots=True, frozen=False, kw_only=True)
class Settings:
    name: str = "Simulation"
    parameters: dict = {}
    reconstruction: Callable = recon.constant
    reconstruction_edge: Callable = recon.constant_edge
    num_flux: Callable = flux.LF()
    nc_flux: Callable = nonconservative_flux.segmentpath()
    compute_dt: Callable = timestepping.constant(dt=0.1)
    time_end: float = 1.0
    truncate_last_time_step: bool = True
    output_snapshots: int = 10
    output_write_all: bool = False
    output_dir: str = "outputs/output"
    output_clean_dir: bool = True
    solver_code_base: str = "python"
    callbacks: [str] = []
    debug: bool = False
    profiling: bool = False


def _initialize_problem(model, mesh):
    model.boundary_conditions.initialize(
        mesh, model.time, model.position, model.distance, model.variables, model.aux_variables, model.parameters, model.sympy_normal
    )

    n_fields = model.n_fields
    n_cells = mesh.n_cells
    n_aux_fields = model.aux_variables.length()

    Q = np.empty((n_fields, n_cells), dtype=float)
    Qaux = np.zeros((n_aux_fields, n_cells), dtype=float)

    Q = model.initial_conditions.apply(mesh.cell_centers, Q)
    Qaux = model.aux_initial_conditions.apply(mesh.cell_centers, Qaux)
    return Q, Qaux


def _get_compute_max_abs_eigenvalue(mesh, pde, settings):
    # reconstruction = settings.reconstruction
    # reconstruction_edge = settings.reconstruction_edge

    # TODO vectorize
    def compute_max_abs_eigenvalue(Q, Qaux, parameters):
        max_abs_eigenvalue = -np.inf
        eigenvalues_i = np.empty(Q.shape[1], dtype=float)
        eigenvalues_j = np.empty(Q.shape[1], dtype=float)
        for i_face in range(mesh.n_faces):
            i_cellA, i_cellB = mesh.face_cells[:, i_face]
            qA = Q[:, i_cellA]
            qB = Q[:, i_cellB]
            qauxA = Qaux[:, i_cellA]
            qauxB = Qaux[:, i_cellB]

            normal = mesh.face_normals[:, i_face]

            evA = pde.eigenvalues(qA, qauxA, parameters, normal)
            evB = pde.eigenvalues(qB, qauxB, parameters, normal)
            max_abs_eigenvalue = max(
                max_abs_eigenvalue, np.max(np.abs(evA))
            )
            max_abs_eigenvalue = max(
                max_abs_eigenvalue, np.max(np.abs(evB))
            )

        assert max_abs_eigenvalue > 10 ** (-8)

        return max_abs_eigenvalue

    def compute_max_abs_eigenvalue_vectorized(Q, Qaux, parameters):
        max_abs_eigenvalue = -np.inf
        eigenvalues_i = np.empty(Q.shape[1], dtype=float)
        eigenvalues_j = np.empty(Q.shape[1], dtype=float)
        i_cellA = mesh.face_cells[0]
        i_cellB = mesh.face_cells[1]
        qA = Q[:, i_cellA]
        qB = Q[:, i_cellB]
        qauxA = Qaux[:, i_cellA]
        qauxB = Qaux[:, i_cellB]

        normal = mesh.face_normals

        evA = pde.eigenvalues(qA, qauxA, parameters, normal)
        evB = pde.eigenvalues(qB, qauxB, parameters, normal)
        max_abs_eigenvalue = max(np.abs(evA).max(), np.abs(evB).max())

        assert max_abs_eigenvalue > 10 ** (-8)

        return max_abs_eigenvalue


    return compute_max_abs_eigenvalue_vectorized


def _get_source(mesh, pde, settings):
    def source(dt, Q, Qaux, parameters, dQ):
        # Loop over the inner elements
        for i_cell in range(mesh.n_inner_cells):
            dQ[:, i_cell] = pde.source(Q[:, i_cell], Qaux[:, i_cell], parameters)
        return dQ


    def source_vectorized(dt, Q, Qaux, parameters, dQ):
        # Loop over the inner elements
        dQ[:, :mesh.n_inner_cells] = pde.source(Q[:, :mesh.n_inner_cells], Qaux[:, :mesh.n_inner_cells], parameters)
        return dQ

    return source_vectorized


def _get_source_jac(mesh, pde, settings):
    def source_jac(dt, Q, Qaux, parameters, dQ):
        # Loop over the inner elements
        for i_cell in range(mesh.n_inner_cells):
            dQ[:, :,i_cell] = pde.source_jacobian(Q[:,i_cell], Qaux[:, i_cell], parameters)
        return dQ

    return source_jac

def _get_semidiscrete_solution_operator_1d(mesh, pde, bcs, settings):
    compute_num_flux = settings.num_flux
    compute_nc_flux = settings.nc_flux

    def operator(dt, Q, Qaux, parameters, dQ):
        dQ = np.zeros_like(dQ)
        for i_face in range(mesh.n_faces):
            #reconstruct
            i_cellA, i_cellB = mesh.face_cells[:, i_face]
            qA = Q[:, i_cellA]
            qB = Q[:, i_cellB]
            qauxA = Qaux[:, i_cellA]
            qauxB = Qaux[:, i_cellB]
            
            normal = mesh.face_normals[:, i_face]
            flux, failed = compute_num_flux(qA, qB, qauxA, qauxB, parameters, normal, pde)
            assert not failed
            if normal[0] > 0:
                nc_flux, failed = compute_nc_flux(qA, qB, qauxA, qauxB, parameters, normal, pde)
            else:
                nc_flux, failed = compute_nc_flux(qB, qA, qauxB, qauxA, parameters, -normal, pde)
            assert not failed


            dQ[:, i_cellA] -= (
                (flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellA]
            )

            dQ[:, i_cellB] += (
                (flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellB]
            )

            dQ[:, i_cellA] += (
                (nc_flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellA]
            )

            dQ[:, i_cellB] += (
                (nc_flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellB]
            )

        return dQ
    return operator

def _get_semidiscrete_solution_operator(mesh, pde, bcs, settings):
    compute_num_flux = settings.num_flux
    compute_nc_flux = settings.nc_flux
    # reconstruction = settings.reconstruction
    # reconstruction_edge = settings.reconstruction_edge

    def operator_rhs_split(dt, Q, Qaux, parameters, dQ):
        dQ = np.zeros_like(dQ)
        for i_face in range(mesh.n_faces):
            #reconstruct
            i_cellA, i_cellB = mesh.face_cells[:, i_face]
            rA = mesh.face_centers[i_face] - mesh.cell_centers[i_cellA]
            rB = mesh.face_centers[i_face] - mesh.cell_centers[i_cellA]
            qA = Q[:, i_cellA]
            qB = Q[:, i_cellB]
            qauxA = Qaux[:, i_cellA]
            qauxB = Qaux[:, i_cellB]
            
            normal = mesh.face_normals[:, i_face]
            flux, failed = compute_num_flux(qA, qB, qauxA, qauxB, parameters, normal, pde)
            assert not failed
            nc_flux, failed = compute_nc_flux(qA, qB, qauxA, qauxB, parameters, normal, pde)
            # nc_flux = np.zeros_like(flux)
            assert not failed


            dQ[:, i_cellA] -= (
                (flux + nc_flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellA]
            )

            dQ[:, i_cellB] += (
                (flux - nc_flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellB]
            )

        return dQ

    def operator_rhs_split_check(dt, Q, Qaux, parameters, dQ):
        dQ = np.zeros_like(dQ)
        flux_check = np.zeros((3, mesh.n_faces))
        nc_flux_check = np.zeros((3, mesh.n_faces))
        dQ_face_check_m = np.zeros((3, mesh.n_faces))
        dQ_face_check_p = np.zeros((3, mesh.n_faces))
        iA = np.zeros((mesh.n_faces), dtype=int)
        iB = np.zeros((mesh.n_faces), dtype=int)
        for i_face in range(mesh.n_faces):
            #reconstruct
            i_cellA, i_cellB = mesh.face_cells[:, i_face]
            qA = Q[:, i_cellA]
            qB = Q[:, i_cellB]
            qauxA = Qaux[:, i_cellA]
            qauxB = Qaux[:, i_cellB]
            
            normal = mesh.face_normals[:, i_face]
            flux, failed = compute_num_flux(qA, qB, qauxA, qauxB, parameters, normal, pde)
            assert not failed
            # nc_flux, failed = compute_nc_flux(qA, qB, qauxA, qauxB, parameters, normal, pde)
            nc_flux = np.zeros_like(flux)
            assert not failed


            dQ[:, i_cellA] -= (
                (flux + nc_flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellA]
            )

            dQ[:, i_cellB] += (
                (flux - nc_flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellB]
            )
            flux_check[:, i_face] = flux
            nc_flux_check[:, i_face] = nc_flux
            iA[i_face] = i_cellA
            iB[i_face] = i_cellB

            dQ_face_check_m[:, i_face] -= ( 
                (flux + nc_flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellA]
            )
            dQ_face_check_p[:, i_face] += (
                (flux - nc_flux)
                * mesh.face_volumes[i_face]
                / mesh.cell_volumes[i_cellB]
            )
        return dQ, flux_check, nc_flux_check, iA, iB, dQ_face_check_m, dQ_face_check_p


    def operator_price_c_cellwise(dt, Q, Qaux, parameters, dQ):
        dQ = np.zeros_like(dQ)
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]

        for cell in range(mesh.n_inner_cells):
            qi = Q[:, cell]
            qauxi = Qaux[:, cell]
            neighbors = mesh.cell_neighbors[cell]
            faces  = mesh.cell_faces[:, cell]
            for neighbor, face in zip(neighbors, faces):
                # outward pointing normal
                if mesh.face_cells[0, face] == cell:
                    normal = mesh.face_normals[:, face]
                    sv_i = mesh.face_subvolumes[face, 0]
                    sv_j = mesh.face_subvolumes[face, 1]
                else:
                    normal = -mesh.face_normals[:, face]
                    sv_j = mesh.face_subvolumes[face, 0]
                    sv_i = mesh.face_subvolumes[face, 1]
                face_volume = mesh.face_volumes[face]
                qj = Q[:, neighbor]
                qauxj = Qaux[:, neighbor]
                vol = mesh.cell_volumes[cell] 
                nc_flux, failed = compute_nc_flux(qi, qj, qauxi, qauxj, parameters, normal, sv_i, sv_j, face_volume, dt, pde)
                # nc_flux, failed = compute_nc_flux(qi, qj, qauxi, qauxj, parameters, normal, sv_i, sv_j, vol, dt, pde)

                dQ[:, cell] -= (
                    (nc_flux)
                    * face_volume
                    / vol
                )
        return dQ

    def operator_price_c(dt, Q, Qaux, parameters, dQ):
        dQ = np.zeros_like(dQ)
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]

        ##########################
        # reconstruction
        ##########################
        limiter = get_limiter(name='Venkatakrishnan')
        n_fields = Q.shape[0]
        # gradQ = np.zeros((n_fields, mesh.dimension, mesh.n_cells), dtype=float)
        phi = np.zeros_like(Q)
        gradQ = np.einsum('...dn, kn ->k...d', mesh.lsq_gradQ, Q)
        deltaQ = np.einsum('...hn, kn->kh...', mesh.deltaQ, Q)
        # gradQ = np.zeros_like(gradQ)


        # delta_Q = np.zeros((Q.shape[0], Q.shape[1], mesh.n_faces_per_cell+1), dtype=float)
        # delta_Q[:, :mesh.n_inner_cells, :mesh.n_faces_per_cell] = Q[:, mesh.cell_neighbors[:mesh.n_inner_cells, :mesh.n_faces_per_cell]] - np.repeat(Q[:, :mesh.n_inner_cells, np.newaxis], mesh.n_faces_per_cell, axis=2)
        # delta_Q[:, mesh.n_inner_cells:, :] = Q[:, mesh.cell_neighbors[mesh.n_inner_cells:, :mesh.n_faces_per_cell+1]] - np.repeat(Q[:, mesh.n_inner_cells:, np.newaxis], mesh.n_faces_per_cell+1, axis=2)
        # delta_Q_max = np.max(delta_Q, axis=2)
        # delta_Q_max = np.where(delta_Q_max < 0, 0, delta_Q_max)
        # delta_Q_min = np.min(delta_Q, axis=2)
        # delta_Q_min = np.where(delta_Q_min > 0, 0, delta_Q_min)

        ## DEBUG COPY GRADIENT
        # delta_Q_max[:, mesh.boundary_face_ghosts] = delta_Q_max[:, mesh.boundary_face_cells]
        # delta_Q_min[:, mesh.boundary_face_ghosts] = delta_Q_min[:, mesh.boundary_face_cells]

        # rij = mesh.face_centers - mesh.cell_centers[:, mesh.face_cells[0].T].T
        # gradQ_face_cell = gradQ[:, :, mesh.face_cells[0]]
        # grad_Q_rij = np.einsum('kmd, md->km', gradQ_face_cell, rij)
        # phi_ij = np.ones((n_fields, mesh.n_faces), dtype=float)
        # phi_ij = np.where(grad_Q_rij > 0, limiter((), phi_ij) 
        # phi_ij = np.where(grad_Q_rij < 0, limiter((delta_Q_min[:, mesh.face_cells[0]])/(grad_Q_rij)), phi_ij) 
        # phi_ij[grad_Q_rij > 0] = limiter( (delta_Q_max[:, mesh.face_cells[0]])[grad_Q_rij > 0] /(grad_Q_rij)[grad_Q_rij > 0])
        # phi_ij[grad_Q_rij < 0] = limiter( (delta_Q_min[:, mesh.face_cells[0]])[grad_Q_rij < 0] /(grad_Q_rij)[grad_Q_rij < 0])
        # phi0 = np.min(phi_ij[:, mesh.cell_faces], axis=1)

        # rij = mesh.face_centers - mesh.cell_centers[:, mesh.face_cells[1].T].T
        # gradQ_face_cell = gradQ[:, :, mesh.face_cells[1]]
        # grad_Q_rij = np.einsum('ij..., ...j->i...', gradQ_face_cell, rij)
        # phi_ij = np.ones((n_fields, mesh.n_faces), dtype=float)
        # phi_ij = np.where(grad_Q_rij > 0, limiter((delta_Q_max[:, mesh.face_cells[1]])/(grad_Q_rij)), phi_ij) 
        # phi_ij = np.where(grad_Q_rij < 0, limiter((delta_Q_min[:, mesh.face_cells[1]])/(grad_Q_rij)), phi_ij) 
        # phi_ij[grad_Q_rij > 0] = limiter( (delta_Q_max[:, mesh.face_cells[1]])[grad_Q_rij > 0] /(grad_Q_rij)[grad_Q_rij > 0])
        # phi_ij[grad_Q_rij < 0] = limiter( (delta_Q_min[:, mesh.face_cells[1]])[grad_Q_rij < 0] /(grad_Q_rij)[grad_Q_rij < 0])
        # phi1 = np.min(phi_ij[:, mesh.cell_faces], axis=1)

        # phi[:, :mesh.n_inner_cells] = np.min((phi0, phi1), axis=0)

        # Qaux[i_aux_recon:i_aux_recon+mesh.dimension] = gradQ[0][:]
        # for i in range(model.n_fields):
        #     Qaux[i_aux_recon+mesh.dimension+i, :] = phi[i]

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]

        # rA = mesh.face_centers - mesh.cell_centers[:, iA].T
        # rB = mesh.face_centers - mesh.cell_centers[:, iB].T

        # rN = np.empty((Q.shape[1], mesh.n_faces_per_cell))

        # for i_c in range(mesh.n_inner_cells):
        #     for i_f in range(mesh.n_faces_per_cell):
        #         cell_face = mesh.cell_faces[i_f, i_c]
        #         dist = mesh.face_centers[cell_face] - mesh.cell_centers[:, i_c].T
        #         dqdn = gradQ[:, i_c, :] @ dist
        #         q_const = Q[:, i_c]
        #         q_recon = q_const + dqdn

        # i_cell_faces_flat = np.zeros((mesh.n_faces * 2), dtype=int)
        # i_cell_cells_flat = np.zeros((mesh.n_faces * 2), dtype=int)
        # i_cell_cells_flat[::mesh.n_faces_per_cell] = list(range(mesh.n_cells))
        # i_cell_faces_flat[:mesh.face_cells.shape[1]] = mesh.face_cells[0, :]
        # i_cell_faces_flat[mesh.face_cells.shape[1]:] = mesh.face_cells[1, :]

        # for j in range(mesh.n_faces_per_cell):
        #     rN[:, j] = mesh.face_centers[mesh.cell_faces[:, j]] - mesh.cell_centers[:].T

        # dQA = np.einsum('k...d, ...d->k...', gradQ[:, iA, :], rA)
        # dQB = np.einsum('k...d, ...d->k...', gradQ[:, iB, :], rB)        

        # qAm = Q[:, iA]
        # qBm= Q[:, iB]
        # qA = Q[:, iA] + dQA
        # qB = Q[:, iB] + dQB

        # qmin = np.min(np.array([qAm, qBm]), axis=0)
        # qmax = np.max(np.array([qAm, qBm]), axis=0)

        # def phi(q):
        #     # return (q + np.abs(q)) / (1 + np.abs(q))
        #     return np.max([ np.zeros_like(q), np.min([np.ones_like(q), q], axis=0) ], axis=0)

        # def compute_gamma(umin,umax, uface, ucell, eps=0.0):
        #    theta_min = (umin-ucell)/(uface-ucell)
        #    theta_max = (umax-ucell)/(uface-ucell)
        #    out = np.ones_like(theta_min)
        #    out = np.where(uface < ucell - eps, phi(theta_min), out)
        #    out = np.where(uface > ucell - eps, phi(theta_max), out)
        #    return out

        # gammaA = compute_gamma(qmin, qmax, qA, qAm)
        # gammaB = compute_gamma(qmin, qmax, qB, qBm)

        # gamma_cell = np.infty * np.ones((Q.shape[0], mesh.n_cells), dtype=int)
        # #TODO vectorize
        # for c, gamma in zip(iA, gammaA.T):
        #     gamma_cell[:, c] = np.min([gamma_cell[:, c], gamma], axis=0)
        # for c, gamma in zip(iB, gammaB.T):
        #     gamma_cell[:, c] = np.min([gamma_cell[:, c], gamma], axis=0)

        ##########################

        ### Reconstruction ala 10.1016/S0017-9310(02)00330-7
        def phi(q):
            # return (q + np.abs(q)) / (1 + np.abs(q))
            return np.max([ np.zeros_like(q), np.min([np.ones_like(q), q], axis=0) ], axis=0)

        rA = mesh.face_centers - mesh.cell_centers[:, iA].T
        rB = mesh.face_centers - mesh.cell_centers[:, iB].T

        qA = Q[:, iA]
        qB = Q[:, iB]
        rBA = mesh.cell_centers[:, iB].T - mesh.cell_centers[:, iA].T
        dQA_dn = np.einsum('k...d, ...d->k...', gradQ[:, iA, :], -rBA)
        dQB_dn = np.einsum('k...d, ...d->k...', gradQ[:, iB, :], rBA)        
        dQA = np.einsum('k...d, ...d->k...', gradQ[:, iA, :], rA)
        dQB = np.einsum('k...d, ...d->k...', gradQ[:, iB, :], rB)        

        eps = 10**(-14)
        rA = (2*dQA_dn)/(qB - qA + eps) - 1
        rB = (2*dQB_dn)/(qA - qB + eps) - 1

        qA = Q[:, iA] + 0.5*phi(rA) * (dQA)
        qB = Q[:, iB] + 0.5*phi(rB) * (dQB)
        ##


        # qA = Q[:, iA] 
        # qB = Q[:, iB] 
        # qA = Q[:, iA] + gamma_cell[:, iA] * dQA
        # qB = Q[:, iB] + gamma_cell[:, iB] * dQB
        # qA = Q[:, iA] + dQA
        # qB = Q[:, iB] + dQB
        qauxA = Qaux[:, iA]
        qauxB = Qaux[:, iB]
        normals = mesh.face_normals
        face_volumes = mesh.face_volumes
        cell_volumesA = mesh.cell_volumes[iA]
        cell_volumesB = mesh.cell_volumes[iB]
        svA = mesh.face_subvolumes[:, 0]
        svB = mesh.face_subvolumes[:, 1]

        ### TODO CHECK!!
        # for some reason, I need to flip the normal to make it work... is it inward pointing??

        nc_fluxA, failed = compute_nc_flux(qA, qB, qauxA, qauxB, parameters, -normals, svA, svB, face_volumes, dt, pde)
        assert not failed
        nc_fluxB, failed = compute_nc_flux(qB, qA, qauxB, qauxA, parameters, normals, svB, svA, face_volumes, dt, pde)
        # nc_flux = np.zeros_like(flux)
        assert not failed

        compute_ader_flux = ader_flux.quadrature()
        ader_flux_Q, failed = compute_ader_flux(Q, gradQ, Qaux, parameters, mesh.cell_volumes, dt, pde)
        assert not failed
        

        # I add/substract the contributions for the inner cells, based on the faces
        for faces in mesh.cell_faces:
            # I need to know if the cell_face is part of A or B and only add that contribution
            iA_masked = (iA[faces] == np.array(list(range(mesh.n_inner_cells))))
            iB_masked = (iB[faces] == np.array(list(range(mesh.n_inner_cells))))

            dQ[:, :mesh.n_inner_cells][:, iA_masked] -= (
                (nc_fluxA)
                * face_volumes
                / cell_volumesA
            )[:, faces][:, iA_masked]

            dQ[:, :mesh.n_inner_cells][:, iB_masked] -= (
                (nc_fluxB)
                * face_volumes
                / cell_volumesB
            )[:, faces][:, iB_masked]

        return dQ - 1./(dt*mesh.cell_volumes) * ader_flux_Q 
        # return dQ 

    def operator_rhs_split_vectorized(dt, Q, Qaux, parameters, dQ):
        dQ = np.zeros_like(dQ)
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]

        qA = Q[:, iA] 
        qB = Q[:, iB] 
        qauxA = Qaux[:, iA]
        qauxB = Qaux[:, iB]
        normals = mesh.face_normals
        face_volumes = mesh.face_volumes
        cell_volumesA = mesh.cell_volumes[iA]
        cell_volumesB = mesh.cell_volumes[iB]

        flux, failed = compute_num_flux(qA, qB, qauxA, qauxB, parameters, normals, pde)
        assert not failed
        nc_flux, failed = compute_nc_flux(qA, qB, qauxA, qauxB, parameters, normals, pde)
        # nc_flux = np.zeros_like(flux)
        assert not failed

        # I add/substract the contributions for the inner cells, based on the faces
        for faces in mesh.cell_faces:
            # I need to know if the cell_face is part of A or B and only add that contribution
            iA_masked = (iA[faces] == np.array(list(range(mesh.n_inner_cells))))
            iB_masked = (iB[faces] == np.array(list(range(mesh.n_inner_cells))))

            dQ[:, :mesh.n_inner_cells][:, iA_masked] -= (
                (flux + nc_flux)
                * face_volumes
                / cell_volumesA
            )[:, faces][:, iA_masked]

            dQ[:, :mesh.n_inner_cells][:, iB_masked] += (
                (flux - nc_flux)
                * face_volumes
                / cell_volumesB
            )[:, faces][:, iB_masked]

        # I also want to update the ghost cells for higher order methods (otherwise I need bcs in integrator)
        faces = mesh.boundary_face_face_indices
        # I need to know if the cell_face is part of A or B and only add that contribution
        iA_masked = (iA[faces] == np.array(list(range(mesh.n_inner_cells, mesh.n_cells))))
        iB_masked = (iB[faces] == np.array(list(range(mesh.n_inner_cells, mesh.n_cells))))

        dQ[:, mesh.n_inner_cells:][:, iA_masked] -= (
            (flux + nc_flux)
            * face_volumes
            / cell_volumesA
        )[:, faces][:, iA_masked]

        dQ[:, mesh.n_inner_cells:][:, iB_masked] += (
            (flux - nc_flux)
            * face_volumes
            / cell_volumesB
        )[:, faces][:, iB_masked]

        return dQ

    def operator_rhs_split_reconstructed_vectorized(dt, Q, Qaux, parameters, dQ):
        dQ = np.zeros_like(dQ)

        i_recon_start = Q.shape[0]
        # phi = Qaux[i_recon_start+mesh.dimension:]
        # gradQ = np.zeros((Q.shape[0], Q.shape[1], 2), dtype=float)
        # gradQ[:,:,:] = Qaux[i_recon_start:i_recon_start+mesh.dimension]

        limiter = get_limiter(name='Venkatakrishnan')

        # reconstruction
        n_fields = Q.shape[0]
        gradQ = np.zeros((n_fields, mesh.dimension, mesh.n_cells), dtype=float)
        phi = np.zeros_like(Q)
        gradQ = np.einsum('...dj, kj ->kd...', mesh.lsq_gradQ, Q)
        delta_Q = np.zeros((Q.shape[0], Q.shape[1], mesh.n_faces_per_cell+1), dtype=float)
        delta_Q[:, :mesh.n_inner_cells, :mesh.n_faces_per_cell] = Q[:, mesh.cell_neighbors[:mesh.n_inner_cells, :mesh.n_faces_per_cell]] - np.repeat(Q[:, :mesh.n_inner_cells, np.newaxis], mesh.n_faces_per_cell, axis=2)
        # delta_Q[:, mesh.n_inner_cells:, :] = Q[:, mesh.cell_neighbors[mesh.n_inner_cells:, :mesh.n_faces_per_cell+1]] - np.repeat(Q[:, mesh.n_inner_cells:, np.newaxis], mesh.n_faces_per_cell+1, axis=2)
        delta_Q_max = np.max(delta_Q, axis=2)
        # delta_Q_max = np.where(delta_Q_max < 0, 0, delta_Q_max)
        delta_Q_min = np.min(delta_Q, axis=2)
        # delta_Q_min = np.where(delta_Q_min > 0, 0, delta_Q_min)

        ## DEBUG COPY GRADIENT
        delta_Q_max[:, mesh.boundary_face_ghosts] = delta_Q_max[:, mesh.boundary_face_cells]
        delta_Q_min[:, mesh.boundary_face_ghosts] = delta_Q_min[:, mesh.boundary_face_cells]

        rij = mesh.face_centers - mesh.cell_centers[:, mesh.face_cells[0].T].T
        gradQ_face_cell = gradQ[:, :, mesh.face_cells[0]]
        grad_Q_rij = np.einsum('ij..., ...j->i...', gradQ_face_cell, rij)
        phi_ij = np.ones((n_fields, mesh.n_faces), dtype=float)
        # phi_ij = np.where(grad_Q_rij > 0, limiter((), phi_ij) 
        # phi_ij = np.where(grad_Q_rij < 0, limiter((delta_Q_min[:, mesh.face_cells[0]])/(grad_Q_rij)), phi_ij) 
        phi_ij[grad_Q_rij > 0] = limiter( (delta_Q_max[:, mesh.face_cells[0]])[grad_Q_rij > 0] /(grad_Q_rij)[grad_Q_rij > 0])
        phi_ij[grad_Q_rij < 0] = limiter( (delta_Q_min[:, mesh.face_cells[0]])[grad_Q_rij < 0] /(grad_Q_rij)[grad_Q_rij < 0])
        phi0 = np.min(phi_ij[:, mesh.cell_faces], axis=1)

        rij = mesh.face_centers - mesh.cell_centers[:, mesh.face_cells[1].T].T
        gradQ_face_cell = gradQ[:, :, mesh.face_cells[1]]
        grad_Q_rij = np.einsum('ij..., ...j->i...', gradQ_face_cell, rij)
        phi_ij = np.ones((n_fields, mesh.n_faces), dtype=float)
        # phi_ij = np.where(grad_Q_rij > 0, limiter((delta_Q_max[:, mesh.face_cells[1]])/(grad_Q_rij)), phi_ij) 
        # phi_ij = np.where(grad_Q_rij < 0, limiter((delta_Q_min[:, mesh.face_cells[1]])/(grad_Q_rij)), phi_ij) 
        phi_ij[grad_Q_rij > 0] = limiter( (delta_Q_max[:, mesh.face_cells[1]])[grad_Q_rij > 0] /(grad_Q_rij)[grad_Q_rij > 0])
        phi_ij[grad_Q_rij < 0] = limiter( (delta_Q_min[:, mesh.face_cells[1]])[grad_Q_rij < 0] /(grad_Q_rij)[grad_Q_rij < 0])
        phi1 = np.min(phi_ij[:, mesh.cell_faces], axis=1)

        phi[:, :mesh.n_inner_cells] = np.min((phi0, phi1), axis=0)

        # Qaux[i_aux_recon:i_aux_recon+mesh.dimension] = gradQ[0][:]
        # for i in range(model.n_fields):
        #     Qaux[i_aux_recon+mesh.dimension+i, :] = phi[i]

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]

        rA = mesh.face_centers - mesh.cell_centers[:, iA].T
        rB = mesh.face_centers - mesh.cell_centers[:, iB].T

        qA = Q[:, iA] + phi[:, iA] * np.einsum('kd..., ...d->k...', gradQ[:, :, iA], rA)
        qB = Q[:, iB] + phi[:, iB] * np.einsum('kd..., ...d->k...', gradQ[:, :, iB], rB)
        qauxA = Qaux[:, iA]
        qauxB = Qaux[:, iB]
        normals = mesh.face_normals
        face_volumes = mesh.face_volumes
        cell_volumesA = mesh.cell_volumes[iA]
        cell_volumesB = mesh.cell_volumes[iB]

        flux, failed = compute_num_flux(qA, qB, qauxA, qauxB, parameters, normals, pde)
        assert not failed
        nc_flux, failed = compute_nc_flux(qA, qB, qauxA, qauxB, parameters, normals, pde)
        # nc_flux = np.zeros_like(flux)
        assert not failed

        # dQcheck = np.zeros_like(dQ)
        # dQcheck, flux_check, nc_flux_check, iA_check, iB_check, dQ_m, dQ_p = operator_rhs_split_check(dt, Q, Qaux, parameters, dQcheck)

        # for i_face in range(mesh.n_faces):
        #     i_cellA, i_cellB = mesh.face_cells[:, i_face]

        #     dQ[:, i_cellA] -= (
        #         (flux + nc_flux)[:, i_face]
        #         * mesh.face_volumes[i_face]
        #         / mesh.cell_volumes[i_cellA]
        #     )

        #     dQ[:, i_cellB] += (
        #         (flux - nc_flux)[:, i_face]
        #         * mesh.face_volumes[i_face]
        #         / mesh.cell_volumes[i_cellB]
        #     )



        # I add/substract the contributions for the inner cells, based on the faces
        for faces in mesh.cell_faces:
            # I need to know if the cell_face is part of A or B and only add that contribution
            iA_masked = (iA[faces] == np.array(list(range(mesh.n_inner_cells))))
            iB_masked = (iB[faces] == np.array(list(range(mesh.n_inner_cells))))

            dQ[:, :mesh.n_inner_cells][:, iA_masked] -= (
                (flux + nc_flux)
                * face_volumes
                / cell_volumesA
            )[:, faces][:, iA_masked]

            dQ[:, :mesh.n_inner_cells][:, iB_masked] += (
                (flux - nc_flux)
                * face_volumes
                / cell_volumesB
            )[:, faces][:, iB_masked]

        # I also want to update the ghost cells for higher order methods (otherwise I need bcs in integrator)
        faces = mesh.boundary_face_face_indices
        # I need to know if the cell_face is part of A or B and only add that contribution
        iA_masked = (iA[faces] == np.array(list(range(mesh.n_inner_cells, mesh.n_cells))))
        iB_masked = (iB[faces] == np.array(list(range(mesh.n_inner_cells, mesh.n_cells))))

        dQ[:, mesh.n_inner_cells:][:, iA_masked] -= (
            (flux + nc_flux)
            * face_volumes
            / cell_volumesA
        )[:, faces][:, iA_masked]

        dQ[:, mesh.n_inner_cells:][:, iB_masked] += (
            (flux - nc_flux)
            * face_volumes
            / cell_volumesB
        )[:, faces][:, iB_masked]



        # for faces in mesh.cell_faces:
        #     cellsA = iA[faces]
        #     i_cellsA1 = np.sort(np.unique(cellsA, return_index=True)[1])
        #     cellsA1 = cellsA[i_cellsA1]
        #     facesA1 = faces[i_cellsA1]
        #     i_cellsA2 = np.setdiff1d(np.array(list(range(64))),  i_cellsA1)
        #     cellsA2 = cellsA[i_cellsA2]
        #     facesA2 = faces[i_cellsA2]

        #     cellsB = iB[faces]
        #     i_cellsB1 = np.sort(np.unique(cellsB, return_index=True)[1])
        #     cellsB1 = cellsB[i_cellsB1]
        #     facesB1 = faces[i_cellsB1]
        #     i_cellsB2 = np.setdiff1d(np.array(list(range(64))),  i_cellsB1)
        #     cellsB2 = cellsB[i_cellsB2]
        #     facesB2 = faces[i_cellsB2]

        #     dQm[:, cellsA1] -= (
        #         (flux + nc_flux)
        #         * face_volumes
        #         / cell_volumesA
        #     )[:, facesA1]
        #     dQm[:, cellsA2] -= (
        #         (flux + nc_flux)
        #         * face_volumes
        #         / cell_volumesA
        #     )[:, facesA2]

        #     dQp[:, cellsB1] += (
        #         (flux - nc_flux)
        #         * face_volumes
        #         / cell_volumesB
        #     )[:, facesB1]

        #     dQp[:, cellsB2] += (
        #         (flux - nc_flux)
        #         * face_volumes
        #         / cell_volumesB
        #     )[:, facesB2]
        # dQ = 0.5*(dQm + dQp)


        

        return dQ
            
    return operator_price_c
    # return operator_1d_cell_centered
    # return operator_rhs_split_vectorized


# def _get_semidiscrete_solution_operator(mesh, pde, boundary_conditions, settings):
#     compute_num_flux = settings.num_flux
#     compute_nc_flux = settings.nc_flux
#     reconstruction = settings.reconstruction
#     reconstruction_edge = settings.reconstruction_edge

#     def operator_rhs_split(dt, Q, Qaux, parameters, dQ):
#         # Loop over the inner elements
#         for i_elem in range(mesh.n_elements):
#             for i_th_neighbor in range(mesh.element_n_neighbors[i_elem]):
#                 i_neighbor = mesh.element_neighbors[i_elem, i_th_neighbor]
#                 if i_elem > i_neighbor:
#                     continue
#                 i_face = mesh.element_neighbors_face_index[i_elem, i_th_neighbor]
#                 # reconstruct
#                 [Qi, Qauxi], [Qj, Qauxj] = reconstruction(
#                     mesh, [Q, Qaux], i_elem, i_th_neighbor
#                 )

#                 # TODO callout to a requirement of the flux
#                 mesh_props = SimpleNamespace(
#                     dt_dx=dt / (2 * mesh.element_inradius[i_elem])
#                 )
#                 flux, failed = compute_num_flux(
#                     Qi,
#                     Qj,
#                     Qauxi,
#                     Qauxj,
#                     parameters,
#                     mesh.element_face_normals[i_elem, i_face],
#                     pde,
#                     mesh_props=mesh_props,
#                 )
#                 assert not failed
#                 nc_flux, failed = compute_nc_flux(
#                     Qi,
#                     Qj,
#                     Qauxi,
#                     Qauxj,
#                     parameters,
#                     mesh.element_face_normals[i_elem, i_face],
#                     pde,
#                 )
#                 assert not failed

#                 # TODO index map (elem_edge_index) such that I do not need:
#                 # and if statement to avoid double edge computation (as I do now)
#                 # avoid double edge computation
#                 dQ[i_elem] -= (
#                     (flux + nc_flux)
#                     * mesh.element_face_areas[i_elem, i_face]
#                     / mesh.element_volume[i_elem]
#                 )

#                 dQ[i_neighbor] += (
#                     (flux - nc_flux)
#                     * mesh.element_face_areas[i_elem, i_face]
#                     / mesh.element_volume[i_neighbor]
#                 )

#         # Loop over boundary faces
#         for i in range(mesh.n_boundary_elements):
#             i_elem = mesh.boundary_face_corresponding_element[i]
#             i_face = mesh.boundary_face_element_face_index[i]
#             Q_ghost = boundary_conditions.apply(
#                 i,
#                 i_elem,
#                 Q,
#                 Qaux,
#                 parameters,
#                 mesh.element_face_normals[i_elem, i_face],
#             )
#             # i_neighbor = mesh.element_neighbors[i_elem, i_face]
#             # reconstruct
#             [Qi, Qauxi], [Qj, Qauxj] = reconstruction_edge(
#                 mesh, [Q, Qaux], Q_ghost, i_elem
#             )

#             # callout to a requirement of the flux
#             mesh_props = SimpleNamespace(dt_dx=dt / (2 * mesh.element_inradius[i_elem]))
#             flux, failed = compute_num_flux(
#                 Qi,
#                 Qj,
#                 Qauxi,
#                 Qauxj,
#                 parameters,
#                 mesh.element_face_normals[i_elem, i_face],
#                 pde,
#                 mesh_props=mesh_props,
#             )
#             assert not failed
#             nc_flux, failed = compute_nc_flux(
#                 Qi,
#                 Qj,
#                 Qauxi,
#                 Qauxj,
#                 parameters,
#                 mesh.element_face_normals[i_elem, i_face],
#                 pde,
#             )
#             assert not failed

#             dQ[i_elem] -= (
#                 (flux + nc_flux)
#                 * mesh.element_face_areas[i_elem, i_face]
#                 / mesh.element_volume[i_elem]
#             )
#         return dQ

    return operator_rhs_split


def load_runtime_model(model):
    runtime_pde = model.get_pde()
    # runtime_bcs = model.create_python_boundary_interface(printer='numpy')
    runtime_bcs = model.get_boundary_conditions()
    # runtime_bc = model.get_boundary_conditions()
    # model.boundary_conditions.runtime_bc = runtime_bcs
    return runtime_pde, runtime_bcs


def save_model_to_C(model, settings):
    _ = model.create_c_interface(path=os.path.join(settings.output_dir, "c_interface"))
    _ = model.create_c_boundary_interface(
        path=os.path.join(settings.output_dir, "c_interface")
    )


def fvm_c_unsteady_semidiscete(
    mesh,
    model,
    settings,
    ode_solver_flux="RK1",
    ode_solver_source="RK1",
    rebuild_model=True,
    rebuild_mesh=True,
    rebuild_c=True,
):
    io.clean_files(settings.output_dir, ".vtk")
    io.clean_files(settings.output_dir, ".vtk.series")
    io.clean_files(settings.output_dir, "fields.hdf5")
    Q, Qaux = _initialize_problem(model, mesh)
    parameters = model.parameter_values
    io.init_output_directory(settings.output_dir, settings.output_clean_dir)
    _ = io.save_fields(
        settings.output_dir, 0.0, 0.0, 0, Q, Qaux, settings.output_write_all
    )

    if rebuild_mesh:
        mesh.write_to_hdf5(settings.output_dir)

    settings.parameters = model.parameters.to_value_dict(model.parameter_values)
    io.save_settings(settings.output_dir, settings)

    if rebuild_model or rebuild_mesh:
        if os.path.exists(os.path.join(settings.output_dir, "c_interface")):
            shutil.rmtree(os.path.join(settings.output_dir, "c_interface"))
        save_model_to_C(model, settings)
        model.boundary_conditions.append_boundary_map_to_mesh_hdf5(settings.output_dir)
        modify_sympy_c_code.transform_code(settings)

    if rebuild_model or rebuild_mesh or rebuild_c:
        c_interface.build(
            dimension=mesh.dimension,
            n_boundary_conditions=len(model.boundary_conditions.boundary_conditions),
            n_elements=mesh.n_elements,
            n_fields=model.n_fields,
            n_fields_aux=model.aux_variables.length(),
            path_settings=os.path.join(settings.output_dir, "settings.hdf5"),
            path_mesh=os.path.join(settings.output_dir, "mesh.hdf5"),
            path_fields=os.path.join(settings.output_dir, "fields.hdf5"),
            model_path = os.path.join(settings.output_dir, "c_interface/Model"),
            debug = settings.debug,
            profiling = settings.profiling,
        )

    c_interface.run_driver()


def write_field_to_hdf5(filepath: str, time, Q, Qaux):
    main_dir = os.getenv("SMS")
    with h5py.File(os.path.join(main_dir, filepath), "a") as f:
        solution = f.create_group("solution")
        iteration = f.create_group("iteration")
        iteration.create_dataset("time", time)
        iteration.create_dataset("Q", time)
        iteration.create_dataset("Qaux", time)

# TODO vectorize
def apply_boundary_conditions(mesh, time, Q, Qaux, parameters, runtime_bcs):
    """
    mesh.boundary_face_cell relates to the cell where we retreive information. If not periodic, this is the cell adjacent to the ghost cell.
    This cells information is fed into the boundary face function and written to the ghost cell.
    """
    for i_bc_face in range(mesh.n_boundary_faces):
        i_face = mesh.boundary_face_face_indices[i_bc_face]
        i_bc_func = mesh.boundary_face_function_numbers[i_bc_face]
        bc_func = runtime_bcs[i_bc_func]
        q_cell = Q[:, mesh.boundary_face_cells[i_bc_face]]
        qaux_cell = Qaux[:, mesh.boundary_face_cells[ i_bc_face]]
        normal = mesh.face_normals[:, i_face]
        position = mesh.face_centers[i_face]
        # I cannot use the mesh.boudnary_face_cells because this is potenntially a periodic ghost cell. Then the distance is very large, as I go across the grid
        # position_cell = mesh.cell_centers[:, mesh.boundary_face_cells[i_bc_face]]
        position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i_bc_face]]
        # distance = np.linalg.norm(position - position_cell)
        distance = np.linalg.norm(position - position_ghost)
        q_ghost = bc_func(time, position, distance, q_cell, qaux_cell, parameters, normal)
        ### DEBUG
        cells_adjacent_to_ghost = min(mesh.face_cells[:, mesh.boundary_face_face_indices[i_bc_face]])
        ###
        Q[:, mesh.boundary_face_ghosts[i_bc_face]] = q_ghost
    return Q

def get_limiter(name='min'):
    if name == 'min':
        def f(x):
            return np.min((np.ones_like(x), x), axis=0)
        return f
    elif name == "Venkatakrishnan":
        def f(x):
            return (x**2 + 2*x)/(x**2 + x + 2.*np.ones_like(x))
        return f
    elif name == "zero":
        def f(x):
            return np.zeros_like(x)
        return f
    else:
        assert False

def solver_price_c(
    mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
):
    iteration = 0
    time = 0.0

    output_hdf5_path = os.path.join(settings.output_dir, f'{settings.name}.h5')

    assert model.dimension == mesh.dimension

    progressbar = pyprog.ProgressBar(
        "{}: ".format(settings.name),
        "\r",
        settings.time_end,
        complete_symbol="█",
        not_complete_symbol="-",
    )
    progressbar.progress_explain = ""
    progressbar.update()

    Q, Qaux = _initialize_problem(model, mesh)

    parameters = model.parameter_values
    pde, bcs = load_runtime_model(model)
    Q = apply_boundary_conditions(mesh, time, Q, Qaux, parameters, bcs)

    i_snapshot = 0
    dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
    io.init_output_directory(settings.output_dir, settings.output_clean_dir)
    mesh.write_to_hdf5(output_hdf5_path)
    i_snapshot = io.save_fields(
        output_hdf5_path, time, 0, i_snapshot, Q, Qaux, settings.output_write_all
    )

    Qnew = deepcopy(Q)

    space_solution_operator = _get_semidiscrete_solution_operator(mesh, pde, bcs, settings)
    compute_source = _get_source(mesh, pde, settings)
    compute_source_jac = _get_source_jac(mesh, pde, settings)

    compute_max_abs_eigenvalue = _get_compute_max_abs_eigenvalue(
        mesh, pde, settings
    )
    min_inradius = np.min(mesh.cell_inradius)

    time_start = gettime()
    while time < settings.time_end:
        #     # print(f'in loop from process {os.getpid()}')
        Q = deepcopy(Qnew)
        dt = settings.compute_dt(
            Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
        )
        assert dt > 10 ** (-8)
        assert not np.isnan(dt) and np.isfinite(dt)

        #     # add 0.001 safty measure to avoid very small time steps
        if settings.truncate_last_time_step:
            if time + dt * 1.001 > settings.time_end:
                dt = settings.time_end - time + 10 ** (-10)



        Qnew = ode_solver_flux(space_solution_operator, Q, Qaux, parameters, dt)

        Qnew = ode_solver_source(
            compute_source, Qnew, Qaux, parameters, dt, func_jac=compute_source_jac
        )
        Qnew = apply_boundary_conditions(mesh, time, Qnew, Qaux, parameters, bcs)

        # Update solution and time
        time += dt
        iteration += 1
        print(iteration, time, dt)

        i_snapshot = io.save_fields(
            output_hdf5_path,
            time,
            (i_snapshot + 1) * dt_snapshot,
            i_snapshot,
            Qnew,
            Qaux,
            settings.output_write_all,
        )

    print(f'Runtime: {gettime() - time_start}')

    progressbar.end()
    return settings


def jax_fvm_unsteady_semidiscrete(
    mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
):
    iteration = 0
    time = 0.0

    output_hdf5_path = os.path.join(settings.output_dir, f'{settings.name}.h5')

    assert model.dimension == mesh.dimension

    progressbar = pyprog.ProgressBar(
        "{}: ".format(settings.name),
        "\r",
        settings.time_end,
        complete_symbol="█",
        not_complete_symbol="-",
    )
    progressbar.progress_explain = ""
    progressbar.update()

    # reconstruction
    # i_aux_recon = model.aux_variables.length()
    # recon_variables = [f'dQdx_{i}' for i in range(mesh.dimension)] + [f'phi_{f}' for f in range(model.n_fields)]
    # model.aux_variables = register_sympy_attribute(model.aux_variables.get_list() + recon_variables)
    # model.n_aux_fields = model.aux_variables.length()


    Q, Qaux = _initialize_problem(model, mesh)

    # runtime_bcs = model.create_python_boundary_interface(printer='numpy')

    parameters = model.parameter_values
    pde, bcs = load_runtime_model(model)
    Q = apply_boundary_conditions(mesh, time, Q, Qaux, parameters, bcs)

    # for callback in self.callback_function_list_init:
    #     Qnew, kwargs = callback(self, Qnew, **kwargs)

    i_snapshot = 0
    dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
    io.init_output_directory(settings.output_dir, settings.output_clean_dir)
    mesh.write_to_hdf5(output_hdf5_path)
    i_snapshot = io.save_fields(
        output_hdf5_path, time, 0, i_snapshot, Q, Qaux, settings.output_write_all
    )

    # settings.parameters = model.parameters.to_value_dict(model.parameter_values)
    # io.save_settings(settings.output_dir, settings)
    # limiter = get_limiter(name='Venkatakrishnan')
    # limiter = get_limiter(name='zero')

    Qnew = deepcopy(Q)

    space_solution_operator = _get_semidiscrete_solution_operator(mesh, pde, bcs, settings)
    # space_solution_operator = _get_semidiscrete_solution_operator(
    #     mesh, pde, model.boundary_conditions, settings
    # )
    compute_source = _get_source(mesh, pde, settings)
    compute_source_jac = _get_source_jac(mesh, pde, settings)

    compute_max_abs_eigenvalue = _get_compute_max_abs_eigenvalue(
        mesh, pde, settings
    )
    min_inradius = np.min(mesh.cell_inradius)

    # if model.levels > 1:
    #     # enforce_boundary_conditions = model.basis.enforce_boundary_conditions(dim=mesh.dimension, enforced_basis=[2], rhs=np.zeros(1),)
    #     enforce_boundary_conditions = model.basis.enforce_boundary_conditions_lsq2(dim=mesh.dimension)
    # else:
    #     enforce_boundary_conditions  = lambda Q: Q
    enforce_boundary_conditions  = lambda Q: Q

    # limiter = get_limiter(name='Venkatakrishnan')
    # limiter = get_limiter(name='zero')

    # print(f'hi from process {os.getpid()}')
    time_start = gettime()
    while time < settings.time_end:
        #     # print(f'in loop from process {os.getpid()}')
        Q = deepcopy(Qnew)
        if model.name == "ShearShallowFlowPathconservative":
            h, u, v, R11, R12, R22, E11, E12, E22, P11, P12, P22 = model.get_primitives(Q)
            assert (P11 > 0).all
            assert (P12 > 0).all
            assert (P22 > 0).all
            P11 = np.where(P11 <= 10**(-15), 10**(-15), P11)
            P12 = np.where(P12 <= 10**(-15), 10**(-15), P12)
            P22 = np.where(P22 <= 10**(-15), 10**(-15), P22)
            R11 = Q[0] * P11
            R12 = Q[0] * P12
            R22 = Q[0] * P22
            u = Q[1]/Q[0]
            v = Q[2]/Q[0]
            Q[3] =  (1/2 * R11 + 1/2 * Q[0] * u * u)
            Q[4] =  (1/2 * R12 + 1/2 * Q[0] * u * v)
            Q[5] =  (1/2 * R22 + 1/2 * Q[0] * v * v)

        if model.name == "ShearShallowFlow":
            Q[2] = np.where(Q[2] <= 10**(-15), 10**(-15), Q[2])


        dt = settings.compute_dt(
            Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
        )
        assert dt > 10 ** (-6)
        assert not np.isnan(dt) and np.isfinite(dt)

        # if iteration < self.warmup_interations:
        #     dt *= self.dt_relaxation_factor
        # if dt < self.dtmin:
        #     dt = self.dtmin

        # reconstruction
        # gradQ = np.zeros((model.n_fields, mesh.dimension, mesh.n_cells), dtype=float)
        # phi = np.zeros_like(Q)
        # gradQ = np.einsum('...dj, kj ->kd...', mesh.lsq_gradQ, Q)
        # delta_Q = np.zeros((Q.shape[0], Q.shape[1], mesh.n_faces_per_cell+1), dtype=float)
        # delta_Q[:, :mesh.n_inner_cells, :mesh.n_faces_per_cell] = Q[:, mesh.cell_neighbors[:mesh.n_inner_cells, :mesh.n_faces_per_cell]] - np.repeat(Q[:, :mesh.n_inner_cells, np.newaxis], mesh.n_faces_per_cell, axis=2)
        # # delta_Q[:, mesh.n_inner_cells:, :] = Q[:, mesh.cell_neighbors[mesh.n_inner_cells:, :mesh.n_faces_per_cell+1]] - np.repeat(Q[:, mesh.n_inner_cells:, np.newaxis], mesh.n_faces_per_cell+1, axis=2)
        # delta_Q_max = np.max(delta_Q, axis=2)
        # # delta_Q_max = np.where(delta_Q_max < 0, 0, delta_Q_max)
        # delta_Q_min = np.min(delta_Q, axis=2)
        # # delta_Q_min = np.where(delta_Q_min > 0, 0, delta_Q_min)

        # ## DEBUG COPY GRADIENT
        # delta_Q_max[:, mesh.boundary_face_ghosts] = delta_Q_max[:, mesh.boundary_face_cells]
        # delta_Q_min[:, mesh.boundary_face_ghosts] = delta_Q_min[:, mesh.boundary_face_cells]

        # rij = mesh.face_centers - mesh.cell_centers[:, mesh.face_cells[0].T].T
        # gradQ_face_cell = gradQ[:, :, mesh.face_cells[0]]
        # grad_Q_rij = np.einsum('ij..., ...j->i...', gradQ_face_cell, rij)
        # phi_ij = np.ones((model.n_fields, mesh.n_faces), dtype=float)
        # # phi_ij = np.where(grad_Q_rij > 0, limiter((), phi_ij) 
        # # phi_ij = np.where(grad_Q_rij < 0, limiter((delta_Q_min[:, mesh.face_cells[0]])/(grad_Q_rij)), phi_ij) 
        # phi_ij[grad_Q_rij > 0] = limiter( (delta_Q_max[:, mesh.face_cells[0]])[grad_Q_rij > 0] /(grad_Q_rij)[grad_Q_rij > 0])
        # phi_ij[grad_Q_rij < 0] = limiter( (delta_Q_min[:, mesh.face_cells[0]])[grad_Q_rij < 0] /(grad_Q_rij)[grad_Q_rij < 0])
        # phi0 = np.min(phi_ij[:, mesh.cell_faces], axis=1)

        # rij = mesh.face_centers - mesh.cell_centers[:, mesh.face_cells[1].T].T
        # gradQ_face_cell = gradQ[:, :, mesh.face_cells[1]]
        # grad_Q_rij = np.einsum('ij..., ...j->i...', gradQ_face_cell, rij)
        # phi_ij = np.ones((model.n_fields, mesh.n_faces), dtype=float)
        # # phi_ij = np.where(grad_Q_rij > 0, limiter((delta_Q_max[:, mesh.face_cells[1]])/(grad_Q_rij)), phi_ij) 
        # # phi_ij = np.where(grad_Q_rij < 0, limiter((delta_Q_min[:, mesh.face_cells[1]])/(grad_Q_rij)), phi_ij) 
        # phi_ij[grad_Q_rij > 0] = limiter( (delta_Q_max[:, mesh.face_cells[1]])[grad_Q_rij > 0] /(grad_Q_rij)[grad_Q_rij > 0])
        # phi_ij[grad_Q_rij < 0] = limiter( (delta_Q_min[:, mesh.face_cells[1]])[grad_Q_rij < 0] /(grad_Q_rij)[grad_Q_rij < 0])
        # phi1 = np.min(phi_ij[:, mesh.cell_faces], axis=1)


        # phi[:, :mesh.n_inner_cells] = np.min((phi0, phi1), axis=0)

        # Qaux[i_aux_recon:i_aux_recon+mesh.dimension] = gradQ[0][:]
        # for i in range(model.n_fields):
        #     Qaux[i_aux_recon+mesh.dimension+i, :] = phi[i]



        #     # add 0.001 safty measure to avoid very small time steps
        if settings.truncate_last_time_step:
            if time + dt * 1.001 > settings.time_end:
                dt = settings.time_end - time + 10 ** (-10)

        #     # TODO this two things should be 'callouts'!!
        # Qnew = enforce_boundary_conditions(Qnew)
        Qnew = ode_solver_flux(space_solution_operator, Q, Qaux, parameters, dt)
        # Qnew = enforce_boundary_conditions(Qnew)

        ### SSF Energy  DEBUG
        # h = Q[0]
        # u = Q[1] / h
        # P11 = Q[2np.zeros(1)
        # hE = h*(u**2 + settings.parameters['g'] * h + P11)
        # Q[3] = hE

        Qnew = ode_solver_source(
            compute_source, Qnew, Qaux, parameters, dt, func_jac=compute_source_jac
        )

        Qnew = apply_boundary_conditions(mesh, time, Qnew, Qaux, parameters, bcs)
        Qnew = enforce_boundary_conditions(Qnew)
        #     )
        #     # Qnew  += -Q+ode_solver_source(
        #     #     compute_source, Q, Qaux, parameters, dt, func_jac=compute_source_jac
        #     # )
        #     # Qnew = enforce_boundary_conditions(Qnew)

        # Update solution and time
        time += dt
        iteration += 1
        print(iteration, time, dt)

        #     # for callback in self.callback_function_list_post_solvestep:
        #     #     Qnew, kwargs = callback(self, Qnew, **kwargs)

        i_snapshot = io.save_fields(
            output_hdf5_path,
            time,
            (i_snapshot + 1) * dt_snapshot,
            i_snapshot,
            Qnew,
            Qaux,
            settings.output_write_all,
        )

    #     # logger.info(
    #     #     "Iteration: {:6.0f}, Runtime: {:6.2f}, Time: {:2.4f}, dt: {:2.4f}, error: {}".format(
    #     #         iteration, gettime() - time_start, time, dt, error
    #     #     )
    #     # )
    #     # print(f'finished timestep: {os.getpid()}')
    # progressbar.set_stat(min(time, settings.time_end))
    # progressbar.update()
    print(f'Runtime: {gettime() - time_start}')

    progressbar.end()
    return settings


def fvm_unsteady_semidiscrete(
    mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
):
    iteration = 0
    time = 0.0

    assert model.dimension == mesh.dimension

    progressbar = pyprog.ProgressBar(
        "{}: ".format(settings.name),
        "\r",
        settings.time_end,
        complete_symbol="█",
        not_complete_symbol="-",
    )
    progressbar.progress_explain = ""
    progressbar.update()
    # force=True is needed in order to disable old logging root handlers (if multiple test cases are run by one file)
    # otherwise only a logfile will be created for the first testcase but the log file gets deleted by rmtree
    # logging.basicConfig(
    #     filename=os.path.join(
    #         os.path.join(main_dir, self.output_dir), "logfile.log"
    #     ),
    #     filemode="w",
    #     level=self.logging_level,
    #     force=True,
    # )
    # logger = logging.getLogger(__name__ + ":solve_steady")

    # enforce_boundary_conditions = model.basis.enforce_boundary_conditions()

    Q, Qaux = _initialize_problem(model, mesh)
    # Q = enforce_boundary_conditions(Q)
    parameters = model.parameter_values
    Qnew = deepcopy(Q)

    # for callback in self.callback_function_list_init:
    #     Qnew, kwargs = callback(self, Qnew, **kwargs)

    i_snapshot = 0
    dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
    io.init_output_directory(settings.output_dir, settings.output_clean_dir)
    i_snapshot = io.save_fields(
        settings.output_dir, time, 0, i_snapshot, Qnew, Qaux, settings.output_write_all
    )
    mesh.write_to_hdf5(settings.output_dir)

    settings.parameters = model.parameters.to_value_dict(model.parameter_values)
    io.save_settings(settings.output_dir, settings)

    time_start = gettime()

    pde, bc = load_runtime_model(model)


    # map_elements_to_edges = recon.create_map_elements_to_edges(mesh)
    # on_edges_normal, on_edges_length = recon.get_edge_geometry_data(mesh)
    # space_solution_operator = get_semidiscrete_solution_operator(mesh, pde, settings, on_edges_normal, on_edges_length, map_elements_to_edges)
    space_solution_operator = _get_semidiscrete_solution_operator(
        mesh, pde, model.boundary_conditions, settings
    )
    compute_source = _get_source(mesh, pde, settings)
    compute_source_jac = _get_source_jac(mesh, pde, settings)
    compute_max_abs_eigenvalue = _get_compute_max_abs_eigenvalue(
        mesh, pde, model.boundary_conditions, settings
    )
    min_inradius = np.min(mesh.element_inradius)

    # print(f'hi from process {os.getpid()}')
    while time < settings.time_end:
        # print(f'in loop from process {os.getpid()}')
        Q = deepcopy(Qnew)
        dt = settings.compute_dt(
            Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
        )
        assert dt > 10 ** (-6)

        assert not np.isnan(dt) and np.isfinite(dt)

        # if iteration < self.warmup_interations:
        #     dt *= self.dt_relaxation_factor
        # if dt < self.dtmin:
        #     dt = self.dtmin

        # add 0.001 safty measure to avoid very small time steps
        if settings.truncate_last_time_step:
            if time + dt * 1.001 > settings.time_end:
                dt = settings.time_end - time + 10 ** (-10)

        # TODO this two things should be 'callouts'!!
        Qnew = ode_solver_flux(space_solution_operator, Q, Qaux, parameters, dt)
        # Qnew = enforce_boundary_conditions(Qnew)

        hb = lambda t, x : np.cos(x[:, 0] -t)
        Qaux[:,0] = hb(time, mesh.element_center)
        Qaux = Qaux.reshape((Q.shape[0], 1))

        Qnew = ode_solver_source(
            compute_source, Qnew, Qaux, parameters, dt, func_jac=compute_source_jac
        )
        # Qnew  += -Q+ode_solver_source(
        #     compute_source, Q, Qaux, parameters, dt, func_jac=compute_source_jac
        # )
        # Qnew = enforce_boundary_conditions(Qnew)

        # Update solution and time
        time += dt
        iteration += 1

        # for callback in self.callback_function_list_post_solvestep:
        #     Qnew, kwargs = callback(self, Qnew, **kwargs)

        i_snapshot = io.save_fields(
            settings.output_dir,
            time,
            (i_snapshot + 1) * dt_snapshot,
            i_snapshot,
            Qnew,
            Qaux,
            settings.output_write_all,
        )

        # logger.info(
        #     "Iteration: {:6.0f}, Runtime: {:6.2f}, Time: {:2.4f}, dt: {:2.4f}, error: {}".format(
        #         iteration, gettime() - time_start, time, dt, error
        #     )
        # )
        # print(f'finished timestep: {os.getpid()}')
        progressbar.set_stat(min(time, settings.time_end))
        progressbar.update()

    progressbar.end()
    return settings


def fvm_sindy_timestep_generator(
    mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
):
    iteration = 0
    time = 0.0

    assert model.dimension == mesh.dimension

    progressbar = pyprog.ProgressBar(
        "{}: ".format(settings.name),
        "\r",
        settings.time_end,
        complete_symbol="█",
        not_complete_symbol="-",
    )
    progressbar.progress_explain = ""
    progressbar.update()
    # force=True is needed in order to disable old logging root handlers (if multiple test cases are run by one file)
    # otherwise only a logfile will be created for the first testcase but the log file gets deleted by rmtree
    # logging.basicConfig(
    #     filename=os.path.join(
    #         os.path.join(main_dir, self.output_dir), "logfile.log"
    #     ),
    #     filemode="w",
    #     level=self.logging_level,
    #     force=True,
    # )
    # logger = logging.getLogger(__name__ + ":solve_steady")

    Q, Qaux = _initialize_problem(model, mesh)
    parameters = model.parameter_values
    settings.parameters
    Qnew = deepcopy(Q)

    # for callback in self.callback_function_list_init:
    #     Qnew, kwargs = callback(self, Qnew, **kwargs)

    i_snapshot = 0
    i_snapshot_tmp = 0
    dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
    io.init_output_directory(settings.output_dir, settings.output_clean_dir)
    i_snapshot = io.save_fields(
        settings.output_dir, time, 0, i_snapshot, Qnew, Qaux, settings.output_write_all
    )
    i_snapshot_tmp = io.save_fields(
        settings.output_dir,
        time,
        0,
        i_snapshot_tmp,
        Qnew,
        Qaux,
        settings.output_write_all,
        filename="fields_intermediate.hdf5",
    )
    mesh.write_to_hdf5(settings.output_dir)

    settings.parameters = model.parameters.to_value_dict(model.parameter_values)
    io.save_settings(settings.output_dir, settings)

    time_start = gettime()

    pde, bc = load_runtime_model(model)

    space_solution_operator = _get_semidiscrete_solution_operator(
        mesh, pde, bcs, settings
    )
    compute_source = _get_source(mesh, pde, settings)
    compute_source_jac = _get_source_jac(mesh, pde, settings)
    compute_max_abs_eigenvalue = _get_compute_max_abs_eigenvalue(
        mesh, pde, model.boundary_conditions, settings
    )
    min_inradius = np.min(mesh.element_inradius)

    # print(f'hi from process {os.getpid()}')
    while time < settings.time_end:
        assert i_snapshot == i_snapshot_tmp
        Qnew_load, Qaux_load, time_load = io.load_fields_from_hdf5(
            "output_lvl1/fields.hdf5", i_snapshot=i_snapshot - 1
        )
        # map the fields
        Qnew[:, 0] = Qnew_load[:, 0]
        Qnew[:, 1] = Qnew_load[:, 1]

        # print(f'in loop from process {os.getpid()}')
        Q = deepcopy(Qnew)
        dt = settings.compute_dt(
            Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
        )
        assert dt > 10 ** (-6)

        assert not np.isnan(dt) and np.isfinite(dt)

        # if iteration < self.warmup_interations:
        #     dt *= self.dt_relaxation_factor
        # if dt < self.dtmin:
        #     dt = self.dtmin

        # add 0.001 safty measure to avoid very small time steps
        if settings.truncate_last_time_step:
            if time + dt * 1.001 > settings.time_end:
                dt = settings.time_end - time + 10 ** (-10)

        # TODO this two things should be 'callouts'!!
        Qnew = ode_solver_flux(space_solution_operator, Q, Qaux, parameters, dt)

        # TODO print out the snapshots for testing SINDy
        # TODO this is not exactly what I want. I want to use the high fidelity solution as an IC in each time step! and do one time step from there
        # But is this really what I want? Inspired from GNS, this is not a robust way of doing it. However, how it is currenlty, it also does not makes sence, because
        # after some time, I will be very far off the true solution.
        Qnew_save = np.array(Qnew)

        Qnew = ode_solver_source(
            compute_source, Qnew, Qaux, parameters, dt, func_jac=compute_source_jac
        )

        # if time > 0.8:
        #     gradQ = mesh.gradQ(Q)
        #     for q, gradq in zip(Q, gradQ):
        #         u, v, w = reconstruct_uvw(q, gradq, model.levels, model.basis)
        # time, Qnew, Qaux, parameters, settings = callback_post_solve(time, Qnew, Qaux, parameters, settings)
        # Qavg = np.mean(Qavg_5steps, axis=0)
        # error = (
        #     self.compute_Lp_error(Qnew - Qavg, p=2, **kwargs)
        #     / (self.compute_Lp_error(Qavg, p=2, **kwargs) + 10 ** (-10))
        # ).max()
        # Qavg_5steps.pop()
        # Qavg_5steps.insert(0, Qnew)

        # Update solution and time
        time += dt
        iteration += 1

        # for callback in self.callback_function_list_post_solvestep:
        #     Qnew, kwargs = callback(self, Qnew, **kwargs)

        i_snapshot = io.save_fields(
            settings.output_dir,
            time,
            (i_snapshot + 1) * dt_snapshot,
            i_snapshot,
            Qnew,
            Qaux,
            settings.output_write_all,
        )
        i_snapshot_tmp = io.save_fields(
            settings.output_dir,
            time,
            (i_snapshot_tmp + 1) * dt_snapshot,
            i_snapshot_tmp,
            Qnew_save,
            Qaux,
            settings.output_write_all,
            filename="fields_intermediate.hdf5",
        )

        # logger.info(
        #     "Iteration: {:6.0f}, Runtime: {:6.2f}, Time: {:2.4f}, dt: {:2.4f}, error: {}".format(
        #         iteration, gettime() - time_start, time, dt, error
        #     )
        # )
        # print(f'finished timestep: {os.getpid()}')
        progressbar.set_stat(min(time, settings.time_end))
        progressbar.update()

    progressbar.end()
    return settings
