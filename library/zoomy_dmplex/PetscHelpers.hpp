#ifndef PETSC_HELPERS_HPP
#define PETSC_HELPERS_HPP

#include <petscdmplex.h>
#include <petscfv.h>
#include <petscds.h>
#include "Model.H"
#include "Numerics.H"

using Real = PetscScalar;

namespace Zoomy {


    static void RiemannAdapter(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, void *ctx) {
        Real Qaux_L[Model<Real>::n_dof_qaux] = {0.0};
        Real Qaux_R[Model<Real>::n_dof_qaux] = {0.0};
        Real area = 0.0;
        for(int d=0; d<dim; ++d) area += n[d]*n[d];
        area = std::sqrt(area);
        
        Real n_hat[3] = {0.0, 0.0, 0.0};
        if (area > 1e-14) { for(int d=0; d<dim; ++d) n_hat[d] = n[d] / area; } 
        else { for(int d=0; d<dim; ++d) flux[d] = 0.0; return; }
        
        Real flux_per_area[Model<Real>::n_dof_q];
        Numerics<Real>::numerical_flux(xL, xR, Qaux_L, Qaux_R, (const Real*)n_hat, flux_per_area);
        
        for(int i=0; i<Model<Real>::n_dof_q; ++i) flux[i] = flux_per_area[i] * area;
    }

    static PetscErrorCode BoundaryAdapter(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx) {
        const int bc_idx = *(int*)ctx; 
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0}; 
        Real dX = 0.0; 
        Model<Real>::boundary_conditions(bc_idx, xI, Qaux, (const Real*)n, (const Real*)c, time, dX, xG);
        return PETSC_SUCCESS;
    }
}
#endif