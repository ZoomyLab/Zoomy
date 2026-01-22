#ifndef PETSC_HELPERS_HPP
#define PETSC_HELPERS_HPP

#include <petscdmplex.h>
#include <petscfv.h>
#include <petscds.h>
#include "Model.H"
#include "Numerics.H"

using Real = PetscScalar;

namespace Zoomy {

    static void RiemannAdapter(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, 
                               const PetscScalar *xL, const PetscScalar *xR, 
                               PetscInt numConstants, const PetscScalar constants[], 
                               PetscScalar *flux, void *ctx) 
    {
        // 1. Placeholder Aux (Set to 0.0)
        // TODO: Access actual aux data if passed by PETSc context or reconstruction
        Real Qaux_L[Model<Real>::n_dof_qaux] = {0.0};
        Real Qaux_R[Model<Real>::n_dof_qaux] = {0.0};

        // 2. Geometry
        Real area = 0.0;
        for(int d=0; d<dim; ++d) area += n[d]*n[d];
        area = std::sqrt(area);
        
        Real n_hat[3] = {0.0, 0.0, 0.0};
        if (area > 1e-14) { 
            for(int d=0; d<dim; ++d) n_hat[d] = n[d] / area; 
        } else { 
            for(int i=0; i<Model<Real>::n_dof_q; ++i) flux[i] = 0.0; 
            return; 
        }
        
        // 3. Call Numerics
        auto res = Numerics<Real>::numerical_flux(xL, xR, Qaux_L, Qaux_R, constants, n_hat);
        
        // 4. Copy to PETSc Buffer
        for(int i=0; i<Model<Real>::n_dof_q; ++i) {
            flux[i] = res[i] * area;
        }
    }

    static PetscErrorCode InitialConditionAdapter(PetscInt dim, PetscReal time, const PetscReal x[], 
                                                  PetscInt Nf, PetscScalar u[], void *ctx)
    {
        const PetscReal* p = (const PetscReal*)ctx;
        auto res = Model<Real>::initial_condition(x, p);
        for(int i=0; i<Model<Real>::n_dof_q; ++i) u[i] = res[i];
        return PETSC_SUCCESS;
    }

    // NEW: Adapter for Aux Initial Conditions
    static PetscErrorCode InitialAuxConditionAdapter(PetscInt dim, PetscReal time, const PetscReal x[], 
                                                     PetscInt Nf, PetscScalar u[], void *ctx)
    {
        const PetscReal* p = (const PetscReal*)ctx;
        auto res = Model<Real>::initial_aux_condition(x, p);
        for(int i=0; i<Model<Real>::n_dof_qaux; ++i) u[i] = res[i];
        return PETSC_SUCCESS;
    }

    static PetscErrorCode BoundaryAdapter(PetscReal time, const PetscReal *c, const PetscReal *n, 
                                          const PetscScalar *xI, PetscScalar *xG, void *ctx) 
    {
        const int bc_idx = *(int*)ctx; 
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0}; 
        Real dX = 0.0; 
        
        auto res = Model<Real>::boundary_conditions(bc_idx, xI, Qaux, (const Real*)n, (const Real*)c, time, dX);
        
        for(int i=0; i<Model<Real>::n_dof_q; ++i) xG[i] = res[i];
        return PETSC_SUCCESS;
    }
}
#endif