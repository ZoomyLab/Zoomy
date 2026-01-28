#ifndef HIGHERORDERSOLVER_HPP
#define HIGHERORDERSOLVER_HPP

#include "VirtualSolver.hpp"
#include <vector>
#include <algorithm>

class HigherOrderSolver : public VirtualSolver {
private:
    Vec X_old;      // State at time t
    Vec X_low;      // Candidate state from 1st Order
    PetscInt max_steps;
    PetscInt target_order; 

public:
    HigherOrderSolver(PetscInt order) : VirtualSolver(), X_old(NULL), X_low(NULL), max_steps(100), target_order(order) {}
    
    ~HigherOrderSolver() {
        if (X_old) VecDestroy(&X_old);
        if (X_low) VecDestroy(&X_low);
    }

    PetscErrorCode TakeOneStep(PetscReal time, PetscReal dt) override {
        PetscFunctionBeginUser;
        if (!X_old) {
            PetscCall(VecDuplicate(X, &X_old));
            PetscCall(VecDuplicate(X, &X_low));
        }
        
        // 1. Backup state
        PetscCall(VecCopy(X, X_old));

        // 2. Candidate High-Order Step
        SetSolverOrder(2);
        PetscCall(TSSetTime(ts, time));
        PetscCall(TSSetTimeStep(ts, dt));
        PetscCall(TSStep(ts));

        // 3. Validity Check (MOOD)
        std::vector<PetscInt> bad_cells;
        PetscCall(CheckTVD(X_old, X, bad_cells));
        
        PetscInt n_bad_global;
        PetscInt n_bad_local = bad_cells.size();
        PetscCallMPI(MPI_Allreduce(&n_bad_local, &n_bad_global, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

        if (n_bad_global > 0) {
            // 4. Fallback to 1st Order for local blending
            PetscCall(VecCopy(X_old, X_low));
            SetSolverOrder(1);
            PetscCall(TSSetSolution(ts, X_low));
            PetscCall(TSStep(ts));
            
            PetscCall(BlendSolutions(X, X_low, bad_cells));
            PetscCall(TSSetSolution(ts, X));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    void SetSolverOrder(int order) {
        PetscFV fvm;
        DMGetField(dmQ, 0, NULL, (PetscObject*)&fvm);
        
        if (order >= 2) {
            PetscFVSetType(fvm, PETSCFVLEASTSQUARES);
            PetscLimiter lim;
            PetscLimiterCreate(PETSC_COMM_WORLD, &lim);
            PetscLimiterSetType(lim, PETSCLIMITERNONE);
            PetscFVSetLimiter(fvm, lim);
            PetscLimiterDestroy(&lim);
        } else {
            PetscFVSetType(fvm, PETSCFVUPWIND);
        }
    }

    PetscErrorCode CheckTVD(Vec x_old, Vec x_new, std::vector<PetscInt>& bad_cells) {
        PetscFunctionBeginUser;
        bad_cells.clear();
        Vec locX_old, locX_new;
        PetscCall(DMGetLocalVector(dmQ, &locX_old));
        PetscCall(DMGetLocalVector(dmQ, &locX_new));
        PetscCall(DMGlobalToLocalBegin(dmQ, x_old, INSERT_VALUES, locX_old));
        PetscCall(DMGlobalToLocalEnd(dmQ, x_old, INSERT_VALUES, locX_old));
        PetscCall(DMGlobalToLocalBegin(dmQ, x_new, INSERT_VALUES, locX_new));
        PetscCall(DMGlobalToLocalEnd(dmQ, x_new, INSERT_VALUES, locX_new));
        const PetscScalar *old_ptr, *new_ptr;
        PetscCall(VecGetArrayRead(locX_old, &old_ptr));
        PetscCall(VecGetArrayRead(locX_new, &new_ptr));
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd)); 
        const Real eps = 1e-10; 
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt gStart, gEnd;
            PetscCall(DMPlexGetPointGlobal(dmQ, c, &gStart, &gEnd));
            if (gStart < 0) continue; 
            Real u_old = old_ptr[c * Model<Real>::n_dof_q + 0]; 
            Real min_val = u_old; Real max_val = u_old;
            const PetscInt *faces; PetscInt numFaces;
            PetscCall(DMPlexGetCone(dmQ, c, &faces));
            PetscCall(DMPlexGetConeSize(dmQ, c, &numFaces));
            for(int f=0; f<numFaces; ++f) {
                const PetscInt *neighbors; PetscInt numNeighbors;
                PetscCall(DMPlexGetSupport(dmQ, faces[f], &neighbors));
                PetscCall(DMPlexGetSupportSize(dmQ, faces[f], &numNeighbors));
                for(int n=0; n<numNeighbors; ++n) {
                    PetscInt nCell = neighbors[n];
                    if (nCell == c) continue;
                    Real u_neigh = old_ptr[nCell * Model<Real>::n_dof_q + 0];
                    if (u_neigh < min_val) min_val = u_neigh;
                    if (u_neigh > max_val) max_val = u_neigh;
                }
            }
            Real u_new = new_ptr[c * Model<Real>::n_dof_q + 0];
            if (u_new < min_val - eps || u_new > max_val + eps) bad_cells.push_back(c);
        }
        PetscCall(VecRestoreArrayRead(locX_old, &old_ptr));
        PetscCall(VecRestoreArrayRead(locX_new, &new_ptr));
        PetscCall(DMRestoreLocalVector(dmQ, &locX_old));
        PetscCall(DMRestoreLocalVector(dmQ, &locX_new));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode BlendSolutions(Vec x_target, Vec x_source, const std::vector<PetscInt>& cells) {
        PetscFunctionBeginUser;
        Vec loc_target, loc_source;
        PetscCall(DMGetLocalVector(dmQ, &loc_target));
        PetscCall(DMGetLocalVector(dmQ, &loc_source));
        PetscCall(DMGlobalToLocalBegin(dmQ, x_target, INSERT_VALUES, loc_target));
        PetscCall(DMGlobalToLocalEnd(dmQ, x_target, INSERT_VALUES, loc_target));
        PetscCall(DMGlobalToLocalBegin(dmQ, x_source, INSERT_VALUES, loc_source));
        PetscCall(DMGlobalToLocalEnd(dmQ, x_source, INSERT_VALUES, loc_source));
        PetscScalar *t_ptr; const PetscScalar *s_ptr;
        PetscCall(VecGetArray(loc_target, &t_ptr));
        PetscCall(VecGetArrayRead(loc_source, &s_ptr));
        PetscSection section;
        PetscCall(DMGetLocalSection(dmQ, &section));
        for (PetscInt c : cells) {
            PetscInt off;
            PetscCall(PetscSectionGetOffset(section, c, &off));
            for(int d=0; d<Model<Real>::n_dof_q; ++d) {
                t_ptr[off + d] = s_ptr[off + d];
            }
        }
        PetscCall(VecRestoreArray(loc_target, &t_ptr));
        PetscCall(VecRestoreArrayRead(loc_source, &s_ptr));
        PetscCall(DMLocalToGlobalBegin(dmQ, loc_target, INSERT_VALUES, x_target));
        PetscCall(DMLocalToGlobalEnd(dmQ, loc_target, INSERT_VALUES, x_target));
        PetscCall(DMRestoreLocalVector(dmQ, &loc_target));
        PetscCall(DMRestoreLocalVector(dmQ, &loc_source));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

};
#endif