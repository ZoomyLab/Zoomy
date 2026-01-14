#ifndef STANDARD_SOLVER_HPP
#define STANDARD_SOLVER_HPP

#include "VirtualSolver.hpp"

class FirstOrderSolver : public VirtualSolver {
public:
    using VirtualSolver::VirtualSolver; 

    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-ufv_cfl", &cfl, NULL));

        // Overlap = 1 for Standard Solver
        PetscCall(SetupArchitecture(1));
        
        PetscCall(TSMonitorSet(ts, MonitorWrapper, this, NULL));
        PetscCall(TSSetPreStep(ts, PreStepWrapper));

        PetscCall(UpdateBoundaryGhosts(0.0));
        PetscCall(WriteVTU(0, 0.0)); 

        PetscCall(TSSolve(ts, X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    static PetscErrorCode MonitorWrapper(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx) {
        FirstOrderSolver* self = (FirstOrderSolver*)ctx;
        return self->WriteVTU(step, time);
    }

    static PetscErrorCode PreStepWrapper(TS ts) {
        void *ctx; TSGetApplicationContext(ts, &ctx);
        FirstOrderSolver* self = (FirstOrderSolver*)ctx;
        
        PetscReal time; TSGetTime(ts, &time);
        PetscCall(self->UpdateBoundaryGhosts(time));
        
        Vec X_local;
        DMGetLocalVector(self->dmQ, &X_local);
        DMGlobalToLocalBegin(self->dmQ, self->X, INSERT_VALUES, X_local);
        DMGlobalToLocalEnd(self->dmQ, self->X, INSERT_VALUES, X_local);
        const PetscScalar *x_ptr;
        VecGetArrayRead(X_local, &x_ptr);
        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(self->dmQ, 0, &cStart, &cEnd); 
        Real max_eigen_local = 0.0;
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0}; 
        std::vector<Real> lam(Model<Real>::n_dof_q); 
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const Real* Q_cell = &x_ptr[c * Model<Real>::n_dof_q];
            if (Q_cell[0] < 1e-6) continue; 
            for (int d = 0; d < Model<Real>::dimension; ++d) {
                Real n[3] = {0.0, 0.0, 0.0}; n[d] = 1.0;
                Model<Real>::eigenvalues(Q_cell, Qaux, n, lam.data());
                for(Real val : lam) max_eigen_local = std::max(max_eigen_local, std::abs(val));
            }
        }
        VecRestoreArrayRead(X_local, &x_ptr);
        DMRestoreLocalVector(self->dmQ, &X_local);
        Real max_eigen_global;
        MPI_Allreduce(&max_eigen_local, &max_eigen_global, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD);
        Real dt = (max_eigen_global > 1e-12) ? self->cfl * self->minRadius / max_eigen_global : 1e-4; 
        TSSetTimeStep(ts, dt);
        return PETSC_SUCCESS;
    }
};
#endif