#ifndef SOLVER_STRATEGIES_HPP
#define SOLVER_STRATEGIES_HPP

#include <petsc.h>
#include "ModularSolver.hpp"

// Abstract Interface
class SolverStrategy {
public:
    virtual ~SolverStrategy() = default;
    virtual PetscErrorCode SetupTS(TS ts, ModularSolver* solver) = 0;
};

// ============================================================================
// 1. Splitting Strategy
// ============================================================================
class SplittingStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        PetscCall(TSSetRHSFunction(ts, NULL, RHSWrapper, solver));
        PetscCall(TSSetPostStep(ts, SplittingWrapper));
        PetscCall(TSSetType(ts, TSSSP)); 
        PetscCall(TSSSPSetType(ts, TSSSPRK104)); 
        return PETSC_SUCCESS;
    }

private:
    static PetscErrorCode RHSWrapper(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
        return ((ModularSolver*)ctx)->transport->FormRHS(t, X, F);
    }

    static PetscErrorCode SplittingWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        ModularSolver* solver = (ModularSolver*)ctx;
        Vec X_curr; TSGetSolution(ts, &X_curr);
        PetscReal dt; TSGetTimeStep(ts, &dt); 
        if(solver->source_solver) {
             PetscCall(solver->source_solver->Solve(dt, X_curr, solver->A)); 
        }
        return solver->PostStep(ts);
    }
};

// ============================================================================
// 2. IMEX Strategy
// ============================================================================
class IMEXStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        PetscCall(TSSetRHSFunction(ts, NULL, RHSWrapper, solver));
        PetscCall(TSSetIFunction(ts, NULL, IFunctionWrapper, solver));
        
        Mat P;
        PetscCall(DMCreateMatrix(solver->dmQ, &P));
        PetscCall(TSSetIJacobian(ts, NULL, P, IJacobianWrapper, solver));
        PetscCall(MatDestroy(&P));

        PetscCall(TSSetType(ts, TSARKIMEX)); 
        PetscCall(TSARKIMEXSetType(ts, "2e"));
        PetscCall(TSSetPostStep(ts, PostStepWrapper)); 
        return PETSC_SUCCESS;
    }

private:
    static PetscErrorCode PostStepWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        return ((ModularSolver*)ctx)->PostStep(ts);
    }

    static PetscErrorCode RHSWrapper(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
        return ((ModularSolver*)ctx)->transport->FormRHS(t, X, F);
    }

    static PetscErrorCode IFunctionWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, Vec F, void* ctx) {
        ModularSolver* solver = (ModularSolver*)ctx;
        PetscCall(VecCopy(X_dot, F));
        PetscCall(solver->AddImplicitSourceToResidual(X, F, -1.0));
        return PETSC_SUCCESS;
    }

    static PetscErrorCode IJacobianWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, PetscReal a, Mat J, Mat P, void* ctx) {
        return ((ModularSolver*)ctx)->FormSourceJacobian(t, X, a, P);
    }
};

// ============================================================================
// 3. Fully Implicit Strategy
// ============================================================================
class FullyImplicitStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        PetscCall(TSSetType(ts, TSBDF));
        PetscCall(TSBDFSetOrder(ts, 2));

        PetscCall(TSSetIFunction(ts, NULL, IFunctionWrapper, solver));
        
        Mat P;
        PetscCall(DMCreateMatrix(solver->dmQ, &P));
        
        // --- Safety: Allow new nonzeros if the preallocation is slightly off ---
        PetscCall(MatSetOption(P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE)); 

        PetscCall(TSSetIJacobian(ts, P, P, IJacobianWrapper, solver));
        PetscCall(MatDestroy(&P));

        SNES snes;
        PetscCall(TSGetSNES(ts, &snes));
        PetscCall(SNESSetUseMatrixFree(snes, PETSC_TRUE, PETSC_FALSE));

        PetscCall(TSSetPostStep(ts, PostStepWrapper));
        return PETSC_SUCCESS;
    }

private:
    static PetscErrorCode PostStepWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        return ((ModularSolver*)ctx)->PostStep(ts);
    }

    static PetscErrorCode IFunctionWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, Vec F, void* ctx) {
        ModularSolver* solver = (ModularSolver*)ctx;
        PetscCall(VecCopy(X_dot, F));
        PetscCall(solver->AddImplicitSourceToResidual(X, F, -1.0));
        
        Vec R_transport;
        PetscCall(DMGetGlobalVector(solver->dmQ, &R_transport));
        PetscCall(solver->transport->FormRHS(t, X, R_transport));
        PetscCall(VecAXPY(F, -1.0, R_transport));
        PetscCall(DMRestoreGlobalVector(solver->dmQ, &R_transport));
        return PETSC_SUCCESS;
    }

    static PetscErrorCode IJacobianWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, PetscReal a, Mat J, Mat P, void* ctx) {
        ModularSolver* solver = (ModularSolver*)ctx;
        
        // Assemble Preconditioner P
        PetscCall(solver->FormImplicitJacobian(t, X, a, P));

        // Initialize Matrix-Free J
        if (J && J != P) {
            PetscBool is_mffd;
            PetscCall(PetscObjectTypeCompare((PetscObject)J, MATMFFD, &is_mffd));
            if (is_mffd) {
                PetscCall(MatMFFDSetBase(J, X, NULL));
            }
            PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
        }
        return PETSC_SUCCESS;
    }
};

#endif