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
// 1. Splitting Strategy (Explicit Transport + Post-Step Implicit Source)
// ============================================================================
class SplittingStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        PetscCall(TSSetRHSFunction(ts, NULL, RHSWrapper, solver));
        PetscCall(TSSetPostStep(ts, SplittingWrapper));
        PetscCall(TSSetType(ts, TSSSP)); // Strong Stability Preserving
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
        
        // Solve Source locally
        if(solver->source_solver) {
             PetscCall(solver->source_solver->Solve(dt, X_curr, solver->A)); 
        }
        return solver->PostStep(ts);
    }
};

// ============================================================================
// 2. IMEX Strategy (Explicit Transport + Implicit Source in IFunction)
//    Solves friction/source simultaneously with time step (No splitting error)
// ============================================================================
class IMEXStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        // Explicit Part: Fluxes
        PetscCall(TSSetRHSFunction(ts, NULL, RHSWrapper, solver));
        
        // Implicit Part: Source Term
        PetscCall(TSSetIFunction(ts, NULL, IFunctionWrapper, solver));
        PetscCall(TSSetIJacobian(ts, NULL, NULL, IJacobianWrapper, solver));
        
        PetscCall(TSSetType(ts, TSARKIMEX)); 
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

    // G = U_dot - Source(U)
    static PetscErrorCode IFunctionWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, Vec F, void* ctx) {
        ModularSolver* solver = (ModularSolver*)ctx;
        // G = U_dot
        PetscCall(VecCopy(X_dot, F));
        // G -= Source(U)
        PetscCall(solver->AddImplicitSourceToResidual(X, F, -1.0));
        return PETSC_SUCCESS;
    }

    static PetscErrorCode IJacobianWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, PetscReal a, Mat J, Mat P, void* ctx) {
        // J = a*I - dSource/dQ
        return ((ModularSolver*)ctx)->FormSourceJacobian(t, X, a, P);
    }
};

// ============================================================================
// 3. Fully Implicit Strategy (Everything in IFunction)
//    - Relies on -snes_mf_operator for Flux Jacobian
//    - Uses Analytical Source Jacobian for Preconditioning
// ============================================================================
class FullyImplicitStrategy : public SolverStrategy {
public:
    PetscErrorCode SetupTS(TS ts, ModularSolver* solver) override {
        PetscCall(TSSetIFunction(ts, NULL, IFunctionWrapper, solver));
        PetscCall(TSSetIJacobian(ts, NULL, NULL, IJacobianWrapper, solver));
        PetscCall(TSSetType(ts, TSARKIMEX));
        PetscCall(TSSetPostStep(ts, PostStepWrapper));
        return PETSC_SUCCESS;
    }

private:
    static PetscErrorCode PostStepWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        return ((ModularSolver*)ctx)->PostStep(ts);
    }

    // G = U_dot - Source(U) - FluxDiv(U)
    static PetscErrorCode IFunctionWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, Vec F, void* ctx) {
        ModularSolver* solver = (ModularSolver*)ctx;

        // 1. G = U_dot - Source(U)
        PetscCall(VecCopy(X_dot, F));
        PetscCall(solver->AddImplicitSourceToResidual(X, F, -1.0));

        // 2. G -= FluxDiv(U)
        // Transport::FormRHS returns (+FluxDiv). We subtract it.
        Vec Fluxes;
        PetscCall(VecDuplicate(F, &Fluxes));
        PetscCall(solver->transport->FormRHS(t, X, Fluxes));
        PetscCall(VecAXPY(F, -1.0, Fluxes));
        PetscCall(VecDestroy(&Fluxes));
        return PETSC_SUCCESS;
    }

    static PetscErrorCode IJacobianWrapper(TS ts, PetscReal t, Vec X, Vec X_dot, PetscReal a, Mat J, Mat P, void* ctx) {
        // P = a*I - dSource/dQ
        // The Flux Jacobian part is handled by Matrix-Free operator (-snes_mf_operator)
        // This 'P' acts as the Preconditioner.
        return ((ModularSolver*)ctx)->FormSourceJacobian(t, X, a, P);
    }
};

#endif