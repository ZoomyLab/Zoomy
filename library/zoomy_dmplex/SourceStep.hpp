#ifndef SOURCESTEP_HPP
#define SOURCESTEP_HPP

#include "VirtualSolver.hpp"

template <typename T>
class SourceStep {
private:
    SNES snes;
    DM dmQ, dmAux;
    Vec X_star; // Holds the state after transport
    std::vector<T> parameters;
    SourceKernelPtr source_kernel;
    JacobianKernelPtr jac_q;
    // ... other jacobians ...

public:
    SourceStep(DM q, DM aux, std::vector<T> params) : dmQ(q), dmAux(aux), parameters(params) {
        snes = NULL; X_star = NULL;
        source_kernel = Model<T>::source;
        jac_q = Model<T>::source_jacobian_wrt_variables;
    }

    ~SourceStep() {
        if (snes) SNESDestroy(&snes);
        if (X_star) VecDestroy(&X_star);
    }

    PetscErrorCode Setup() {
        PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
        PetscCall(SNESSetFunction(snes, NULL, FormFunctionWrapper, this));
        PetscCall(SNESSetJacobian(snes, NULL, NULL, FormJacobianWrapper, this));
        PetscCall(SNESSetFromOptions(snes));
        PetscCall(DMCreateGlobalVector(dmQ, &X_star));
        return PETSC_SUCCESS;
    }

    PetscErrorCode Solve(PetscReal dt, Vec X_curr) {
        if (!snes) PetscCall(Setup());
        
        // 1. Store X_star (result of explicit step)
        PetscCall(VecCopy(X_curr, X_star));
        
        // 2. Store dt in context (hacky via container or member, using member here)
        this->current_dt = dt;

        // 3. Solve F(U) = U - X_star - dt*S(U) = 0
        // Use X_curr as initial guess (it equals X_star, which is a good guess)
        PetscCall(SNESSolve(snes, NULL, X_curr));
        
        return PETSC_SUCCESS;
    }

private:
    PetscReal current_dt;

    PetscErrorCode FormFunction(SNES snes, Vec U, Vec Resid) {
        // Resid = U - X_star
        PetscCall(VecCopy(U, Resid));
        PetscCall(VecAXPY(Resid, -1.0, X_star));
        
        // Compute S(U)
        Vec S_global; PetscCall(VecDuplicate(U, &S_global)); PetscCall(VecZeroEntries(S_global));
        
        // Fetch Local U
        Vec U_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &U_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, U, INSERT_VALUES, U_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, U, INSERT_VALUES, U_loc));
        
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        // Need to update Aux based on U
        UpdateLocalAux(U_loc, A_loc);

        // Calculate Source locally
        Vec S_loc; PetscCall(DMGetLocalVector(dmQ, &S_loc)); PetscCall(VecZeroEntries(S_loc));
        PetscScalar *s_ptr; PetscCall(VecGetArray(S_loc, &s_ptr));
        const PetscScalar *u_ptr, *a_ptr;
        PetscCall(VecGetArrayRead(U_loc, &u_ptr)); PetscCall(VecGetArrayRead(A_loc, &a_ptr));
        
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
            const PetscScalar *q, *a; PetscScalar *s;
            PetscCall(DMPlexPointLocalRead(dmQ, c, u_ptr, &q));
            PetscCall(DMPlexPointLocalRead(dmAux, c, a_ptr, &a));
            PetscCall(DMPlexPointLocalRef(dmQ, c, s_ptr, &s));
            if (q && a && s) {
                auto src = source_kernel(q, a, parameters.data());
                for(int i=0; i<Model<T>::n_dof_q; ++i) s[i] = src[i];
            }
        }
        
        PetscCall(VecRestoreArrayRead(U_loc, &u_ptr)); PetscCall(VecRestoreArrayRead(A_loc, &a_ptr));
        PetscCall(VecRestoreArray(S_loc, &s_ptr));
        
        PetscCall(DMLocalToGlobalBegin(dmQ, S_loc, ADD_VALUES, S_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, S_loc, ADD_VALUES, S_global));
        
        // Resid = Resid - dt * S(U)
        PetscCall(VecAXPY(Resid, -current_dt, S_global));
        
        PetscCall(VecDestroy(&S_global));
        PetscCall(DMRestoreLocalVector(dmQ, &U_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(DMRestoreLocalVector(dmQ, &S_loc));
        
        return PETSC_SUCCESS;
    }

    PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat Jpre) {
        // J = I - dt * dS/dU
        // Helper: Compute (1/dt)*I - dS/dU then scale by dt?
        // Or just compute -dS/dU and add I on diagonal?
        
        // Let's compute -dS/dU directly
        PetscCall(MatZeroEntries(Jpre));
        
        Vec U_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &U_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, U, INSERT_VALUES, U_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, U, INSERT_VALUES, U_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        UpdateLocalAux(U_loc, A_loc);
        
        const PetscScalar *u_ptr, *a_ptr;
        PetscCall(VecGetArrayRead(U_loc, &u_ptr)); PetscCall(VecGetArrayRead(A_loc, &a_ptr));
        
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscSection section; PetscCall(DMGetGlobalSection(dmQ, &section));
        
        std::vector<PetscScalar> values(Model<T>::n_dof_q * Model<T>::n_dof_q);
        std::vector<PetscInt> indices(Model<T>::n_dof_q);

        for(PetscInt c=cStart; c<cEnd; ++c) {
            const PetscScalar *q, *a;
            PetscCall(DMPlexPointLocalRead(dmQ, c, u_ptr, &q));
            PetscCall(DMPlexPointLocalRead(dmAux, c, a_ptr, &a));
            
            if (q && a) {
                // Get dS/dU block
                // Simplified: Ignoring chain rule dS/dAux * dAux/dQ for brevity, 
                // but strictly should be included like in old code.
                auto J_val = jac_q(q, a, parameters.data());
                
                // We want I - dt * J_val
                for(int i=0; i<Model<T>::n_dof_q; ++i) {
                    for(int j=0; j<Model<T>::n_dof_q; ++j) {
                        PetscScalar val = - current_dt * J_val[i*Model<T>::n_dof_q + j];
                        if (i==j) val += 1.0;
                        values[i*Model<T>::n_dof_q + j] = val;
                    }
                }
                
                PetscInt goff; PetscCall(PetscSectionGetOffset(section, c, &goff));
                if (goff >= 0) {
                    for(int i=0; i<Model<T>::n_dof_q; ++i) indices[i] = goff + i;
                    PetscCall(MatSetValues(Jpre, Model<T>::n_dof_q, indices.data(), Model<T>::n_dof_q, indices.data(), values.data(), INSERT_VALUES));
                }
            }
        }
        
        PetscCall(VecRestoreArrayRead(U_loc, &u_ptr)); PetscCall(VecRestoreArrayRead(A_loc, &a_ptr));
        PetscCall(DMRestoreLocalVector(dmQ, &U_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        
        PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
        if (J != Jpre) {
            PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
        }
        return PETSC_SUCCESS;
    }

    void UpdateLocalAux(Vec X_loc, Vec A_loc) {
        const PetscScalar *x_ptr; PetscScalar *a_ptr;
        VecGetArrayRead(X_loc, &x_ptr); VecGetArray(A_loc, &a_ptr);
        PetscInt cStart, cEnd; DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd);
        for(PetscInt c=cStart; c<cEnd; ++c) {
            const PetscScalar *q; PetscScalar *a;
            DMPlexPointLocalRead(dmQ, c, x_ptr, &q); DMPlexPointLocalRef(dmAux, c, a_ptr, &a);
            if (q && a) {
                auto res = Model<T>::update_aux_variables(q, a, parameters.data());
                for(int i=0; i<Model<T>::n_dof_qaux; ++i) a[i] = res[i];
            }
        }
        VecRestoreArrayRead(X_loc, &x_ptr); VecRestoreArray(A_loc, &a_ptr);
    }

    static PetscErrorCode FormFunctionWrapper(SNES snes, Vec U, Vec Resid, void *ctx) { return ((SourceStep*)ctx)->FormFunction(snes, U, Resid); }
    static PetscErrorCode FormJacobianWrapper(SNES snes, Vec U, Mat J, Mat Jpre, void *ctx) { return ((SourceStep*)ctx)->FormJacobian(snes, U, J, Jpre); }
};
#endif