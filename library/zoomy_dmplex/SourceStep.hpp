#ifndef SOURCESTEP_HPP
#define SOURCESTEP_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <petscvec.h>
#include <petscdmplex.h>
#include "Model.H"

template <typename T>
class SourceStep {
private:
    DM dmQ;
    DM dmAux;
    std::vector<T>& parameters;

    // Local Solver Constants
    const int max_newton_iter = 15;
    const T tol = 1e-8;

public:
    SourceStep(DM dm, DM aux, std::vector<T>& params) : dmQ(dm), dmAux(aux), parameters(params) {}

    ~SourceStep() {}

    // Solve Implicitly: Q_new = Q_old + dt * S(Q_new)
    PetscErrorCode Solve(T dt, Vec X, Vec A) {
        PetscFunctionBeginUser;
        
        PetscScalar *x_arr;
        const PetscScalar *a_arr;
        
        PetscCall(VecGetArray(X, &x_arr));
        PetscCall(VecGetArrayRead(A, &a_arr));
        
        // --- Ownership Ranges & Global Sections (Critical for Parallel Safety) ---
        PetscInt rstart_Q, rstart_A;
        PetscCall(VecGetOwnershipRange(X, &rstart_Q, NULL));
        PetscCall(VecGetOwnershipRange(A, &rstart_A, NULL));

        PetscSection sQ_glob, sAux_glob;
        PetscCall(DMGetGlobalSection(dmQ, &sQ_glob));
        PetscCall(DMGetGlobalSection(dmAux, &sAux_glob));
        // ------------------------------------------------------------------------
        
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        const T* params_ptr = parameters.data();
        const int n_dof = Model<T>::n_dof_q;
        const int n_aux = Model<T>::n_dof_qaux;

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt offQ_glob, offAux_glob;
            PetscCall(PetscSectionGetOffset(sQ_glob, c, &offQ_glob));
            PetscCall(PetscSectionGetOffset(sAux_glob, c, &offAux_glob));
            
            // Only solve for owned cells (non-negative global offset)
            if (offQ_glob >= 0) {
                PetscInt idx_Q = offQ_glob - rstart_Q;
                PetscScalar* q_ptr = &x_arr[idx_Q];
                
                const PetscScalar* aux_ptr = nullptr;
                if (offAux_glob >= 0) {
                    PetscInt idx_Aux = offAux_glob - rstart_A;
                    aux_ptr = &a_arr[idx_Aux];
                }

                SolveLocalCell(dt, q_ptr, aux_ptr, params_ptr, n_dof, n_aux);
            }
        }

        PetscCall(VecRestoreArray(X, &x_arr));
        PetscCall(VecRestoreArrayRead(A, &a_arr));
        
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    void SolveLocalCell(T dt, PetscScalar* q, const PetscScalar* aux_init, const T* params, int n_q, int n_aux) {
        // Local state vectors
        std::vector<PetscScalar> q_old(n_q);
        std::vector<PetscScalar> q_curr(n_q);
        std::vector<PetscScalar> aux_curr(n_aux);
        
        // Linear algebra workspace
        std::vector<PetscScalar> residual(n_q);
        std::vector<PetscScalar> delta(n_q);
        std::vector<PetscScalar> J(n_q * n_q);

        // Initialize state
        for(int i=0; i<n_q; ++i) {
            q_old[i] = q[i];
            q_curr[i] = q[i];
        }
        // Initialize Aux (if available)
        if(aux_init) {
            for(int i=0; i<n_aux; ++i) aux_curr[i] = aux_init[i];
        } else {
            std::fill(aux_curr.begin(), aux_curr.end(), 0.0);
        }

        for(int iter=0; iter < max_newton_iter; ++iter) {
            // 1. Update Aux variables based on current Q (Consistency Step)
            auto updated_aux = Model<T>::update_aux_variables(q_curr.data(), aux_curr.data(), params);
            for(int i=0; i<n_aux; ++i) aux_curr[i] = updated_aux[i];

            // 2. Compute Source Term and Residual
            auto S = Model<T>::source(q_curr.data(), aux_curr.data(), params);
            
            T res_norm = 0.0;
            for(int i=0; i<n_q; ++i) {
                // R = Q_curr - Q_old - dt * S
                residual[i] = q_curr[i] - q_old[i] - dt * S[i];
                res_norm += residual[i] * residual[i];
            }
            if(std::sqrt(res_norm) < tol) break;

            // 3. Compute Analytical Jacobian with Chain Rule
            // J = I - dt * ( dS/dQ + dS/dAux * dAux/dQ )
            
            // Get Jacobians from Model
            auto dS_dQ   = Model<T>::source_jacobian_wrt_variables(q_curr.data(), aux_curr.data(), params);
            auto dS_dAux = Model<T>::source_jacobian_wrt_aux_variables(q_curr.data(), aux_curr.data(), params);
            auto dAux_dQ = Model<T>::update_aux_variables_jacobian_wrt_variables(q_curr.data(), aux_curr.data(), params);

            // Reset J to Identity
            std::fill(J.begin(), J.end(), 0.0);
            for(int i=0; i<n_q; ++i) J[i*n_q + i] = 1.0;

            // Subtract dt * Total_Jacobian
            for(int i=0; i<n_q; ++i) {       // Row (Equation i)
                for(int j=0; j<n_q; ++j) {   // Col (Derivative w.r.t Q_j)
                    
                    // Direct term: dS_i / dQ_j
                    T term = dS_dQ[i*n_q + j];

                    // Chain rule term: sum_k ( dS_i / dAux_k * dAux_k / dQ_j )
                    for(int k=0; k<n_aux; ++k) {
                        // dS_dAux is (n_q x n_aux) flattened -> index [i * n_aux + k]
                        // dAux_dQ is (n_aux x n_q) flattened -> index [k * n_q + j]
                        term += dS_dAux[i * n_aux + k] * dAux_dQ[k * n_q + j];
                    }

                    J[i*n_q + j] -= dt * term;
                }
            }

            // 4. Solve Linear System: J * delta = -Residual
            for(int i=0; i<n_q; ++i) residual[i] = -residual[i];
            
            if(SolveLinearSystem(J, residual, delta, n_q)) {
                for(int i=0; i<n_q; ++i) q_curr[i] += delta[i];
            } else {
                break; // Singular Jacobian or failure
            }
        }

        // Write back result
        for(int i=0; i<n_q; ++i) q[i] = q_curr[i];
    }

    bool SolveLinearSystem(std::vector<PetscScalar>& A, std::vector<PetscScalar>& b, std::vector<PetscScalar>& x, int N) {
        // Gaussian Elimination with Partial Pivoting
        for (int k=0; k<N; ++k) {
            int max_row = k;
            T max_val = std::abs(A[k*N + k]);
            for (int i=k+1; i<N; ++i) {
                if (std::abs(A[i*N + k]) > max_val) {
                    max_val = std::abs(A[i*N + k]);
                    max_row = i;
                }
            }
            if (max_val < 1e-14) return false; 

            if (max_row != k) {
                for (int j=k; j<N; ++j) std::swap(A[k*N + j], A[max_row*N + j]);
                std::swap(b[k], b[max_row]);
            }

            for (int i=k+1; i<N; ++i) {
                T factor = A[i*N + k] / A[k*N + k];
                for (int j=k; j<N; ++j) A[i*N + j] -= factor * A[k*N + j];
                b[i] -= factor * b[k];
            }
        }

        for (int i=N-1; i>=0; --i) {
            T sum = 0.0;
            for (int j=i+1; j<N; ++j) sum += A[i*N + j] * x[j];
            x[i] = (b[i] - sum) / A[i*N + i];
        }
        return true;
    }
};
#endif