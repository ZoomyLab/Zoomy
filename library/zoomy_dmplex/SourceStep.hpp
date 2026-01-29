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
    const T epsilon = 1e-7; 

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
        
        PetscSection sQ, sAux;
        PetscCall(DMGetLocalSection(dmQ, &sQ));
        PetscCall(DMGetLocalSection(dmAux, &sAux));
        
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        const T* params_ptr = parameters.data();
        const int n_dof = Model<T>::n_dof_q;

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt offQ, offAux;
            PetscCall(PetscSectionGetOffset(sQ, c, &offQ));
            PetscCall(PetscSectionGetOffset(sAux, c, &offAux));
            
            if (offQ >= 0) {
                PetscScalar* q_ptr = &x_arr[offQ];
                const PetscScalar* aux_ptr = (offAux >= 0) ? &a_arr[offAux] : nullptr;
                SolveLocalCell(dt, q_ptr, aux_ptr, params_ptr, n_dof);
            }
        }

        PetscCall(VecRestoreArray(X, &x_arr));
        PetscCall(VecRestoreArrayRead(A, &a_arr));
        
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    void SolveLocalCell(T dt, PetscScalar* q, const PetscScalar* aux, const T* params, int n_dof) {
        std::vector<PetscScalar> q_old(n_dof);
        std::vector<PetscScalar> q_curr(n_dof);
        std::vector<PetscScalar> residual(n_dof);
        std::vector<PetscScalar> delta(n_dof);
        std::vector<PetscScalar> J(n_dof * n_dof);

        for(int i=0; i<n_dof; ++i) {
            q_old[i] = q[i];
            q_curr[i] = q[i]; 
        }

        for(int iter=0; iter < max_newton_iter; ++iter) {
            auto S = Model<T>::source(q_curr.data(), aux, params);
            
            T res_norm = 0.0;
            for(int i=0; i<n_dof; ++i) {
                residual[i] = q_curr[i] - q_old[i] - dt * S[i];
                res_norm += residual[i] * residual[i];
            }
            if(std::sqrt(res_norm) < tol) break; 

            // Finite Difference Jacobian
            std::fill(J.begin(), J.end(), 0.0);
            for(int i=0; i<n_dof; ++i) J[i*n_dof + i] = 1.0;

            for(int j=0; j<n_dof; ++j) {
                T save_val = q_curr[j];
                T pert = epsilon * (std::abs(save_val) + 1.0);
                q_curr[j] += pert;
                
                auto S_pert = Model<T>::source(q_curr.data(), aux, params);
                
                for(int i=0; i<n_dof; ++i) {
                    T dS_dq = (S_pert[i] - S[i]) / pert;
                    J[i*n_dof + j] -= dt * dS_dq; 
                }
                q_curr[j] = save_val; 
            }

            for(int i=0; i<n_dof; ++i) residual[i] = -residual[i];
            
            if(SolveLinearSystem(J, residual, delta, n_dof)) {
                for(int i=0; i<n_dof; ++i) q_curr[i] += delta[i];
            } else {
                break; 
            }
        }

        for(int i=0; i<n_dof; ++i) q[i] = q_curr[i];
    }

    bool SolveLinearSystem(std::vector<PetscScalar>& A, std::vector<PetscScalar>& b, std::vector<PetscScalar>& x, int N) {
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