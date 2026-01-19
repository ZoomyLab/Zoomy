#pragma once
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

template <typename T>
struct Model {

    // --- Constants ---
    static constexpr int n_dof_q    = 3;
    static constexpr int n_dof_qaux = 1;
    static constexpr int dimension  = 2;
    static constexpr int n_boundary_tags = 3;

    // --- Helpers ---
    static const std::vector<std::string> get_boundary_tags() { return { "inflow", "outflow", "wall" }; }
    static const std::vector<std::string> get_boundary_tag_ids() { return { "3002", "3003", "3000" }; }


    // --- Kernels ---

    static PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) {
        // Simple Dam Break at x = 0.0
        // Left (x < 0): High water h=2
        // Right (x >= 0): Low water h=1
        
        if (x[0] < 5.0) {
            u[0] = 2.0; // Height
        } else {
            u[0] = 1.0; // Height
        }
        
        // Momentum (hu, hv) = 0
        u[1] = 0.0; 
        if (dim > 1) u[2] = 0.0;
        
        return PETSC_SUCCESS;
    }

    static inline void flux(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    T t0 = 0.5*1.0*9.81*std::pow(Q[0], 2);
    T t1 = (1.0 / std::pow(Q[0], 1));
    T t2 = Q[1]*Q[2]*t1;
    res[0] = Q[1];
    res[1] = Q[2];
    res[2] = std::pow(Q[1], 2)*t1 + t0;
    res[3] = t2;
    res[4] = t2;
    res[5] = std::pow(Q[2], 2)*t1 + t0;
    }


    static inline void dflux(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    }


    static inline void nonconservative_matrix(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
    res[16] = 0;
    res[17] = 0;
    }


    static inline void quasilinear_matrix(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    T t0 = 1.0*1.0*9.81*Q[0];
    T t1 = (1.0 / std::pow(Q[0], 2));
    T t2 = -Q[1]*Q[2]*t1;
    T t3 = (1.0 / std::pow(Q[0], 1));
    T t4 = Q[1]*t3;
    T t5 = Q[2]*t3;
    res[0] = 0;
    res[1] = 0;
    res[2] = 1;
    res[3] = 0;
    res[4] = 0;
    res[5] = 1;
    res[6] = -std::pow(Q[1], 2)*t1 + t0;
    res[7] = t2;
    res[8] = 2*t4;
    res[9] = t5;
    res[10] = 0;
    res[11] = t4;
    res[12] = t2;
    res[13] = -std::pow(Q[2], 2)*t1 + t0;
    res[14] = t5;
    res[15] = 0;
    res[16] = t4;
    res[17] = 2*t5;
    }


    static inline void source(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    }


    static inline void source_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    }


    static inline void source_jacobian_wrt_aux_variables(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    }


    static inline void eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* n,
        T* res)
    {
    T t0 = 1.0*n[0]*Q[1];
    T t1 = 1.0*n[1]*Q[2];
    T t2 = (1.0 / std::pow(Q[0], 2));
    T t3 = 1.0*std::pow(1.0*9.81*std::pow(Q[0], 5), 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0);
    T t4 = Q[0]*t0 + Q[0]*t1;
    res[0] = (t0 + t1)/Q[0];
    res[1] = t2*(t3 + t4);
    res[2] = t2*(-t3 + t4);
    }


    static inline void left_eigenvectors(
        const T* Q,
        const T* Qaux,
        const T* n,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    }


    static inline void right_eigenvectors(
        const T* Q,
        const T* Qaux,
        const T* n,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    }


    static inline void residual(
        const T* Q,
        const T* Qaux,
        const T* X,
        const T time,
        const T dX,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    }


    static inline void interpolate(
        const T* Q,
        const T* Qaux,
        const T* X,
        T* res)
    {
    T t0 = (1.0 / std::pow(Q[0], 1));
    res[0] = 0.0;
    res[1] = Q[0];
    res[2] = Q[1]*t0;
    res[3] = Q[2]*t0;
    res[4] = 0.0;
    res[5] = 9810.0*Q[0]*(1 - X[2]);
    }


    static inline void boundary_conditions(
        const int bc_idx,
        const T* Q,
        const T* Qaux,
        const T* n,
        const T* X,
        const T time,
        const T dX,
        T* res)
    {
    if (bc_idx == 0 || bc_idx == 1 || bc_idx == 2) {
        T t0 = n[1]*Q[2];
        T t1 = n[0]*Q[1] + t0;
        T t2 = 1.0*n[0];
        T t3 = 1.0*n[0]*Q[1] + 1.0*t0;
        T t4 = 1.0*n[1];
        res[0] = Q[0];
        res[1] = 1.0*Q[1] - t1*t2 - t2*t3;
        res[2] = 1.0*Q[2] - t1*t4 - t3*t4;
    }
    }

};
