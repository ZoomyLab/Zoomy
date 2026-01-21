#pragma once
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <algorithm>

template <typename T>
struct Model {
    // --- Constants ---
    static constexpr int n_dof_q    = 4;
    static constexpr int n_dof_qaux = 1;
    static constexpr int dimension  = 2;
    static constexpr int n_boundary_tags = 3;

    // --- Helpers ---
    static const std::vector<std::string> get_boundary_tags() { return { "inflow", "outflow", "wall" }; }

    // --- Kernels ---

    static inline void flux(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    T t0 = Q[1]*Qaux[0];
    T t1 = 0.5*9.81*std::pow(Q[1], 2);
    T t2 = Q[1]*std::pow(Qaux[0], 2);
    T t3 = Q[2]*Q[3]*t2;
    res[0] = 0;
    res[1] = 0;
    res[2] = Q[2]*t0;
    res[3] = Q[3]*t0;
    res[4] = std::pow(Q[2], 2)*t2 + t1;
    res[5] = t3;
    res[6] = t3;
    res[7] = std::pow(Q[3], 2)*t2 + t1;
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
    res[6] = 0;
    res[7] = 0;
    }


    static inline void nonconservative_matrix(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    T t0 = 9.81*Q[1];
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
    res[16] = t0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = t0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    }


    static inline void quasilinear_matrix(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    T t0 = Q[1]*Qaux[0];
    T t1 = 9.81*Q[1];
    T t2 = 1.0*t1;
    T t3 = std::pow(Qaux[0], 2);
    T t4 = Q[3]*t3;
    T t5 = Q[2]*t4;
    T t6 = Q[1]*Q[2]*t3;
    T t7 = Q[1]*t4;
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
    res[10] = Q[2]*Qaux[0];
    res[11] = Q[3]*Qaux[0];
    res[12] = t0;
    res[13] = 0;
    res[14] = 0;
    res[15] = t0;
    res[16] = t1;
    res[17] = 0;
    res[18] = std::pow(Q[2], 2)*t3 + t2;
    res[19] = t5;
    res[20] = 2*t6;
    res[21] = t7;
    res[22] = 0;
    res[23] = t6;
    res[24] = 0;
    res[25] = t1;
    res[26] = t5;
    res[27] = std::pow(Q[3], 2)*t3 + t2;
    res[28] = t7;
    res[29] = 0;
    res[30] = t6;
    res[31] = 2*t7;
    }


    static inline void source(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
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
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
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
    }


    static inline void eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* n,
        T* res)
    {
    T t0 = Q[1] > 1e-06;
    T t1 = std::pow(Qaux[0], 2);
    T t2 = n[0]*Q[2];
    T t3 = t1*t2;
    T t4 = 1.0*Q[1];
    T t5 = n[1]*Q[3];
    T t6 = t1*t4*t5 + t3*t4;
    T t7 = std::pow(n[0], 2);
    T t8 = std::pow(Q[1], 2);
    T t9 = 0.5*Qaux[0];
    T t10 = 9.81*t8*t9;
    T t11 = std::pow(n[1], 2);
    T t12 = 0.125*t1;
    T t13 = std::pow(Q[2], 2)*t7;
    T t14 = std::pow(Q[3], 2)*t11;
    T t15 = std::pow(Qaux[0], 4)*t8;
    T t16 = 0.5*t15;
    T t17 = 1.4142135623730951*std::pow(t10*t11 + t10*t7 + t12*t13 + t12*t14 + t13*t16 + t14*t16 + 1.0*t15*t2*t5 + 0.25*t3*t5, 1.0/2.0);
    T t18 = t2*t9 + t5*t9 + t6;
    res[0] = ((t0) ? (0) : (0));
    res[1] = ((t0) ? (t6) : (0));
    res[2] = ((t0) ? (-t17 + t18) : (0));
    res[3] = ((t0) ? (t17 + t18) : (0));
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
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
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
    res[9] = 0;
    res[10] = 0;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 0;
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
    res[3] = 0;
    }


    static inline void interpolate(
        const T* Q,
        const T* Qaux,
        const T* X,
        T* res)
    {
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2]*Qaux[0];
    res[3] = Q[3]*Qaux[0];
    res[4] = 0.0;
    res[5] = 9810.0*Q[1]*(1 - X[2]);
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
        res[3] = Q[3];
    }
    }

};
