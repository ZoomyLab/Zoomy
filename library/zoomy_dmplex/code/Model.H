#pragma once
#include <cmath>
#include <array>

#ifndef ZOOMY_SIMPLE_ARRAY
#define ZOOMY_SIMPLE_ARRAY
template <typename T, int N>
struct SimpleArray {
    T data[N];
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
    T* begin() { return data; }
    const T* begin() const { return data; }
    T* end() { return data + N; }
    const T* end() const { return data + N; }
};
#endif

#include <vector>
#include <string>
#include <algorithm>

#ifdef __CUDACC__
#define PORTABLE_FN __host__ __device__
#else
#define PORTABLE_FN
#endif

template <typename T>
struct Model {
    static constexpr int n_dof_q    = 6;
    static constexpr int n_dof_qaux = 1;
    static constexpr int dimension  = 2;
    static constexpr int n_boundary_tags = 1;
    static const std::vector<std::string> get_boundary_tags() { return { "default" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "chezy_C", "eps", "ex", "ey", "ez", "rho", "nu", "lamda", "c_slipmod", "l_bl", "l_turb", "kappa" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 50.0, 1e-06, 0.0, 0.0, 1.0, 1000.0, 1e-06, 0.1, 0.0, 0.1, 0.1, 0.41 }; }
    PORTABLE_FN static inline SimpleArray<T, 12> flux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Q[1]*Qaux[0];
    T t1 = (1.0/2.0)*p[5]*p[0]*std::pow(Q[1], 2);
    T t2 = Q[1]*std::pow(Qaux[0], 2);
    T t3 = (1.0/3.0)*t2;
    T t4 = Q[2]*t2;
    T t5 = Q[3]*Q[5]*t3 + Q[4]*t4;
    T t6 = Q[4]*t2;
    T t7 = Q[3]*t6 + Q[5]*t4;
    SimpleArray<T, 12> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = Q[2]*t0;
    res[3] = Q[4]*t0;
    res[4] = std::pow(Q[2], 2)*t2 + std::pow(Q[3], 2)*t3 + t1;
    res[5] = t5;
    res[6] = 2*Q[3]*t4;
    res[7] = t7;
    res[8] = t5;
    res[9] = std::pow(Q[4], 2)*t2 + std::pow(Q[5], 2)*t3 + t1;
    res[10] = t7;
    res[11] = 2*Q[5]*t6;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 12> dflux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 12> res;
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
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 72> nonconservative_matrix(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = p[5]*p[0]*Q[1];
    T t1 = -Q[2]*Qaux[0];
    T t2 = -Q[4]*Qaux[0];
    SimpleArray<T, 72> res;
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
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = t0;
    res[25] = 0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = 0;
    res[36] = 0;
    res[37] = 0;
    res[38] = 0;
    res[39] = 0;
    res[40] = 0;
    res[41] = 0;
    res[42] = t1;
    res[43] = 0;
    res[44] = 0;
    res[45] = 0;
    res[46] = 0;
    res[47] = t1;
    res[48] = 0;
    res[49] = t0;
    res[50] = 0;
    res[51] = 0;
    res[52] = 0;
    res[53] = 0;
    res[54] = 0;
    res[55] = 0;
    res[56] = 0;
    res[57] = 0;
    res[58] = 0;
    res[59] = 0;
    res[60] = 0;
    res[61] = 0;
    res[62] = 0;
    res[63] = 0;
    res[64] = 0;
    res[65] = 0;
    res[66] = t2;
    res[67] = 0;
    res[68] = 0;
    res[69] = 0;
    res[70] = 0;
    res[71] = t2;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 72> quasilinear_matrix(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Q[2]*Qaux[0];
    T t1 = Q[4]*Qaux[0];
    T t2 = Q[1]*Qaux[0];
    T t3 = p[5]*p[0]*Q[1];
    T t4 = std::pow(Qaux[0], 2);
    T t5 = (1.0/3.0)*t4;
    T t6 = Q[4]*t4;
    T t7 = Q[5]*t5;
    T t8 = Q[2]*t6 + Q[3]*t7;
    T t9 = Q[2]*t4;
    T t10 = Q[1]*t9;
    T t11 = 2*t10;
    T t12 = Q[1]*t6;
    T t13 = Q[1]*t4;
    T t14 = Q[3]*t13;
    T t15 = Q[1]*t7;
    T t16 = Q[1]*Q[3]*t5;
    T t17 = Q[3]*t6 + Q[5]*t9;
    T t18 = Q[5]*t13;
    T t19 = -t0;
    T t20 = 2*t12;
    T t21 = -t1;
    SimpleArray<T, 72> res;
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
    res[14] = t0;
    res[15] = t1;
    res[16] = t2;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = t2;
    res[22] = 0;
    res[23] = 0;
    res[24] = t3;
    res[25] = 0;
    res[26] = std::pow(Q[2], 2)*t4 + std::pow(Q[3], 2)*t5 + t3;
    res[27] = t8;
    res[28] = t11;
    res[29] = t12;
    res[30] = (2.0/3.0)*t14;
    res[31] = t15;
    res[32] = 0;
    res[33] = t10;
    res[34] = 0;
    res[35] = t16;
    res[36] = 0;
    res[37] = 0;
    res[38] = 2*Q[3]*t9;
    res[39] = t17;
    res[40] = 2*t14;
    res[41] = t18;
    res[42] = t11 + t19;
    res[43] = t12;
    res[44] = 0;
    res[45] = t14;
    res[46] = 0;
    res[47] = t10 + t19;
    res[48] = 0;
    res[49] = t3;
    res[50] = t8;
    res[51] = std::pow(Q[4], 2)*t4 + std::pow(Q[5], 2)*t5 + t3;
    res[52] = t12;
    res[53] = 0;
    res[54] = t15;
    res[55] = 0;
    res[56] = t10;
    res[57] = t20;
    res[58] = t16;
    res[59] = (2.0/3.0)*t18;
    res[60] = 0;
    res[61] = 0;
    res[62] = t17;
    res[63] = 2*Q[5]*t6;
    res[64] = t18;
    res[65] = 0;
    res[66] = t12 + t21;
    res[67] = 0;
    res[68] = t14;
    res[69] = 2*t18;
    res[70] = t10;
    res[71] = t20 + t21;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = Q[1] > p[2];
    T t1 = (1.0 / std::pow(Q[1], 1));
    T t2 = n[0]*Q[2] + n[1]*Q[4];
    T t3 = n[0]*Q[3];
    T t4 = n[1]*Q[5];
    T t5 = std::pow(n[0], 2);
    T t6 = std::pow(n[1], 2);
    T t7 = p[5]*p[0]*std::pow(Q[1], 3);
    T t8 = std::pow(std::pow(Q[3], 2)*t5 + std::pow(Q[5], 2)*t6 + 2*t3*t4 + t5*t7 + t6*t7, 1.0/2.0);
    T t9 = (1.0/3.0)*std::pow(3, 1.0/2.0);
    T t10 = t3*t9;
    T t11 = t4*t9;
    SimpleArray<T, 6> res;
    res[0] = ((t0) ? (0) : (0));
    res[1] = ((t0) ? (t1*t2) : (0));
    res[2] = ((t0) ? (t1*(t2 - t8)) : (0));
    res[3] = ((t0) ? (t1*(t2 + t8)) : (0));
    res[4] = ((t0) ? (t1*(-t10 - t11 + t2)) : (0));
    res[5] = ((t0) ? (t1*(t10 + t11 + t2)) : (0));
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 36> left_eigenvectors(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    SimpleArray<T, 36> res;
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
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 36> right_eigenvectors(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    SimpleArray<T, 36> res;
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
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = 0;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = 0;
    res[28] = 0;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> source(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Qaux[0]/(p[8]*p[6]);
    T t1 = 1.0*t0;
    T t2 = 3.0*t0;
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = -Q[2]*t1 - Q[3]*t1;
    res[3] = -Q[2]*t2 - Q[3]*t2;
    res[4] = -Q[4]*t1 - Q[5]*t1;
    res[5] = -Q[4]*t2 - Q[5]*t2;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 36> source_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Qaux[0]/(p[8]*p[6]);
    T t1 = -1.0*t0;
    T t2 = -3.0*t0;
    SimpleArray<T, 36> res;
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
    res[14] = t1;
    res[15] = t2;
    res[16] = 0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = t1;
    res[21] = t2;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = 0;
    res[28] = t1;
    res[29] = t2;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = 0;
    res[34] = t1;
    res[35] = t2;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> source_jacobian_wrt_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = 1/(p[8]*p[6]);
    T t1 = 1.0*t0;
    T t2 = 3.0*t0;
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = -Q[2]*t1 - Q[3]*t1;
    res[3] = -Q[2]*t2 - Q[3]*t2;
    res[4] = -Q[4]*t1 - Q[5]*t1;
    res[5] = -Q[4]*t2 - Q[5]*t2;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> project_2d_to_3d(
        const T* X,
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Qaux[0]*(1 - 2*X[2]);
    SimpleArray<T, 6> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2]*Qaux[0] + Q[3]*t0;
    res[3] = Q[4]*Qaux[0] + Q[5]*t0;
    res[4] = 0;
    res[5] = 9810.0*Q[1]*(1 - X[2]);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> project_3d_to_2d(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> residual(
        const T time,
        const T* X,
        const T dX,
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> interpolate(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 6> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2];
    res[3] = Q[3];
    res[4] = Q[4];
    res[5] = Q[5];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> initial_condition(
        const T* X,
        const T* p)
    {
    SimpleArray<T, 6> res;
    res[0] = 1.0;
    res[1] = ((X[0] < 5) ? (2.0) : (0.0));
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> initial_aux_condition(
        const T* X,
        const T* p)
    {
    SimpleArray<T, 1> res;
    res[0] = ((X[0] < 5) ? (0.5) : (1.0/p[2]));
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> update_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Q[1]/(100.0*p[2] + Q[1]);
    SimpleArray<T, 6> res;
    res[0] = Q[0];
    res[1] = ((false) ? (0) : (Q[1]));
    res[2] = Q[2]*t0;
    res[3] = Q[3]*t0;
    res[4] = Q[4]*t0;
    res[5] = Q[5]*t0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> update_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 1> res;
    res[0] = Q[1]/(p[2] + std::pow(Q[1], 2));
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 36> update_variables_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = 100.0*p[2] + Q[1];
    T t1 = (1.0 / std::pow(t0, 1));
    T t2 = Q[1]/std::pow(t0, 2);
    T t3 = Q[1]*t1;
    SimpleArray<T, 36> res;
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = ((false) ? (0) : (1));
    res[8] = Q[2]*t1 - Q[2]*t2;
    res[9] = Q[3]*t1 - Q[3]*t2;
    res[10] = Q[4]*t1 - Q[4]*t2;
    res[11] = Q[5]*t1 - Q[5]*t2;
    res[12] = 0;
    res[13] = 0;
    res[14] = t3;
    res[15] = 0;
    res[16] = 0;
    res[17] = 0;
    res[18] = 0;
    res[19] = 0;
    res[20] = 0;
    res[21] = t3;
    res[22] = 0;
    res[23] = 0;
    res[24] = 0;
    res[25] = 0;
    res[26] = 0;
    res[27] = 0;
    res[28] = t3;
    res[29] = 0;
    res[30] = 0;
    res[31] = 0;
    res[32] = 0;
    res[33] = 0;
    res[34] = 0;
    res[35] = t3;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> update_aux_variables_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Q[1], 2);
    T t1 = p[2] + t0;
    SimpleArray<T, 6> res;
    res[0] = 0;
    res[1] = -2*t0/std::pow(t1, 2) + (1.0 / std::pow(t1, 1));
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> boundary_conditions(
        const int bc_idx,
        const T* Q,
        const T* Qaux,
        const T* n,
        const T* X,
        const T time,
        const T dX)
    {
    if (bc_idx == 0) {
        SimpleArray<T, 6> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = Q[2];
        res[3] = Q[3];
        res[4] = Q[4];
        res[5] = Q[5];
        return res;
    }
    SimpleArray<T, 6> default_res = {0};
    return default_res;
    }

};
