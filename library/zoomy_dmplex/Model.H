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
    static constexpr int n_dof_q    = 4;
    static constexpr int n_dof_qaux = 1;
    static constexpr int dimension  = 2;
    static constexpr int n_boundary_tags = 3;
    static const std::vector<std::string> get_boundary_tags() { return { "inflow", "outflow", "wall" }; }
    PORTABLE_FN static inline SimpleArray<T, 8> flux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Q[1]*Qaux[0];
    T t1 = 0.5*p[0]*std::pow(Q[1], 2);
    T t2 = Q[1]*std::pow(Qaux[0], 2);
    T t3 = Q[2]*Q[3]*t2;
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = Q[2]*t0;
    res[3] = Q[3]*t0;
    res[4] = std::pow(Q[2], 2)*t2 + t1;
    res[5] = t3;
    res[6] = t3;
    res[7] = std::pow(Q[3], 2)*t2 + t1;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 8> dflux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 8> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 32> nonconservative_matrix(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = p[0]*Q[1];
    SimpleArray<T, 32> res;
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
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 32> quasilinear_matrix(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = Q[1]*Qaux[0];
    T t1 = p[0]*Q[1];
    T t2 = 1.0*t1;
    T t3 = std::pow(Qaux[0], 2);
    T t4 = Q[3]*t3;
    T t5 = Q[2]*t4;
    T t6 = Q[1]*Q[2]*t3;
    T t7 = Q[1]*t4;
    SimpleArray<T, 32> res;
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
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> eigenvalues(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T t0 = Q[1] > p[2];
    T t1 = std::pow(Qaux[0], 2);
    T t2 = n[0]*Q[2];
    T t3 = t1*t2;
    T t4 = 1.0*Q[1];
    T t5 = n[1]*Q[3];
    T t6 = t1*t4*t5 + t3*t4;
    T t7 = std::pow(n[0], 2);
    T t8 = std::pow(Q[1], 2);
    T t9 = 0.5*Qaux[0];
    T t10 = p[0]*t8*t9;
    T t11 = std::pow(n[1], 2);
    T t12 = 0.125*t1;
    T t13 = std::pow(Q[2], 2)*t7;
    T t14 = std::pow(Q[3], 2)*t11;
    T t15 = std::pow(Qaux[0], 4)*t8;
    T t16 = 0.5*t15;
    T t17 = 1.4142135623730951*std::pow(t10*t11 + t10*t7 + t12*t13 + t12*t14 + t13*t16 + t14*t16 + 1.0*t15*t2*t5 + 0.25*t3*t5, 1.0/2.0);
    T t18 = t2*t9 + t5*t9 + t6;
    SimpleArray<T, 4> res;
    res[0] = ((t0) ? (0) : (0));
    res[1] = ((t0) ? (t6) : (0));
    res[2] = ((t0) ? (-t17 + t18) : (0));
    res[3] = ((t0) ? (t17 + t18) : (0));
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> left_eigenvectors(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    SimpleArray<T, 16> res;
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
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> right_eigenvectors(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    SimpleArray<T, 16> res;
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
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> source(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Qaux[0], 2);
    T t1 = 1.0*Qaux[0]*std::pow(std::pow(Q[2], 2)*t0 + std::pow(Q[3], 2)*t0, 1.0/2.0)/std::pow(p[1], 2);
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = -Q[2]*t1;
    res[3] = -Q[3]*t1;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> source_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Q[2], 2);
    T t1 = std::pow(Qaux[0], 2);
    T t2 = std::pow(Q[3], 2);
    T t3 = std::pow(t0*t1 + t1*t2, 1.0/2.0);
    T t4 = 1.0/std::pow(p[1], 2);
    T t5 = Qaux[0]*t3*t4;
    T t6 = std::pow(Qaux[0], 3)*t4/t3;
    T t7 = -Q[2]*Q[3]*t6;
    SimpleArray<T, 16> res;
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
    res[10] = -t0*t6 - t5;
    res[11] = t7;
    res[12] = 0;
    res[13] = 0;
    res[14] = t7;
    res[15] = -t2*t6 - t5;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> source_jacobian_wrt_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    T t0 = std::pow(Q[2], 2);
    T t1 = std::pow(Qaux[0], 2);
    T t2 = std::pow(Q[3], 2);
    T t3 = std::pow(t0*t1 + t1*t2, 1.0/2.0);
    T t4 = 1.0/std::pow(p[1], 2);
    T t5 = t3*t4;
    T t6 = Qaux[0]*t4*(Qaux[0]*t0 + Qaux[0]*t2)/t3;
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = -Q[2]*t5 - Q[2]*t6;
    res[3] = -Q[3]*t5 - Q[3]*t6;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 6> project_2d_to_3d(
        const T* X,
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 6> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2]*Qaux[0];
    res[3] = Q[3]*Qaux[0];
    res[4] = 0.0;
    res[5] = 9810.0*Q[1]*(1 - X[2]);
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> project_3d_to_2d(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> residual(
        const T time,
        const T* X,
        const T dX,
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> interpolate(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 4> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2];
    res[3] = Q[3];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> initial_condition(
        const T* X,
        const T* p)
    {
    SimpleArray<T, 4> res;
    res[0] = 1.0;
    res[1] = ((X[0] < 5) ? (2.0) : (1.0));
    res[2] = 0;
    res[3] = 0;
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

    PORTABLE_FN static inline SimpleArray<T, 4> update_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 4> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2];
    res[3] = Q[3];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> update_aux_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 1> res;
    res[0] = ((Q[1] > p[2]) ? (1.0/Q[1]) : (1.0/p[2]));
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 16> update_variables_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 16> res;
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 1;
    res[6] = 0;
    res[7] = 0;
    res[8] = 0;
    res[9] = 0;
    res[10] = 1;
    res[11] = 0;
    res[12] = 0;
    res[13] = 0;
    res[14] = 0;
    res[15] = 1;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> update_aux_variables_jacobian_wrt_variables(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 4> res;
    res[0] = 0;
    res[1] = ((Q[1] > p[2]) ? (-1.0/std::pow(Q[1], 2)) : (0));
    res[2] = 0;
    res[3] = 0;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 4> boundary_conditions(
        const int bc_idx,
        const T* Q,
        const T* Qaux,
        const T* n,
        const T* X,
        const T time,
        const T dX)
    {
    if (bc_idx == 0 || bc_idx == 1 || bc_idx == 2) {
        SimpleArray<T, 4> res;
        res[0] = Q[0];
        res[1] = Q[1];
        res[2] = Q[2];
        res[3] = Q[3];
        return res;
    }
    SimpleArray<T, 4> default_res = {0};
    return default_res;
    }

};
