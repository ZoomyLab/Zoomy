#pragma once
#include <cmath>
#include <array>
#include "Model.H"

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
#include <algorithm>

#ifdef __CUDACC__
#define PORTABLE_FN __host__ __device__
#else
#define PORTABLE_FN
#endif

template <typename T>
struct Numerics {
    static constexpr int n_dof_q = 4;
    PORTABLE_FN static inline SimpleArray<T, 4> numerical_flux(
        const T* Q_minus,
        const T* Q_plus,
        const T* Qaux_minus,
        const T* Qaux_plus,
        const T* p,
        const T* n)
    {
    T t0 = std::max(Q_minus[0], Q_plus[0]);
    T t1 = -t0;
    T t2 = std::max(0.0, Q_minus[0] + Q_minus[1] + t1);
    T t3 = t2/std::max(9.9999999999999995e-7, Q_minus[1]);
    T t4 = std::max(0.0, Q_plus[0] + Q_plus[1] + t1);
    T t5 = t4/std::max(9.9999999999999995e-7, Q_plus[1]);
    T v_arg_0[] = { t0, t2, Q_minus[2]*t3, Q_minus[3]*t3 };
    T v_arg_1[] = { t0, t4, Q_plus[2]*t5, Q_plus[3]*t5 };
    auto v_call_0 = Model<T>::flux(v_arg_0, Qaux_minus, p);
    auto v_call_1 = Model<T>::flux(v_arg_1, Qaux_plus, p);
    auto v_call_2 = Model<T>::eigenvalues(v_arg_0, Qaux_minus, p, n);
    auto v_call_3 = Model<T>::eigenvalues(v_arg_1, Qaux_plus, p, n);
    T t6 = 0.5*n[0];
    T t7 = 0.5*n[1];
    T t8 = -std::max(Q_minus[0], Q_plus[0]);
    T t9 = std::max(0.0, Q_minus[0] + Q_minus[1] + t8);
    T t10 = std::max(0.0, Q_plus[0] + Q_plus[1] + t8);
    T t11 = 0.5*std::max(std::fabs(v_call_2[0]), std::max(std::fabs(v_call_2[1]), std::max(std::fabs(v_call_2[2]), std::max(std::fabs(v_call_2[3]), std::max(std::fabs(v_call_3[0]), std::max(std::fabs(v_call_3[1]), std::max(std::fabs(v_call_3[2]), std::fabs(v_call_3[3]))))))));
    T t12 = t9/std::max(9.9999999999999995e-7, Q_minus[1]);
    T t13 = (1.0 / std::pow(std::max(9.9999999999999995e-7, Q_plus[1]), 1));
    SimpleArray<T, 4> res;
    res[0] = t6*v_call_0[0] + t6*v_call_1[0] + t7*v_call_0[1] + t7*v_call_1[1];
    res[1] = -t11*(t10 - t9) + t6*v_call_0[2] + t6*v_call_1[2] + t7*v_call_0[3] + t7*v_call_1[3];
    res[2] = -t11*(-Q_minus[2]*t12 + Q_plus[2]*t10*t13) + t6*v_call_0[4] + t6*v_call_1[4] + t7*v_call_0[5] + t7*v_call_1[5];
    res[3] = -t11*(-Q_minus[3]*t12 + Q_plus[3]*t10*t13) + t6*v_call_0[6] + t6*v_call_1[6] + t7*v_call_0[7] + t7*v_call_1[7];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    auto v_call_0 = Model<T>::eigenvalues(Q, Qaux, p, n);
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(v_call_0[0]), std::max(std::fabs(v_call_0[1]), std::max(std::fabs(v_call_0[2]), std::fabs(v_call_0[3]))));
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 5> overwrite_q_qaux(
        const T* Q,
        const T* Qaux,
        const T* p)
    {
    SimpleArray<T, 5> res;
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2];
    res[3] = Q[3];
    res[4] = ((Q[1] > p[2]) ? (1.0/Q[1]) : (p[2]));
    return res;
    }

};
