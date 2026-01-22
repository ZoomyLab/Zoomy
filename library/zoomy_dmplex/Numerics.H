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
    auto v_call_0 = Model<T>::flux(Q_minus, Qaux_minus, p);
    auto v_call_1 = Model<T>::flux(Q_plus, Qaux_plus, p);
    auto v_call_2 = Model<T>::eigenvalues(Q_minus, Qaux_minus, p, n);
    auto v_call_3 = Model<T>::eigenvalues(Q_plus, Qaux_plus, p, n);
    T t0 = 0.5*n[0];
    T t1 = 0.5*n[1];
    T t2 = 0.5*std::max(std::fabs(v_call_2[0]), std::max(std::fabs(v_call_2[1]), std::max(std::fabs(v_call_2[2]), std::max(std::fabs(v_call_2[3]), std::max(std::fabs(v_call_3[0]), std::max(std::fabs(v_call_3[1]), std::max(std::fabs(v_call_3[2]), std::fabs(v_call_3[3]))))))));
    SimpleArray<T, 4> res;
    res[0] = t0*v_call_0[0] + t0*v_call_1[0] + t1*v_call_0[1] + t1*v_call_1[1] - t2*(-Q_minus[0] + Q_plus[0]);
    res[1] = t0*v_call_0[2] + t0*v_call_1[2] + t1*v_call_0[3] + t1*v_call_1[3] - t2*(-Q_minus[1] + Q_plus[1]);
    res[2] = t0*v_call_0[4] + t0*v_call_1[4] + t1*v_call_0[5] + t1*v_call_1[5] - t2*(-Q_minus[2] + Q_plus[2]);
    res[3] = t0*v_call_0[6] + t0*v_call_1[6] + t1*v_call_0[7] + t1*v_call_1[7] - t2*(-Q_minus[3] + Q_plus[3]);
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
