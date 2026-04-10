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

#ifdef __CUDACC__
#define PORTABLE_FN __host__ __device__
#else
#define PORTABLE_FN
#endif

/**
 * Rusanov (Local Lax-Friedrichs) flux for 2D scalar advection.
 *
 * F_hat = 0.5 * (F(uL) + F(uR)) . n - 0.5 * smax * (uR - uL)
 *
 * where F(u) . n = (ax * nx + ay * ny) * u
 * and smax = |ax * nx + ay * ny|
 */
template <typename T>
struct Numerics {
    static constexpr int n_dof_q = 1;

    PORTABLE_FN static inline SimpleArray<T, 1> numerical_flux(
        const T* Q_minus, const T* Q_plus,
        const T* Qaux_minus, const T* Qaux_plus,
        const T* p, const T* n)
    {
        T a_dot_n = p[0] * n[0] + p[1] * n[1];
        T smax = std::abs(a_dot_n);

        SimpleArray<T, 1> res;
        res[0] = 0.5 * (a_dot_n * Q_minus[0] + a_dot_n * Q_plus[0])
               - 0.5 * smax * (Q_plus[0] - Q_minus[0]);
        return res;
    }

    // No non-conservative terms for scalar advection
    PORTABLE_FN static inline SimpleArray<T, 2> numerical_fluctuations(
        const T* Q_minus, const T* Q_plus,
        const T* Qaux_minus, const T* Qaux_plus,
        const T* p, const T* n)
    {
        SimpleArray<T, 2> res;
        res[0] = 0;
        res[1] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q, const T* Qaux, const T* p, const T* n)
    {
        SimpleArray<T, 1> res;
        res[0] = std::abs(p[0] * n[0] + p[1] * n[1]);
        return res;
    }
};
