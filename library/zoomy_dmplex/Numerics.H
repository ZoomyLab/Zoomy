#pragma once
#include <cmath>
#include <algorithm>
#include <vector>

#define PORTABLE_FN
template <typename T>
struct Numerics {

    // --- Numerics Constants ---
    static constexpr int n_dof_q = 3;

    // --- Kernels ---

    PORTABLE_FN static inline void numerical_flux(
        const T* Q_minus,
        const T* Q_plus,
        const T* Qaux_minus,
        const T* Qaux_plus,
        const T* n,
        T* res)
    {
    T t0 = 0.5*n[0];
    T t1 = Q_minus[1]*t0;
    T t2 = 0.5*n[1];
    T t3 = Q_minus[2]*t2;
    T t4 = Q_plus[1]*t0;
    T t5 = Q_plus[2]*t2;
    T t6 = (1.0 / std::pow(Q_minus[0], 1));
    T t7 = 1.0*n[0];
    T t8 = 1.0*n[1];
    T t9 = Q_minus[1]*t7 + Q_minus[2]*t8;
    T t10 = (1.0 / std::pow(Q_plus[0], 1));
    T t11 = Q_plus[1]*t7 + Q_plus[2]*t8;
    T t12 = 1.0*9.81;
    T t13 = 1.0*std::pow(t12, 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0);
    T t14 = std::pow(Q_minus[0], 3.0/2.0)*t13;
    T t15 = std::pow(Q_plus[0], 3.0/2.0)*t13;
    T t16 = 0.5*std::max(t10*std::fabs(t11), std::max(t6*std::fabs(t9), std::max(t10*std::fabs(t11 + t15), std::max(t6*std::fabs(t14 + t9), std::max(t10*std::fabs(t11 - t15), t6*std::fabs(t14 - t9))))));
    T t17 = 0.5*t12;
    T t18 = std::pow(Q_minus[0], 2)*t17;
    T t19 = std::pow(Q_plus[0], 2)*t17;
    res[0] = t1 - t16*(-Q_minus[0] + Q_plus[0]) + t3 + t4 + t5;
    res[1] = Q_minus[1]*t3*t6 + Q_plus[1]*t10*t5 + t0*(std::pow(Q_minus[1], 2)*t6 + t18) + t0*(std::pow(Q_plus[1], 2)*t10 + t19) - t16*(-Q_minus[1] + Q_plus[1]);
    res[2] = Q_minus[2]*t1*t6 + Q_plus[2]*t10*t4 - t16*(-Q_minus[2] + Q_plus[2]) + t2*(std::pow(Q_minus[2], 2)*t6 + t18) + t2*(std::pow(Q_plus[2], 2)*t10 + t19);
    }


    PORTABLE_FN static inline void local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* n,
        T* res)
    {
    T t0 = (1.0 / std::pow(Q[0], 1));
    T t1 = 1.0*std::pow(Q[0], 3.0/2.0)*std::pow(1.0*9.81, 1.0/2.0)*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0);
    T t2 = 1.0*n[0]*Q[1] + 1.0*n[1]*Q[2];
    res[0] = std::max(t0*(t1 + t2), t0*std::fabs(t1 - t2));
    }

};
