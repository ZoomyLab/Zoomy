#pragma once
#include <cmath>
#include <algorithm>
#include <vector>

template <typename T>
struct Numerics {

    // --- Numerics Constants ---
    static constexpr int n_dof_q = 3;

    // --- Kernels ---

    static inline void numerical_flux(
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
    T t8 = Q_minus[1]*t7;
    T t9 = 1.0*n[1];
    T t10 = Q_minus[2]*t9;
    T t11 = (1.0 / std::pow(Q_plus[0], 1));
    T t12 = Q_plus[1]*t7;
    T t13 = Q_plus[2]*t9;
    T t14 = std::pow(Q_minus[0], 2);
    T t15 = (1.0 / std::pow(t14, 1));
    T t16 = 1.0*9.81;
    T t17 = 1.0*std::pow(std::pow(n[0], 2) + std::pow(n[1], 2), 1.0/2.0);
    T t18 = t17*std::pow(std::pow(Q_minus[0], 5)*t16, 1.0/2.0);
    T t19 = Q_minus[0]*t10 + Q_minus[0]*t8;
    T t20 = std::pow(Q_plus[0], 2);
    T t21 = (1.0 / std::pow(t20, 1));
    T t22 = t17*std::pow(std::pow(Q_plus[0], 5)*t16, 1.0/2.0);
    T t23 = Q_plus[0]*t12 + Q_plus[0]*t13;
    T t24 = 0.5*std::max(std::fabs(t11*(t12 + t13)), std::max(std::fabs(t15*(t18 + t19)), std::max(std::fabs(t21*(t22 + t23)), std::max(std::fabs(t6*(t10 + t8)), std::max(std::fabs(t15*(t18 - t19)), std::fabs(t21*(t22 - t23)))))));
    T t25 = 0.5*t16;
    T t26 = t14*t25;
    T t27 = t20*t25;
    res[0] = t1 - t24*(-Q_minus[0] + Q_plus[0]) + t3 + t4 + t5;
    res[1] = Q_minus[1]*t3*t6 + Q_plus[1]*t11*t5 + t0*(std::pow(Q_minus[1], 2)*t6 + t26) + t0*(std::pow(Q_plus[1], 2)*t11 + t27) - t24*(-Q_minus[1] + Q_plus[1]);
    res[2] = Q_minus[2]*t1*t6 + Q_plus[2]*t11*t4 + t2*(std::pow(Q_minus[2], 2)*t6 + t26) + t2*(std::pow(Q_plus[2], 2)*t11 + t27) - t24*(-Q_minus[2] + Q_plus[2]);
    }

};
