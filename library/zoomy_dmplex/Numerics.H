#pragma once
#include <cmath>
#include <algorithm>
#include <vector>

#define PORTABLE_FN
template <typename T>
struct Numerics {

    // --- Numerics Constants ---
    static constexpr int n_dof_q = 4;

    // --- Kernels ---

    PORTABLE_FN static inline void numerical_flux(
        const T* Q_minus,
        const T* Q_plus,
        const T* Qaux_minus,
        const T* Qaux_plus,
        const T* n,
        T* res)
    {
    T t0 = Q_minus[1] > 1e-06;
    T t1 = Q_plus[1] > 1e-06;
    T t2 = std::pow(Qaux_minus[0], 2);
    T t3 = Q_minus[1]*t2;
    T t4 = 1.0*t3;
    T t5 = Q_minus[2]*n[0];
    T t6 = Q_minus[3]*n[1];
    T t7 = t4*t5 + t4*t6;
    T t8 = std::pow(Qaux_plus[0], 2);
    T t9 = Q_plus[1]*t8;
    T t10 = 1.0*t9;
    T t11 = Q_plus[2]*n[0];
    T t12 = Q_plus[3]*n[1];
    T t13 = t10*t11 + t10*t12;
    T t14 = t5*t6;
    T t15 = std::pow(n[0], 2);
    T t16 = std::pow(Q_minus[1], 2);
    T t17 = 0.5*9.81;
    T t18 = t16*t17;
    T t19 = Qaux_minus[0]*t18;
    T t20 = std::pow(n[1], 2);
    T t21 = 0.125*t2;
    T t22 = std::pow(Q_minus[2], 2);
    T t23 = t15*t22;
    T t24 = std::pow(Q_minus[3], 2);
    T t25 = t20*t24;
    T t26 = std::pow(Qaux_minus[0], 4)*t16;
    T t27 = 0.5*t26;
    T t28 = 1.4142135623730951*std::pow(0.25*t14*t2 + 1.0*t14*t26 + t15*t19 + t19*t20 + t21*t23 + t21*t25 + t23*t27 + t25*t27, 1.0/2.0);
    T t29 = 0.5*n[0];
    T t30 = Q_minus[2]*t29;
    T t31 = Qaux_minus[0]*t30;
    T t32 = 0.5*n[1];
    T t33 = Q_minus[3]*t32;
    T t34 = Qaux_minus[0]*t33;
    T t35 = t31 + t34 + t7;
    T t36 = t11*t12;
    T t37 = std::pow(Q_plus[1], 2);
    T t38 = t17*t37;
    T t39 = Qaux_plus[0]*t38;
    T t40 = 0.125*t8;
    T t41 = std::pow(Q_plus[2], 2);
    T t42 = t15*t41;
    T t43 = std::pow(Q_plus[3], 2);
    T t44 = t20*t43;
    T t45 = std::pow(Qaux_plus[0], 4)*t37;
    T t46 = 0.5*t45;
    T t47 = 1.4142135623730951*std::pow(t15*t39 + t20*t39 + 1.0*t36*t45 + 0.25*t36*t8 + t40*t42 + t40*t44 + t42*t46 + t44*t46, 1.0/2.0);
    T t48 = Q_plus[2]*t29;
    T t49 = Qaux_plus[0]*t48;
    T t50 = Q_plus[3]*t32;
    T t51 = Qaux_plus[0]*t50;
    T t52 = t13 + t49 + t51;
    T t53 = 0.5*std::max(std::fabs(((t0) ? (0) : (0))), std::max(std::fabs(((t0) ? (t7) : (0))), std::max(std::fabs(((t1) ? (0) : (0))), std::max(std::fabs(((t1) ? (t13) : (0))), std::max(std::fabs(((t0) ? (t28 + t35) : (0))), std::max(std::fabs(((t1) ? (t47 + t52) : (0))), std::max(std::fabs(((t0) ? (-t28 + t35) : (0))), std::fabs(((t1) ? (-t47 + t52) : (0))))))))));
    res[0] = -t53*(-Q_minus[0] + Q_plus[0]);
    res[1] = Q_minus[1]*t31 + Q_minus[1]*t34 + Q_plus[1]*t49 + Q_plus[1]*t51 - t53*(-Q_minus[1] + Q_plus[1]);
    res[2] = Q_minus[2]*t3*t33 + Q_plus[2]*t50*t9 + t29*(t18 + t22*t3) + t29*(t38 + t41*t9) - t53*(-Q_minus[2] + Q_plus[2]);
    res[3] = Q_minus[3]*t3*t30 + Q_plus[3]*t48*t9 + t32*(t18 + t24*t3) + t32*(t38 + t43*t9) - t53*(-Q_minus[3] + Q_plus[3]);
    }


    PORTABLE_FN static inline void local_max_abs_eigenvalue(
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
    res[0] = std::max(std::fabs(((t0) ? (0) : (0))), std::max(std::fabs(((t0) ? (t6) : (0))), std::max(std::fabs(((t0) ? (t17 + t18) : (0))), std::fabs(((t0) ? (-t17 + t18) : (0))))));
    }


    PORTABLE_FN static inline void update_Q(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2];
    res[3] = Q[3];
    }


    PORTABLE_FN static inline void update_Qaux(
        const T* Q,
        const T* Qaux,
        T* res)
    {
    res[0] = Q[0];
    res[1] = Q[1];
    res[2] = Q[2];
    res[3] = Q[3];
    }

};
