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
    T v_arg_1[] = { p[0], p[1], p[2] };
    T v_arg_2[] = { t0, t4, Q_plus[2]*t5, Q_plus[3]*t5 };
    T v_arg_3[] = { n[0], n[1] };
    auto v_call_0 = Model<T>::flux(v_arg_0, Qaux_minus, v_arg_1);
    auto v_call_1 = Model<T>::flux(v_arg_2, Qaux_plus, v_arg_1);
    auto v_call_2 = Model<T>::eigenvalues(v_arg_0, Qaux_minus, v_arg_1, v_arg_3);
    auto v_call_3 = Model<T>::eigenvalues(v_arg_2, Qaux_plus, v_arg_1, v_arg_3);
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

    PORTABLE_FN static inline SimpleArray<T, 8> nonconservative_fluctuations(
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
    T t3 = 0.5*t2;
    T t4 = std::max(0.0, Q_plus[0] + Q_plus[1] + t1);
    T t5 = 0.5*t4;
    T t6 = (1.0 / std::pow(std::max(9.9999999999999995e-7, Q_minus[1]), 1));
    T t7 = t3*t6;
    T t8 = (1.0 / std::pow(std::max(9.9999999999999995e-7, Q_plus[1]), 1));
    T t9 = t5*t8;
    T t10 = 0.1127016653792583*t2;
    T t11 = 0.8872983346207417*t4;
    T t12 = t10*t6;
    T t13 = t11*t8;
    T t14 = 0.8872983346207417*t2;
    T t15 = 0.1127016653792583*t4;
    T t16 = t14*t6;
    T t17 = t15*t8;
    T t18 = t2*t6;
    T t19 = t4*t8;
    T v_arg_0[] = { t0, t3 + t5, Q_minus[2]*t7 + Q_plus[2]*t9, Q_minus[3]*t7 + Q_plus[3]*t9 };
    T v_arg_1[] = { 0.5*Qaux_minus[0] + 0.5*Qaux_plus[0], 0.5*Qaux_minus[1] + 0.5*Qaux_plus[1], 0.5*Qaux_minus[2] + 0.5*Qaux_plus[2], 0.5*Qaux_minus[3] + 0.5*Qaux_plus[3] };
    T v_arg_2[] = { p[0], p[1], p[2] };
    T v_arg_3[] = { t0, t10 + t11, Q_minus[2]*t12 + Q_plus[2]*t13, Q_minus[3]*t12 + Q_plus[3]*t13 };
    T v_arg_4[] = { 0.1127016653792583*Qaux_minus[0] + 0.8872983346207417*Qaux_plus[0], 0.1127016653792583*Qaux_minus[1] + 0.8872983346207417*Qaux_plus[1], 0.1127016653792583*Qaux_minus[2] + 0.8872983346207417*Qaux_plus[2], 0.1127016653792583*Qaux_minus[3] + 0.8872983346207417*Qaux_plus[3] };
    T v_arg_5[] = { t0, t14 + t15, Q_minus[2]*t16 + Q_plus[2]*t17, Q_minus[3]*t16 + Q_plus[3]*t17 };
    T v_arg_6[] = { 0.8872983346207417*Qaux_minus[0] + 0.1127016653792583*Qaux_plus[0], 0.8872983346207417*Qaux_minus[1] + 0.1127016653792583*Qaux_plus[1], 0.8872983346207417*Qaux_minus[2] + 0.1127016653792583*Qaux_plus[2], 0.8872983346207417*Qaux_minus[3] + 0.1127016653792583*Qaux_plus[3] };
    T v_arg_7[] = { t0, t2, Q_minus[2]*t18, Q_minus[3]*t18 };
    T v_arg_8[] = { n[0], n[1] };
    T v_arg_9[] = { t0, t4, Q_plus[2]*t19, Q_plus[3]*t19 };
    auto v_call_0 = Model<T>::nonconservative_matrix(v_arg_0, v_arg_1, v_arg_2);
    auto v_call_1 = Model<T>::nonconservative_matrix(v_arg_3, v_arg_4, v_arg_2);
    auto v_call_2 = Model<T>::nonconservative_matrix(v_arg_5, v_arg_6, v_arg_2);
    auto v_call_3 = Model<T>::eigenvalues(v_arg_7, Qaux_minus, v_arg_2, v_arg_8);
    auto v_call_4 = Model<T>::eigenvalues(v_arg_9, Qaux_plus, v_arg_2, v_arg_8);
    T t20 = 0.44444444444444442*n[0];
    T t21 = 0.27777777777777785*n[0];
    T t22 = 0.44444444444444442*n[1];
    T t23 = 0.27777777777777785*n[1];
    T t24 = -std::max(Q_minus[0], Q_plus[0]);
    T t25 = std::max(0.0, Q_minus[0] + Q_minus[1] + t24);
    T t26 = std::max(0.0, Q_plus[0] + Q_plus[1] + t24);
    T t27 = -0.5*t25 + 0.5*t26;
    T t28 = t25/std::max(9.9999999999999995e-7, Q_minus[1]);
    T t29 = (1.0 / std::pow(std::max(9.9999999999999995e-7, Q_plus[1]), 1));
    T t30 = -0.5*Q_minus[2]*t28 + 0.5*Q_plus[2]*t26*t29;
    T t31 = -0.5*Q_minus[3]*t28 + 0.5*Q_plus[3]*t26*t29;
    T t32 = t27*(t20*v_call_0[2] + t21*v_call_1[2] + t21*v_call_2[2] + t22*v_call_0[3] + t23*v_call_1[3] + t23*v_call_2[3]) + t30*(t20*v_call_0[4] + t21*v_call_1[4] + t21*v_call_2[4] + t22*v_call_0[5] + t23*v_call_1[5] + t23*v_call_2[5]) + t31*(t20*v_call_0[6] + t21*v_call_1[6] + t21*v_call_2[6] + t22*v_call_0[7] + t23*v_call_1[7] + t23*v_call_2[7]);
    T t33 = std::max(std::fabs(v_call_3[0]), std::max(std::fabs(v_call_3[1]), std::max(std::fabs(v_call_3[2]), std::max(std::fabs(v_call_3[3]), std::max(std::fabs(v_call_4[0]), std::max(std::fabs(v_call_4[1]), std::max(std::fabs(v_call_4[2]), std::fabs(v_call_4[3]))))))));
    T t34 = t27*t33;
    T t35 = t27*(t20*v_call_0[10] + t21*v_call_1[10] + t21*v_call_2[10] + t22*v_call_0[11] + t23*v_call_1[11] + t23*v_call_2[11]) + t30*(t20*v_call_0[12] + t21*v_call_1[12] + t21*v_call_2[12] + t22*v_call_0[13] + t23*v_call_1[13] + t23*v_call_2[13]) + t31*(t20*v_call_0[14] + t21*v_call_1[14] + t21*v_call_2[14] + t22*v_call_0[15] + t23*v_call_1[15] + t23*v_call_2[15]);
    T t36 = t30*t33;
    T t37 = t27*(t20*v_call_0[18] + t21*v_call_1[18] + t21*v_call_2[18] + t22*v_call_0[19] + t23*v_call_1[19] + t23*v_call_2[19]) + t30*(t20*v_call_0[20] + t21*v_call_1[20] + t21*v_call_2[20] + t22*v_call_0[21] + t23*v_call_1[21] + t23*v_call_2[21]) + t31*(t20*v_call_0[22] + t21*v_call_1[22] + t21*v_call_2[22] + t22*v_call_0[23] + t23*v_call_1[23] + t23*v_call_2[23]);
    T t38 = t31*t33;
    T t39 = t27*(t20*v_call_0[26] + t21*v_call_1[26] + t21*v_call_2[26] + t22*v_call_0[27] + t23*v_call_1[27] + t23*v_call_2[27]) + t30*(t20*v_call_0[28] + t21*v_call_1[28] + t21*v_call_2[28] + t22*v_call_0[29] + t23*v_call_1[29] + t23*v_call_2[29]) + t31*(t20*v_call_0[30] + t21*v_call_1[30] + t21*v_call_2[30] + t22*v_call_0[31] + t23*v_call_1[31] + t23*v_call_2[31]);
    SimpleArray<T, 8> res;
    res[0] = t32;
    res[1] = t34 + t35;
    res[2] = t36 + t37;
    res[3] = t38 + t39;
    res[4] = t32;
    res[5] = -t34 + t35;
    res[6] = -t36 + t37;
    res[7] = -t38 + t39;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T v_arg_0[] = { Q[0], Q[1], Q[2], Q[3] };
    T v_arg_1[] = { Qaux[0] };
    T v_arg_2[] = { p[0], p[1], p[2] };
    T v_arg_3[] = { n[0], n[1] };
    auto v_call_0 = Model<T>::eigenvalues(v_arg_0, v_arg_1, v_arg_2, v_arg_3);
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(v_call_0[0]), std::max(std::fabs(v_call_0[1]), std::max(std::fabs(v_call_0[2]), std::fabs(v_call_0[3]))));
    return res;
    }

};
