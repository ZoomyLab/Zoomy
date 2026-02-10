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
    static constexpr int n_dof_q = 6;
    PORTABLE_FN static inline SimpleArray<T, 6> numerical_flux(
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
    T v_arg_0[] = { t0, t2, Q_minus[2]*t3, Q_minus[3]*t3, Q_minus[4]*t3, Q_minus[5]*t3 };
    T v_arg_1[] = { (1.0 / std::pow(p[2] + t2, 1)) };
    T v_arg_2[] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12] };
    T v_arg_3[] = { t0, t4, Q_plus[2]*t5, Q_plus[3]*t5, Q_plus[4]*t5, Q_plus[5]*t5 };
    T v_arg_4[] = { (1.0 / std::pow(p[2] + t4, 1)) };
    T v_arg_5[] = { n[0], n[1] };
    auto v_call_0 = Model<T>::flux(v_arg_0, v_arg_1, v_arg_2);
    auto v_call_1 = Model<T>::flux(v_arg_3, v_arg_4, v_arg_2);
    auto v_call_2 = Model<T>::eigenvalues(v_arg_0, v_arg_1, v_arg_2, v_arg_5);
    auto v_call_3 = Model<T>::eigenvalues(v_arg_3, v_arg_4, v_arg_2, v_arg_5);
    T t6 = 0.5*n[0];
    T t7 = 0.5*n[1];
    T t8 = -std::max(Q_minus[0], Q_plus[0]);
    T t9 = std::max(0.0, Q_minus[0] + Q_minus[1] + t8);
    T t10 = std::max(0.0, Q_plus[0] + Q_plus[1] + t8);
    T t11 = 0.5*std::max(std::fabs(v_call_2[0]), std::max(std::fabs(v_call_2[1]), std::max(std::fabs(v_call_2[2]), std::max(std::fabs(v_call_2[3]), std::max(std::fabs(v_call_2[4]), std::max(std::fabs(v_call_2[5]), std::max(std::fabs(v_call_3[0]), std::max(std::fabs(v_call_3[1]), std::max(std::fabs(v_call_3[2]), std::max(std::fabs(v_call_3[3]), std::max(std::fabs(v_call_3[4]), std::fabs(v_call_3[5]))))))))))));
    SimpleArray<T, 6> res;
    res[0] = t6*v_call_0[0] + t6*v_call_1[0] + t7*v_call_0[1] + t7*v_call_1[1];
    res[1] = -t11*(t10 - t9) + t6*v_call_0[2] + t6*v_call_1[2] + t7*v_call_0[3] + t7*v_call_1[3];
    res[2] = -t11*(-Q_minus[2]*t9/std::max(9.9999999999999995e-7, Q_minus[1]) + Q_plus[2]*t10/std::max(9.9999999999999995e-7, Q_plus[1])) + t6*v_call_0[4] + t6*v_call_1[4] + t7*v_call_0[5] + t7*v_call_1[5];
    res[3] = t6*v_call_0[6] + t6*v_call_1[6] + t7*v_call_0[7] + t7*v_call_1[7];
    res[4] = t6*v_call_0[8] + t6*v_call_1[8] + t7*v_call_0[9] + t7*v_call_1[9];
    res[5] = t6*v_call_0[10] + t6*v_call_1[10] + t7*v_call_0[11] + t7*v_call_1[11];
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 12> numerical_fluctuations(
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
    T t10 = (1.0 / std::pow(p[2] + t2, 1));
    T t11 = (1.0 / std::pow(p[2] + t4, 1));
    T t12 = 0.1127016653792583*t2;
    T t13 = 0.8872983346207417*t4;
    T t14 = t12*t6;
    T t15 = t13*t8;
    T t16 = 0.8872983346207417*t2;
    T t17 = 0.1127016653792583*t4;
    T t18 = t16*t6;
    T t19 = t17*t8;
    T t20 = t2*t6;
    T t21 = t4*t8;
    T v_arg_0[] = { t0, t3 + t5, Q_minus[2]*t7 + Q_plus[2]*t9, Q_minus[3]*t7 + Q_plus[3]*t9, Q_minus[4]*t7 + Q_plus[4]*t9, Q_minus[5]*t7 + Q_plus[5]*t9 };
    T v_arg_1[] = { 0.5*t10 + 0.5*t11 };
    T v_arg_2[] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12] };
    T v_arg_3[] = { t0, t12 + t13, Q_minus[2]*t14 + Q_plus[2]*t15, Q_minus[3]*t14 + Q_plus[3]*t15, Q_minus[4]*t14 + Q_plus[4]*t15, Q_minus[5]*t14 + Q_plus[5]*t15 };
    T v_arg_4[] = { 0.1127016653792583*t10 + 0.8872983346207417*t11 };
    T v_arg_5[] = { t0, t16 + t17, Q_minus[2]*t18 + Q_plus[2]*t19, Q_minus[3]*t18 + Q_plus[3]*t19, Q_minus[4]*t18 + Q_plus[4]*t19, Q_minus[5]*t18 + Q_plus[5]*t19 };
    T v_arg_6[] = { 0.8872983346207417*t10 + 0.1127016653792583*t11 };
    T v_arg_7[] = { t0, t2, Q_minus[2]*t20, Q_minus[3]*t20, Q_minus[4]*t20, Q_minus[5]*t20 };
    T v_arg_8[] = { t10 };
    T v_arg_9[] = { n[0], n[1] };
    T v_arg_10[] = { t0, t4, Q_plus[2]*t21, Q_plus[3]*t21, Q_plus[4]*t21, Q_plus[5]*t21 };
    T v_arg_11[] = { t11 };
    auto v_call_0 = Model<T>::nonconservative_matrix(v_arg_0, v_arg_1, v_arg_2);
    auto v_call_1 = Model<T>::nonconservative_matrix(v_arg_3, v_arg_4, v_arg_2);
    auto v_call_2 = Model<T>::nonconservative_matrix(v_arg_5, v_arg_6, v_arg_2);
    auto v_call_3 = Model<T>::eigenvalues(v_arg_7, v_arg_8, v_arg_2, v_arg_9);
    auto v_call_4 = Model<T>::eigenvalues(v_arg_10, v_arg_11, v_arg_2, v_arg_9);
    T t22 = 0.44444444444444442*n[0];
    T t23 = 0.27777777777777785*n[0];
    T t24 = 0.44444444444444442*n[1];
    T t25 = 0.27777777777777785*n[1];
    T t26 = -std::max(Q_minus[0], Q_plus[0]);
    T t27 = std::max(0.0, Q_minus[0] + Q_minus[1] + t26);
    T t28 = std::max(0.0, Q_plus[0] + Q_plus[1] + t26);
    T t29 = -0.5*t27 + 0.5*t28;
    T t30 = t27/std::max(9.9999999999999995e-7, Q_minus[1]);
    T t31 = (1.0 / std::pow(std::max(9.9999999999999995e-7, Q_plus[1]), 1));
    T t32 = -0.5*Q_minus[2]*t30 + 0.5*Q_plus[2]*t28*t31;
    T t33 = -0.5*Q_minus[3]*t30 + 0.5*Q_plus[3]*t28*t31;
    T t34 = -0.5*Q_minus[4]*t30 + 0.5*Q_plus[4]*t28*t31;
    T t35 = -0.5*Q_minus[5]*t30 + 0.5*Q_plus[5]*t28*t31;
    T t36 = t29*(t22*v_call_0[2] + t23*v_call_1[2] + t23*v_call_2[2] + t24*v_call_0[3] + t25*v_call_1[3] + t25*v_call_2[3]) + t32*(t22*v_call_0[4] + t23*v_call_1[4] + t23*v_call_2[4] + t24*v_call_0[5] + t25*v_call_1[5] + t25*v_call_2[5]) + t33*(t22*v_call_0[6] + t23*v_call_1[6] + t23*v_call_2[6] + t24*v_call_0[7] + t25*v_call_1[7] + t25*v_call_2[7]) + t34*(t22*v_call_0[8] + t23*v_call_1[8] + t23*v_call_2[8] + t24*v_call_0[9] + t25*v_call_1[9] + t25*v_call_2[9]) + t35*(t22*v_call_0[10] + t23*v_call_1[10] + t23*v_call_2[10] + t24*v_call_0[11] + t25*v_call_1[11] + t25*v_call_2[11]);
    T t37 = t29*(t22*v_call_0[14] + t23*v_call_1[14] + t23*v_call_2[14] + t24*v_call_0[15] + t25*v_call_1[15] + t25*v_call_2[15]) + t32*(t22*v_call_0[16] + t23*v_call_1[16] + t23*v_call_2[16] + t24*v_call_0[17] + t25*v_call_1[17] + t25*v_call_2[17]) + t33*(t22*v_call_0[18] + t23*v_call_1[18] + t23*v_call_2[18] + t24*v_call_0[19] + t25*v_call_1[19] + t25*v_call_2[19]) + t34*(t22*v_call_0[20] + t23*v_call_1[20] + t23*v_call_2[20] + t24*v_call_0[21] + t25*v_call_1[21] + t25*v_call_2[21]) + t35*(t22*v_call_0[22] + t23*v_call_1[22] + t23*v_call_2[22] + t24*v_call_0[23] + t25*v_call_1[23] + t25*v_call_2[23]);
    T t38 = t29*(t22*v_call_0[26] + t23*v_call_1[26] + t23*v_call_2[26] + t24*v_call_0[27] + t25*v_call_1[27] + t25*v_call_2[27]) + t32*(t22*v_call_0[28] + t23*v_call_1[28] + t23*v_call_2[28] + t24*v_call_0[29] + t25*v_call_1[29] + t25*v_call_2[29]) + t33*(t22*v_call_0[30] + t23*v_call_1[30] + t23*v_call_2[30] + t24*v_call_0[31] + t25*v_call_1[31] + t25*v_call_2[31]) + t34*(t22*v_call_0[32] + t23*v_call_1[32] + t23*v_call_2[32] + t24*v_call_0[33] + t25*v_call_1[33] + t25*v_call_2[33]) + t35*(t22*v_call_0[34] + t23*v_call_1[34] + t23*v_call_2[34] + t24*v_call_0[35] + t25*v_call_1[35] + t25*v_call_2[35]);
    T t39 = std::max(std::fabs(v_call_3[0]), std::max(std::fabs(v_call_3[1]), std::max(std::fabs(v_call_3[2]), std::max(std::fabs(v_call_3[3]), std::max(std::fabs(v_call_3[4]), std::max(std::fabs(v_call_3[5]), std::max(std::fabs(v_call_4[0]), std::max(std::fabs(v_call_4[1]), std::max(std::fabs(v_call_4[2]), std::max(std::fabs(v_call_4[3]), std::max(std::fabs(v_call_4[4]), std::fabs(v_call_4[5]))))))))))));
    T t40 = t33*t39;
    T t41 = t29*(t22*v_call_0[38] + t23*v_call_1[38] + t23*v_call_2[38] + t24*v_call_0[39] + t25*v_call_1[39] + t25*v_call_2[39]) + t32*(t22*v_call_0[40] + t23*v_call_1[40] + t23*v_call_2[40] + t24*v_call_0[41] + t25*v_call_1[41] + t25*v_call_2[41]) + t33*(t22*v_call_0[42] + t23*v_call_1[42] + t23*v_call_2[42] + t24*v_call_0[43] + t25*v_call_1[43] + t25*v_call_2[43]) + t34*(t22*v_call_0[44] + t23*v_call_1[44] + t23*v_call_2[44] + t24*v_call_0[45] + t25*v_call_1[45] + t25*v_call_2[45]) + t35*(t22*v_call_0[46] + t23*v_call_1[46] + t23*v_call_2[46] + t24*v_call_0[47] + t25*v_call_1[47] + t25*v_call_2[47]);
    T t42 = t29*(t22*v_call_0[50] + t23*v_call_1[50] + t23*v_call_2[50] + t24*v_call_0[51] + t25*v_call_1[51] + t25*v_call_2[51]) + t32*(t22*v_call_0[52] + t23*v_call_1[52] + t23*v_call_2[52] + t24*v_call_0[53] + t25*v_call_1[53] + t25*v_call_2[53]) + t33*(t22*v_call_0[54] + t23*v_call_1[54] + t23*v_call_2[54] + t24*v_call_0[55] + t25*v_call_1[55] + t25*v_call_2[55]) + t34*(t22*v_call_0[56] + t23*v_call_1[56] + t23*v_call_2[56] + t24*v_call_0[57] + t25*v_call_1[57] + t25*v_call_2[57]) + t35*(t22*v_call_0[58] + t23*v_call_1[58] + t23*v_call_2[58] + t24*v_call_0[59] + t25*v_call_1[59] + t25*v_call_2[59]);
    T t43 = t35*t39;
    T t44 = t29*(t22*v_call_0[62] + t23*v_call_1[62] + t23*v_call_2[62] + t24*v_call_0[63] + t25*v_call_1[63] + t25*v_call_2[63]) + t32*(t22*v_call_0[64] + t23*v_call_1[64] + t23*v_call_2[64] + t24*v_call_0[65] + t25*v_call_1[65] + t25*v_call_2[65]) + t33*(t22*v_call_0[66] + t23*v_call_1[66] + t23*v_call_2[66] + t24*v_call_0[67] + t25*v_call_1[67] + t25*v_call_2[67]) + t34*(t22*v_call_0[68] + t23*v_call_1[68] + t23*v_call_2[68] + t24*v_call_0[69] + t25*v_call_1[69] + t25*v_call_2[69]) + t35*(t22*v_call_0[70] + t23*v_call_1[70] + t23*v_call_2[70] + t24*v_call_0[71] + t25*v_call_1[71] + t25*v_call_2[71]);
    SimpleArray<T, 12> res;
    res[0] = t36;
    res[1] = t37;
    res[2] = t38;
    res[3] = t40 + t41;
    res[4] = t42;
    res[5] = t43 + t44;
    res[6] = t36;
    res[7] = t37;
    res[8] = t38;
    res[9] = -t40 + t41;
    res[10] = t42;
    res[11] = -t43 + t44;
    return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> local_max_abs_eigenvalue(
        const T* Q,
        const T* Qaux,
        const T* p,
        const T* n)
    {
    T v_arg_0[] = { Q[0], Q[1], Q[2], Q[3], Q[4], Q[5] };
    T v_arg_1[] = { Qaux[0] };
    T v_arg_2[] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12] };
    T v_arg_3[] = { n[0], n[1] };
    auto v_call_0 = Model<T>::eigenvalues(v_arg_0, v_arg_1, v_arg_2, v_arg_3);
    SimpleArray<T, 1> res;
    res[0] = std::max(std::fabs(v_call_0[0]), std::max(std::fabs(v_call_0[1]), std::max(std::fabs(v_call_0[2]), std::max(std::fabs(v_call_0[3]), std::max(std::fabs(v_call_0[4]), std::fabs(v_call_0[5]))))));
    return res;
    }

};
