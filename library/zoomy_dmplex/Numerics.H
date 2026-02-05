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
    T v_arg_1[] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12] };
    T v_arg_2[] = { t0, t4, Q_plus[2]*t5, Q_plus[3]*t5, Q_plus[4]*t5, Q_plus[5]*t5 };
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
    T t11 = 0.5*std::max(std::fabs(v_call_2[0]), std::max(std::fabs(v_call_2[1]), std::max(std::fabs(v_call_2[2]), std::max(std::fabs(v_call_2[3]), std::max(std::fabs(v_call_2[4]), std::max(std::fabs(v_call_2[5]), std::max(std::fabs(v_call_3[0]), std::max(std::fabs(v_call_3[1]), std::max(std::fabs(v_call_3[2]), std::max(std::fabs(v_call_3[3]), std::max(std::fabs(v_call_3[4]), std::fabs(v_call_3[5]))))))))))));
    T t12 = t9/std::max(9.9999999999999995e-7, Q_minus[1]);
    T t13 = (1.0 / std::pow(std::max(9.9999999999999995e-7, Q_plus[1]), 1));
    SimpleArray<T, 6> res;
    res[0] = t6*v_call_0[0] + t6*v_call_1[0] + t7*v_call_0[1] + t7*v_call_1[1];
    res[1] = -t11*(t10 - t9) + t6*v_call_0[2] + t6*v_call_1[2] + t7*v_call_0[3] + t7*v_call_1[3];
    res[2] = -t11*(-Q_minus[2]*t12 + Q_plus[2]*t10*t13) + t6*v_call_0[4] + t6*v_call_1[4] + t7*v_call_0[5] + t7*v_call_1[5];
    res[3] = -t11*(-Q_minus[3]*t12 + Q_plus[3]*t10*t13) + t6*v_call_0[6] + t6*v_call_1[6] + t7*v_call_0[7] + t7*v_call_1[7];
    res[4] = -t11*(-Q_minus[4]*t12 + Q_plus[4]*t10*t13) + t6*v_call_0[8] + t6*v_call_1[8] + t7*v_call_0[9] + t7*v_call_1[9];
    res[5] = -t11*(-Q_minus[5]*t12 + Q_plus[5]*t10*t13) + t6*v_call_0[10] + t6*v_call_1[10] + t7*v_call_0[11] + t7*v_call_1[11];
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
    T v_arg_0[] = { 0.5*Q_minus[0] + 0.5*Q_plus[0], 0.5*Q_minus[1] + 0.5*Q_plus[1], 0.5*Q_minus[2] + 0.5*Q_plus[2], 0.5*Q_minus[3] + 0.5*Q_plus[3], 0.5*Q_minus[4] + 0.5*Q_plus[4], 0.5*Q_minus[5] + 0.5*Q_plus[5] };
    T v_arg_1[] = { 0.5*Qaux_minus[0] + 0.5*Qaux_plus[0] };
    T v_arg_2[] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12] };
    T v_arg_3[] = { 0.1127016653792583*Q_minus[0] + 0.8872983346207417*Q_plus[0], 0.1127016653792583*Q_minus[1] + 0.8872983346207417*Q_plus[1], 0.1127016653792583*Q_minus[2] + 0.8872983346207417*Q_plus[2], 0.1127016653792583*Q_minus[3] + 0.8872983346207417*Q_plus[3], 0.1127016653792583*Q_minus[4] + 0.8872983346207417*Q_plus[4], 0.1127016653792583*Q_minus[5] + 0.8872983346207417*Q_plus[5] };
    T v_arg_4[] = { 0.1127016653792583*Qaux_minus[0] + 0.8872983346207417*Qaux_plus[0] };
    T v_arg_5[] = { 0.8872983346207417*Q_minus[0] + 0.1127016653792583*Q_plus[0], 0.8872983346207417*Q_minus[1] + 0.1127016653792583*Q_plus[1], 0.8872983346207417*Q_minus[2] + 0.1127016653792583*Q_plus[2], 0.8872983346207417*Q_minus[3] + 0.1127016653792583*Q_plus[3], 0.8872983346207417*Q_minus[4] + 0.1127016653792583*Q_plus[4], 0.8872983346207417*Q_minus[5] + 0.1127016653792583*Q_plus[5] };
    T v_arg_6[] = { 0.8872983346207417*Qaux_minus[0] + 0.1127016653792583*Qaux_plus[0] };
    T v_arg_7[] = { n[0], n[1] };
    auto v_call_0 = Model<T>::nonconservative_matrix(v_arg_0, v_arg_1, v_arg_2);
    auto v_call_1 = Model<T>::nonconservative_matrix(v_arg_3, v_arg_4, v_arg_2);
    auto v_call_2 = Model<T>::nonconservative_matrix(v_arg_5, v_arg_6, v_arg_2);
    auto v_call_3 = Model<T>::eigenvalues(Q_minus, Qaux_minus, v_arg_2, v_arg_7);
    auto v_call_4 = Model<T>::eigenvalues(Q_plus, Qaux_plus, v_arg_2, v_arg_7);
    T t0 = std::max(std::fabs(v_call_3[0]), std::max(std::fabs(v_call_3[1]), std::max(std::fabs(v_call_3[2]), std::max(std::fabs(v_call_3[3]), std::max(std::fabs(v_call_3[4]), std::max(std::fabs(v_call_3[5]), std::max(std::fabs(v_call_4[0]), std::max(std::fabs(v_call_4[1]), std::max(std::fabs(v_call_4[2]), std::max(std::fabs(v_call_4[3]), std::max(std::fabs(v_call_4[4]), std::fabs(v_call_4[5]))))))))))));
    T t1 = -0.5*Q_minus[0] + 0.5*Q_plus[0];
    T t2 = t0*t1;
    T t3 = 0.44444444444444442*n[0];
    T t4 = 0.27777777777777785*n[0];
    T t5 = 0.44444444444444442*n[1];
    T t6 = 0.27777777777777785*n[1];
    T t7 = -0.5*Q_minus[1] + 0.5*Q_plus[1];
    T t8 = -0.5*Q_minus[2] + 0.5*Q_plus[2];
    T t9 = -0.5*Q_minus[3] + 0.5*Q_plus[3];
    T t10 = -0.5*Q_minus[4] + 0.5*Q_plus[4];
    T t11 = -0.5*Q_minus[5] + 0.5*Q_plus[5];
    T t12 = t1*(t3*v_call_0[0] + t4*v_call_1[0] + t4*v_call_2[0] + t5*v_call_0[1] + t6*v_call_1[1] + t6*v_call_2[1]) + t10*(t3*v_call_0[8] + t4*v_call_1[8] + t4*v_call_2[8] + t5*v_call_0[9] + t6*v_call_1[9] + t6*v_call_2[9]) + t11*(t3*v_call_0[10] + t4*v_call_1[10] + t4*v_call_2[10] + t5*v_call_0[11] + t6*v_call_1[11] + t6*v_call_2[11]) + t7*(t3*v_call_0[2] + t4*v_call_1[2] + t4*v_call_2[2] + t5*v_call_0[3] + t6*v_call_1[3] + t6*v_call_2[3]) + t8*(t3*v_call_0[4] + t4*v_call_1[4] + t4*v_call_2[4] + t5*v_call_0[5] + t6*v_call_1[5] + t6*v_call_2[5]) + t9*(t3*v_call_0[6] + t4*v_call_1[6] + t4*v_call_2[6] + t5*v_call_0[7] + t6*v_call_1[7] + t6*v_call_2[7]);
    T t13 = t0*t7;
    T t14 = t1*(t3*v_call_0[12] + t4*v_call_1[12] + t4*v_call_2[12] + t5*v_call_0[13] + t6*v_call_1[13] + t6*v_call_2[13]) + t10*(t3*v_call_0[20] + t4*v_call_1[20] + t4*v_call_2[20] + t5*v_call_0[21] + t6*v_call_1[21] + t6*v_call_2[21]) + t11*(t3*v_call_0[22] + t4*v_call_1[22] + t4*v_call_2[22] + t5*v_call_0[23] + t6*v_call_1[23] + t6*v_call_2[23]) + t7*(t3*v_call_0[14] + t4*v_call_1[14] + t4*v_call_2[14] + t5*v_call_0[15] + t6*v_call_1[15] + t6*v_call_2[15]) + t8*(t3*v_call_0[16] + t4*v_call_1[16] + t4*v_call_2[16] + t5*v_call_0[17] + t6*v_call_1[17] + t6*v_call_2[17]) + t9*(t3*v_call_0[18] + t4*v_call_1[18] + t4*v_call_2[18] + t5*v_call_0[19] + t6*v_call_1[19] + t6*v_call_2[19]);
    T t15 = t0*t8;
    T t16 = t1*(t3*v_call_0[24] + t4*v_call_1[24] + t4*v_call_2[24] + t5*v_call_0[25] + t6*v_call_1[25] + t6*v_call_2[25]) + t10*(t3*v_call_0[32] + t4*v_call_1[32] + t4*v_call_2[32] + t5*v_call_0[33] + t6*v_call_1[33] + t6*v_call_2[33]) + t11*(t3*v_call_0[34] + t4*v_call_1[34] + t4*v_call_2[34] + t5*v_call_0[35] + t6*v_call_1[35] + t6*v_call_2[35]) + t7*(t3*v_call_0[26] + t4*v_call_1[26] + t4*v_call_2[26] + t5*v_call_0[27] + t6*v_call_1[27] + t6*v_call_2[27]) + t8*(t3*v_call_0[28] + t4*v_call_1[28] + t4*v_call_2[28] + t5*v_call_0[29] + t6*v_call_1[29] + t6*v_call_2[29]) + t9*(t3*v_call_0[30] + t4*v_call_1[30] + t4*v_call_2[30] + t5*v_call_0[31] + t6*v_call_1[31] + t6*v_call_2[31]);
    T t17 = t0*t9;
    T t18 = t1*(t3*v_call_0[36] + t4*v_call_1[36] + t4*v_call_2[36] + t5*v_call_0[37] + t6*v_call_1[37] + t6*v_call_2[37]) + t10*(t3*v_call_0[44] + t4*v_call_1[44] + t4*v_call_2[44] + t5*v_call_0[45] + t6*v_call_1[45] + t6*v_call_2[45]) + t11*(t3*v_call_0[46] + t4*v_call_1[46] + t4*v_call_2[46] + t5*v_call_0[47] + t6*v_call_1[47] + t6*v_call_2[47]) + t7*(t3*v_call_0[38] + t4*v_call_1[38] + t4*v_call_2[38] + t5*v_call_0[39] + t6*v_call_1[39] + t6*v_call_2[39]) + t8*(t3*v_call_0[40] + t4*v_call_1[40] + t4*v_call_2[40] + t5*v_call_0[41] + t6*v_call_1[41] + t6*v_call_2[41]) + t9*(t3*v_call_0[42] + t4*v_call_1[42] + t4*v_call_2[42] + t5*v_call_0[43] + t6*v_call_1[43] + t6*v_call_2[43]);
    T t19 = t0*t10;
    T t20 = t1*(t3*v_call_0[48] + t4*v_call_1[48] + t4*v_call_2[48] + t5*v_call_0[49] + t6*v_call_1[49] + t6*v_call_2[49]) + t10*(t3*v_call_0[56] + t4*v_call_1[56] + t4*v_call_2[56] + t5*v_call_0[57] + t6*v_call_1[57] + t6*v_call_2[57]) + t11*(t3*v_call_0[58] + t4*v_call_1[58] + t4*v_call_2[58] + t5*v_call_0[59] + t6*v_call_1[59] + t6*v_call_2[59]) + t7*(t3*v_call_0[50] + t4*v_call_1[50] + t4*v_call_2[50] + t5*v_call_0[51] + t6*v_call_1[51] + t6*v_call_2[51]) + t8*(t3*v_call_0[52] + t4*v_call_1[52] + t4*v_call_2[52] + t5*v_call_0[53] + t6*v_call_1[53] + t6*v_call_2[53]) + t9*(t3*v_call_0[54] + t4*v_call_1[54] + t4*v_call_2[54] + t5*v_call_0[55] + t6*v_call_1[55] + t6*v_call_2[55]);
    T t21 = t0*t11;
    T t22 = t1*(t3*v_call_0[60] + t4*v_call_1[60] + t4*v_call_2[60] + t5*v_call_0[61] + t6*v_call_1[61] + t6*v_call_2[61]) + t10*(t3*v_call_0[68] + t4*v_call_1[68] + t4*v_call_2[68] + t5*v_call_0[69] + t6*v_call_1[69] + t6*v_call_2[69]) + t11*(t3*v_call_0[70] + t4*v_call_1[70] + t4*v_call_2[70] + t5*v_call_0[71] + t6*v_call_1[71] + t6*v_call_2[71]) + t7*(t3*v_call_0[62] + t4*v_call_1[62] + t4*v_call_2[62] + t5*v_call_0[63] + t6*v_call_1[63] + t6*v_call_2[63]) + t8*(t3*v_call_0[64] + t4*v_call_1[64] + t4*v_call_2[64] + t5*v_call_0[65] + t6*v_call_1[65] + t6*v_call_2[65]) + t9*(t3*v_call_0[66] + t4*v_call_1[66] + t4*v_call_2[66] + t5*v_call_0[67] + t6*v_call_1[67] + t6*v_call_2[67]);
    SimpleArray<T, 12> res;
    res[0] = t12 + t2;
    res[1] = t13 + t14;
    res[2] = t15 + t16;
    res[3] = t17 + t18;
    res[4] = t19 + t20;
    res[5] = t21 + t22;
    res[6] = t12 - t2;
    res[7] = -t13 + t14;
    res[8] = -t15 + t16;
    res[9] = -t17 + t18;
    res[10] = -t19 + t20;
    res[11] = -t21 + t22;
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
