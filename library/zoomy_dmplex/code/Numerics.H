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

    PORTABLE_FN static inline SimpleArray<T, 12> nonconservative_fluctuations(
        const T* Q_minus,
        const T* Q_plus,
        const T* Qaux_minus,
        const T* Qaux_plus,
        const T* p,
        const T* n)
    {
    T v_arg_0[] = { Q[0], Q[1], Q[2], Q[3], Q[4], Q[5] };
    T v_arg_1[] = { Qaux[0] };
    T v_arg_2[] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12] };
    T v_arg_3[] = { n[0], n[1] };
    auto v_call_0 = Model<T>::nonconservative_matrix(v_arg_0, v_arg_1, v_arg_2);
    auto v_call_1 = Model<T>::eigenvalues(Q_minus, Qaux_minus, v_arg_2, v_arg_3);
    auto v_call_2 = Model<T>::eigenvalues(Q_plus, Qaux_plus, v_arg_2, v_arg_3);
    T t0 = std::max(std::fabs(v_call_1[0]), std::max(std::fabs(v_call_1[1]), std::max(std::fabs(v_call_1[2]), std::max(std::fabs(v_call_1[3]), std::max(std::fabs(v_call_1[4]), std::max(std::fabs(v_call_1[5]), std::max(std::fabs(v_call_2[0]), std::max(std::fabs(v_call_2[1]), std::max(std::fabs(v_call_2[2]), std::max(std::fabs(v_call_2[3]), std::max(std::fabs(v_call_2[4]), std::fabs(v_call_2[5]))))))))))));
    T t1 = -0.5*Q_minus[0] + 0.5*Q_plus[0];
    T t2 = t0*t1;
    T t3 = 1.0000000000000002*n[0];
    T t4 = 1.0000000000000002*n[1];
    T t5 = -0.5*Q_minus[1] + 0.5*Q_plus[1];
    T t6 = -0.5*Q_minus[2] + 0.5*Q_plus[2];
    T t7 = -0.5*Q_minus[3] + 0.5*Q_plus[3];
    T t8 = -0.5*Q_minus[4] + 0.5*Q_plus[4];
    T t9 = -0.5*Q_minus[5] + 0.5*Q_plus[5];
    T t10 = t1*(t3*v_call_0[0] + t4*v_call_0[1]) + t5*(t3*v_call_0[2] + t4*v_call_0[3]) + t6*(t3*v_call_0[4] + t4*v_call_0[5]) + t7*(t3*v_call_0[6] + t4*v_call_0[7]) + t8*(t3*v_call_0[8] + t4*v_call_0[9]) + t9*(t3*v_call_0[10] + t4*v_call_0[11]);
    T t11 = t0*t5;
    T t12 = t1*(t3*v_call_0[12] + t4*v_call_0[13]) + t5*(t3*v_call_0[14] + t4*v_call_0[15]) + t6*(t3*v_call_0[16] + t4*v_call_0[17]) + t7*(t3*v_call_0[18] + t4*v_call_0[19]) + t8*(t3*v_call_0[20] + t4*v_call_0[21]) + t9*(t3*v_call_0[22] + t4*v_call_0[23]);
    T t13 = t0*t6;
    T t14 = t1*(t3*v_call_0[24] + t4*v_call_0[25]) + t5*(t3*v_call_0[26] + t4*v_call_0[27]) + t6*(t3*v_call_0[28] + t4*v_call_0[29]) + t7*(t3*v_call_0[30] + t4*v_call_0[31]) + t8*(t3*v_call_0[32] + t4*v_call_0[33]) + t9*(t3*v_call_0[34] + t4*v_call_0[35]);
    T t15 = t0*t7;
    T t16 = t1*(t3*v_call_0[36] + t4*v_call_0[37]) + t5*(t3*v_call_0[38] + t4*v_call_0[39]) + t6*(t3*v_call_0[40] + t4*v_call_0[41]) + t7*(t3*v_call_0[42] + t4*v_call_0[43]) + t8*(t3*v_call_0[44] + t4*v_call_0[45]) + t9*(t3*v_call_0[46] + t4*v_call_0[47]);
    T t17 = t0*t8;
    T t18 = t1*(t3*v_call_0[48] + t4*v_call_0[49]) + t5*(t3*v_call_0[50] + t4*v_call_0[51]) + t6*(t3*v_call_0[52] + t4*v_call_0[53]) + t7*(t3*v_call_0[54] + t4*v_call_0[55]) + t8*(t3*v_call_0[56] + t4*v_call_0[57]) + t9*(t3*v_call_0[58] + t4*v_call_0[59]);
    T t19 = t0*t9;
    T t20 = t1*(t3*v_call_0[60] + t4*v_call_0[61]) + t5*(t3*v_call_0[62] + t4*v_call_0[63]) + t6*(t3*v_call_0[64] + t4*v_call_0[65]) + t7*(t3*v_call_0[66] + t4*v_call_0[67]) + t8*(t3*v_call_0[68] + t4*v_call_0[69]) + t9*(t3*v_call_0[70] + t4*v_call_0[71]);
    SimpleArray<T, 12> res;
    res[0] = t10 + t2;
    res[1] = t11 + t12;
    res[2] = t13 + t14;
    res[3] = t15 + t16;
    res[4] = t17 + t18;
    res[5] = t19 + t20;
    res[6] = t10 - t2;
    res[7] = -t11 + t12;
    res[8] = -t13 + t14;
    res[9] = -t15 + t16;
    res[10] = -t17 + t18;
    res[11] = -t19 + t20;
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
