#pragma once
#include "List.H"
#include "vector.H"
#include "scalar.H"

namespace Model
{
constexpr int n_dof_q    = 8;
constexpr int n_dof_qaux = 16;
constexpr int dimension  = 2;
const Foam::List<Foam::word> map_boundary_tag_to_function_index{ "wall", "inflow", "outflow" };

inline Foam::List<Foam::List<Foam::scalar>> flux_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = Foam::pow(Q[3], 2)*t0;
        Foam::scalar t2 = Foam::pow(Q[4], 2)*t0;
        Foam::scalar t3 = Q[2]*t0;
        Foam::scalar t4 = 2*t3;
        Foam::scalar t5 = Q[3]*t0;
        Foam::scalar t6 = Q[6]*t5;
        Foam::scalar t7 = Q[4]*t0;
        Foam::scalar t8 = Q[7]*t7;
        Foam::scalar t9 = Q[5]*t0;
        res[0][0] = 0;
        res[1][0] = Q[2];
        res[2][0] = (1.0/2.0)*1.0*9.81*Foam::pow(Q[1], 2) + Foam::pow(Q[2], 2)*t0 + (1.0/3.0)*t1 + (1.0/5.0)*t2;
        res[3][0] = Q[3]*t4 + (4.0/5.0)*Q[4]*t5;
        res[4][0] = Q[4]*t4 + (2.0/3.0)*t1 + (2.0/7.0)*t2;
        res[5][0] = Q[5]*t3 + (1.0/3.0)*t6 + (1.0/5.0)*t8;
        res[6][0] = Q[3]*t9 + Q[6]*t3 + (2.0/5.0)*Q[6]*t7 + (2.0/5.0)*Q[7]*t5;
        res[7][0] = Q[4]*t9 + Q[7]*t3 + (2.0/3.0)*t6 + (2.0/7.0)*t8;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = Q[3]*t0;
        Foam::scalar t3 = Q[6]*t2;
        Foam::scalar t4 = Q[4]*t0;
        Foam::scalar t5 = Q[7]*t4;
        Foam::scalar t6 = Q[5]*t0;
        Foam::scalar t7 = Foam::pow(Q[6], 2)*t0;
        Foam::scalar t8 = Foam::pow(Q[7], 2)*t0;
        Foam::scalar t9 = 2*t6;
        res[0][0] = 0;
        res[1][0] = Q[5];
        res[2][0] = Q[5]*t1 + (1.0/3.0)*t3 + (1.0/5.0)*t5;
        res[3][0] = Q[3]*t6 + Q[6]*t1 + (2.0/5.0)*Q[6]*t4 + (2.0/5.0)*Q[7]*t2;
        res[4][0] = Q[4]*t6 + Q[7]*t1 + (2.0/3.0)*t3 + (2.0/7.0)*t5;
        res[5][0] = (1.0/2.0)*1.0*9.81*Foam::pow(Q[1], 2) + Foam::pow(Q[5], 2)*t0 + (1.0/3.0)*t7 + (1.0/5.0)*t8;
        res[6][0] = (4.0/5.0)*Q[6]*Q[7]*t0 + Q[6]*t9;
        res[7][0] = Q[7]*t9 + (2.0/3.0)*t7 + (2.0/7.0)*t8;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_jacobian_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(8, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = Foam::pow(Q[3], 2)*t0;
        Foam::scalar t2 = Foam::pow(Q[4], 2)*t0;
        Foam::scalar t3 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t4 = Q[2]*t3;
        Foam::scalar t5 = 2*t4;
        Foam::scalar t6 = Q[3]*t3;
        Foam::scalar t7 = (2.0/3.0)*t6;
        Foam::scalar t8 = Q[4]*t3;
        Foam::scalar t9 = (2.0/5.0)*t8;
        Foam::scalar t10 = Q[2]*t0;
        Foam::scalar t11 = 2*t10;
        Foam::scalar t12 = Q[3]*t0;
        Foam::scalar t13 = Q[6]*t12;
        Foam::scalar t14 = Q[4]*t0;
        Foam::scalar t15 = Q[7]*t14;
        Foam::scalar t16 = Q[5]*t3;
        Foam::scalar t17 = Q[6]*t3;
        Foam::scalar t18 = Q[7]*t3;
        Foam::scalar t19 = Q[5]*t0;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[0][6] = 0;
        res[0][7] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 1;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[1][6] = 0;
        res[1][7] = 0;
        res[2][0] = 0;
        res[2][1] = 1.0*9.81*Q[1] - Foam::pow(Q[2], 2)*t0 - 1.0/3.0*t1 - 1.0/5.0*t2;
        res[2][2] = t5;
        res[2][3] = t7;
        res[2][4] = t9;
        res[2][5] = 0;
        res[2][6] = 0;
        res[2][7] = 0;
        res[3][0] = 0;
        res[3][1] = -Q[3]*t11 - 4.0/5.0*Q[4]*t12;
        res[3][2] = 2*t6;
        res[3][3] = t5 + (4.0/5.0)*t8;
        res[3][4] = (4.0/5.0)*t6;
        res[3][5] = 0;
        res[3][6] = 0;
        res[3][7] = 0;
        res[4][0] = 0;
        res[4][1] = -Q[4]*t11 - 2.0/3.0*t1 - 2.0/7.0*t2;
        res[4][2] = 2*t8;
        res[4][3] = (4.0/3.0)*t6;
        res[4][4] = t5 + (4.0/7.0)*t8;
        res[4][5] = 0;
        res[4][6] = 0;
        res[4][7] = 0;
        res[5][0] = 0;
        res[5][1] = -Q[5]*t10 - 1.0/3.0*t13 - 1.0/5.0*t15;
        res[5][2] = t16;
        res[5][3] = (1.0/3.0)*t17;
        res[5][4] = (1.0/5.0)*t18;
        res[5][5] = t4;
        res[5][6] = (1.0/3.0)*t6;
        res[5][7] = (1.0/5.0)*t8;
        res[6][0] = 0;
        res[6][1] = -Q[3]*t19 - Q[6]*t10 - 2.0/5.0*Q[6]*t14 - 2.0/5.0*Q[7]*t12;
        res[6][2] = t17;
        res[6][3] = t16 + (2.0/5.0)*t18;
        res[6][4] = (2.0/5.0)*t17;
        res[6][5] = t6;
        res[6][6] = t4 + t9;
        res[6][7] = (2.0/5.0)*t6;
        res[7][0] = 0;
        res[7][1] = -Q[4]*t19 - Q[7]*t10 - 2.0/3.0*t13 - 2.0/7.0*t15;
        res[7][2] = t18;
        res[7][3] = (2.0/3.0)*t17;
        res[7][4] = t16 + (2.0/7.0)*t18;
        res[7][5] = t8;
        res[7][6] = t7;
        res[7][7] = t4 + (2.0/7.0)*t8;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_jacobian_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(8, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = Q[3]*t0;
        Foam::scalar t3 = Q[6]*t2;
        Foam::scalar t4 = Q[4]*t0;
        Foam::scalar t5 = Q[7]*t4;
        Foam::scalar t6 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t7 = Q[5]*t6;
        Foam::scalar t8 = Q[6]*t6;
        Foam::scalar t9 = Q[7]*t6;
        Foam::scalar t10 = Q[2]*t6;
        Foam::scalar t11 = Q[3]*t6;
        Foam::scalar t12 = Q[4]*t6;
        Foam::scalar t13 = Q[5]*t0;
        Foam::scalar t14 = (2.0/5.0)*t9;
        Foam::scalar t15 = (2.0/3.0)*t8;
        Foam::scalar t16 = Foam::pow(Q[6], 2)*t0;
        Foam::scalar t17 = Foam::pow(Q[7], 2)*t0;
        Foam::scalar t18 = 2*t7;
        Foam::scalar t19 = 2*t13;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[0][6] = 0;
        res[0][7] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 1;
        res[1][6] = 0;
        res[1][7] = 0;
        res[2][0] = 0;
        res[2][1] = -Q[5]*t1 - 1.0/3.0*t3 - 1.0/5.0*t5;
        res[2][2] = t7;
        res[2][3] = (1.0/3.0)*t8;
        res[2][4] = (1.0/5.0)*t9;
        res[2][5] = t10;
        res[2][6] = (1.0/3.0)*t11;
        res[2][7] = (1.0/5.0)*t12;
        res[3][0] = 0;
        res[3][1] = -Q[3]*t13 - Q[6]*t1 - 2.0/5.0*Q[6]*t4 - 2.0/5.0*Q[7]*t2;
        res[3][2] = t8;
        res[3][3] = t14 + t7;
        res[3][4] = (2.0/5.0)*t8;
        res[3][5] = t11;
        res[3][6] = t10 + (2.0/5.0)*t12;
        res[3][7] = (2.0/5.0)*t11;
        res[4][0] = 0;
        res[4][1] = -Q[4]*t13 - Q[7]*t1 - 2.0/3.0*t3 - 2.0/7.0*t5;
        res[4][2] = t9;
        res[4][3] = t15;
        res[4][4] = t7 + (2.0/7.0)*t9;
        res[4][5] = t12;
        res[4][6] = (2.0/3.0)*t11;
        res[4][7] = t10 + (2.0/7.0)*t12;
        res[5][0] = 0;
        res[5][1] = 1.0*9.81*Q[1] - Foam::pow(Q[5], 2)*t0 - 1.0/3.0*t16 - 1.0/5.0*t17;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = t18;
        res[5][6] = t15;
        res[5][7] = t14;
        res[6][0] = 0;
        res[6][1] = -4.0/5.0*Q[6]*Q[7]*t0 - Q[6]*t19;
        res[6][2] = 0;
        res[6][3] = 0;
        res[6][4] = 0;
        res[6][5] = 2*t8;
        res[6][6] = t18 + (4.0/5.0)*t9;
        res[6][7] = (4.0/5.0)*t8;
        res[7][0] = 0;
        res[7][1] = -Q[7]*t19 - 2.0/3.0*t16 - 2.0/7.0*t17;
        res[7][2] = 0;
        res[7][3] = 0;
        res[7][4] = 0;
        res[7][5] = 2*t9;
        res[7][6] = (4.0/3.0)*t8;
        res[7][7] = t18 + (4.0/7.0)*t9;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(8, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = (2.0/5.0)*t0;
        Foam::scalar t3 = Q[5]*t0;
        Foam::scalar t4 = Q[7]*t0;
        Foam::scalar t5 = Q[6]*t0;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[0][6] = 0;
        res[0][7] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[1][6] = 0;
        res[1][7] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[2][4] = 0;
        res[2][5] = 0;
        res[2][6] = 0;
        res[2][7] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = -Q[4]*t2 - t1;
        res[3][4] = Q[3]*t2;
        res[3][5] = 0;
        res[3][6] = 0;
        res[3][7] = 0;
        res[4][0] = 0;
        res[4][1] = 0;
        res[4][2] = 0;
        res[4][3] = -2*Q[3]*t0;
        res[4][4] = -2.0/7.0*Q[4]*t0 - t1;
        res[4][5] = 0;
        res[4][6] = 0;
        res[4][7] = 0;
        res[5][0] = 0;
        res[5][1] = 0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = 0;
        res[5][6] = 0;
        res[5][7] = 0;
        res[6][0] = 0;
        res[6][1] = 0;
        res[6][2] = 0;
        res[6][3] = -t3 - 1.0/5.0*t4;
        res[6][4] = (1.0/5.0)*t5;
        res[6][5] = 0;
        res[6][6] = 0;
        res[6][7] = 0;
        res[7][0] = 0;
        res[7][1] = 0;
        res[7][2] = 0;
        res[7][3] = -t5;
        res[7][4] = -t3 - 1.0/7.0*t4;
        res[7][5] = 0;
        res[7][6] = 0;
        res[7][7] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(8, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = Q[4]*t0;
        Foam::scalar t3 = Q[3]*t0;
        Foam::scalar t4 = Q[5]*t0;
        Foam::scalar t5 = Q[7]*t0;
        Foam::scalar t6 = Q[6]*t0;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[0][6] = 0;
        res[0][7] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[1][6] = 0;
        res[1][7] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[2][4] = 0;
        res[2][5] = 0;
        res[2][6] = 0;
        res[2][7] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
        res[3][4] = 0;
        res[3][5] = 0;
        res[3][6] = -t1 - 1.0/5.0*t2;
        res[3][7] = (1.0/5.0)*t3;
        res[4][0] = 0;
        res[4][1] = 0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = 0;
        res[4][5] = 0;
        res[4][6] = -t3;
        res[4][7] = -t1 - 1.0/7.0*t2;
        res[5][0] = 0;
        res[5][1] = 0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = 0;
        res[5][6] = 0;
        res[5][7] = 0;
        res[6][0] = 0;
        res[6][1] = 0;
        res[6][2] = 0;
        res[6][3] = 0;
        res[6][4] = 0;
        res[6][5] = 0;
        res[6][6] = -t4 - 1.0/5.0*t5;
        res[6][7] = (1.0/5.0)*t6;
        res[7][0] = 0;
        res[7][1] = 0;
        res[7][2] = 0;
        res[7][3] = 0;
        res[7][4] = 0;
        res[7][5] = 0;
        res[7][6] = -t6;
        res[7][7] = -t4 - 1.0/7.0*t5;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(8, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = Foam::pow(Q[3], 2)*t0;
        Foam::scalar t2 = Foam::pow(Q[4], 2)*t0;
        Foam::scalar t3 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t4 = Q[2]*t3;
        Foam::scalar t5 = Q[3]*t3;
        Foam::scalar t6 = (2.0/3.0)*t5;
        Foam::scalar t7 = Q[4]*t3;
        Foam::scalar t8 = (2.0/5.0)*t7;
        Foam::scalar t9 = Q[2]*t0;
        Foam::scalar t10 = 2*t9;
        Foam::scalar t11 = Q[3]*t0;
        Foam::scalar t12 = t4 + t8;
        Foam::scalar t13 = t4 + (2.0/7.0)*t7;
        Foam::scalar t14 = Q[6]*t11;
        Foam::scalar t15 = Q[4]*t0;
        Foam::scalar t16 = Q[7]*t15;
        Foam::scalar t17 = Q[6]*t3;
        Foam::scalar t18 = (1.0/3.0)*t17;
        Foam::scalar t19 = Q[7]*t3;
        Foam::scalar t20 = (1.0/5.0)*t19;
        Foam::scalar t21 = Q[5]*t0;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[0][6] = 0;
        res[0][7] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 1;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[1][6] = 0;
        res[1][7] = 0;
        res[2][0] = 0;
        res[2][1] = 1.0*9.81*Q[1] - Foam::pow(Q[2], 2)*t0 - 1.0/3.0*t1 - 1.0/5.0*t2;
        res[2][2] = 2*t4;
        res[2][3] = t6;
        res[2][4] = t8;
        res[2][5] = 0;
        res[2][6] = 0;
        res[2][7] = 0;
        res[3][0] = 0;
        res[3][1] = -Q[3]*t10 - 4.0/5.0*Q[4]*t11;
        res[3][2] = 2*t5;
        res[3][3] = t12;
        res[3][4] = (6.0/5.0)*t5;
        res[3][5] = 0;
        res[3][6] = 0;
        res[3][7] = 0;
        res[4][0] = 0;
        res[4][1] = -Q[4]*t10 - 2.0/3.0*t1 - 2.0/7.0*t2;
        res[4][2] = 2*t7;
        res[4][3] = -t6;
        res[4][4] = t13;
        res[4][5] = 0;
        res[4][6] = 0;
        res[4][7] = 0;
        res[5][0] = 0;
        res[5][1] = -Q[5]*t9 - 1.0/3.0*t14 - 1.0/5.0*t16;
        res[5][2] = Q[5]*t3;
        res[5][3] = t18;
        res[5][4] = t20;
        res[5][5] = t4;
        res[5][6] = (1.0/3.0)*t5;
        res[5][7] = (1.0/5.0)*t7;
        res[6][0] = 0;
        res[6][1] = -Q[3]*t21 - 2.0/5.0*Q[6]*t15 - Q[6]*t9 - 2.0/5.0*Q[7]*t11;
        res[6][2] = t17;
        res[6][3] = t20;
        res[6][4] = (3.0/5.0)*t17;
        res[6][5] = t5;
        res[6][6] = t12;
        res[6][7] = (2.0/5.0)*t5;
        res[7][0] = 0;
        res[7][1] = -Q[4]*t21 - Q[7]*t9 - 2.0/3.0*t14 - 2.0/7.0*t16;
        res[7][2] = t19;
        res[7][3] = -t18;
        res[7][4] = (1.0/7.0)*t19;
        res[7][5] = t7;
        res[7][6] = t6;
        res[7][7] = t13;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(8, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = Q[3]*t0;
        Foam::scalar t3 = Q[6]*t2;
        Foam::scalar t4 = Q[4]*t0;
        Foam::scalar t5 = Q[7]*t4;
        Foam::scalar t6 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t7 = Q[5]*t6;
        Foam::scalar t8 = Q[6]*t6;
        Foam::scalar t9 = (1.0/3.0)*t8;
        Foam::scalar t10 = Q[7]*t6;
        Foam::scalar t11 = Q[3]*t6;
        Foam::scalar t12 = (1.0/3.0)*t11;
        Foam::scalar t13 = Q[4]*t6;
        Foam::scalar t14 = (1.0/5.0)*t13;
        Foam::scalar t15 = Q[5]*t0;
        Foam::scalar t16 = (2.0/5.0)*t10;
        Foam::scalar t17 = (2.0/3.0)*t8;
        Foam::scalar t18 = Foam::pow(Q[6], 2)*t0;
        Foam::scalar t19 = Foam::pow(Q[7], 2)*t0;
        Foam::scalar t20 = 2*t15;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[0][6] = 0;
        res[0][7] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 1;
        res[1][6] = 0;
        res[1][7] = 0;
        res[2][0] = 0;
        res[2][1] = -Q[5]*t1 - 1.0/3.0*t3 - 1.0/5.0*t5;
        res[2][2] = t7;
        res[2][3] = t9;
        res[2][4] = (1.0/5.0)*t10;
        res[2][5] = Q[2]*t6;
        res[2][6] = t12;
        res[2][7] = t14;
        res[3][0] = 0;
        res[3][1] = -Q[3]*t15 - Q[6]*t1 - 2.0/5.0*Q[6]*t4 - 2.0/5.0*Q[7]*t2;
        res[3][2] = t8;
        res[3][3] = t16 + t7;
        res[3][4] = (2.0/5.0)*t8;
        res[3][5] = t11;
        res[3][6] = t14;
        res[3][7] = (3.0/5.0)*t11;
        res[4][0] = 0;
        res[4][1] = -Q[4]*t15 - Q[7]*t1 - 2.0/3.0*t3 - 2.0/7.0*t5;
        res[4][2] = t10;
        res[4][3] = t17;
        res[4][4] = (2.0/7.0)*t10 + t7;
        res[4][5] = t13;
        res[4][6] = -t12;
        res[4][7] = (1.0/7.0)*t13;
        res[5][0] = 0;
        res[5][1] = 1.0*9.81*Q[1] - Foam::pow(Q[5], 2)*t0 - 1.0/3.0*t18 - 1.0/5.0*t19;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = 2*t7;
        res[5][6] = t17;
        res[5][7] = t16;
        res[6][0] = 0;
        res[6][1] = -4.0/5.0*Q[6]*Q[7]*t0 - Q[6]*t20;
        res[6][2] = 0;
        res[6][3] = 0;
        res[6][4] = 0;
        res[6][5] = 2*t8;
        res[6][6] = (3.0/5.0)*t10 + t7;
        res[6][7] = t8;
        res[7][0] = 0;
        res[7][1] = -Q[7]*t20 - 2.0/3.0*t18 - 2.0/7.0*t19;
        res[7][2] = 0;
        res[7][3] = 0;
        res[7][4] = 0;
        res[7][5] = 2*t10;
        res[7][6] = t9;
        res[7][7] = (3.0/7.0)*t10 + t7;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> eigenvalues(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& n)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = n.x()*Q[2] + n.y()*Q[5];
        Foam::scalar t2 = t0*t1;
        Foam::scalar t3 = Foam::pow(Q[1], 3.0/2.0)*Foam::pow(1.0*9.81, 1.0/2.0)*Foam::pow(Foam::pow(n.x(), 2) + Foam::pow(n.y(), 2), 1.0/2.0);
        res[0][0] = 0;
        res[1][0] = t2;
        res[2][0] = t2;
        res[3][0] = t2;
        res[4][0] = t2;
        res[5][0] = t2;
        res[6][0] = t0*(t1 - t3);
        res[7][0] = t0*(t1 + t3);
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> left_eigenvectors(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(8, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[0][6] = 0;
        res[0][7] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[1][6] = 0;
        res[1][7] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[2][4] = 0;
        res[2][5] = 0;
        res[2][6] = 0;
        res[2][7] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
        res[3][4] = 0;
        res[3][5] = 0;
        res[3][6] = 0;
        res[3][7] = 0;
        res[4][0] = 0;
        res[4][1] = 0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = 0;
        res[4][5] = 0;
        res[4][6] = 0;
        res[4][7] = 0;
        res[5][0] = 0;
        res[5][1] = 0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = 0;
        res[5][6] = 0;
        res[5][7] = 0;
        res[6][0] = 0;
        res[6][1] = 0;
        res[6][2] = 0;
        res[6][3] = 0;
        res[6][4] = 0;
        res[6][5] = 0;
        res[6][6] = 0;
        res[6][7] = 0;
        res[7][0] = 0;
        res[7][1] = 0;
        res[7][2] = 0;
        res[7][3] = 0;
        res[7][4] = 0;
        res[7][5] = 0;
        res[7][6] = 0;
        res[7][7] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> right_eigenvectors(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(8, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[0][6] = 0;
        res[0][7] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[1][6] = 0;
        res[1][7] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[2][4] = 0;
        res[2][5] = 0;
        res[2][6] = 0;
        res[2][7] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
        res[3][4] = 0;
        res[3][5] = 0;
        res[3][6] = 0;
        res[3][7] = 0;
        res[4][0] = 0;
        res[4][1] = 0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = 0;
        res[4][5] = 0;
        res[4][6] = 0;
        res[4][7] = 0;
        res[5][0] = 0;
        res[5][1] = 0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = 0;
        res[5][6] = 0;
        res[5][7] = 0;
        res[6][0] = 0;
        res[6][1] = 0;
        res[6][2] = 0;
        res[6][3] = 0;
        res[6][4] = 0;
        res[6][5] = 0;
        res[6][6] = 0;
        res[6][7] = 0;
        res[7][0] = 0;
        res[7][1] = 0;
        res[7][2] = 0;
        res[7][3] = 0;
        res[7][4] = 0;
        res[7][5] = 0;
        res[7][6] = 0;
        res[7][7] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> source(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
        res[4][0] = 0;
        res[5][0] = 0;
        res[6][0] = 0;
        res[7][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> residual(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
        res[4][0] = 0;
        res[5][0] = 0;
        res[6][0] = 0;
        res[7][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> source_implicit(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(8, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
        res[4][0] = 0;
        res[5][0] = 0;
        res[6][0] = 0;
        res[7][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> interpolate(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& X)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = 2*X.z() - 1;
        Foam::scalar t2 = Foam::pow(t1, 2);
        Foam::scalar t3 = t0*(1.5*t2 - 0.5);
        Foam::scalar t4 = Q[2]*t0;
        Foam::scalar t5 = -t0*t1;
        Foam::scalar t6 = Q[3]*t5 + t4;
        Foam::scalar t7 = Q[5]*t0;
        Foam::scalar t8 = Q[6]*t5 + t7;
        Foam::scalar t9 = Foam::pow(X.z(), 2);
        Foam::scalar t10 = X.z() - t9;
        Foam::scalar t11 = 2*Foam::pow(X.z(), 3) + X.z() - 3*t9;
        Foam::scalar t12 = t0*t10;
        Foam::scalar t13 = t0*t11;
        Foam::scalar t14 = t0*((3.0/2.0)*t2 - 1.0/2.0);
        res[0][0] = Q[0];
        res[1][0] = Q[1];
        res[2][0] = Q[4]*t3 + t6;
        res[3][0] = Q[7]*t3 + t8;
        res[4][0] = -Q[1]*(X.z()*Qaux[10] + Qaux[11]*t10 + Qaux[12]*t11) - Q[1]*(X.z()*Qaux[2] + Qaux[3]*t10 + Qaux[4]*t11) - Qaux[1]*(X.z()*t4 + Q[3]*t12 + Q[4]*t13) - Qaux[9]*(X.z()*t7 + Q[6]*t12 + Q[7]*t13) + (X.z()*Qaux[1] + Qaux[0])*(Q[4]*t14 + t6) + (X.z()*Qaux[9] + Qaux[8])*(Q[7]*t14 + t8);
        res[5][0] = 9810.0*Q[1]*(1 - X.z());
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> boundary_conditions(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& n,
    const Foam::vector& X,
    const Foam::scalar& time,
    const Foam::scalar& dX)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(8, 0.0));
    Foam::scalar t0 = 1.0*n.x();
        Foam::scalar t1 = 1.0*n.y();
        Foam::scalar t2 = Q[2]*t0 + Q[5]*t1;
        Foam::scalar t3 = -n.x()*t2;
        Foam::scalar t4 = Q[3]*t0 + Q[6]*t1;
        Foam::scalar t5 = -n.x()*t4;
        Foam::scalar t6 = Q[4]*t0 + Q[7]*t1;
        Foam::scalar t7 = -n.x()*t6;
        Foam::scalar t8 = -n.y()*t2;
        Foam::scalar t9 = -n.y()*t4;
        Foam::scalar t10 = -n.y()*t6;
        res[0][0] = Q[0];
        res[0][1] = Q[1];
        res[0][2] = t3;
        res[0][3] = t5;
        res[0][4] = t7;
        res[0][5] = t8;
        res[0][6] = t9;
        res[0][7] = t10;
        res[1][0] = Q[0];
        res[1][1] = Q[1];
        res[1][2] = t3;
        res[1][3] = t5;
        res[1][4] = t7;
        res[1][5] = t8;
        res[1][6] = t9;
        res[1][7] = t10;
        res[2][0] = Q[0];
        res[2][1] = Q[1];
        res[2][2] = t3;
        res[2][3] = t5;
        res[2][4] = t7;
        res[2][5] = t8;
        res[2][6] = t9;
        res[2][7] = t10;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> basis(
    const Foam::scalar& z)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = 2*z - 1;
        res[0][0] = 1;
        res[1][0] = -t0;
        res[2][0] = (3.0/2.0)*Foam::pow(t0, 2) - 1.0/2.0;
    return res;
}
        
} // namespace Model
