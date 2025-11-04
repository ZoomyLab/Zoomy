#pragma once
#include "List.H"
#include "vector.H"
#include "scalar.H"

namespace Model
{
constexpr int n_dof_q    = 6;
constexpr int n_dof_qaux = 12;
constexpr int dimension  = 2;
const Foam::List<Foam::word> map_boundary_tag_to_function_index{ "wall", "inflow", "outflow" };

inline Foam::List<Foam::List<Foam::scalar>> flux_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = (1.0/3.0)*t0;
        Foam::scalar t2 = Q[2]*t0;
        res[0][0] = 0;
        res[1][0] = Q[2];
        res[2][0] = (1.0/2.0)*1.0*9.81*Foam::pow(Q[1], 2) + Foam::pow(Q[2], 2)*t0 + Foam::pow(Q[3], 2)*t1;
        res[3][0] = 2*Q[3]*t2;
        res[4][0] = Q[3]*Q[5]*t1 + Q[4]*t2;
        res[5][0] = Q[3]*Q[4]*t0 + Q[5]*t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = Q[3]*t0;
        res[0][0] = 0;
        res[1][0] = Q[4];
        res[2][0] = Q[4]*t1 + (1.0/3.0)*Q[5]*t2;
        res[3][0] = Q[4]*t2 + Q[5]*t1;
        res[4][0] = (1.0/2.0)*1.0*9.81*Foam::pow(Q[1], 2) + Foam::pow(Q[4], 2)*t0 + (1.0/3.0)*Foam::pow(Q[5], 2)*t0;
        res[5][0] = 2*Q[4]*Q[5]*t0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_jacobian_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(6, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = (1.0/3.0)*t0;
        Foam::scalar t2 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t3 = Q[2]*t2;
        Foam::scalar t4 = 2*t3;
        Foam::scalar t5 = Q[3]*t2;
        Foam::scalar t6 = Q[2]*t0;
        Foam::scalar t7 = Q[4]*t2;
        Foam::scalar t8 = Q[5]*t2;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 1;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[2][0] = 0;
        res[2][1] = 1.0*9.81*Q[1] - Foam::pow(Q[2], 2)*t0 - Foam::pow(Q[3], 2)*t1;
        res[2][2] = t4;
        res[2][3] = (2.0/3.0)*t5;
        res[2][4] = 0;
        res[2][5] = 0;
        res[3][0] = 0;
        res[3][1] = -2*Q[3]*t6;
        res[3][2] = 2*t5;
        res[3][3] = t4;
        res[3][4] = 0;
        res[3][5] = 0;
        res[4][0] = 0;
        res[4][1] = -Q[3]*Q[5]*t1 - Q[4]*t6;
        res[4][2] = t7;
        res[4][3] = (1.0/3.0)*t8;
        res[4][4] = t3;
        res[4][5] = (1.0/3.0)*t5;
        res[5][0] = 0;
        res[5][1] = -Q[3]*Q[4]*t0 - Q[5]*t6;
        res[5][2] = t8;
        res[5][3] = t7;
        res[5][4] = t5;
        res[5][5] = t3;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_jacobian_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(6, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = Q[3]*t0;
        Foam::scalar t3 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t4 = Q[4]*t3;
        Foam::scalar t5 = Q[5]*t3;
        Foam::scalar t6 = Q[2]*t3;
        Foam::scalar t7 = Q[3]*t3;
        Foam::scalar t8 = 2*t4;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 1;
        res[1][5] = 0;
        res[2][0] = 0;
        res[2][1] = -Q[4]*t1 - 1.0/3.0*Q[5]*t2;
        res[2][2] = t4;
        res[2][3] = (1.0/3.0)*t5;
        res[2][4] = t6;
        res[2][5] = (1.0/3.0)*t7;
        res[3][0] = 0;
        res[3][1] = -Q[4]*t2 - Q[5]*t1;
        res[3][2] = t5;
        res[3][3] = t4;
        res[3][4] = t7;
        res[3][5] = t6;
        res[4][0] = 0;
        res[4][1] = 1.0*9.81*Q[1] - Foam::pow(Q[4], 2)*t0 - 1.0/3.0*Foam::pow(Q[5], 2)*t0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = t8;
        res[4][5] = (2.0/3.0)*t5;
        res[5][0] = 0;
        res[5][1] = -2*Q[4]*Q[5]*t0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 2*t5;
        res[5][5] = t8;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(6, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[2][4] = 0;
        res[2][5] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = -Q[2]*t0;
        res[3][4] = 0;
        res[3][5] = 0;
        res[4][0] = 0;
        res[4][1] = 0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = 0;
        res[4][5] = 0;
        res[5][0] = 0;
        res[5][1] = 0;
        res[5][2] = 0;
        res[5][3] = -Q[4]*t0;
        res[5][4] = 0;
        res[5][5] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(6, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[2][4] = 0;
        res[2][5] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
        res[3][4] = 0;
        res[3][5] = -Q[2]*t0;
        res[4][0] = 0;
        res[4][1] = 0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = 0;
        res[4][5] = 0;
        res[5][0] = 0;
        res[5][1] = 0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = -Q[4]*t0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(6, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = (1.0/3.0)*t0;
        Foam::scalar t2 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t3 = Q[2]*t2;
        Foam::scalar t4 = Q[3]*t2;
        Foam::scalar t5 = Q[2]*t0;
        Foam::scalar t6 = Q[5]*t2;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 1;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[2][0] = 0;
        res[2][1] = 1.0*9.81*Q[1] - Foam::pow(Q[2], 2)*t0 - Foam::pow(Q[3], 2)*t1;
        res[2][2] = 2*t3;
        res[2][3] = (2.0/3.0)*t4;
        res[2][4] = 0;
        res[2][5] = 0;
        res[3][0] = 0;
        res[3][1] = -2*Q[3]*t5;
        res[3][2] = 2*t4;
        res[3][3] = t3;
        res[3][4] = 0;
        res[3][5] = 0;
        res[4][0] = 0;
        res[4][1] = -Q[3]*Q[5]*t1 - Q[4]*t5;
        res[4][2] = Q[4]*t2;
        res[4][3] = (1.0/3.0)*t6;
        res[4][4] = t3;
        res[4][5] = (1.0/3.0)*t4;
        res[5][0] = 0;
        res[5][1] = -Q[3]*Q[4]*t0 - Q[5]*t5;
        res[5][2] = t6;
        res[5][3] = 0;
        res[5][4] = t4;
        res[5][5] = t3;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(6, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = Q[3]*t0;
        Foam::scalar t3 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t4 = Q[4]*t3;
        Foam::scalar t5 = Q[5]*t3;
        Foam::scalar t6 = Q[3]*t3;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 1;
        res[1][5] = 0;
        res[2][0] = 0;
        res[2][1] = -Q[4]*t1 - 1.0/3.0*Q[5]*t2;
        res[2][2] = t4;
        res[2][3] = (1.0/3.0)*t5;
        res[2][4] = Q[2]*t3;
        res[2][5] = (1.0/3.0)*t6;
        res[3][0] = 0;
        res[3][1] = -Q[4]*t2 - Q[5]*t1;
        res[3][2] = t5;
        res[3][3] = t4;
        res[3][4] = t6;
        res[3][5] = 0;
        res[4][0] = 0;
        res[4][1] = 1.0*9.81*Q[1] - Foam::pow(Q[4], 2)*t0 - 1.0/3.0*Foam::pow(Q[5], 2)*t0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = 2*t4;
        res[4][5] = (2.0/3.0)*t5;
        res[5][0] = 0;
        res[5][1] = -2*Q[4]*Q[5]*t0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 2*t5;
        res[5][5] = t4;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> eigenvalues(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& n)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = n.x()*Q[2] + n.y()*Q[4];
        Foam::scalar t2 = (1.0/3.0)*Foam::pow(3, 1.0/2.0);
        Foam::scalar t3 = n.x()*Q[3];
        Foam::scalar t4 = t2*t3;
        Foam::scalar t5 = n.y()*Q[5];
        Foam::scalar t6 = t2*t5;
        Foam::scalar t7 = Foam::pow(n.x(), 2);
        Foam::scalar t8 = Foam::pow(n.y(), 2);
        Foam::scalar t9 = 1.0*9.81*Foam::pow(Q[1], 3);
        Foam::scalar t10 = Foam::pow(Foam::pow(Q[3], 2)*t7 + Foam::pow(Q[5], 2)*t8 + 2*t3*t5 + t7*t9 + t8*t9, 1.0/2.0);
        res[0][0] = 0;
        res[1][0] = t0*t1;
        res[2][0] = t0*(t1 - t4 - t6);
        res[3][0] = t0*(t1 + t4 + t6);
        res[4][0] = t0*(t1 - t10);
        res[5][0] = t0*(t1 + t10);
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> left_eigenvectors(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(6, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[2][4] = 0;
        res[2][5] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
        res[3][4] = 0;
        res[3][5] = 0;
        res[4][0] = 0;
        res[4][1] = 0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = 0;
        res[4][5] = 0;
        res[5][0] = 0;
        res[5][1] = 0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> right_eigenvectors(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(6, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[0][4] = 0;
        res[0][5] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[1][4] = 0;
        res[1][5] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[2][4] = 0;
        res[2][5] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
        res[3][4] = 0;
        res[3][5] = 0;
        res[4][0] = 0;
        res[4][1] = 0;
        res[4][2] = 0;
        res[4][3] = 0;
        res[4][4] = 0;
        res[4][5] = 0;
        res[5][0] = 0;
        res[5][1] = 0;
        res[5][2] = 0;
        res[5][3] = 0;
        res[5][4] = 0;
        res[5][5] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> source(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
        res[4][0] = 0;
        res[5][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> residual(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
        res[4][0] = 0;
        res[5][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> source_implicit(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
        res[4][0] = 0;
        res[5][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> interpolate(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& X)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(6, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = Q[2]*t0;
        Foam::scalar t2 = t0*(1 - 2*X.z());
        Foam::scalar t3 = Q[3]*t2 + t1;
        Foam::scalar t4 = Q[4]*t0;
        Foam::scalar t5 = Q[5]*t2 + t4;
        Foam::scalar t6 = -Foam::pow(X.z(), 2) + X.z();
        Foam::scalar t7 = t0*t6;
        res[0][0] = Q[0];
        res[1][0] = Q[1];
        res[2][0] = t3;
        res[3][0] = t5;
        res[4][0] = -Q[1]*(X.z()*Qaux[2] + Qaux[3]*t6) - Q[1]*(X.z()*Qaux[8] + Qaux[9]*t6) - Qaux[1]*(X.z()*t1 + Q[3]*t7) - Qaux[7]*(X.z()*t4 + Q[5]*t7) + t3*(X.z()*Qaux[1] + Qaux[0]) + t5*(X.z()*Qaux[7] + Qaux[6]);
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
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(6, 0.0));
    Foam::scalar t0 = 1.0*n.x();
        Foam::scalar t1 = 1.0*n.y();
        Foam::scalar t2 = Q[2]*t0 + Q[4]*t1;
        Foam::scalar t3 = -n.x()*t2;
        Foam::scalar t4 = Q[3]*t0 + Q[5]*t1;
        Foam::scalar t5 = -n.x()*t4;
        Foam::scalar t6 = -n.y()*t2;
        Foam::scalar t7 = -n.y()*t4;
        res[0][0] = Q[0];
        res[0][1] = Q[1];
        res[0][2] = t3;
        res[0][3] = t5;
        res[0][4] = t6;
        res[0][5] = t7;
        res[1][0] = Q[0];
        res[1][1] = Q[1];
        res[1][2] = t3;
        res[1][3] = t5;
        res[1][4] = t6;
        res[1][5] = t7;
        res[2][0] = Q[0];
        res[2][1] = Q[1];
        res[2][2] = t3;
        res[2][3] = t5;
        res[2][4] = t6;
        res[2][5] = t7;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> basis(
    const Foam::scalar& z)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(2, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 1;
        res[1][0] = 1 - 2*z;
    return res;
}
        
} // namespace Model
