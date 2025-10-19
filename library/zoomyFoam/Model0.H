#pragma once
#include "List.H"
#include "vector.H"
#include "scalar.H"

namespace Model
{
constexpr int n_dof_q    = 4;
constexpr int n_dof_qaux = 8;
constexpr int dimension  = 2;
const Foam::List<Foam::word> map_boundary_tag_to_function_index{ "wall", "inflow", "outflow" };

inline Foam::List<Foam::List<Foam::scalar>> flux_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        res[0][0] = 0;
        res[1][0] = Q[2];
        res[2][0] = (1.0/2.0)*1.0*9.81*Foam::pow(Q[1], 2) + Foam::pow(Q[2], 2)*t0;
        res[3][0] = Q[2]*Q[3]*t0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        res[0][0] = 0;
        res[1][0] = Q[3];
        res[2][0] = Q[2]*Q[3]*t0;
        res[3][0] = (1.0/2.0)*1.0*9.81*Foam::pow(Q[1], 2) + Foam::pow(Q[3], 2)*t0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_jacobian_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(4, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t2 = Q[2]*t1;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 1;
        res[1][3] = 0;
        res[2][0] = 0;
        res[2][1] = 1.0*9.81*Q[1] - Foam::pow(Q[2], 2)*t0;
        res[2][2] = 2*t2;
        res[2][3] = 0;
        res[3][0] = 0;
        res[3][1] = -Q[2]*Q[3]*t0;
        res[3][2] = Q[3]*t1;
        res[3][3] = t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_jacobian_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(4, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t2 = Q[3]*t1;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 1;
        res[2][0] = 0;
        res[2][1] = -Q[2]*Q[3]*t0;
        res[2][2] = t2;
        res[2][3] = Q[2]*t1;
        res[3][0] = 0;
        res[3][1] = 1.0*9.81*Q[1] - Foam::pow(Q[3], 2)*t0;
        res[3][2] = 0;
        res[3][3] = 2*t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(4, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(4, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(4, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t2 = Q[2]*t1;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 1;
        res[1][3] = 0;
        res[2][0] = 0;
        res[2][1] = 1.0*9.81*Q[1] - Foam::pow(Q[2], 2)*t0;
        res[2][2] = 2*t2;
        res[2][3] = 0;
        res[3][0] = 0;
        res[3][1] = -Q[2]*Q[3]*t0;
        res[3][2] = Q[3]*t1;
        res[3][3] = t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(4, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 2));
        Foam::scalar t1 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t2 = Q[3]*t1;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 1;
        res[2][0] = 0;
        res[2][1] = -Q[2]*Q[3]*t0;
        res[2][2] = t2;
        res[2][3] = Q[2]*t1;
        res[3][0] = 0;
        res[3][1] = 1.0*9.81*Q[1] - Foam::pow(Q[3], 2)*t0;
        res[3][2] = 0;
        res[3][3] = 2*t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> eigenvalues(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& n)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[1], 1));
        Foam::scalar t1 = n.x()*Q[2] + n.y()*Q[3];
        Foam::scalar t2 = Foam::pow(Q[1], 3.0/2.0)*Foam::pow(1.0*9.81, 1.0/2.0)*Foam::pow(Foam::pow(n.x(), 2) + Foam::pow(n.y(), 2), 1.0/2.0);
        res[0][0] = 0;
        res[1][0] = t0*t1;
        res[2][0] = t0*(t1 - t2);
        res[3][0] = t0*(t1 + t2);
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> left_eigenvectors(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(4, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> right_eigenvectors(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(4, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[0][3] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[1][3] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
        res[2][3] = 0;
        res[3][0] = 0;
        res[3][1] = 0;
        res[3][2] = 0;
        res[3][3] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> source(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> residual(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> source_implicit(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
        res[3][0] = 0;
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
        Foam::scalar t2 = Q[3]*t0;
        Foam::scalar t3 = X.z()*Q[1];
        Foam::scalar t4 = X.z()*Qaux[1];
        Foam::scalar t5 = X.z()*Qaux[5];
        res[0][0] = Q[0];
        res[1][0] = Q[1];
        res[2][0] = t1;
        res[3][0] = t2;
        res[4][0] = Q[2]*t0*(Qaux[0] + t4) + Q[3]*t0*(Qaux[4] + t5) - Qaux[2]*t3 - Qaux[6]*t3 - t1*t4 - t2*t5;
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
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(4, 0.0));
    Foam::scalar t0 = 1.0*n.x()*Q[2] + 1.0*n.y()*Q[3];
        Foam::scalar t1 = -n.x()*t0;
        Foam::scalar t2 = -n.y()*t0;
        res[0][0] = Q[0];
        res[0][1] = Q[1];
        res[0][2] = t1;
        res[0][3] = t2;
        res[1][0] = Q[0];
        res[1][1] = Q[1];
        res[1][2] = t1;
        res[1][3] = t2;
        res[2][0] = Q[0];
        res[2][1] = Q[1];
        res[2][2] = t1;
        res[2][3] = t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> basis(
    const Foam::scalar& z)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(1, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 1;
    return res;
}
        
} // namespace Model
