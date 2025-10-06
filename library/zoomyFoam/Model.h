#pragma once
#include "List.H"
#include "vector.H"
#include "scalar.H"

namespace Model
{
constexpr int n_dof_q    = 3;
constexpr int n_dof_qaux = 2;
constexpr int dimension  = 2;

inline Foam::List<Foam::List<Foam::scalar>> flux_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[0], 1));
        res[0][0] = Q[1];
        res[1][0] = (1.0/2.0)*9.81*Foam::pow(Q[0], 2) + Foam::pow(Q[1], 2)*t0;
        res[2][0] = Q[1]*Q[2]*t0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[0], 1));
        res[0][0] = Q[2];
        res[1][0] = Q[1]*Q[2]*t0;
        res[2][0] = (1.0/2.0)*9.81*Foam::pow(Q[0], 2) + Foam::pow(Q[2], 2)*t0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_jacobian_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[0], 2));
        Foam::scalar t1 = (1.0 / Foam::pow(Q[0], 1));
        Foam::scalar t2 = Q[1]*t1;
        res[0][0] = 0;
        res[0][1] = 1;
        res[0][2] = 0;
        res[1][0] = 9.81*Q[0] - Foam::pow(Q[1], 2)*t0;
        res[1][1] = 2*t2;
        res[1][2] = 0;
        res[2][0] = -Q[1]*Q[2]*t0;
        res[2][1] = Q[2]*t1;
        res[2][2] = t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> flux_jacobian_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[0], 2));
        Foam::scalar t1 = (1.0 / Foam::pow(Q[0], 1));
        Foam::scalar t2 = Q[2]*t1;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 1;
        res[1][0] = -Q[1]*Q[2]*t0;
        res[1][1] = t2;
        res[1][2] = Q[1]*t1;
        res[2][0] = 9.81*Q[0] - Foam::pow(Q[2], 2)*t0;
        res[2][1] = 0;
        res[2][2] = 2*t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> nonconservative_matrix_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_x(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[0], 2));
        Foam::scalar t1 = (1.0 / Foam::pow(Q[0], 1));
        Foam::scalar t2 = Q[1]*t1;
        res[0][0] = 0;
        res[0][1] = 1;
        res[0][2] = 0;
        res[1][0] = 9.81*Q[0] - Foam::pow(Q[1], 2)*t0;
        res[1][1] = 2*t2;
        res[1][2] = 0;
        res[2][0] = -Q[1]*Q[2]*t0;
        res[2][1] = Q[2]*t1;
        res[2][2] = t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> quasilinear_matrix_y(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[0], 2));
        Foam::scalar t1 = (1.0 / Foam::pow(Q[0], 1));
        Foam::scalar t2 = Q[2]*t1;
        res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 1;
        res[1][0] = -Q[1]*Q[2]*t0;
        res[1][1] = t2;
        res[1][2] = Q[1]*t1;
        res[2][0] = 9.81*Q[0] - Foam::pow(Q[2], 2)*t0;
        res[2][1] = 0;
        res[2][2] = 2*t2;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> eigenvalues(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& n)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = n.x()*Q[1];
        Foam::scalar t1 = n.y()*Q[2];
        Foam::scalar t2 = (1.0 / Foam::pow(Q[0], 2));
        Foam::scalar t3 = Foam::pow(9.81*Foam::pow(Q[0], 5), 1.0/2.0)*Foam::pow(Foam::pow(n.x(), 2) + Foam::pow(n.y(), 2), 1.0/2.0);
        Foam::scalar t4 = Q[0]*t0 + Q[0]*t1;
        res[0][0] = (t0 + t1)/Q[0];
        res[1][0] = t2*(t3 + t4);
        res[2][0] = t2*(-t3 + t4);
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> left_eigenvectors(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> right_eigenvectors(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(3, 0.0));
    res[0][0] = 0;
        res[0][1] = 0;
        res[0][2] = 0;
        res[1][0] = 0;
        res[1][1] = 0;
        res[1][2] = 0;
        res[2][0] = 0;
        res[2][1] = 0;
        res[2][2] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> source(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> residual(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> source_implicit(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(3, Foam::List<Foam::scalar>(1, 0.0));
    res[0][0] = 0;
        res[1][0] = 0;
        res[2][0] = 0;
    return res;
}
        

inline Foam::List<Foam::List<Foam::scalar>> interpolate(
    const Foam::List<Foam::scalar>& Q,
    const Foam::List<Foam::scalar>& Qaux,
    const Foam::vector& X)
{
    auto res = Foam::List<Foam::List<Foam::scalar>>(5, Foam::List<Foam::scalar>(1, 0.0));
    Foam::scalar t0 = (1.0 / Foam::pow(Q[0], 1));
        res[0][0] = Q[0];
        res[1][0] = Q[1]*t0;
        res[2][0] = Q[2]*t0;
        res[3][0] = 0;
        res[4][0] = 9810.0*Q[0]*(1 - X.z());
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
    auto res = Foam::List<Foam::List<Foam::scalar>>(4, Foam::List<Foam::scalar>(3, 0.0));
    Foam::scalar t0 = n.y()*Q[2];
        Foam::scalar t1 = 1.0*n.x()*Q[1] + 1.0*t0;
        Foam::scalar t2 = 1.0*n.x()*Q[1] + 1.0*t0;
        Foam::scalar t3 = -n.x()*t1 - n.x()*t2 + 1.0*Q[1];
        Foam::scalar t4 = -n.y()*t1 - n.y()*t2 + 1.0*Q[2];
        res[0][0] = Q[0];
        res[0][1] = t3;
        res[0][2] = t4;
        res[1][0] = Q[0];
        res[1][1] = t3;
        res[1][2] = t4;
        res[2][0] = Q[0];
        res[2][1] = t3;
        res[2][2] = t4;
        res[3][0] = Q[0];
        res[3][1] = t3;
        res[3][2] = t4;
    return res;
}
        
} // namespace Model
