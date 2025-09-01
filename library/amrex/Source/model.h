
#pragma once
#include <AMReX_Array4.H>
#include <AMReX_Vector.H>

class Model {
public:
    static constexpr int n_dof_q    = 8;
    static constexpr int n_dof_qaux = 1;
    static constexpr int dimension  = 2;



    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1>
    flux_x ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,1>{};
        amrex::Real t0 = Qaux(0)*amrex::Math::powi<2>(Q(3));
        amrex::Real t1 = Qaux(0)*amrex::Math::powi<2>(Q(4));
        amrex::Real t2 = Qaux(0)*Q(2);
        amrex::Real t3 = 2*t2;
        amrex::Real t4 = Qaux(0)*Q(3);
        amrex::Real t5 = Q(6)*t4;
        amrex::Real t6 = Qaux(0)*Q(4);
        amrex::Real t7 = Q(7)*t6;
        amrex::Real t8 = Qaux(0)*Q(5);
        res(0,0) = 0;
        res(1,0) = Q(2);
        res(2,0) = (1.0/2.0)*1.0*9.81*amrex::Math::powi<2>(Q(1)) + Qaux(0)*amrex::Math::powi<2>(Q(2)) + (1.0/3.0)*t0 + (1.0/5.0)*t1;
        res(3,0) = Q(3)*t3 + (4.0/5.0)*Q(4)*t4;
        res(4,0) = Q(4)*t3 + (2.0/3.0)*t0 + (2.0/7.0)*t1;
        res(5,0) = Q(5)*t2 + (1.0/3.0)*t5 + (1.0/5.0)*t7;
        res(6,0) = Q(3)*t8 + Q(6)*t2 + (2.0/5.0)*Q(6)*t6 + (2.0/5.0)*Q(7)*t4;
        res(7,0) = Q(4)*t8 + Q(7)*t2 + (2.0/3.0)*t5 + (2.0/7.0)*t7;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1>
    flux_y ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,1>{};
        amrex::Real t0 = Qaux(0)*Q(2);
        amrex::Real t1 = Qaux(0)*Q(3);
        amrex::Real t2 = Q(6)*t1;
        amrex::Real t3 = Qaux(0)*Q(4);
        amrex::Real t4 = Q(7)*t3;
        amrex::Real t5 = Qaux(0)*Q(5);
        amrex::Real t6 = Qaux(0)*amrex::Math::powi<2>(Q(6));
        amrex::Real t7 = Qaux(0)*amrex::Math::powi<2>(Q(7));
        amrex::Real t8 = 2*t5;
        res(0,0) = 0;
        res(1,0) = Q(5);
        res(2,0) = Q(5)*t0 + (1.0/3.0)*t2 + (1.0/5.0)*t4;
        res(3,0) = Q(3)*t5 + Q(6)*t0 + (2.0/5.0)*Q(6)*t3 + (2.0/5.0)*Q(7)*t1;
        res(4,0) = Q(4)*t5 + Q(7)*t0 + (2.0/3.0)*t2 + (2.0/7.0)*t4;
        res(5,0) = (1.0/2.0)*1.0*9.81*amrex::Math::powi<2>(Q(1)) + Qaux(0)*amrex::Math::powi<2>(Q(5)) + (1.0/3.0)*t6 + (1.0/5.0)*t7;
        res(6,0) = (4.0/5.0)*Qaux(0)*Q(6)*Q(7) + Q(6)*t8;
        res(7,0) = Q(7)*t8 + (2.0/3.0)*t6 + (2.0/7.0)*t7;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1>
    flux_jacobian_x ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,1>{};
        amrex::Real t0 = Qaux(0)*amrex::Math::powi<2>(Q(3));
        amrex::Real t1 = Qaux(0)*amrex::Math::powi<2>(Q(4));
        amrex::Real t2 = Qaux(0)*Q(2);
        amrex::Real t3 = 2*t2;
        amrex::Real t4 = Qaux(0)*Q(3);
        amrex::Real t5 = Q(6)*t4;
        amrex::Real t6 = Qaux(0)*Q(4);
        amrex::Real t7 = Q(7)*t6;
        amrex::Real t8 = Qaux(0)*Q(5);
        res(0,0) = 0;
        res(1,0) = Q(2);
        res(2,0) = (1.0/2.0)*1.0*9.81*amrex::Math::powi<2>(Q(1)) + Qaux(0)*amrex::Math::powi<2>(Q(2)) + (1.0/3.0)*t0 + (1.0/5.0)*t1;
        res(3,0) = Q(3)*t3 + (4.0/5.0)*Q(4)*t4;
        res(4,0) = Q(4)*t3 + (2.0/3.0)*t0 + (2.0/7.0)*t1;
        res(5,0) = Q(5)*t2 + (1.0/3.0)*t5 + (1.0/5.0)*t7;
        res(6,0) = Q(3)*t8 + Q(6)*t2 + (2.0/5.0)*Q(6)*t6 + (2.0/5.0)*Q(7)*t4;
        res(7,0) = Q(4)*t8 + Q(7)*t2 + (2.0/3.0)*t5 + (2.0/7.0)*t7;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1>
    flux_jacobian_y ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,1>{};
        amrex::Real t0 = Qaux(0)*Q(2);
        amrex::Real t1 = Qaux(0)*Q(3);
        amrex::Real t2 = Q(6)*t1;
        amrex::Real t3 = Qaux(0)*Q(4);
        amrex::Real t4 = Q(7)*t3;
        amrex::Real t5 = Qaux(0)*Q(5);
        amrex::Real t6 = Qaux(0)*amrex::Math::powi<2>(Q(6));
        amrex::Real t7 = Qaux(0)*amrex::Math::powi<2>(Q(7));
        amrex::Real t8 = 2*t5;
        res(0,0) = 0;
        res(1,0) = Q(5);
        res(2,0) = Q(5)*t0 + (1.0/3.0)*t2 + (1.0/5.0)*t4;
        res(3,0) = Q(3)*t5 + Q(6)*t0 + (2.0/5.0)*Q(6)*t3 + (2.0/5.0)*Q(7)*t1;
        res(4,0) = Q(4)*t5 + Q(7)*t0 + (2.0/3.0)*t2 + (2.0/7.0)*t4;
        res(5,0) = (1.0/2.0)*1.0*9.81*amrex::Math::powi<2>(Q(1)) + Qaux(0)*amrex::Math::powi<2>(Q(5)) + (1.0/3.0)*t6 + (1.0/5.0)*t7;
        res(6,0) = (4.0/5.0)*Qaux(0)*Q(6)*Q(7) + Q(6)*t8;
        res(7,0) = Q(7)*t8 + (2.0/3.0)*t6 + (2.0/7.0)*t7;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,8>
    nonconservative_matrix_x ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,8>{};
        amrex::Real t0 = Qaux(0)*Q(2);
        amrex::Real t1 = Qaux(0)*Q(4);
        amrex::Real t2 = Qaux(0)*Q(3);
        amrex::Real t3 = Qaux(0)*Q(5);
        amrex::Real t4 = Qaux(0)*Q(7);
        amrex::Real t5 = Qaux(0)*Q(6);
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(0,4) = 0;
        res(0,5) = 0;
        res(0,6) = 0;
        res(0,7) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(1,4) = 0;
        res(1,5) = 0;
        res(1,6) = 0;
        res(1,7) = 0;
        res(2,0) = -1.0*9.81*Q(1);
        res(2,1) = 0;
        res(2,2) = 0;
        res(2,3) = 0;
        res(2,4) = 0;
        res(2,5) = 0;
        res(2,6) = 0;
        res(2,7) = 0;
        res(3,0) = 0;
        res(3,1) = 0;
        res(3,2) = 0;
        res(3,3) = -t0 - 1.0/5.0*t1;
        res(3,4) = (1.0/5.0)*t2;
        res(3,5) = 0;
        res(3,6) = 0;
        res(3,7) = 0;
        res(4,0) = 0;
        res(4,1) = 0;
        res(4,2) = 0;
        res(4,3) = -t2;
        res(4,4) = -t0 - 1.0/7.0*t1;
        res(4,5) = 0;
        res(4,6) = 0;
        res(4,7) = 0;
        res(5,0) = 0;
        res(5,1) = 0;
        res(5,2) = 0;
        res(5,3) = 0;
        res(5,4) = 0;
        res(5,5) = 0;
        res(5,6) = 0;
        res(5,7) = 0;
        res(6,0) = 0;
        res(6,1) = 0;
        res(6,2) = 0;
        res(6,3) = -t3 - 1.0/5.0*t4;
        res(6,4) = (1.0/5.0)*t5;
        res(6,5) = 0;
        res(6,6) = 0;
        res(6,7) = 0;
        res(7,0) = 0;
        res(7,1) = 0;
        res(7,2) = 0;
        res(7,3) = -t5;
        res(7,4) = -t3 - 1.0/7.0*t4;
        res(7,5) = 0;
        res(7,6) = 0;
        res(7,7) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,8>
    nonconservative_matrix_y ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,8>{};
        amrex::Real t0 = Qaux(0)*Q(2);
        amrex::Real t1 = Qaux(0)*Q(4);
        amrex::Real t2 = Qaux(0)*Q(3);
        amrex::Real t3 = Qaux(0)*Q(5);
        amrex::Real t4 = Qaux(0)*Q(7);
        amrex::Real t5 = Qaux(0)*Q(6);
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(0,4) = 0;
        res(0,5) = 0;
        res(0,6) = 0;
        res(0,7) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(1,4) = 0;
        res(1,5) = 0;
        res(1,6) = 0;
        res(1,7) = 0;
        res(2,0) = 0;
        res(2,1) = 0;
        res(2,2) = 0;
        res(2,3) = 0;
        res(2,4) = 0;
        res(2,5) = 0;
        res(2,6) = 0;
        res(2,7) = 0;
        res(3,0) = 0;
        res(3,1) = 0;
        res(3,2) = 0;
        res(3,3) = 0;
        res(3,4) = 0;
        res(3,5) = 0;
        res(3,6) = -t0 - 1.0/5.0*t1;
        res(3,7) = (1.0/5.0)*t2;
        res(4,0) = 0;
        res(4,1) = 0;
        res(4,2) = 0;
        res(4,3) = 0;
        res(4,4) = 0;
        res(4,5) = 0;
        res(4,6) = -t2;
        res(4,7) = -t0 - 1.0/7.0*t1;
        res(5,0) = -1.0*9.81*Q(1);
        res(5,1) = 0;
        res(5,2) = 0;
        res(5,3) = 0;
        res(5,4) = 0;
        res(5,5) = 0;
        res(5,6) = 0;
        res(5,7) = 0;
        res(6,0) = 0;
        res(6,1) = 0;
        res(6,2) = 0;
        res(6,3) = 0;
        res(6,4) = 0;
        res(6,5) = 0;
        res(6,6) = -t3 - 1.0/5.0*t4;
        res(6,7) = (1.0/5.0)*t5;
        res(7,0) = 0;
        res(7,1) = 0;
        res(7,2) = 0;
        res(7,3) = 0;
        res(7,4) = 0;
        res(7,5) = 0;
        res(7,6) = -t5;
        res(7,7) = -t3 - 1.0/7.0*t4;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,8>
    quasilinear_matrix_x ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,8>{};
        amrex::Real t0 = -1.0*9.81*Q(1);
        amrex::Real t1 = amrex::Math::powi<2>(Qaux(0));
        amrex::Real t2 = amrex::Math::powi<2>(Q(3))*t1;
        amrex::Real t3 = amrex::Math::powi<2>(Q(4))*t1;
        amrex::Real t4 = Qaux(0)*Q(2);
        amrex::Real t5 = Qaux(0)*Q(3);
        amrex::Real t6 = (2.0/3.0)*t5;
        amrex::Real t7 = Qaux(0)*Q(4);
        amrex::Real t8 = (2.0/5.0)*t7;
        amrex::Real t9 = Q(2)*t1;
        amrex::Real t10 = 2*t9;
        amrex::Real t11 = Q(3)*t1;
        amrex::Real t12 = (1.0/3.0)*t5;
        amrex::Real t13 = Q(6)*t11;
        amrex::Real t14 = Q(4)*t1;
        amrex::Real t15 = Q(7)*t14;
        amrex::Real t16 = Qaux(0)*Q(6);
        amrex::Real t17 = (1.0/3.0)*t16;
        amrex::Real t18 = Qaux(0)*Q(7);
        amrex::Real t19 = (1.0/5.0)*t18;
        amrex::Real t20 = Q(5)*t1;
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(0,4) = 0;
        res(0,5) = 0;
        res(0,6) = 0;
        res(0,7) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 1;
        res(1,3) = 0;
        res(1,4) = 0;
        res(1,5) = 0;
        res(1,6) = 0;
        res(1,7) = 0;
        res(2,0) = t0;
        res(2,1) = -amrex::Math::powi<2>(Q(2))*t1 - t0 - 1.0/3.0*t2 - 1.0/5.0*t3;
        res(2,2) = 2*t4;
        res(2,3) = t6;
        res(2,4) = t8;
        res(2,5) = 0;
        res(2,6) = 0;
        res(2,7) = 0;
        res(3,0) = 0;
        res(3,1) = -Q(3)*t10 - 4.0/5.0*Q(4)*t11;
        res(3,2) = 2*t5;
        res(3,3) = t4 + (3.0/5.0)*t7;
        res(3,4) = t5;
        res(3,5) = 0;
        res(3,6) = 0;
        res(3,7) = 0;
        res(4,0) = 0;
        res(4,1) = -Q(4)*t10 - 2.0/3.0*t2 - 2.0/7.0*t3;
        res(4,2) = 2*t7;
        res(4,3) = t12;
        res(4,4) = t4 + (3.0/7.0)*t7;
        res(4,5) = 0;
        res(4,6) = 0;
        res(4,7) = 0;
        res(5,0) = 0;
        res(5,1) = -Q(5)*t9 - 1.0/3.0*t13 - 1.0/5.0*t15;
        res(5,2) = Qaux(0)*Q(5);
        res(5,3) = t17;
        res(5,4) = t19;
        res(5,5) = t4;
        res(5,6) = t12;
        res(5,7) = (1.0/5.0)*t7;
        res(6,0) = 0;
        res(6,1) = -Q(3)*t20 - 2.0/5.0*Q(6)*t14 - Q(6)*t9 - 2.0/5.0*Q(7)*t11;
        res(6,2) = t16;
        res(6,3) = t19;
        res(6,4) = (3.0/5.0)*t16;
        res(6,5) = t5;
        res(6,6) = t4 + t8;
        res(6,7) = (2.0/5.0)*t5;
        res(7,0) = 0;
        res(7,1) = -Q(4)*t20 - Q(7)*t9 - 2.0/3.0*t13 - 2.0/7.0*t15;
        res(7,2) = t18;
        res(7,3) = -t17;
        res(7,4) = (1.0/7.0)*t18;
        res(7,5) = t7;
        res(7,6) = t6;
        res(7,7) = t4 + (2.0/7.0)*t7;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,8>
    quasilinear_matrix_y ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,8>{};
        amrex::Real t0 = amrex::Math::powi<2>(Qaux(0));
        amrex::Real t1 = Q(2)*t0;
        amrex::Real t2 = Q(3)*t0;
        amrex::Real t3 = Q(6)*t2;
        amrex::Real t4 = Q(4)*t0;
        amrex::Real t5 = Q(7)*t4;
        amrex::Real t6 = Qaux(0)*Q(5);
        amrex::Real t7 = Qaux(0)*Q(6);
        amrex::Real t8 = (1.0/3.0)*t7;
        amrex::Real t9 = Qaux(0)*Q(7);
        amrex::Real t10 = Qaux(0)*Q(3);
        amrex::Real t11 = (1.0/3.0)*t10;
        amrex::Real t12 = Qaux(0)*Q(4);
        amrex::Real t13 = (1.0/5.0)*t12;
        amrex::Real t14 = Q(5)*t0;
        amrex::Real t15 = (2.0/5.0)*t9;
        amrex::Real t16 = (2.0/3.0)*t7;
        amrex::Real t17 = -1.0*9.81*Q(1);
        amrex::Real t18 = amrex::Math::powi<2>(Q(6))*t0;
        amrex::Real t19 = amrex::Math::powi<2>(Q(7))*t0;
        amrex::Real t20 = 2*t14;
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(0,4) = 0;
        res(0,5) = 0;
        res(0,6) = 0;
        res(0,7) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(1,4) = 0;
        res(1,5) = 1;
        res(1,6) = 0;
        res(1,7) = 0;
        res(2,0) = 0;
        res(2,1) = -Q(5)*t1 - 1.0/3.0*t3 - 1.0/5.0*t5;
        res(2,2) = t6;
        res(2,3) = t8;
        res(2,4) = (1.0/5.0)*t9;
        res(2,5) = Qaux(0)*Q(2);
        res(2,6) = t11;
        res(2,7) = t13;
        res(3,0) = 0;
        res(3,1) = -Q(3)*t14 - Q(6)*t1 - 2.0/5.0*Q(6)*t4 - 2.0/5.0*Q(7)*t2;
        res(3,2) = t7;
        res(3,3) = t15 + t6;
        res(3,4) = (2.0/5.0)*t7;
        res(3,5) = t10;
        res(3,6) = t13;
        res(3,7) = (3.0/5.0)*t10;
        res(4,0) = 0;
        res(4,1) = -Q(4)*t14 - Q(7)*t1 - 2.0/3.0*t3 - 2.0/7.0*t5;
        res(4,2) = t9;
        res(4,3) = t16;
        res(4,4) = t6 + (2.0/7.0)*t9;
        res(4,5) = t12;
        res(4,6) = -t11;
        res(4,7) = (1.0/7.0)*t12;
        res(5,0) = t17;
        res(5,1) = -amrex::Math::powi<2>(Q(5))*t0 - t17 - 1.0/3.0*t18 - 1.0/5.0*t19;
        res(5,2) = 0;
        res(5,3) = 0;
        res(5,4) = 0;
        res(5,5) = 2*t6;
        res(5,6) = t16;
        res(5,7) = t15;
        res(6,0) = 0;
        res(6,1) = -4.0/5.0*Q(6)*Q(7)*t0 - Q(6)*t20;
        res(6,2) = 0;
        res(6,3) = 0;
        res(6,4) = 0;
        res(6,5) = 2*t7;
        res(6,6) = t6 + (3.0/5.0)*t9;
        res(6,7) = t7;
        res(7,0) = 0;
        res(7,1) = -Q(7)*t20 - 2.0/3.0*t18 - 2.0/7.0*t19;
        res(7,2) = 0;
        res(7,3) = 0;
        res(7,4) = 0;
        res(7,5) = 2*t9;
        res(7,6) = t8;
        res(7,7) = t6 + (3.0/7.0)*t9;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1>
    eigenvalues ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
    amrex::SmallMatrix<amrex::Real,2,1> const& normal) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,1>{};
        amrex::Real t0 = normal(0)*Q(2) + normal(1)*Q(5);
        amrex::Real t1 = Q(1) > 1.0e-8;
        amrex::Real t2 = (1.0/3.0)*std::pow(3, 1.0/2.0);
        amrex::Real t3 = normal(0)*Q(3);
        amrex::Real t4 = t2*t3;
        amrex::Real t5 = normal(1)*Q(6);
        amrex::Real t6 = t2*t5;
        amrex::Real t7 = (1.0/5.0)*std::pow(15, 1.0/2.0);
        amrex::Real t8 = t3*t7;
        amrex::Real t9 = t5*t7;
        amrex::Real t10 = amrex::Math::powi<2>(normal(0));
        amrex::Real t11 = amrex::Math::powi<2>(normal(1));
        amrex::Real t12 = 1.0*9.81*amrex::Math::powi<3>(Q(1));
        amrex::Real t13 = std::pow(amrex::Math::powi<2>(Q(3))*t10 + amrex::Math::powi<2>(Q(6))*t11 + t10*t12 + t11*t12 + 2*t3*t5, 1.0/2.0);
        res(0,0) = 0;
        res(1,0) = ((t1) ? (
   Qaux(0)*t0
)
: (
   0
));
        res(2,0) = ((t1) ? (
   Qaux(0)*(t0 - t4 - t6)
)
: (
   0
));
        res(3,0) = ((t1) ? (
   Qaux(0)*(t0 + t4 + t6)
)
: (
   0
));
        res(4,0) = ((t1) ? (
   Qaux(0)*(t0 - t8 - t9)
)
: (
   0
));
        res(5,0) = ((t1) ? (
   Qaux(0)*(t0 + t8 + t9)
)
: (
   0
));
        res(6,0) = ((t1) ? (
   Qaux(0)*(t0 + t13)
)
: (
   0
));
        res(7,0) = ((t1) ? (
   Qaux(0)*(t0 - t13)
)
: (
   0
));
        return res;

    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,8>
    left_eigenvectors ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,8>{};
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(0,4) = 0;
        res(0,5) = 0;
        res(0,6) = 0;
        res(0,7) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(1,4) = 0;
        res(1,5) = 0;
        res(1,6) = 0;
        res(1,7) = 0;
        res(2,0) = 0;
        res(2,1) = 0;
        res(2,2) = 0;
        res(2,3) = 0;
        res(2,4) = 0;
        res(2,5) = 0;
        res(2,6) = 0;
        res(2,7) = 0;
        res(3,0) = 0;
        res(3,1) = 0;
        res(3,2) = 0;
        res(3,3) = 0;
        res(3,4) = 0;
        res(3,5) = 0;
        res(3,6) = 0;
        res(3,7) = 0;
        res(4,0) = 0;
        res(4,1) = 0;
        res(4,2) = 0;
        res(4,3) = 0;
        res(4,4) = 0;
        res(4,5) = 0;
        res(4,6) = 0;
        res(4,7) = 0;
        res(5,0) = 0;
        res(5,1) = 0;
        res(5,2) = 0;
        res(5,3) = 0;
        res(5,4) = 0;
        res(5,5) = 0;
        res(5,6) = 0;
        res(5,7) = 0;
        res(6,0) = 0;
        res(6,1) = 0;
        res(6,2) = 0;
        res(6,3) = 0;
        res(6,4) = 0;
        res(6,5) = 0;
        res(6,6) = 0;
        res(6,7) = 0;
        res(7,0) = 0;
        res(7,1) = 0;
        res(7,2) = 0;
        res(7,3) = 0;
        res(7,4) = 0;
        res(7,5) = 0;
        res(7,6) = 0;
        res(7,7) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,8>
    right_eigenvectors ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,8>{};
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(0,4) = 0;
        res(0,5) = 0;
        res(0,6) = 0;
        res(0,7) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(1,4) = 0;
        res(1,5) = 0;
        res(1,6) = 0;
        res(1,7) = 0;
        res(2,0) = 0;
        res(2,1) = 0;
        res(2,2) = 0;
        res(2,3) = 0;
        res(2,4) = 0;
        res(2,5) = 0;
        res(2,6) = 0;
        res(2,7) = 0;
        res(3,0) = 0;
        res(3,1) = 0;
        res(3,2) = 0;
        res(3,3) = 0;
        res(3,4) = 0;
        res(3,5) = 0;
        res(3,6) = 0;
        res(3,7) = 0;
        res(4,0) = 0;
        res(4,1) = 0;
        res(4,2) = 0;
        res(4,3) = 0;
        res(4,4) = 0;
        res(4,5) = 0;
        res(4,6) = 0;
        res(4,7) = 0;
        res(5,0) = 0;
        res(5,1) = 0;
        res(5,2) = 0;
        res(5,3) = 0;
        res(5,4) = 0;
        res(5,5) = 0;
        res(5,6) = 0;
        res(5,7) = 0;
        res(6,0) = 0;
        res(6,1) = 0;
        res(6,2) = 0;
        res(6,3) = 0;
        res(6,4) = 0;
        res(6,5) = 0;
        res(6,6) = 0;
        res(6,7) = 0;
        res(7,0) = 0;
        res(7,1) = 0;
        res(7,2) = 0;
        res(7,3) = 0;
        res(7,4) = 0;
        res(7,5) = 0;
        res(7,6) = 0;
        res(7,7) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1>
    source ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,1>{};
        amrex::Real t0 = 9.81*Q(1);
        amrex::Real t1 = amrex::Math::powi<2>(Qaux(0))*1e-06;
        amrex::Real t2 = 12*t1;
        amrex::Real t3 = 0.03333333333333333/(0.001*1000.0);
        amrex::Real t4 = t3*(Qaux(0)*Q(2) + Qaux(0)*Q(3) + Qaux(0)*Q(4));
        amrex::Real t5 = 60*t1;
        amrex::Real t6 = t3*(Qaux(0)*Q(5) + Qaux(0)*Q(6) + Qaux(0)*Q(7));
        res(0,0) = 0;
        res(1,0) = 0;
        res(2,0) = 0.0*t0;
        res(3,0) = -Q(3)*t2 - 3.0*t4;
        res(4,0) = -Q(4)*t5 - 5.0*t4;
        res(5,0) = 0.0*t0;
        res(6,0) = -Q(6)*t2 - 3.0*t6;
        res(7,0) = -Q(7)*t5 - 5.0*t6;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1>
    residual ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,1>{};
        res(0,0) = 0;
        res(1,0) = 0;
        res(2,0) = 0;
        res(3,0) = 0;
        res(4,0) = 0;
        res(5,0) = 0;
        res(6,0) = 0;
        res(7,0) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1>
    source_implicit ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,8,1>{};
        res(0,0) = 0;
        res(1,0) = 0;
        res(2,0) = 0;
        res(3,0) = 0;
        res(4,0) = 0;
        res(5,0) = 0;
        res(6,0) = 0;
        res(7,0) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1>
    interpolate_3d ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
    amrex::SmallMatrix<amrex::Real,3,1> const& X) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,6,1>{};
        amrex::Real t0 = 2*X(2) - 1;
        amrex::Real t1 = -Qaux(0)*t0;
        amrex::Real t2 = Qaux(0)*(1.5*amrex::Math::powi<2>(t0) - 0.5);
        res(0,0) = Q(0);
        res(1,0) = Q(1);
        res(2,0) = Qaux(0)*Q(2) + Q(3)*t1 + Q(4)*t2;
        res(3,0) = Qaux(0)*Q(5) + Q(6)*t1 + Q(7)*t2;
        res(4,0) = 0;
        res(5,0) = 9.81*Q(1)*1000.0*(1 - X(2));
        return res;
    }

        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,8>
    boundary_conditions ( amrex::SmallMatrix<amrex::Real,8,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
    amrex::SmallMatrix<amrex::Real,2,1> const& normal, 
    amrex::SmallMatrix<amrex::Real,3,1> const& position,
    amrex::Real const& time,
    amrex::Real const& dX) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,8>{};
        res(0,0) = Q(0);
        res(0,1) = Q(1);
        res(0,2) = Q(2);
        res(0,3) = Q(3);
        res(0,4) = Q(4);
        res(0,5) = Q(5);
        res(0,6) = Q(6);
        res(0,7) = Q(7);
        res(1,0) = Q(0);
        res(1,1) = Q(1);
        res(1,2) = Q(2);
        res(1,3) = Q(3);
        res(1,4) = Q(4);
        res(1,5) = Q(5);
        res(1,6) = Q(6);
        res(1,7) = Q(7);
        res(2,0) = Q(0);
        res(2,1) = Q(1);
        res(2,2) = Q(2);
        res(2,3) = Q(3);
        res(2,4) = Q(4);
        res(2,5) = Q(5);
        res(2,6) = Q(6);
        res(2,7) = Q(7);
        res(3,0) = Q(0);
        res(3,1) = Q(1);
        res(3,2) = Q(2);
        res(3,3) = Q(3);
        res(3,4) = Q(4);
        res(3,5) = Q(5);
        res(3,6) = Q(6);
        res(3,7) = Q(7);
        return res;

    }
        
};
                