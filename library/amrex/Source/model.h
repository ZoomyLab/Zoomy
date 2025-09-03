
#pragma once
#include <AMReX_Array4.H>
#include <AMReX_Vector.H>

class Model {
public:
    static constexpr int n_dof_q    = 4;
    static constexpr int n_dof_qaux = 1;
    static constexpr int dimension  = 2;



    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1>
    flux_x ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,1>{};
        res(0,0) = 0;
        res(1,0) = Q(2);
        res(2,0) = (1.0/2.0)*1.0*9.81*amrex::Math::powi<2>(Q(1)) + Qaux(0)*amrex::Math::powi<2>(Q(2));
        res(3,0) = Qaux(0)*Q(2)*Q(3);
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1>
    flux_y ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,1>{};
        res(0,0) = 0;
        res(1,0) = Q(3);
        res(2,0) = Qaux(0)*Q(2)*Q(3);
        res(3,0) = (1.0/2.0)*1.0*9.81*amrex::Math::powi<2>(Q(1)) + Qaux(0)*amrex::Math::powi<2>(Q(3));
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1>
    flux_jacobian_x ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,1>{};
        res(0,0) = 0;
        res(1,0) = Q(2);
        res(2,0) = (1.0/2.0)*1.0*9.81*amrex::Math::powi<2>(Q(1)) + Qaux(0)*amrex::Math::powi<2>(Q(2));
        res(3,0) = Qaux(0)*Q(2)*Q(3);
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1>
    flux_jacobian_y ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,1>{};
        res(0,0) = 0;
        res(1,0) = Q(3);
        res(2,0) = Qaux(0)*Q(2)*Q(3);
        res(3,0) = (1.0/2.0)*1.0*9.81*amrex::Math::powi<2>(Q(1)) + Qaux(0)*amrex::Math::powi<2>(Q(3));
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,4>
    nonconservative_matrix_x ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,4>{};
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(2,0) = 1.0*9.81*Q(1);
        res(2,1) = 0;
        res(2,2) = 0;
        res(2,3) = 0;
        res(3,0) = 0;
        res(3,1) = 0;
        res(3,2) = 0;
        res(3,3) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,4>
    nonconservative_matrix_y ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,4>{};
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(2,0) = 0;
        res(2,1) = 0;
        res(2,2) = 0;
        res(2,3) = 0;
        res(3,0) = 1.0*9.81*Q(1);
        res(3,1) = 0;
        res(3,2) = 0;
        res(3,3) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,4>
    quasilinear_matrix_x ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,4>{};
        amrex::Real t0 = 1.0*9.81*Q(1);
        amrex::Real t1 = amrex::Math::powi<2>(Qaux(0));
        amrex::Real t2 = Qaux(0)*Q(2);
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 1;
        res(1,3) = 0;
        res(2,0) = t0;
        res(2,1) = -amrex::Math::powi<2>(Q(2))*t1 + t0;
        res(2,2) = 2*t2;
        res(2,3) = 0;
        res(3,0) = 0;
        res(3,1) = -Q(2)*Q(3)*t1;
        res(3,2) = Qaux(0)*Q(3);
        res(3,3) = t2;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,4>
    quasilinear_matrix_y ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,4>{};
        amrex::Real t0 = amrex::Math::powi<2>(Qaux(0));
        amrex::Real t1 = Qaux(0)*Q(3);
        amrex::Real t2 = 1.0*9.81*Q(1);
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 1;
        res(2,0) = 0;
        res(2,1) = -Q(2)*Q(3)*t0;
        res(2,2) = t1;
        res(2,3) = Qaux(0)*Q(2);
        res(3,0) = t2;
        res(3,1) = -amrex::Math::powi<2>(Q(3))*t0 + t2;
        res(3,2) = 0;
        res(3,3) = 2*t1;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1>
    eigenvalues ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
    amrex::SmallMatrix<amrex::Real,2,1> const& normal) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,1>{};
        amrex::Real t0 = normal(0)*Q(2);
        amrex::Real t1 = normal(1)*Q(3);
        amrex::Real t2 = amrex::Math::powi<2>(Qaux(0));
        amrex::Real t3 = std::pow(1.0*9.81*amrex::Math::powi<5>(Q(1)), 1.0/2.0)*std::pow(amrex::Math::powi<2>(normal(0)) + amrex::Math::powi<2>(normal(1)), 1.0/2.0);
        amrex::Real t4 = Q(1)*t0 + Q(1)*t1;
        res(0,0) = 0;
        res(1,0) = Qaux(0)*(t0 + t1);
        res(2,0) = t2*(t3 + t4);
        res(3,0) = t2*(-t3 + t4);
        return res;

    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,4>
    left_eigenvectors ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,4>{};
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(2,0) = 0;
        res(2,1) = 0;
        res(2,2) = 0;
        res(2,3) = 0;
        res(3,0) = 0;
        res(3,1) = 0;
        res(3,2) = 0;
        res(3,3) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,4>
    right_eigenvectors ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,4>{};
        res(0,0) = 0;
        res(0,1) = 0;
        res(0,2) = 0;
        res(0,3) = 0;
        res(1,0) = 0;
        res(1,1) = 0;
        res(1,2) = 0;
        res(1,3) = 0;
        res(2,0) = 0;
        res(2,1) = 0;
        res(2,2) = 0;
        res(2,3) = 0;
        res(3,0) = 0;
        res(3,1) = 0;
        res(3,2) = 0;
        res(3,3) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1>
    source ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,1>{};
        res(0,0) = 0;
        res(1,0) = 0;
        res(2,0) = 0;
        res(3,0) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1>
    residual ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,1>{};
        res(0,0) = 0;
        res(1,0) = 0;
        res(2,0) = 0;
        res(3,0) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1>
    source_implicit ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,1>{};
        res(0,0) = 0;
        res(1,0) = 0;
        res(2,0) = 0;
        res(3,0) = 0;
        return res;
    }
        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1>
    interpolate_3d ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
    amrex::SmallMatrix<amrex::Real,3,1> const& X) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,6,1>{};
        res(0,0) = Q(0);
        res(1,0) = Q(1);
        res(2,0) = Qaux(0)*Q(2);
        res(3,0) = Qaux(0)*Q(3);
        res(4,0) = 0;
        res(5,0) = 9.81*Q(1)*1000.0*(1 - X(2));
        return res;
    }

        


    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,4>
    boundary_conditions ( amrex::SmallMatrix<amrex::Real,4,1> const& Q,
    amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
    amrex::SmallMatrix<amrex::Real,2,1> const& normal, 
    amrex::SmallMatrix<amrex::Real,3,1> const& position,
    amrex::Real const& time,
    amrex::Real const& dX) noexcept
    {
        auto res = amrex::SmallMatrix<amrex::Real,4,4>{};
        res(0,0) = Q(0);
        res(0,1) = Q(1);
        res(0,2) = Q(2);
        res(0,3) = Q(3);
        res(1,0) = Q(0);
        res(1,1) = Q(1);
        res(1,2) = Q(2);
        res(1,3) = Q(3);
        res(2,0) = Q(0);
        res(2,1) = Q(1);
        res(2,2) = Q(2);
        res(2,3) = Q(3);
        res(3,0) = Q(0);
        res(3,1) = Q(1);
        res(3,2) = Q(2);
        res(3,3) = Q(3);
        return res;

    }
        
};
                