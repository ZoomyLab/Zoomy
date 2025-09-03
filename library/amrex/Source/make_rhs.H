#ifndef MAKE_RHS
#define MAKE_RHS

#include "AMReX_Array4.H"
#include "constants.H"
#include "riemann.H"
#include "model.h"

/**
 * Define rhs on a cell by cell basis
 */

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
VecQ
make_flux (int i, int j, amrex::Array4<amrex::Real const> const& sol_array, amrex::Array4<amrex::Real const> const& sol_array_aux, amrex::Real dx, amrex::Real dy, amrex::Real dt)
{
    int n_dof = Model::n_dof_q;
    int n_dof_aux = Model::n_dof_qaux;
    VecQ Qij, Qx_lo, Qx_hi, Qy_lo, Qy_hi;
    VecQaux Qauxij, Qauxx_lo, Qauxx_hi, Qauxy_lo, Qauxy_hi;
    for (int k=0;k<n_dof; ++k) {
        Qij(k,0)   = sol_array(i,j,0,k);
        Qx_lo(k,0) = sol_array(i-1,j,0,k);
        Qx_hi(k,0) = sol_array(i+1,j,0,k);
        Qy_lo(k,0) = sol_array(i,j-1,0,k);
        Qy_hi(k,0) = sol_array(i,j+1,0,k);
    }

    for (int k=0;k<n_dof_aux; ++k) {
        Qauxij(k,0)   = sol_array_aux(i,j,0,k);
        Qauxx_lo(k,0) = sol_array_aux(i-1,j,0,k);
        Qauxx_hi(k,0) = sol_array_aux(i+1,j,0,k);
        Qauxy_lo(k,0) = sol_array_aux(i,j-1,0,k);
        Qauxy_hi(k,0) = sol_array_aux(i,j+1,0,k);
    }

    Vec2 normal_xp = makeSmallMatrix<2, 1>({+1., 0.});
    Vec2 normal_xm = makeSmallMatrix<2, 1>({-1., 0.});
    Vec2 normal_yp = makeSmallMatrix<2, 1>({0., +1.});
    Vec2 normal_ym = makeSmallMatrix<2, 1>({0., -1.});

    // VecQ fx_lo = rusanov(Qij, Qx_hi, Qauxij, Qauxx_hi, normal_xp, +1, dx*dy, dx*dy, dy, Model::quasilinear_matrix_x, dt);
    // VecQ fx_hi = rusanov(Qij, Qx_lo, Qauxij, Qauxx_lo, normal_xm, -1, dx*dy, dx*dy, dy, Model::quasilinear_matrix_x, dt);
    // VecQ fy_lo = rusanov(Qij, Qy_hi, Qauxij, Qauxy_hi, normal_yp, +1, dx*dy, dx*dy, dx, Model::quasilinear_matrix_y, dt);
    // VecQ fy_hi = rusanov(Qij, Qy_lo, Qauxij, Qauxy_lo, normal_ym, -1, dx*dy, dx*dy, dx, Model::quasilinear_matrix_y, dt);

    // VecQ fx_lo = price_c(Qij, Qx_hi, Qauxij, Qauxx_hi, normal_xp, +1, dx*dy, dx*dy, dy, Model::quasilinear_matrix_x, dt);
    // VecQ fx_hi = price_c(Qij, Qx_lo, Qauxij, Qauxx_lo, normal_xm, -1, dx*dy, dx*dy, dy, Model::quasilinear_matrix_x, dt);
    // VecQ fy_lo = price_c(Qij, Qy_hi, Qauxij, Qauxy_hi, normal_yp, +1, dx*dy, dx*dy, dx, Model::quasilinear_matrix_y, dt);
    // VecQ fy_hi = price_c(Qij, Qy_lo, Qauxij, Qauxy_lo, normal_ym, -1, dx*dy, dx*dy, dx, Model::quasilinear_matrix_y, dt);

    // https://doi.org/10.1090%2FS0025-5718-09-02317-5
    VecQ fx_lo = rusanov(Qx_hi,Qij, Qauxx_hi, Qauxij, normal_xp, -1, dx*dy, dx*dy, dy, Model::quasilinear_matrix_x, dt);
    VecQ fx_hi = rusanov(Qij, Qx_lo, Qauxij, Qauxx_lo, normal_xm, +1, dx*dy, dx*dy, dy, Model::quasilinear_matrix_x, dt);
    VecQ fy_lo = rusanov(Qy_hi, Qij, Qauxy_hi, Qauxij, normal_yp, -1, dx*dy, dx*dy, dx, Model::quasilinear_matrix_y, dt);
    VecQ fy_hi = rusanov(Qij, Qy_lo, Qauxij, Qauxy_lo, normal_ym, +1, dx*dy, dx*dy, dx, Model::quasilinear_matrix_y, dt);

    VecQ rhs = -(fx_hi + fx_lo) - (fy_hi + fy_lo) ;


    return rhs;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
VecQ
make_rhs (int i, int j, amrex::Array4<amrex::Real const> const& sol_array, amrex::Array4<amrex::Real const> const& sol_array_aux, amrex::Real dx, amrex::Real dy, amrex::Real dt)
{
    int n_dof = Model::n_dof_q;
    int n_dof_aux = Model::n_dof_qaux;
    VecQ Qij;
    VecQaux Qauxij;
    for (int k=0;k<n_dof; ++k) {
        Qij(k,0)   = sol_array(i,j,0,k);
    }

    for (int k=0;k<n_dof_aux; ++k) {
        Qauxij(k,0)   = sol_array_aux(i,j,0,k);
    }
    VecQ rhs = VecQ::Zero();
    rhs = Model::source(Qij, Qauxij);
    return rhs;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
VecQ
make_rhs_explicit (int i, int j, amrex::Array4<amrex::Real const> const& sol_array, amrex::Array4<amrex::Real const> const& sol_array_aux, amrex::Real dx, amrex::Real dy, amrex::Real dt)
{
    int n_dof = Model::n_dof_q;
    int n_dof_aux = Model::n_dof_qaux;
    VecQ Qij;
    VecQaux Qauxij;
    for (int k=0;k<n_dof; ++k) {
        Qij(k,0)   = sol_array(i,j,0,k);
    }

    for (int k=0;k<n_dof_aux; ++k) {
        Qauxij(k,0)   = sol_array_aux(i,j,0,k);
    }
    VecQ rhs = Qij;
    return rhs;
    amrex::Real h = Qij(idx_h,0);
    amrex::Real hinv = Qauxij(0,0);
    // if (h < eps) return rhs;
    // amrex::Real factor = h / amrex::max(h, eps);
    // rhs = factor*factor*Model::source(Qij, Qauxij);
    amrex::Real q = hinv * std::sqrt(Qij(2, 0) * Qij(2, 0) + Qij(3, 0) * Qij(3, 0));
    amrex::Real Cf = 9.81 / 30. / 30.;
    rhs(2, 0) = Qij(2, 0) / (1. + dt * Cf * q);
    rhs(3, 0) = Qij(3, 0) / (1. + dt * Cf * q);
    // rhs = Model::source(Qij, Q

    return rhs;
}
#endif
