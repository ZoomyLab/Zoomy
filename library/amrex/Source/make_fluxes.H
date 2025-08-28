#ifndef MAKE_FLUXES
#define MAKE_FLUXES

#include "AMReX_Array4.H"
#include "model.h"

/**
 * Define rhs on a cell by cell basis
 */

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
flux_xlo (int i, int j, int comp, amrex::Array4<amrex::Real const> const& sol_array, amrex::Real dx, amrex::Real dy)
{
    amrex::Real flux = 0.5 * (flux_x(i,j,comp, sol_array) + flux_x(i-1,j,comp, sol_array));


    return flux;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
flux_xhi (int i, int j, int comp, amrex::Array4<amrex::Real const> const& sol_array, amrex::Real dx, amrex::Real dy)
{
    amrex::Real flux = 0.5 * (flux_x(i,j,comp, sol_array) + flux_x(i+1,j,comp, sol_array));

    return flux;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
flux_ylo (int i, int j, int comp, amrex::Array4<amrex::Real const> const& sol_array, amrex::Real dx, amrex::Real dy)
{

    amrex::Real flux = 0.5 * (flux_y(i,j,comp, sol_array) + flux_y(i,j-1,comp, sol_array));

    return flux;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
flux_yhi (int i, int j, int comp, amrex::Array4<amrex::Real const> const& sol_array, amrex::Real dx, amrex::Real dy)
{

    amrex::Real flux = 0.5 * (flux_y(i,j,comp, sol_array) + flux_y(i,j+1,comp, sol_array));

    return flux;
}

#endif
