#ifndef CONSTANTS
#define CONSTANTS

#include <AMReX_SmallMatrix.H>
#include <initializer_list>
#include <cassert>

#include "model.h"

typedef amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> VecQ;
typedef amrex::SmallMatrix<amrex::Real, Model::n_dof_q, Model::n_dof_q> MatQ;
typedef amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> VecQaux;
typedef amrex::SmallMatrix<amrex::Real, 6, 1> VecQ3d;
typedef amrex::SmallMatrix<amrex::Real, 3, 1> Vec3;
typedef amrex::SmallMatrix<amrex::Real, 2, 1> Vec2;

inline constexpr amrex::Real eps = 1e-8;
inline constexpr int idx_h = 1;
inline constexpr int idx_b = 0;



template <int R, int C>
amrex::SmallMatrix<amrex::Real, R, C> makeSmallMatrix(const std::initializer_list<amrex::Real>& vals)
{
    assert(vals.size() == R * C && "Initializer list size must match SmallMatrix size");

    amrex::SmallMatrix<amrex::Real, R, C> mat;
    auto it = vals.begin();
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            mat(i,j) = *it++;
        }
    }
    return mat;
}

template <typename T, int NRows, int NCols, typename CharT>
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void check_nan(const amrex::SmallMatrix<T, NRows, NCols>& v, const CharT* error_message)
{
    for (int i = 0; i < NRows; ++i) {
        for (int j = 0; j < NCols; ++j) {
            if (std::isnan(v(i, j))) {
                amrex::Print() << "NaN detected in " << error_message
                               << " at (" << i << "," << j << ")\n";
            }
        }
    }
}

#endif
