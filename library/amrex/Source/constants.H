#ifndef CONSTANTS
#define CONSTANTS

#include <AMReX_SmallMatrix.H>
#include <initializer_list>
#include <cassert>

#include "model.h"

typedef amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> VecQ;
typedef amrex::SmallMatrix<amrex::Real, Model::n_dof_q, Model::n_dof_q> MatQ;
typedef amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> VecQaux;
typedef amrex::SmallMatrix<amrex::Real, 5, 1> VecQ3d;
typedef amrex::SmallMatrix<amrex::Real, 3, 1> Vec3;
typedef amrex::SmallMatrix<amrex::Real, 2, 1> Vec2;


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



#endif
