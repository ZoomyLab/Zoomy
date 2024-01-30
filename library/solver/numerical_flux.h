#include "define.h"
#include "model.h"
#include "max_abs_eigenvalue.h"

#include <cmath>
#include <iostream>

#ifndef NUMERICAL_FLUX_H
#define NUMERICAL_FLUX_H

void rusanov(const realArr& qi, const realArr& qj, const realArr& qauxi, const realArr& qauxj, const realArr& param, const realArr& normal, Model& model, realArr& F)
{
    const int dim = normal.extent(0);
    const int n_fields = qi.extent(0);
    F = realArr("F", n_fields);
    realArr eigenvalues_i = realArr("eigenvalues_i", n_fields);
    realArr eigenvalues_j = realArr("eigenvalues_j", n_fields);
    realArr2 Fi = realArr2("Fi", dim, n_fields);
    realArr2 Fj = realArr2("Fj", dim, n_fields);
    model.flux(qi, qauxi, param, Fi);
    model.flux(qj, qauxj, param, Fj);
    model.eigenvalues(qi, qauxi, param, normal, eigenvalues_i);
    model.eigenvalues(qj, qauxj, param, normal, eigenvalues_j);
    const double abs_max_ev = max(max_abs(eigenvalues_i), max_abs(eigenvalues_j));
    for (int d = 0; d < dim; ++d)
    {
        for (int i = 0; i < n_fields; ++i)
        {
            F(i) += 0.5 * (Fi(d, i) + Fj(d, i)) * normal(d) ;
        }
    }
    for (int i = 0; i < n_fields; ++i)
    {
        F(i) -= 0.5 * abs_max_ev * (qj(i) - qi(i));
    }
}
#endif // NUMERICAL_FLUX_H