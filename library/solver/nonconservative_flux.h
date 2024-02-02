#include "define.h"
#include "model.h"
#include "max_abs_eigenvalue.h"

#include <cmath>
#include <iostream>

#ifndef NONCONSERVATIVE_FLUX_H
#define NONCONSERVATIVE_FLUX_H

void gauss(int order, realArr& xi, realArr& wi)
{
    if (order == 1)
    {
        xi = realArr("xi", 1);
        wi = realArr("wi", 1);
        xi[0] = 0.0;
        wi[0] = 1.;

    }
    else if (order == 3)
    {
        xi = realArr("xi", 3);
        wi = realArr("wi", 3);
        xi[0] = -sqrt(1.0 / 3.0);
        xi[1] = 0.0;
        xi[2] = sqrt(1.0 / 3.0);
        wi[0] = 5.0 / 9.0;
        wi[1] = 8.0 / 9.0;
        wi[2] = 5.0 / 9.0;
    }
}

void shift_integration(realArr& xi, realArr& wi)
{
    const int n_int = xi.extent(0);
    for (int i = 0; i < n_int; ++i)
    {
        xi(i) = 0.5 * (xi(i) + 1.0);
        wi(i) = 0.5 * (wi(i));
    }
}

void segmentpath(const realArr& qi, const realArr& qj, const realArr& qauxi, const realArr& qauxj, const realArr& param, const realArr& normal, Model& model, realArr& NC)
{
    realArr xi;
    realArr wi;
    gauss(3, xi, wi);
    shift_integration(xi, wi);

    const int n_int = xi.extent(0);
    const int dim = normal.extent(0);
    const int n_fields = qi.extent(0);
    const int n_fields_aux = qauxi.extent(0);
    NC = realArr("NC", n_fields);
    realArr3 NC_mat = realArr3("NC_mat" , dim, n_fields, n_fields);

    realArr qs = realArr("qs", n_fields);
    realArr qauxs = realArr("qs", n_fields_aux);
    auto B = [&model, &qi, &qj, &qauxi, &qauxj, &param, &n_fields, &n_fields_aux, &dim, &normal, &xi, &wi, &qs, &qauxs, &n_int](double s, realArr3 & NC_mat)
    {
        for (int i = 0; i < n_fields; ++i)
        {
            qs(i) = qi(i) + s * (qj(i) - qi(i));
        }
        for (int i = 0; i < n_fields_aux; ++i)
        {
            qauxs(i) = qauxi(i) + s * (qauxj(i) - qauxi(i));
        }
        model.nonconservative_matrix(qs, qauxs, param, NC_mat);
    };
    
    for (int i_int = 0; i_int < n_int; ++i_int)
    {
        B(xi(i_int), NC_mat);
        for (int d = 0; d < dim; ++d)
        {
            for (int i = 0; i < n_fields; ++i)
            {
                for (int j = 0; j < n_fields; ++j)
                {
                    NC(i) += -0.5*wi(i_int) * NC_mat(d, i, j) * (qj(j) - qi(j)) * normal(d);
                }
            }
        }
    }
}

#endif // NONCONSERVATIVE_FLUX_H