#include "define.h"
#include "model.h"
#include "max_abs_eigenvalue.h"

#include <cmath>
#include <iostream>

#ifndef NONCONSERVATIVE_FLUX_H
#define NONCONSERVATIVE_FLUX_H

void segmentpath(const realArr& qi, const realArr& qj, const realArr& qauxi, const realArr& qauxj, const realArr& param, const realArr& normal, Model& model, realArr& NC)
{
    const int dim = normal.extent(0);
    const int n_fields = qi.extent(0);
    NC = realArr("NC", n_fields);
}

#endif // NONCONSERVATIVE_FLUX_H