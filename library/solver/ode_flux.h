#include "define.h"
#include "model.h"
#include "fvm.h"

#ifndef ODE_FLUX_FLUX_H
#define ODE_FLUX_FLUX_H


void RK1(FluxSolutionOperator& FSO, const realArr2 Q, const realArr2 Qaux, const realArr param, double dt, realArr2& out)
{
    const int n_elements = Q.extent(0);
    const int n_fields = Q.extent(1);
    realArr2 dQ("dQ", n_elements, n_fields);
    FSO.evaluate(Q, Qaux, param, dQ);
    for (int i = 0; i < n_elements; ++i)
    {
        for (int j = 0; j < n_fields; ++j)
        {
            out(i, j) = Q(i, j) + dt * dQ(i, j);
        }
    }
}


#if ODE_FLUX == 1
    #define FLUX_INTEGRATOR RK1
#elif ODE_FLUX == 2
    #define FLUX_INTEGRATOR RK3
#elif ODE_FLUX == 3
    #define FLUX_INTEGRATOR RK3
#else
    #error "Invalid method. (ODE_FLUX)"
#endif


#endif // ODE_FLUX_H
