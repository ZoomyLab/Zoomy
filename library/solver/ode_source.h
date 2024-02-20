#include "define.h"
#include "model.h"

#ifndef ODE_SOURCE_H
#define ODE_SOURCE_H


void RK1(Model& model, const realArr Q, const realArr Qaux, const realArr param, double dt, realArr& out)
{
    const int n_fields = Q.extent(0);
    realArr fun = realArr("fun", n_fields);
    model.source(Q, Qaux, param, fun);
    // realArr2 jac = realArr("jac", n_fields, n_fields);
    // model.source_jacobian(Q, Qaux, param, jac);
    for (int j = 0; j < n_fields; ++j)
    {
        out(j) = Q(j) + dt * fun(j);
    }
}

#if ODE_SOURCE == 1
    #define SOURCE_INTEGRATOR RK1
#elif ODE_SOURCE == 2
    #define SOURCE_INTEGRATOR RK3
#elif ODE_SOURCE == 3
    #define SOURCE_INTEGRATOR RK3
#else
    #error "Invalid method. (ODE_FLUX)"
#endif

#endif // ODE_SOURCE_H
