#include "define.h"
#include "model.h"
#include "linear_system.h"

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

void BDF1(Model& model, const realArr Q, const realArr Qaux, const realArr param, double dt, realArr& out)
{
    const int n_fields = Q.extent(0);
    realArr fun = realArr("fun", n_fields);
    model.source(Q, Qaux, param, fun);
    realArr2 jac = realArr2("jac", n_fields, n_fields);
    model.source_jacobian(Q, Qaux, param, jac);
    realArr2 A = realArr2("A", n_fields, n_fields);
    realArr rhs = realArr("rhs", n_fields);


    for (int j = 0; j < n_fields; ++j)
    {
        rhs(j) = Q(j) + dt * fun(j);
        for (int k = 0; k < n_fields; ++k)
        {
            A(j, k) = -dt * jac(j, k);
            rhs(j) += -dt * jac(j, k) * Q(k);
        }
        A(j, j) += 1.;
    }


    int err = solve_small_linear_system(A, rhs, out);
    if (out(0) <= 0.)
    {
        for (int j = 0; j < param.extent(0); ++j)
        {
            std::cout << "param(" << j << ") = " << param(j) << std::endl;
        }
        for (int j = 0; j < n_fields; ++j)
        {
            std::cout << "out(" << j << ") = " << out(j) << std::endl;
        }
        for (int j = 0; j < n_fields; ++j)
        {
            std::cout << "Q(" << j << ") = " << Q(j) << std::endl;
        }
        for (int j = 0; j < n_fields; ++j)
        {
            for (int k = 0; k < n_fields; ++k)
            {
                std::cout << "J(" << j << ", " << k << ") = " << jac(j, k) << std::endl;
            }

        }
        for (int j = 0; j < n_fields; ++j)
        {
            for (int k = 0; k < n_fields; ++k)
            {
                std::cout << "A(" << j << ", " << k << ") = " << A(j, k) << std::endl;
            }

        }
        std::cout << "----------------" << std::endl;
    }
    bool bnan = false;
    for (int i = 0; i < n_fields; ++i)
    {
        if (std::isnan(out(i))) bnan = true;

    }
    if (bnan)
    {
        std::cout << "NaNs found" << std::endl;
        for (int i = 0; i < n_fields; ++i)
        {
            std::cout << "Q(" << i << ") = " << Q(i) << std::endl;
            for (int j = 0; j < n_fields; ++j)
            {
                std::cout << "A(" << i << ", " << j << ") = " << A(i, j) << std::endl;
            }
        }
        for (int i = 0; i < n_fields; ++i)
        {
            std::cout << "sol(" << i << ") = " << out(i) << std::endl;
        }
        std::cout << "----------------" << std::endl;
    }
}

#if ODE_SOURCE == 1
    #define SOURCE_INTEGRATOR RK1
#elif ODE_SOURCE == 2
    #define SOURCE_INTEGRATOR RK3
#elif ODE_SOURCE == 3
    #define SOURCE_INTEGRATOR RK3
#elif ODE_SOURCE == -1
    #define SOURCE_INTEGRATOR BDF1
#else
    #error "Invalid method. (ODE_FLUX)"
#endif

#endif // ODE_SOURCE_H
