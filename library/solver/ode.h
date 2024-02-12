#include "define.h"
#include "model.h"
#include "space_solution_operator.h"

#ifndef ODE_H
#define ODE_H

void RK1(OdeOperator& func, const realArr2 Q, const realArr2 Qaux, const realArr param, double dt, realArr2& out)
{
    const int n_elements = Q.extent(0);
    const int n_fields = Q.extent(1);
    realArr2 dQ("dQ", n_elements, n_fields);
    func.evaluate(Q, Qaux, param, dQ);
    for (int i = 0; i < n_elements; ++i)
    {
        for (int j = 0; j < n_fields; ++j)
        {
            out(i, j) = Q(i, j) + dt * dQ(i, j);
        }
    }
}


class Integrator
{
private:
    void (*method)(OdeOperator&, const realArr2, const realArr2, const realArr, double, realArr2&);
public:
    Integrator(std::string method) {
        if (method == "RK1")
        {
            this->method = RK1;
        }
    }
    void evaluate(OdeOperator& func, const realArr2 Q, const realArr2 Qaux, const realArr param, double dt, realArr2& out)
    {
        this->method(func, Q, Qaux, param, dt, out);
    }
};

#endif // ODE_H