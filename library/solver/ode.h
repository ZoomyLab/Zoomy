#include "define.h"

typedef void (*F)(const realArr2, const realArr2, const realArr, realArr2& out);
typedef void (*dFdQ)(const realArr2, const realArr2, const realArr, realArr3& out);

void RK1(F func, const realArr2 Q, const realArr2 Qaux, const realArr param, double dt, realArr2& out, dFdQ func_jac=nullptr)
{
    realArr2 dQ("dQ", Q.extent(0), Q.extent(1));
    func(Q, Qaux, param, dQ);
    for (int i = 0; i < Q.extent(0); ++i)
    {
        for (int j = 0; j < Q.extent(1); ++j)
        {
            out(i, j) = Q(i, j) + dt * dQ(i, j);
        }
    }
}
