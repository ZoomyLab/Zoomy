#ifndef MODEL_H
#define MODEL_H

#include "../../outputs/output_c/c_interface/Model/model_code.h"
#include "define.h"

class Model
{
public:
    void flux(const realArr Q, const realArr Qaux, const realArr parameters, realArr2& Qout)
    {
        sympy::flux_x(Q.data(), Qaux.data(), parameters.data(), Qout.data());
        #if DIMENSION > 1
            sympy::flux_y(Q.data(), Qaux.data(), parameters.data(), get_last2(Qout, 1).data());
        #endif
        #if DIMENSION > 2
            sympy::flux_z(Q.data(), Qaux.data(), parameters.data(), get_last2(Qout, 2).data());
        #endif
    }

    void flux_jacobian(const realArr Q, const realArr Qaux, const realArr parameters, realArr3& Qout)
    {
        sympy::flux_jacobian_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
        #if DIMENSION > 1
            sympy::flux_jacobian_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
        #endif
        #if DIMENSION > 2
            sympy::flux_jacobian_z(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 2).data());
        #endif
    }

    void nonconservative_matrix(const realArr Q, const realArr Qaux, const realArr parameters, realArr3& Qout)
    {
        sympy::nonconservative_matrix_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
        #if DIMENSION > 1
            sympy::nonconservative_matrix_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
        #endif
        #if DIMENSION > 2
            sympy::nonconservative_matrix_z(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 2).data());
        #endif
    }

    void quasilinear_matrix(const realArr Q, const realArr Qaux, const realArr parameters, realArr3& Qout)
    {
        sympy::quasilinear_matrix_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
        #if DIMENSION > 1
            sympy::quasilinear_matrix_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
        #endif
        #if DIMENSION > 2
            sympy::quasilinear_matrix_z(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 2).data());
        #endif
    }

    void source(const realArr Q, const realArr Qaux, const realArr parameters, realArr& Qout)
    {
        sympy::source(Q.data(), Qaux.data(), parameters.data(), Qout.data());
    }

    void source_jacobian(const realArr Q, const realArr Qaux, const realArr parameters, realArr2& Qout)
    {
        sympy::source_jacobian(Q.data(), Qaux.data(), parameters.data(), Qout.data());
    }
};

#endif // MODEL_HH