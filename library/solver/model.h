#ifndef MODEL_H
#define MODEL_H

#include "../../outputs/output_c/c_interface/Model/model_code.h"
#include "define.h"

template <int>
class Model
{
public:
    const int dimension = 1;

    void flux(const realArr Q, const realArr Qaux, const realArr parameters, realArr2& Qout)
    {
        std::cerr << "flux function not implemented for this dimension" << std::endl;
        std::exit(1);
    }

    void flux_jacobian(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
    {
        std::cerr << "flux_jacobian function not implemented for this dimension" << std::endl;
        std::exit(1);
    }

    void nonconservative_matrix(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
    {
        std::cerr << "nonconservative_matrix function not implemented for this dimension" << std::endl;
        std::exit(1);
    }

    void quasilinear_matrix(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
    {
        std::cerr << "quasilinear_matrix function not implemented for this dimension" << std::endl;
        std::exit(1);
    }

    void source(realArr& Q, realArr& Qaux, realArr& parameters, realArr& Qout)
    {
        source(Q.data(), Qaux.data(), parameters.data(), Qout.data());
    }

    void source_jacobian(realArr& Q, realArr& Qaux, realArr& parameters, realArr2& Qout)
    {
        source_jacobian(Q.data(), Qaux.data(), parameters.data(), Qout.data());
    }
};

// Specialization for DIM = 1
template <>
void Model<1>::flux(const realArr Q, const realArr Qaux, const realArr parameters, realArr2& Qout)
{
    flux_x(Q.data(), Qaux.data(), parameters.data(), Qout.data());
}

// Specialization for DIM = 2
template <>
void Model<2>::flux(const realArr Q, const realArr Qaux, const realArr parameters, realArr2& Qout)
{
    flux_x(Q.data(), Qaux.data(), parameters.data(), get_last2(Qout, 0).data());
    flux_y(Q.data(), Qaux.data(), parameters.data(), get_last2(Qout, 1).data());
}

// Specialization for DIM = 3
template <>
void Model<3>::flux(const realArr Q, const realArr Qaux, const realArr parameters, realArr2& Qout)
{
    flux_x(Q.data(), Qaux.data(), parameters.data(), get_last2(Qout, 0).data());
    flux_y(Q.data(), Qaux.data(), parameters.data(), get_last2(Qout, 1).data());
    flux_z(Q.data(), Qaux.data(), parameters.data(), get_last2(Qout, 2).data());
}

// Specialization for DIM = 1
template <>
void Model<1>::flux_jacobian(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    flux_jacobian_x(Q.data(), Qaux.data(), parameters.data(), Qout.data());
}

// Specialization for DIM = 2
template <>
void Model<2>::flux_jacobian(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    flux_jacobian_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
    flux_jacobian_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
}

// Specialization for DIM = 3
template <>
void Model<3>::flux_jacobian(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    flux_jacobian_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
    flux_jacobian_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
    flux_jacobian_z(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 2).data());
}

// Specialization for DIM = 1
template <>
void Model<1>::nonconservative_matrix(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    nonconservative_matrix_x(Q.data(), Qaux.data(), parameters.data(), Qout.data());
}

// Specialization for DIM = 2
template <>
void Model<2>::nonconservative_matrix(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    nonconservative_matrix_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
    nonconservative_matrix_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
}

// Specialization for DIM = 3
template <>
void Model<3>::nonconservative_matrix(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    nonconservative_matrix_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
    nonconservative_matrix_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
    nonconservative_matrix_z(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 2).data());
}

// Specialization for DIM = 1
template <>
void Model<1>::quasilinear_matrix(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    quasilinear_matrix_x(Q.data(), Qaux.data(), parameters.data(), Qout.data());
}

// Specialization for DIM = 2
template <>
void Model<2>::quasilinear_matrix(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    quasilinear_matrix_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
    quasilinear_matrix_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
}

// Specialization for DIM = 3
template <>
void Model<3>::quasilinear_matrix(realArr& Q, realArr& Qaux, realArr& parameters, realArr3& Qout)
{
    quasilinear_matrix_x(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 0).data());
    quasilinear_matrix_y(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 1).data());
    quasilinear_matrix_z(Q.data(), Qaux.data(), parameters.data(), get_last3(Qout, 2).data());
}


#endif // MODEL_HH