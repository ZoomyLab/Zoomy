#ifndef MODEL_H
#define MODEL_H

#include "../../outputs/output_c/c_interface/Model/model_code.h"

template <int DIM>
class Model
{
public:
    const int dimension = 1;
    void flux(std::vector<double>& Q, std::vector<double>& Qaux, std::vector<double>& parameters, std::vector<std::vector<double>>& Qout)
    {
        std::cerr << "flux function not implemented for this dimension" << std::endl;
        std::exit(1);
    }
};

// Specialization for DIM = 1
template <>
void Model<1>::flux(std::vector<double>& Q, std::vector<double>& Qaux, std::vector<double>& parameters, std::vector<std::vector<double>>& Qout)
{
    Qout[0][0] = 9999.;
    // flux_x(Q, Qaux, parameters, out[0]);
}

// // Specialization for DIM = 2
// template <>
// void Model<2>::flux(double *Q, double *Qaux, double *parameters, double **out)
// {
//     flux_x(Q, Qaux, parameters, out[0]);
//     flux_y(Q, Qaux, parameters, out[1]);
// }

// // Specialization for DIM = 3
// template <>
// void Model<3>::flux(double *Q, double *Qaux, double *parameters, double **out)
// {
//     flux_x(Q, Qaux, parameters, out[0]);
//     flux_y(Q, Qaux, parameters, out[1]);
//     flux_z(Q, Qaux, parameters, out[2]);
// }

#endif // MODEL_HH