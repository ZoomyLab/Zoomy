#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include "../../outputs/output_c/c_interface/Model/boundary_conditions_code.h"

template <int N_BOUNDARY_CONDITIONS>
class BoundaryConditions
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
void BoundaryConditions<1>::flux(std::vector<double>& Q, std::vector<double>& Qaux, std::vector<double>& parameters, std::vector<std::vector<double>>& Qout)
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

#endif // BOUNDARY_CONDITIONS_HH