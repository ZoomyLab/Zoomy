#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include "../../outputs/output_c/c_interface/Model/boundary_conditions_code.h"
#include "define.h"
#include <vector>

using BoundaryConditionFunc = void (*)(double*, double*, double*, double*, double*);

// Class to hold all boundary condition functions

class BoundaryConditions {
public:
    // Vector to hold all boundary condition functions
    std::vector<BoundaryConditionFunc> boundary_conditions;

    BoundaryConditions() 
    {
        #if N_BOUNDARY_CONDITIONS > 0
            boundary_conditions.push_back(sympy::boundary_condition_0);
        #endif
        #if N_BOUNDARY_CONDITIONS > 1
            boundary_conditions.push_back(sympy::boundary_condition_1);
        #endif
        #if N_BOUNDARY_CONDITIONS > 2
            boundary_conditions.push_back(sympy::boundary_condition_2);
        #endif
        #if N_BOUNDARY_CONDITIONS > 3
            boundary_conditions.push_back(sympy::boundary_condition_3);
        #endif
        #if N_BOUNDARY_CONDITIONS > 4
            boundary_conditions.push_back(sympy::boundary_condition_4);
        #endif
        #if N_BOUNDARY_CONDITIONS > 5
            boundary_conditions.push_back(sympy::boundary_condition_5);
        #endif
        #if N_BOUNDARY_CONDITIONS > 6
            boundary_conditions.push_back(sympy::boundary_condition_6);
        #endif
        #if N_BOUNDARY_CONDITIONS > 7
            boundary_conditions.push_back(sympy::boundary_condition_7);
        #endif
        #if N_BOUNDARY_CONDITIONS > 8
            boundary_conditions.push_back(sympy::boundary_condition_8);
        #endif
        #if N_BOUNDARY_CONDITIONS > 9
            boundary_conditions.push_back(sympy::boundary_condition_9);
        #endif
    }

    // // Constructor
    // BoundaryConditions();

    // Function to apply a specific boundary condition
    void apply(int index, const realArr& Q, const realArr& Qaux, const realArr& parameters, const realArr& normal, realArr& Qout) {
        boundary_conditions[index](Q.data(), Qaux.data(), parameters.data(), normal.data(), Qout.data());
    }
};

// template <>
// BoundaryConditions<1>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
// }

// template <>
// BoundaryConditions<2>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
//     boundary_conditions.push_back(boundary_condition_1);
// }

// template <>
// BoundaryConditions<3>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
//     boundary_conditions.push_back(boundary_condition_1);
//     boundary_conditions.push_back(boundary_condition_2);
// }

// template <>
// BoundaryConditions<4>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
//     boundary_conditions.push_back(boundary_condition_1);
//     boundary_conditions.push_back(boundary_condition_2);
//     boundary_conditions.push_back(boundary_condition_3);
// }

// template <>
// BoundaryConditions<5>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
//     boundary_conditions.push_back(boundary_condition_1);
//     boundary_conditions.push_back(boundary_condition_2);
//     boundary_conditions.push_back(boundary_condition_3);
//     boundary_conditions.push_back(boundary_condition_4);
// }

// template <>
// BoundaryConditions<6>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
//     boundary_conditions.push_back(boundary_condition_1);
//     boundary_conditions.push_back(boundary_condition_2);
//     boundary_conditions.push_back(boundary_condition_3);
//     boundary_conditions.push_back(boundary_condition_4);
//     boundary_conditions.push_back(boundary_condition_5);
// }

// template <>
// BoundaryConditions<7>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
//     boundary_conditions.push_back(boundary_condition_1);
//     boundary_conditions.push_back(boundary_condition_2);
//     boundary_conditions.push_back(boundary_condition_3);
//     boundary_conditions.push_back(boundary_condition_4);
//     boundary_conditions.push_back(boundary_condition_5);
//     boundary_conditions.push_back(boundary_condition_6);
// }

// template <>
// BoundaryConditions<8>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
//     boundary_conditions.push_back(boundary_condition_1);
//     boundary_conditions.push_back(boundary_condition_2);
//     boundary_conditions.push_back(boundary_condition_3);
//     boundary_conditions.push_back(boundary_condition_4);
//     boundary_conditions.push_back(boundary_condition_5);
//     boundary_conditions.push_back(boundary_condition_6);
//     boundary_conditions.push_back(boundary_condition_7);
// }

// template <>
// BoundaryConditions<9>::BoundaryConditions() {
//     boundary_conditions.push_back(boundary_condition_0);
//     boundary_conditions.push_back(boundary_condition_1);
//     boundary_conditions.push_back(boundary_condition_2);
//     boundary_conditions.push_back(boundary_condition_3);
//     boundary_conditions.push_back(boundary_condition_4);
//     boundary_conditions.push_back(boundary_condition_5);
//     boundary_conditions.push_back(boundary_condition_6);
//     boundary_conditions.push_back(boundary_condition_7);
//     boundary_conditions.push_back(boundary_condition_7);
//     boundary_conditions.push_back(boundary_condition_8);
// }


// // ... and so on for each value of N_BOUNDARY_CONDITIONS

#endif // BOUNDARY_CONDITIONS_HH