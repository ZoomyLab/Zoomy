#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "ModularSolver.hpp"

class Solver : public ModularSolver {
public:
    Solver(int order = 1, bool implicit_source = false) : ModularSolver() {
        SetFluxKernel(Numerics<Real>::numerical_flux);
        SetImplicitSource(implicit_source);
        SetExplicitSource(!implicit_source);

        if (order == 1) {
            SetReconstruction(PCM);
        } else if (order == 2) {
            if (this->rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[INFO] Solver: Using 2nd order Linear Reconstruction.\n");
            SetReconstruction(LINEAR); 
            SetGradientMethod(LEAST_SQUARES); 
            SetLeastSquaresOrder(1);
            SetLimiter(true); 
        } else {
            if (this->rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[WARNING] Solver: Order %d not supported. Fallback to 1st order.\n", order);
            SetReconstruction(PCM);
        }
    }
};
#endif