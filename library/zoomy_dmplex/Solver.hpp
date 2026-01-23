#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "ModularSolver.hpp"

class Solver : public ModularSolver {
public:
    Solver(int order = 1) : ModularSolver() {
        AddConservativeFlux(Numerics<Real>::numerical_flux);
        AddSource(Model<Real>::source);

        if (order >= 2) {
            if (this->rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[WARNING] Solver: 2nd order reconstruction not yet implemented. Falling back to 1st order.\n");
        }
    }
};

#endif