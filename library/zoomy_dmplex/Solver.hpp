#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "ModularSolver.hpp"

class Solver : public ModularSolver {
public:
    // Configurable Solver
    // Modes:
    // 1. Explicit Flux + Explicit Source
    // 2. Explicit Flux + Implicit Source
    // 3. Second Order (Reconstruction)
    
    Solver(int order = 1, bool implicit_source = false) : ModularSolver() {
        // 1. Set Flux (Default is Rusanov/Numerical Flux)
        SetFluxKernel(Numerics<Real>::numerical_flux);

        // 2. Set Source Strategy
        if (implicit_source) {
            SetExplicitSource(false);
            SetImplicitSource(true); // Triggers TSARKIMEX
        } else {
            SetExplicitSource(true); // Triggers TSEULER/SSP with source in RHS
            SetImplicitSource(false);
        }

        // 3. Set Order / Reconstruction
        if (order == 1) {
            SetReconstruction(PCM);
        } else if (order == 2) {
            if (this->rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[INFO] Solver: Using 2nd order Linear Reconstruction.\n");
            SetReconstruction(LINEAR); 
            // Note: LINEAR requires gradient computation which isn't fully wired in ComputeFaceFluxes yet
            // but the hook is there.
        } else {
            if (this->rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[WARNING] Solver: Order %d not supported. Fallback to 1st order.\n", order);
            SetReconstruction(PCM);
        }
    }
};

#endif