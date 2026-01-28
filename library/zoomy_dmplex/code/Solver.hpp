#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "ModularSolver.hpp"

// ------------------------------------------
// 1. Conservative Solver (Standard)
// ------------------------------------------
class ConservativeSolver : public ModularSolver {
public:
    ConservativeSolver(int order = 1, bool implicit_source = false) : ModularSolver() {
        // Register standard conservative flux
        SetFluxKernel(Numerics<Real>::numerical_flux);
        
        // Disable Non-Conservative part
        // (Even though Numerics::nonconservative_fluctuations exists and returns zero,
        //  passing nullptr allows TransportStep to skip the loop entirely)
        SetNonConsFluxKernel(nullptr); 

        SetImplicitSource(implicit_source);
        SetExplicitSource(!implicit_source);

        if (order == 1) {
            SetReconstruction(PCM);
        } else if (order == 2) {
            if (this->rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[INFO] ConservativeSolver: Using 2nd order.\n");
            SetReconstruction(LINEAR); 
            SetGradientMethod(LEAST_SQUARES); 
            SetLeastSquaresOrder(1);
            SetLimiter(true); 
        } else {
            SetReconstruction(PCM);
        }
    }
};

// ------------------------------------------
// 2. Non-Conservative Solver (Quasilinear + Conservative)
// ------------------------------------------
class NonConservativeSolver : public ModularSolver {
public:
    NonConservativeSolver(int order = 1, bool implicit_source = false) : ModularSolver() {
        // 1. Register Conservative Flux (e.g., g*h^2/2)
        SetFluxKernel(Numerics<Real>::numerical_flux);
        
        // 2. Register Non-Conservative Fluctuations (e.g., g*h*db/dx)
        SetNonConsFluxKernel(Numerics<Real>::nonconservative_fluctuations);
        
        SetImplicitSource(implicit_source);
        SetExplicitSource(!implicit_source);

        if (order == 1) {
            SetReconstruction(PCM);
        } else if (order == 2) {
            if (this->rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[INFO] NonConservativeSolver: Using 2nd order.\n");
            SetReconstruction(LINEAR); 
            SetGradientMethod(LEAST_SQUARES); 
            SetLeastSquaresOrder(1);
            SetLimiter(true); 
        } else {
            SetReconstruction(PCM);
        }
    }
};

// Default Alias
using Solver = ConservativeSolver; 
// using Solver = NonConservativeSolver; // Switch to this if needed

#endif