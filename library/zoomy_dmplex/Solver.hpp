#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "ModularSolver.hpp"

// A dummy flux kernel that returns zero
// Used when we want to rely solely on the Non-Conservative Flux
template <typename T>
SimpleArray<T, Model<T>::n_dof_q> ZeroFlux(
    const T*, const T*, const T*, const T*, 
    const T*, const T*) 
{
    SimpleArray<T, Model<T>::n_dof_q> res;
    for(int i=0; i<Model<T>::n_dof_q; ++i) res[i] = 0.0;
    return res;
}

// ------------------------------------------
// 1. Conservative Solver (Standard)
// ------------------------------------------
class ConservativeSolver : public ModularSolver {
public:
    ConservativeSolver(int order = 1, bool implicit_source = false) : ModularSolver() {
        // Standard Conservative Flux
        SetFluxKernel(Numerics<Real>::numerical_flux);
        SetNonConsFluxKernel(nullptr); // Disable Non-Cons

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
// 2. Quasilinear Solver (Non-Conservative)
// ------------------------------------------
class QuasilinearSolver : public ModularSolver {
public:
    QuasilinearSolver(int order = 1, bool implicit_source = false) : ModularSolver() {
        // Register 'numerical_flux' as the NON-Conservative kernel
        SetNonConsFluxKernel(Numerics<Real>::numerical_flux);
        
        // Disable the Conservative Flux (set to zero) so we don't double count
        SetFluxKernel(ZeroFlux<Real>);

        SetImplicitSource(implicit_source);
        SetExplicitSource(!implicit_source);

        if (order == 1) {
            SetReconstruction(PCM);
        } else if (order == 2) {
            if (this->rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[INFO] QuasilinearSolver: Using 2nd order.\n");
            SetReconstruction(LINEAR); 
            SetGradientMethod(LEAST_SQUARES); 
            SetLeastSquaresOrder(1);
            SetLimiter(true); 
        } else {
            SetReconstruction(PCM);
        }
    }
};

// Type alias for backward compatibility if needed, or default choice
using Solver = ConservativeSolver; 

#endif