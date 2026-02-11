#include <petsc.h>
#include <iostream>
#include <memory>

#include "ModularSolver.hpp"
#include "MOODSolver.hpp"       // [NEW] Include the MOOD Solver header
#include "SolverStrategies.hpp"

static char help[] = "Shallow Water Moments Solver (MOOD)\n";

int main(int argc, char **argv) {
    PetscFunctionBeginUser;
    // 1. Initialize PETSc
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    { // 2. START OF SCOPE BLOCK
        
        // Choose your time-stepping strategy. 
        // MOOD is compatible with both Splitting and IMEX.
        // Use IMEX if your source terms are stiff.
        auto strategy = std::make_shared<IMEXStrategy>();
        
        // [CHANGE 1] Instantiate MOODSolver instead of ModularSolver
        MOODSolver solver;
        solver.SetStrategy(strategy);
        
        // [CHANGE 2] Set Reconstruction to LINEAR (2nd Order)
        // MOOD relies on a High-Order "Predictor" step. 
        // If you set this to PCM, the solver will have nothing to lower.
        solver.SetReconstruction(LINEAR); 
        
        // [Standard] Set Flux Kernels
        solver.SetFluxKernel(Numerics<Real>::numerical_flux); 
        solver.SetNonConsFluxKernel(Numerics<Real>::numerical_fluctuations);

        if (solver.rank == 0) std::cout << "[MAIN] Starting MOOD Simulation..." << std::endl;
        
        PetscCall(solver.Run(argc, argv));

    } // 3. END OF SCOPE BLOCK 
    // The 'solver' destructor is called automatically here.
    // All PETSc objects (DM, Vec, TS) are destroyed while MPI is still open.

    // 4. Finalize PETSc safely
    return PetscFinalize();
}