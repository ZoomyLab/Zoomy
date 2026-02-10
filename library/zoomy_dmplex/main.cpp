#include <petsc.h>
#include <iostream>
#include <memory>

#include "ModularSolver.hpp"
#include "SolverStrategies.hpp"

static char help[] = "Shallow Water Moments Solver)\n";

int main(int argc, char **argv) {
    PetscFunctionBeginUser;
    // 1. Initialize PETSc
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    { // 2. START OF SCOPE BLOCK
        auto strategy = std::make_shared<IMEXStrategy>();
        
        // Solver is created inside this block
        ModularSolver solver;
        solver.SetStrategy(strategy);
        
        // Standard setup and run
        solver.SetReconstruction(PCM); 
        solver.SetFluxKernel(Numerics<Real>::numerical_flux); 
        solver.SetNonConsFluxKernel(Numerics<Real>::numerical_fluctuations);

        if (solver.rank == 0) std::cout << "[MAIN] Starting Simulation..." << std::endl;
        
        PetscCall(solver.Run(argc, argv));

    } // 3. END OF SCOPE BLOCK 
    // The 'solver' destructor is called automatically here.
    // All PETSc objects (DM, Vec, TS) are destroyed while MPI is still open.

    // 4. Finalize PETSc safely
    return PetscFinalize();
}
