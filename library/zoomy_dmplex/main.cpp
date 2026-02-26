#include <petsc.h>
#include <iostream>
#include <memory>

#include "ModularSolver.hpp"
#include "MOODSolver.hpp"       
#include "SolverStrategies.hpp"

static char help[] = "Shallow Water Moments Solver (Fully Implicit MOOD)\n";

int main(int argc, char **argv) {
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    { 
        auto strategy = std::make_shared<FullyImplicitStrategy>();
        
        MOODSolver solver;
        solver.SetStrategy(strategy);
        solver.SetReconstruction(LINEAR); 
        solver.SetFluxKernel(Numerics<Real>::numerical_flux); 
        solver.SetNonConsFluxKernel(Numerics<Real>::numerical_fluctuations);

        if (solver.rank == 0) std::cout << "[MAIN] Starting MOOD Simulation (Fully Implicit)..." << std::endl;
        
        PetscCall(solver.Run(argc, argv));
    } 

    return PetscFinalize();
}