#include <petsc.h>
#include <iostream>
#include <memory>

// Architecture
#include "ModularSolver.hpp"
#include "SolverStrategies.hpp"
#include "MOODSolver.hpp"

// Physics & Numerics
#include "Model.H"
#include "Numerics.H" 

static char help[] = "Shallow Water Moments Solver (IMEX Default)\n";

int main(int argc, char **argv) {
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    // =========================================================================
    // 1. CONFIGURE THE STRATEGY (PHYSICS)
    // =========================================================================
    // We use the IMEX Strategy: 
    // - Explicit Transport (Fluxes)
    // - Implicit Source (Friction) in the IFunction
    // This prevents the "Time Step Collapse" on steep slopes.
    auto strategy = std::make_shared<IMEXStrategy>();

    // =========================================================================
    // 2. CONFIGURE THE SOLVER (NUMERICS)
    // =========================================================================
    
    // --- OPTION A: 1st Order IMEX (DEFAULT) ---
    // Fast, stable, diffusive. Good for testing and very steep terrain.
    auto solver = std::make_shared<ModularSolver>();
    solver->SetStrategy(strategy);
    solver->SetReconstruction(PCM); // 1st Order (Piecewise Constant)

    // --- OPTION B: 2nd Order MOOD (COMMENTED OUT) ---
    // High-order accuracy with fallback stability.
    // auto solver = std::make_shared<MOODSolver>(strategy);
    // solver->SetReconstruction(LINEAR); // Starts at 2nd Order

    // =========================================================================
    // 3. REGISTER KERNELS
    // =========================================================================
    
    // A. Conservative Flux (e.g., Rusanov/HLL for SWE)
    solver->SetFluxKernel(Numerics<Real>::numerical_flux); 

    // B. Non-Conservative Fluctuations (CRITICAL for Path-Conservative)
    solver->SetNonConsFluxKernel(Numerics<Real>::numerical_fluctuations);

    // =========================================================================
    // 4. RUN
    // =========================================================================
    
    if (solver->rank == 0) {
        std::cout << "[MAIN] Starting Simulation..." << std::endl;
    }

    // Run() handles initialization, mesh loading, and the time loop
    PetscCall(solver->Run(argc, argv));

    PetscCall(PetscFinalize());
    return 0;
}