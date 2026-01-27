#include <petsc.h>

// The unified CPU solver (replaces FirstOrder/HigherOrder)
#include "Solver.hpp" 

// GPU Support
#ifdef ENABLE_GPU
#include "GPUFirstOrderSolver.hpp"
#endif

static char help[] = "Modular FVM Solver (CPU/GPU)\n";

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    {
        PetscInt order = 1;
        
        // Default: true if compiled for GPU, false otherwise
#ifdef ENABLE_GPU
        PetscBool use_gpu = PETSC_TRUE;
#else
        PetscBool use_gpu = PETSC_FALSE;
#endif

        PetscCall(PetscOptionsGetInt(NULL, NULL, "-order", &order, NULL));
        // Allow user override via command line (e.g., ./solver_gpu -gpu 0 to run CPU reference)
        PetscCall(PetscOptionsGetBool(NULL, NULL, "-gpu", &use_gpu, NULL));

        VirtualSolver* solver = nullptr;

        if (use_gpu) {
#ifdef ENABLE_GPU
            if (order == 1) {
                PetscPrintf(PETSC_COMM_WORLD, "--- Solver: GPU First Order ---\n");
                // Assuming GPUFirstOrderSolver still exists and inherits from VirtualSolver
                solver = new GPUFirstOrderSolver(); 
            } else {
                PetscPrintf(PETSC_COMM_WORLD, "Error: GPU Order %d not implemented yet.\n", order);
                PetscFinalize();
                return 1;
            }
#else
            PetscPrintf(PETSC_COMM_WORLD, "Error: CPU binary cannot run in GPU mode. Use ./solver_gpu\n");
            PetscFinalize();
            return 1;
#endif
        } else {
            // --- CPU Path ---
            // Much cleaner now: single class handles both 1st and higher order logic
            if (order == 1) {
                PetscPrintf(PETSC_COMM_WORLD, "--- Solver: CPU First Order ---\n");
            } else {
                PetscPrintf(PETSC_COMM_WORLD, "--- Solver: CPU High Order (Target: %d) ---\n", order);
            }
            
            // Just instantiate the unified Solver
            solver = new Solver(order, false);
        }

        if (solver) {
            PetscCall(solver->Run(argc, argv));
            delete solver;
        }
    }
    PetscCall(PetscFinalize());
    return 0;
}