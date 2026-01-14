#include "FirstOrderSolver.hpp"
#include "HigherOrderSolver.hpp"

#ifdef ENABLE_GPU
#include "GPUFirstOrderSolver.hpp"
#endif

static char help[] = "Modular FVM Solver (CPU/GPU)\n";

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    {
        PetscInt order = 1;
        
        // Default use_gpu to true if compiled for GPU, false otherwise
#ifdef ENABLE_GPU
        PetscBool use_gpu = PETSC_TRUE;
#else
        PetscBool use_gpu = PETSC_FALSE;
#endif

        PetscCall(PetscOptionsGetInt(NULL, NULL, "-order", &order, NULL));
        // Still allow the user to override via command line if they really want to
        PetscCall(PetscOptionsGetBool(NULL, NULL, "-gpu", &use_gpu, NULL));

        VirtualSolver* solver = nullptr;

        if (use_gpu) {
#ifdef ENABLE_GPU
            if (order == 1) {
                PetscPrintf(PETSC_COMM_WORLD, "--- Solver: GPU First Order ---\n");
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
            if (order > 1) {
                PetscPrintf(PETSC_COMM_WORLD, "--- Solver: CPU MOOD (Target Order: %d) ---\n", order);
                solver = new HigherOrderSolver(order);
            } else {
                PetscPrintf(PETSC_COMM_WORLD, "--- Solver: CPU First Order ---\n");
                solver = new FirstOrderSolver();
            }
        }

        if (solver) {
            PetscCall(solver->Run(argc, argv));
            delete solver;
        }
    }
    PetscCall(PetscFinalize());
    return 0;
}