#include "FirstOrderSolver.hpp"
#include "HigherOrderSolver.hpp"

static char help[] = "Modular FVM Solver (Order 1 = Upwind, Order 2+ = MOOD)\n";

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    {
        PetscInt order = 1;
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-order", &order, NULL));

        VirtualSolver* solver = nullptr;
        if (order > 1) {
            PetscPrintf(PETSC_COMM_WORLD, "--- Solver: MOOD (Target Order: %d, Fallback: 1) ---\n", order);
            solver = new HigherOrderSolver(order);
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "--- Solver: Standard (Order: 1) ---\n");
            solver = new FirstOrderSolver();
        }

        PetscCall(solver->Run(argc, argv));
        delete solver;
    }
    PetscCall(PetscFinalize());
    return 0;
}