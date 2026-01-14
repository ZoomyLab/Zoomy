#ifndef FIRSTORDERSOLVER_HPP
#define FIRSTORDERSOLVER_HPP

#include "VirtualSolver.hpp"

class FirstOrderSolver : public VirtualSolver {
private:
    PetscInt max_steps;

public:
    FirstOrderSolver() : VirtualSolver(), max_steps(100) {}

    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-ufv_cfl", &cfl, NULL));
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-ts_max_steps", &max_steps, NULL));

        // 1. Setup (Standard Overlap 1)
        PetscCall(SetupArchitecture(1));
        
        // 2. Initial State
        PetscCall(UpdateBoundaryGhosts(0.0));
        PetscCall(WriteVTU(0, 0.0));

        PetscPrintf(PETSC_COMM_WORLD, "--- Starting First Order Solver (Manual Loop) ---\n");

        PetscReal time = 0.0;
        PetscInt step = 0;

        // 3. Explicit Time Loop
        while (step < max_steps) {
            // A. Calculate Adaptive Time Step (Uses shared VirtualSolver logic)
            PetscReal dt = ComputeTimeStep();
            
            // B. Configure TS
            PetscCall(TSSetTime(ts, time));
            PetscCall(TSSetTimeStep(ts, dt));
            
            // C. Take One Step
            PetscCall(TSSetSolution(ts, X));
            PetscCall(TSStep(ts));
            
            // D. Update State
            time += dt;
            step++;
            
            PetscPrintf(PETSC_COMM_WORLD, "Step %3d | Time %.5g | dt %.5g\n", step, (double)time, (double)dt);
            
            // E. Output
            PetscCall(WriteVTU(step, time));
            PetscCall(UpdateBoundaryGhosts(time));
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }
};
#endif