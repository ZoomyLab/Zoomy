#ifndef FIRSTORDERSOLVER_HPP
#define FIRSTORDERSOLVER_HPP

#include "VirtualSolver.hpp"

class FirstOrderSolver : public VirtualSolver {
public:
    PetscErrorCode TakeOneStep(PetscReal time, PetscReal dt) override {
        PetscFunctionBeginUser;
        SetSolverOrder(1);
        PetscCall(TSSetTime(ts, time));
        PetscCall(TSSetTimeStep(ts, dt));
        PetscCall(TSSetSolution(ts, X));
        PetscCall(TSStep(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};
#endif