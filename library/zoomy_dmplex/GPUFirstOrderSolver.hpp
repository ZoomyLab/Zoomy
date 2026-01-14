#ifndef GPU_FIRST_ORDER_SOLVER_HPP
#define GPU_FIRST_ORDER_SOLVER_HPP

#include "VirtualSolver.hpp"

// Forward declaration to hide CUDA details from the compiler when compiling main.cpp
struct GPUMesh;

class GPUFirstOrderSolver : public VirtualSolver {
public:
    GPUFirstOrderSolver();
    virtual ~GPUFirstOrderSolver();

    // The entry point that main.cpp calls
    PetscErrorCode Run(int argc, char **argv) override;

private:
    // Opaque pointer to GPU data
    GPUMesh* gpu_mesh; 
    
    // Config
    PetscInt max_steps;
    
    // Helpers
    PetscErrorCode UploadTopology();
    void FreeTopology();
    PetscReal ComputeDT_GPU();
    PetscErrorCode ComputeRHS_GPU(Vec X, Vec F);
    PetscErrorCode UpdateSolution_GPU(PetscReal dt, Vec X, Vec F);
};

#endif