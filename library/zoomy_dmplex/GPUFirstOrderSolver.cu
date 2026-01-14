#include "GPUFirstOrderSolver.hpp"
#include "Numerics.H" 

#include <cuda_runtime.h>
#include <petscdevice.h> 

// Use double precision for PETSc compatibility
using Real = PetscReal;

// --- GPU Data Structure Definition ---
struct GPUMesh {
    PetscInt n_faces;
    PetscInt n_cells;
    
    // Device Pointers
    PetscInt *L;
    PetscInt *R;
    PetscReal *normals;      // [n_faces * 3] (Assuming 3D capacity, used as 2D)
    PetscReal *cell_volumes; // [n_cells]
    
    // Temporary Device Buffers for Reductions
    PetscReal *d_eigenvalues;
};

// --- CUDA Helper: Atomic Add for Double ---
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// ============================================================================
//                                 KERNELS
// ============================================================================

__global__ void FluxKernel(
    int n_faces, 
    const PetscInt* L, const PetscInt* R, 
    const PetscReal* normals, 
    const PetscScalar* Q, const PetscScalar* Aux,
    PetscScalar* RHS
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    // 1. Connectivity
    int cL = L[f]; 
    int cR = R[f];

    // 2. Data Pointers
    int stride = Numerics<Real>::n_dof_q;
    const PetscScalar* qL = &Q[cL * stride];
    const PetscScalar* qR = &Q[cR * stride];
    const PetscScalar* auxL = (Aux) ? &Aux[cL * stride] : nullptr; 
    const PetscScalar* auxR = (Aux) ? &Aux[cR * stride] : nullptr; 

    // 3. Geometry (Assuming 2D for now, taking first 2 components)
    // Numerics expects a pointer to the normal vector
    const PetscReal* n = &normals[f * 2]; 

    // 4. Flux Calculation (Physics)
    PetscScalar flux[Numerics<Real>::n_dof_q];
    Numerics<Real>::numerical_flux(qL, qR, auxL, auxR, n, flux);

    // 5. Accumulate to Cells
    for(int i=0; i<stride; ++i) {
        atomicAdd(&RHS[cL * stride + i], -flux[i]);
        atomicAdd(&RHS[cR * stride + i],  flux[i]);
    }
}

__global__ void TimeStepKernel(
    int n_cells, 
    const PetscScalar* Q, const PetscScalar* Aux,
    PetscReal* max_eigen_per_cell
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    int stride = Numerics<Real>::n_dof_q;
    const PetscScalar* q = &Q[c * stride];

    if (q[0] < 1e-6) { max_eigen_per_cell[c] = 0.0; return; }

    const PetscScalar* aux = (Aux) ? &Aux[c * stride] : nullptr; 

    // Check Cartesian directions for max wave speed
    PetscReal local_max = 0.0;
    PetscReal res[1];
    
    // X Direction
    PetscReal n_x[3] = {1.0, 0.0, 0.0};
    Numerics<Real>::local_max_abs_eigenvalue(q, aux, n_x, res);
    if(res[0] > local_max) local_max = res[0];

    // Y Direction
    PetscReal n_y[3] = {0.0, 1.0, 0.0};
    Numerics<Real>::local_max_abs_eigenvalue(q, aux, n_y, res);
    if(res[0] > local_max) local_max = res[0];
    
    max_eigen_per_cell[c] = local_max;
}

__global__ void UpdateKernel(
    int n_cells, PetscReal dt,
    const PetscReal* volumes,
    const PetscScalar* RHS,
    PetscScalar* Q
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    int stride = Numerics<Real>::n_dof_q;
    PetscReal inv_vol = 1.0 / volumes[c];

    for(int i=0; i<stride; ++i) {
        // Q_new = Q_old + (dt/Vol) * RHS
        int idx = c * stride + i;
        Q[idx] += dt * RHS[idx] * inv_vol;
    }
}

// ============================================================================
//                            CLASS IMPLEMENTATION
// ============================================================================

GPUFirstOrderSolver::GPUFirstOrderSolver() : VirtualSolver(), max_steps(1000) {
    gpu_mesh = new GPUMesh();
    gpu_mesh->L = nullptr; // Initialize to null
}

GPUFirstOrderSolver::~GPUFirstOrderSolver() {
    FreeTopology();
    delete gpu_mesh;
}

PetscErrorCode GPUFirstOrderSolver::Run(int argc, char **argv) {
    PetscFunctionBeginUser;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-ufv_cfl", &cfl, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-ts_max_steps", &max_steps, NULL));

    // 1. Setup Architecture (Base Class - CPU Side)
    PetscCall(SetupArchitecture(1));
    
    // 2. Upload Mesh to GPU
    PetscCall(UploadTopology());

    // 3. Prepare Vectors on GPU
    Vec F; 
    PetscCall(VecDuplicate(X, &F));
    
    // Mark them as CUDA vectors so PETSc allocates device memory
    PetscCall(VecSetType(X, VECCUDA));
    PetscCall(VecSetType(F, VECCUDA));
    if(A) PetscCall(VecSetType(A, VECCUDA));

    PetscPrintf(PETSC_COMM_WORLD, "--- GPU First Order Solver Started ---\n");

    PetscReal time = 0.0;
    PetscInt step = 0;

    // Initial Output (Automatically pulls data back to Host)
    PetscCall(WriteVTU(step, time));

    // 4. Time Loop
    while(step < max_steps) {
        // A. Adaptive Time Step
        PetscReal dt = ComputeDT_GPU();
        
        // B. Compute Fluxes (RHS)
        PetscCall(ComputeRHS_GPU(X, F));

        // C. Update Solution
        PetscCall(UpdateSolution_GPU(dt, X, F));

        time += dt;
        step++;

        if (step % 10 == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Step %4d | Time %.5e | dt %.5e\n", step, time, dt);
        }
        if (step % 100 == 0) {
            PetscCall(WriteVTU(step, time));
        }
    }

    PetscCall(VecDestroy(&F));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GPUFirstOrderSolver::UploadTopology() {
    PetscFunctionBeginUser;
    // 1. Get Sizes
    PetscInt fStart, fEnd, cStart, cEnd;
    DMPlexGetHeightStratum(dmQ, 1, &fStart, &fEnd);
    DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd);
    
    gpu_mesh->n_faces = fEnd - fStart;
    gpu_mesh->n_cells = cEnd - cStart;

    // 2. Extract Connectivity (Host)
    std::vector<PetscInt> h_L(gpu_mesh->n_faces);
    std::vector<PetscInt> h_R(gpu_mesh->n_faces);
    
    for(PetscInt f = fStart; f < fEnd; ++f) {
        const PetscInt *supp;
        DMPlexGetSupport(dmQ, f, &supp);
        PetscInt suppSize;
        DMPlexGetSupportSize(dmQ, f, &suppSize);
        h_L[f - fStart] = supp[0];
        h_R[f - fStart] = (suppSize > 1) ? supp[1] : supp[0];
    }

    // 3. Extract Geometry (Host)
    Vec faceGeom, cellGeom;
    DMGetGeometryFVM(dmQ, &faceGeom, &cellGeom, NULL);
    
    // Faces (Normals)
    const PetscScalar* f_ptr;
    VecGetArrayRead(faceGeom, &f_ptr);
    const PetscFVFaceGeom* faces = (const PetscFVFaceGeom*)f_ptr;
    std::vector<PetscReal> h_normals(gpu_mesh->n_faces * 2);
    for(int i=0; i<gpu_mesh->n_faces; ++i) {
        h_normals[i*2 + 0] = faces[i].normal[0];
        h_normals[i*2 + 1] = faces[i].normal[1];
    }
    VecRestoreArrayRead(faceGeom, &f_ptr);

    // Cells (Volumes)
    const PetscScalar* c_ptr;
    VecGetArrayRead(cellGeom, &c_ptr);
    const PetscFVCellGeom* cells = (const PetscFVCellGeom*)c_ptr;
    std::vector<PetscReal> h_vols(gpu_mesh->n_cells);
    for(int i=0; i<gpu_mesh->n_cells; ++i) {
        h_vols[i] = cells[i].volume;
    }
    VecRestoreArrayRead(cellGeom, &c_ptr);

    // 4. Allocate and Copy to GPU
    cudaMalloc(&gpu_mesh->L, gpu_mesh->n_faces * sizeof(PetscInt));
    cudaMalloc(&gpu_mesh->R, gpu_mesh->n_faces * sizeof(PetscInt));
    cudaMalloc(&gpu_mesh->normals, gpu_mesh->n_faces * 2 * sizeof(PetscReal));
    cudaMalloc(&gpu_mesh->cell_volumes, gpu_mesh->n_cells * sizeof(PetscReal));
    cudaMalloc(&gpu_mesh->d_eigenvalues, gpu_mesh->n_cells * sizeof(PetscReal));

    cudaMemcpy(gpu_mesh->L, h_L.data(), gpu_mesh->n_faces * sizeof(PetscInt), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mesh->R, h_R.data(), gpu_mesh->n_faces * sizeof(PetscInt), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mesh->normals, h_normals.data(), gpu_mesh->n_faces * 2 * sizeof(PetscReal), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mesh->cell_volumes, h_vols.data(), gpu_mesh->n_cells * sizeof(PetscReal), cudaMemcpyHostToDevice);

    PetscFunctionReturn(PETSC_SUCCESS);
}

void GPUFirstOrderSolver::FreeTopology() {
    if (gpu_mesh->L) cudaFree(gpu_mesh->L);
    if (gpu_mesh->R) cudaFree(gpu_mesh->R);
    if (gpu_mesh->normals) cudaFree(gpu_mesh->normals);
    if (gpu_mesh->cell_volumes) cudaFree(gpu_mesh->cell_volumes);
    if (gpu_mesh->d_eigenvalues) cudaFree(gpu_mesh->d_eigenvalues);
}

PetscReal GPUFirstOrderSolver::ComputeDT_GPU() {
    // 1. Get Pointers
    const PetscScalar *d_X, *d_A = nullptr;
    VecCUDAGetArrayRead(X, &d_X);
    if (A) VecCUDAGetArrayRead(A, &d_A);

    // 2. Launch Kernel
    int blockSize = 256;
    int numBlocks = (gpu_mesh->n_cells + blockSize - 1) / blockSize;
    
    TimeStepKernel<<<numBlocks, blockSize>>>(
        gpu_mesh->n_cells, d_X, d_A, gpu_mesh->d_eigenvalues
    );

    // 3. Restore Vectors
    VecCUDARestoreArrayRead(X, &d_X);
    if (A) VecCUDARestoreArrayRead(A, &d_A);

    // 4. Reduce Max (Naive Host Copy for simplicity - optimized versions use Thrust)
    std::vector<PetscReal> h_eigen(gpu_mesh->n_cells);
    cudaMemcpy(h_eigen.data(), gpu_mesh->d_eigenvalues, gpu_mesh->n_cells * sizeof(PetscReal), cudaMemcpyDeviceToHost);
    
    PetscReal max_eigen = 0.0;
    for(auto val : h_eigen) if(val > max_eigen) max_eigen = val;

    // 5. Global Reduce (MPI)
    PetscReal max_eigen_global;
    MPI_Allreduce(&max_eigen, &max_eigen_global, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD);

    if (max_eigen_global > 1e-12) {
        return cfl * minRadius / max_eigen_global;
    }
    return 1e-4;
}

PetscErrorCode GPUFirstOrderSolver::ComputeRHS_GPU(Vec X, Vec F) {
    const PetscScalar *d_X, *d_A = nullptr;
    PetscScalar *d_F;

    VecCUDAGetArrayRead(X, &d_X);
    if(A) VecCUDAGetArrayRead(A, &d_A);
    VecCUDAGetArray(F, &d_F);

    // Zero RHS
    cudaMemset(d_F, 0, gpu_mesh->n_cells * Numerics<Real>::n_dof_q * sizeof(PetscScalar));

    int blockSize = 256;
    int numBlocks = (gpu_mesh->n_faces + blockSize - 1) / blockSize;

    FluxKernel<<<numBlocks, blockSize>>>(
        gpu_mesh->n_faces,
        gpu_mesh->L, gpu_mesh->R,
        gpu_mesh->normals,
        d_X, d_A, d_F
    );

    VecCUDARestoreArrayRead(X, &d_X);
    if(A) VecCUDARestoreArrayRead(A, &d_A);
    VecCUDARestoreArray(F, &d_F);
    return PETSC_SUCCESS;
}

PetscErrorCode GPUFirstOrderSolver::UpdateSolution_GPU(PetscReal dt, Vec X, Vec F) {
    PetscScalar *d_X;
    const PetscScalar *d_F;
    
    VecCUDAGetArray(X, &d_X);
    VecCUDAGetArrayRead(F, &d_F);

    int blockSize = 256;
    int numBlocks = (gpu_mesh->n_cells + blockSize - 1) / blockSize;

    UpdateKernel<<<numBlocks, blockSize>>>(
        gpu_mesh->n_cells, dt, gpu_mesh->cell_volumes, d_F, d_X
    );

    VecCUDARestoreArray(X, &d_X);
    VecCUDARestoreArrayRead(F, &d_F);
    return PETSC_SUCCESS;
}