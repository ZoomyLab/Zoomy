static char help[] = "Simplified 1st Order FVM Solver (Zoomy-Core)\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscfv.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <set>

// --- Import Generated Kernels ---
#include "Model.H"
#include "Numerics.H"

using Real = PetscScalar;

class FVMSolver {
private:
    DM          dm;
    TS          ts;
    PetscDS     prob;
    Vec         X;
    PetscMPIInt rank;
    
    // Config
    PetscReal   cfl;
    PetscReal   minRadius;
    std::vector<PetscInt> bc_ids_storage; // To keep ID memory valid for PETSc

public:
    FVMSolver() : cfl(0.5), minRadius(0.0) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dm = NULL; ts = NULL; X = NULL; prob = NULL;
    }

    ~FVMSolver() {
        if (X)  VecDestroy(&X);
        if (ts) TSDestroy(&ts);
        if (dm) DMDestroy(&dm);
    }

    PetscErrorCode Run(int argc, char **argv) {
        PetscFunctionBeginUser;
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-ufv_cfl", &cfl, NULL));

        PetscCall(SetupMesh());
        
        // --- 1. Global Boundary Check ---
        // Verify Mesh <-> Model consistency before setting up solvers
        PetscCall(CheckBoundaryConditionCoverage());

        PetscCall(SetupDiscretization()); 
        PetscCall(SetupTimeStepping());
        PetscCall(SetupInitialConditions());

        PetscCall(TSSolve(ts, X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    // -------------------------------------------------------------------------
    // MESH SETUP
    // -------------------------------------------------------------------------
    PetscErrorCode SetupMesh() {
        PetscFunctionBeginUser;
        
        // Load Mesh
        PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
        PetscCall(DMSetType(dm, DMPLEX));
        PetscCall(DMSetFromOptions(dm)); 

        // Essential for FVM: Valid Face Geometry & Ghost Cells
        PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
        
        // Distribute mesh
        PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

        // Construct Ghost Cells for Multi-Core
        {
            DM gdm;
            PetscCall(DMPlexConstructGhostCells(dm, NULL, NULL, &gdm));
            PetscCall(DMDestroy(&dm));
            dm = gdm;
        }

        // Pre-compute cell geometry
        PetscCall(DMPlexGetGeometryFVM(dm, NULL, NULL, &minRadius));
        PetscPrintf(PETSC_COMM_WORLD, "  [Mesh] Min Cell Radius (dx) = %g\n", (double)minRadius);

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // -------------------------------------------------------------------------
    // SAFETY CHECK: MESH BOUNDARIES VS MODEL
    // -------------------------------------------------------------------------
    PetscErrorCode CheckBoundaryConditionCoverage() {
        PetscFunctionBeginUser;
        
        // 1. Gather all Boundary IDs defined in Model.H
        auto model_ids_str = Model<Real>::get_boundary_tag_ids();
        std::set<PetscInt> model_defined_ids;
        for(const auto& s : model_ids_str) {
            model_defined_ids.insert(std::stoi(s));
        }

        // 2. Get all Boundary IDs present in the Mesh ("Face Sets" label)
        DMLabel label;
        PetscCall(DMGetLabel(dm, "Face Sets", &label));
        if (!label) {
             // If no face sets exist, assume closed periodic or sphere (warn slightly)
             PetscPrintf(PETSC_COMM_WORLD, "  [Check] Warning: No 'Face Sets' found in mesh. Assuming no physical boundaries.\n");
             PetscFunctionReturn(PETSC_SUCCESS);
        }

        IS face_is;
        PetscCall(DMLabelGetValueIS(label, &face_is));
        
        const PetscInt *mesh_ids;
        PetscInt n_mesh_ids;
        if (face_is) {
            PetscCall(ISGetLocalSize(face_is, &n_mesh_ids));
            PetscCall(ISGetIndices(face_is, &mesh_ids));
        } else {
            n_mesh_ids = 0;
            mesh_ids = NULL;
        }

        // 3. Verify: Does every Mesh ID exist in Model?
        std::vector<PetscInt> unknown_ids;
        for(PetscInt i=0; i<n_mesh_ids; ++i) {
            if (model_defined_ids.find(mesh_ids[i]) == model_defined_ids.end()) {
                unknown_ids.push_back(mesh_ids[i]);
            }
        }

        if (face_is) {
            PetscCall(ISRestoreIndices(face_is, &mesh_ids));
            PetscCall(ISDestroy(&face_is));
        }

        // 4. Report Errors
        PetscInt local_err = unknown_ids.empty() ? 0 : 1;
        PetscInt global_err = 0;
        PetscCallMPI(MPI_Allreduce(&local_err, &global_err, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));

        if (global_err) {
            // Let every rank with an error print it (synchronized to avoid garbled text)
            for (auto id : unknown_ids) {
                PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  [Error] Rank %d: Mesh has Boundary ID %d, but Model.H does not define it!\n", rank, id);
            }
            PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Aborting due to undefined boundary conditions in mesh.");
        } 
        
        PetscPrintf(PETSC_COMM_WORLD, "  [Check] All mesh boundaries are covered by Model.H.\n");
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // -------------------------------------------------------------------------
    // DISCRETIZATION
    // -------------------------------------------------------------------------
    PetscErrorCode SetupDiscretization() {
        PetscFunctionBeginUser;

        PetscFV fvm;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvm));
        
        // FORCE 1st Order (Upwind) for Stability
        PetscCall(PetscFVSetType(fvm, PETSCFVUPWIND)); 
        
        PetscCall(PetscFVSetNumComponents(fvm, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvm, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_q; ++i) {
            char name[32]; snprintf(name, 32, "Field_%d", i);
            PetscCall(PetscFVSetComponentName(fvm, i, name));
        }
        
        PetscCall(DMAddField(dm, NULL, (PetscObject)fvm));
        PetscCall(DMCreateDS(dm));
        PetscCall(DMGetDS(dm, &prob));
        
        // Riemann Solver (Rusanov)
        PetscCall(PetscDSSetRiemannSolver(prob, 0, RiemannAdapter));

        // Map Boundaries
        DMLabel label;
        PetscCall(DMGetLabel(dm, "Face Sets", &label));
        
        auto names = Model<Real>::get_boundary_tags();
        auto ids   = Model<Real>::get_boundary_tag_ids();

        bc_ids_storage.resize(names.size());

        for(size_t i=0; i<names.size(); ++i) {
            PetscInt id_val = std::stoi(ids[i]);
            bc_ids_storage[i] = (PetscInt)i; // Model Context Index

            PetscCall(PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, 
                                         names[i].c_str(), 
                                         label, 
                                         1, &id_val, 
                                         0, 0, NULL, 
                                         (PetscVoidFn *)BoundaryAdapter, 
                                         NULL, 
                                         &bc_ids_storage[i], 
                                         NULL));
        }

        PetscCall(PetscFVDestroy(&fvm));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // -------------------------------------------------------------------------
    // ADAPTIVE TIME STEPPING
    // -------------------------------------------------------------------------
    static PetscErrorCode PreStepWrapper(TS ts) {
        PetscFunctionBeginUser;
        void *ctx;
        PetscCall(TSGetApplicationContext(ts, &ctx));
        FVMSolver* solver = static_cast<FVMSolver*>(ctx);
        PetscCall(solver->ComputeTimeStep(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode ComputeTimeStep(TS ts) {
        PetscFunctionBeginUser;
        
        Vec X_global, X_local;
        PetscCall(TSGetSolution(ts, &X_global));
        PetscCall(DMGetLocalVector(dm, &X_local));
        
        // Update Ghosts
        PetscCall(DMGlobalToLocalBegin(dm, X_global, INSERT_VALUES, X_local));
        PetscCall(DMGlobalToLocalEnd(dm, X_global, INSERT_VALUES, X_local));

        const PetscScalar *x_ptr;
        PetscCall(VecGetArrayRead(X_local, &x_ptr));

        // Iterate over CELLS (Height 0)
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd)); 

        Real max_eigen_local = 0.0;
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0}; 
        std::vector<Real> lam(Model<Real>::n_dof_q); 
        
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const Real* Q_cell = &x_ptr[c * Model<Real>::n_dof_q];

            // Filter dry cells
            if (Q_cell[0] < 1e-6) continue; 

            for (int d = 0; d < Model<Real>::dimension; ++d) {
                Real n[3] = {0.0, 0.0, 0.0};
                n[d] = 1.0;
                Model<Real>::eigenvalues(Q_cell, Qaux, n, lam.data());
                for(Real val : lam) max_eigen_local = std::max(max_eigen_local, std::abs(val));
            }
        }
        
        PetscCall(VecRestoreArrayRead(X_local, &x_ptr));
        PetscCall(DMRestoreLocalVector(dm, &X_local));

        // Global Reduce
        Real max_eigen_global;
        PetscCallMPI(MPI_Allreduce(&max_eigen_local, &max_eigen_global, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));

        Real dt;
        if (max_eigen_global > 1e-12) dt = cfl * minRadius / max_eigen_global;
        else dt = 1e-4; // Fallback for initial flat state

        PetscPrintf(PETSC_COMM_WORLD, "  [Step] Max Eigen: %g | dt: %g\n", (double)max_eigen_global, (double)dt);

        PetscCall(TSSetTimeStep(ts, dt));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupTimeStepping() {
        PetscFunctionBeginUser;
        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
        PetscCall(TSSetDM(ts, dm));
        PetscCall(TSSetApplicationContext(ts, this));
        
        PetscCall(TSSetType(ts, TSSSP)); // Strong Stability Preserving
        PetscCall(TSSetMaxTime(ts, 1.0));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
        
        PetscCall(TSSetPreStep(ts, PreStepWrapper));

        PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL));
        PetscCall(DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, NULL));

        PetscCall(TSSetFromOptions(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // -------------------------------------------------------------------------
    // HELPERS & ADAPTERS
    // -------------------------------------------------------------------------
    PetscErrorCode SetupInitialConditions() {
        PetscFunctionBeginUser;
        PetscCall(DMCreateGlobalVector(dm, &X));
        PetscCall(PetscObjectSetName((PetscObject)X, "solution"));

        PetscErrorCode (*ic_funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*);
        ic_funcs[0] = InitialCondition;
        void* ctxs[1] = {NULL};
        
        PetscCall(DMProjectFunction(dm, 0.0, ic_funcs, ctxs, INSERT_ALL_VALUES, X));
        PetscCall(TSSetSolution(ts, X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], 
                                           PetscInt Nf, PetscScalar *u, void *ctx) {
        Real x_dam = 5.0; 
        for(int i=0; i<Model<Real>::n_dof_q; ++i) u[i] = 0.0;
        
        // Simple Dam Break Setup
        if (x[0] <= x_dam) u[0] = 2.0; 
        else               u[0] = 1.0;
        return PETSC_SUCCESS;
    }

    // Wrapper: Numerics::numerical_flux -> PETSc Riemann
    static void RiemannAdapter(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, 
                               const PetscScalar *xL, const PetscScalar *xR, 
                               PetscInt numConstants, const PetscScalar constants[], 
                               PetscScalar *flux, void *ctx) {
        Real Qaux_L[Model<Real>::n_dof_qaux] = {0.0};
        Real Qaux_R[Model<Real>::n_dof_qaux] = {0.0};
        
        Real area = 0.0;
        for(int d=0; d<dim; ++d) area += n[d]*n[d];
        area = std::sqrt(area);

        Real n_hat[3] = {0.0, 0.0, 0.0};
        if (area > 1e-14) {
            for(int d=0; d<dim; ++d) n_hat[d] = n[d] / area;
        } else {
             for(int d=0; d<dim; ++d) flux[d] = 0.0;
             return;
        }

        Real flux_per_area[Model<Real>::n_dof_q];
        Numerics<Real>::numerical_flux(xL, xR, Qaux_L, Qaux_R, (const Real*)n_hat, flux_per_area);

        for(int i=0; i<Model<Real>::n_dof_q; ++i) {
            flux[i] = flux_per_area[i] * area;
        }
    }

    // Wrapper: Model::boundary_conditions -> PETSc BC
    static PetscErrorCode BoundaryAdapter(PetscReal time, const PetscReal *c, const PetscReal *n, 
                                          const PetscScalar *xI, PetscScalar *xG, void *ctx) {
        const int bc_idx = *(int*)ctx; 
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0};
        Real dX = 0.0; 
        
        Model<Real>::boundary_conditions(bc_idx, xI, Qaux, (const Real*)n, (const Real*)c, time, dX, xG);
        return PETSC_SUCCESS;
    }
};

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    {
        FVMSolver solver; 
        PetscCall(solver.Run(argc, argv));
    }
    PetscCall(PetscFinalize());
    return 0;
}