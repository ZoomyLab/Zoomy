static char help[] = "Zoomy-Core Driven Finite Volume Solver using PETSc DMPlex.\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscfv.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cmath>

// --- Import Generated Code ---
#include "Model.H"
#include "Numerics.H"

using Real = PetscScalar;

// =========================================================================
//  BASE SOLVER CLASS
// =========================================================================
class PetscSolver {
protected:
    DM          dm;
    TS          ts;
    PetscDS     prob;
    Vec         X;
    MPI_Comm    comm;
    PetscMPIInt rank;
    
    // Time Stepping Parameters
    PetscReal   cfl;
    PetscReal   minRadius; // dx_min

public:
    PetscSolver() : comm(PETSC_COMM_WORLD), cfl(0.5) { 
        MPI_Comm_rank(comm, &rank);
        dm = NULL; ts = NULL; prob = NULL; X = NULL;
    }

    virtual ~PetscSolver() {
        if (X)    VecDestroy(&X);
        if (ts)   TSDestroy(&ts);
        if (dm)   DMDestroy(&dm);
    }

    // --- High-Level Driver ---
    PetscErrorCode Run(int argc, char **argv) {
        PetscFunctionBeginUser;
        
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-ufv_cfl", &cfl, NULL));

        PetscCall(SetupMesh());
        PetscCall(SetupDiscretization()); 
        PetscCall(SetupTimeStepping());
        PetscCall(SetupInitialConditions());

        PetscCall(TSSolve(ts, X));

        PetscFunctionReturn(PETSC_SUCCESS);
    }

protected:
    PetscErrorCode SetupMesh() {
        PetscFunctionBeginUser;
        PetscCall(DMCreate(comm, &dm));
        PetscCall(DMSetType(dm, DMPLEX));
        PetscCall(DMSetFromOptions(dm)); 
        
        // --- 1. Ensure Adjacency is set for FV ---
        // FIX: Use PETSC_TRUE / PETSC_FALSE
        PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));

        // --- 2. Construct Ghost Cells ---
        {
            DM gdm;
            PetscCall(DMPlexConstructGhostCells(dm, NULL, NULL, &gdm));
            PetscCall(DMDestroy(&dm)); 
            dm = gdm; 
        }

        PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

        // Compute geometry once
        PetscCall(DMPlexGetGeometryFVM(dm, NULL, NULL, &minRadius));
        PetscPrintf(comm, "  [Mesh] Min Cell Radius (dx) = %g\n", (double)minRadius);

        PetscInt meshDim;
        PetscCall(DMGetDimension(dm, &meshDim));
        if (meshDim != Model<Real>::dimension) {
            PetscPrintf(comm, "Warning: Mesh dimension (%d) != Model dimension (%d)\n", meshDim, Model<Real>::dimension);
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    virtual PetscErrorCode SetupDiscretization() = 0; 
    
    virtual PetscErrorCode ComputeTimeStep(TS ts) = 0;

    static PetscErrorCode PreStepWrapper(TS ts) {
        PetscFunctionBeginUser;
        void *ctx;
        PetscCall(TSGetApplicationContext(ts, &ctx));
        PetscSolver* solver = static_cast<PetscSolver*>(ctx);
        PetscCall(solver->ComputeTimeStep(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    virtual PetscErrorCode SetupTimeStepping() {
        PetscFunctionBeginUser;
        PetscCall(TSCreate(comm, &ts));
        PetscCall(TSSetDM(ts, dm));
        PetscCall(TSSetApplicationContext(ts, this)); 
        
        PetscCall(TSSetType(ts, TSSSP)); 
        PetscCall(TSSetMaxTime(ts, 1.0));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
        
        PetscCall(TSSetPreStep(ts, PreStepWrapper));

        PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL));
        PetscCall(DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, NULL));

        PetscCall(TSSetFromOptions(ts)); 
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupInitialConditions() {
        PetscFunctionBeginUser;
        PetscCall(DMCreateGlobalVector(dm, &X));
        PetscCall(PetscObjectSetName((PetscObject)X, "solution"));

        PetscErrorCode (*ic_funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*);
        ic_funcs[0] = InitialConditionWrapper;
        void* ctxs[1] = {NULL};
        
        PetscCall(DMProjectFunction(dm, 0.0, ic_funcs, ctxs, INSERT_ALL_VALUES, X));
        PetscCall(TSSetSolution(ts, X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode InitialConditionWrapper(PetscInt dim, PetscReal time, const PetscReal x[], 
                                                  PetscInt Nf, PetscScalar *u, void *ctx) {
        Real x_dam = 5.0; 
        Real h_L   = 2.0; 
        Real h_R   = 1.0; 

        for(int i=0; i<Model<Real>::n_dof_q; ++i) u[i] = 0.0;

        if (x[0] <= x_dam) u[0] = h_L;
        else               u[0] = h_R;
        
        return PETSC_SUCCESS;
    }
};

// =========================================================================
//  FVM SOLVER IMPLEMENTATION
// =========================================================================
class FVMSolver : public PetscSolver {
protected:
    PetscFV fvm;
    std::vector<int> bc_indices; 

public:
    FVMSolver() { fvm = NULL; }
    ~FVMSolver() {
        if (fvm) PetscFVDestroy(&fvm);
    }

    // --- ADAPTIVE TIME STEPPING LOGIC ---
    PetscErrorCode ComputeTimeStep(TS ts) override {
        PetscFunctionBeginUser;
        
        Vec X_global, X_local;
        PetscCall(TSGetSolution(ts, &X_global));
        PetscCall(DMGetLocalVector(dm, &X_local));
        
        PetscCall(DMGlobalToLocalBegin(dm, X_global, INSERT_VALUES, X_local));
        PetscCall(DMGlobalToLocalEnd(dm, X_global, INSERT_VALUES, X_local));

        const PetscScalar *x_ptr;
        PetscCall(VecGetArrayRead(X_local, &x_ptr));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetDepthStratum(dm, 0, &cStart, &cEnd)); 

        Real global_max_eigenvalue = 0.0;
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0}; 
        std::vector<Real> lam(Model<Real>::n_dof_q); 
        
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const Real* Q_cell = &x_ptr[c * Model<Real>::n_dof_q];

            // Safety: Skip Dry Cells
            if (Q_cell[0] < 1e-6) continue;

            Real max_lam_cell = 0.0;
            for (int d = 0; d < Model<Real>::dimension; ++d) {
                Real n[3] = {0.0, 0.0, 0.0};
                n[d] = 1.0;
                Model<Real>::eigenvalues(Q_cell, Qaux, n, lam.data());
                for(Real val : lam) max_lam_cell = std::max(max_lam_cell, std::abs(val));
            }
            global_max_eigenvalue = std::max(global_max_eigenvalue, max_lam_cell);
        }
        
        PetscCall(VecRestoreArrayRead(X_local, &x_ptr));
        PetscCall(DMRestoreLocalVector(dm, &X_local));

        Real all_max_eigenvalue;
        PetscCallMPI(MPI_Allreduce(&global_max_eigenvalue, &all_max_eigenvalue, 1, MPIU_REAL, MPI_MAX, comm));

        Real dt;
        if (all_max_eigenvalue > 1e-12) dt = cfl * minRadius / all_max_eigenvalue;
        else dt = 1e-3; 

        PetscCall(TSSetTimeStep(ts, dt));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

protected:
    PetscErrorCode SetupDiscretization() override {
        PetscFunctionBeginUser;
        PetscCall(PetscFVCreate(comm, &fvm));
        PetscCall(ConfigureFV()); 

        PetscCall(PetscFVSetNumComponents(fvm, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvm, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_q; ++i) {
            char name[32]; snprintf(name, 32, "Field_%d", i);
            PetscCall(PetscFVSetComponentName(fvm, i, name));
        }
        PetscCall(PetscFVSetFromOptions(fvm));

        PetscCall(DMAddField(dm, NULL, (PetscObject)fvm));
        PetscCall(DMCreateDS(dm));
        PetscCall(DMGetDS(dm, &prob));

        PetscCall(PetscDSSetRiemannSolver(prob, 0, RiemannAdapter));
        PetscCall(SetupBoundaries());

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    virtual PetscErrorCode ConfigureFV() {
        PetscFunctionBeginUser;
        PetscCall(PetscFVSetType(fvm, PETSCFVLEASTSQUARES));
        PetscLimiter lim;
        PetscCall(PetscFVGetLimiter(fvm, &lim));
        PetscCall(PetscLimiterSetType(lim, PETSCLIMITERMINMOD));
        PetscCall(PetscLimiterSetFromOptions(lim));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    PetscErrorCode SetupBoundaries() {
        PetscFunctionBeginUser;
        
        std::vector<std::string> names = Model<Real>::get_boundary_tags();
        std::vector<std::string> id_strs = Model<Real>::get_boundary_tag_ids();
        
        if (names.size() != id_strs.size()) SETERRQ(comm, PETSC_ERR_ARG_SIZ, "Model Error: Tags/IDs size mismatch!");

        DMLabel faceSetsLabel = NULL;
        PetscCall(DMGetLabel(dm, "Face Sets", &faceSetsLabel));
        
        std::vector<std::string> missing_bcs;
        bc_indices.resize(names.size());

        for (size_t i = 0; i < names.size(); ++i) {
            bc_indices[i] = (int)i; 
            PetscInt expected_id = std::stoi(id_strs[i]);
            
            // FIX: Use C++ bool and PETSC_TRUE/FALSE check
            bool local_has = false;
            if (faceSetsLabel) {
                PetscBool has;
                PetscCall(DMLabelHasStratum(faceSetsLabel, expected_id, &has));
                if(has == PETSC_TRUE) local_has = true;
            }

            // FIX: Use MPI_C_BOOL for C++ bool types
            bool global_has = false;
            PetscCallMPI(MPI_Allreduce(&local_has, &global_has, 1, MPI_C_BOOL, MPI_LOR, comm));

            if (global_has) {
                if (local_has) {
                    std::string func_name = "ZoomyBC_" + names[i];
                    PetscCall(PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, 
                                                 func_name.c_str(), faceSetsLabel, 
                                                 1, &expected_id, 
                                                 0, 0, NULL, 
                                                 (PetscVoidFn *)BoundaryAdapter, 
                                                 NULL, &bc_indices[i], NULL));
                }
            } else {
                char buffer[128];
                snprintf(buffer, sizeof(buffer), "%s (ID %d)", names[i].c_str(), expected_id);
                missing_bcs.push_back(std::string(buffer));
            }
        }

        if (!missing_bcs.empty()) {
            PetscPrintf(comm, "\n[ERROR] MISSING BOUNDARY IDs (Global Check):\n");
            for(const auto& err : missing_bcs) PetscPrintf(comm, " ! %s\n", err.c_str());
            SETERRQ(comm, PETSC_ERR_USER, "Aborting due to missing boundary conditions.");
        }
        
        PetscFunctionReturn(PETSC_SUCCESS);
    }

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

    static PetscErrorCode BoundaryAdapter(PetscReal time, const PetscReal *c, const PetscReal *n, 
                                          const PetscScalar *xI, PetscScalar *xG, void *ctx) {
        const int bc_idx = *(int*)ctx;
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0};
        Real dX = 0.0;
        Model<Real>::boundary_conditions(bc_idx, xI, Qaux, (const Real*)n, (const Real*)c, time, dX, xG);
        return PETSC_SUCCESS;
    }
};

class FVMTestSolver : public FVMSolver {
protected:
    PetscErrorCode ConfigureFV() override {
        PetscFunctionBeginUser;
        PetscCall(PetscFVSetType(fvm, PETSCFVUPWIND));
        PetscPrintf(comm, "  [TestMode] Forced 1st Order (Upwind)\n");
        PetscFunctionReturn(PETSC_SUCCESS);
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