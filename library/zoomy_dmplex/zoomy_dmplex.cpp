static char help[] = "1st Order FVM (True Single-File Output)\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscfv.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include "Model.H"
#include "Numerics.H"

using Real = PetscScalar;

class FVMSolver {
private:
    DM          dmMesh;     // Base Topology
    DM          dmQ;        // Primary DM (Solver)
    DM          dmAux;      // Auxiliary DM
    DM          dmOut;      // Output DM (Combined Layout)
    
    TS          ts;
    PetscDS     prob;
    
    Vec         X;          // Solution Vector
    Vec         A;          // Aux Vector
    Vec         X_out;      // Combined Output Vector
    
    PetscMPIInt rank;
    PetscReal   cfl;
    PetscReal   minRadius;
    std::vector<PetscInt> bc_ids_storage; 
    
    struct StepData { PetscInt step; PetscReal time; };
    std::vector<StepData> time_series;

public:
    FVMSolver() : cfl(0.5), minRadius(0.0) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dmMesh = NULL; dmQ = NULL; dmAux = NULL; dmOut = NULL;
        ts = NULL; X = NULL; A = NULL; X_out = NULL; prob = NULL;
    }

    ~FVMSolver() {
        if (X)      VecDestroy(&X);
        if (A)      VecDestroy(&A);
        if (X_out)  VecDestroy(&X_out);
        if (ts)     TSDestroy(&ts);
        
        if (dmQ)    DMDestroy(&dmQ);
        if (dmAux)  DMDestroy(&dmAux);
        if (dmOut)  DMDestroy(&dmOut);
        if (dmMesh) DMDestroy(&dmMesh);
    }

    PetscErrorCode Run(int argc, char **argv) {
        PetscFunctionBeginUser;
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-ufv_cfl", &cfl, NULL));

        PetscCall(SetupBaseMesh());
        PetscCall(SetupPrimaryDM());
        PetscCall(SetupAuxiliaryDM());
        
        // Setup the merged output structure
        PetscCall(SetupOutputDM()); 
        
        PetscCall(SetupTimeStepping());
        PetscCall(SetupInitialConditions());

        PetscCall(UpdateBoundaryGhosts(0.0));

        PetscCall(MonitorSeries(ts, 0, 0.0, X, this));

        PetscCall(TSSolve(ts, X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    // -------------------------------------------------------------------------
    // SETUP
    // -------------------------------------------------------------------------
    PetscErrorCode SetupBaseMesh() {
        PetscFunctionBeginUser;
        PetscCall(DMCreate(PETSC_COMM_WORLD, &dmMesh));
        PetscCall(DMSetType(dmMesh, DMPLEX));
        PetscCall(DMSetFromOptions(dmMesh)); 

        DM dmDist = NULL;
        PetscCall(DMPlexDistribute(dmMesh, 1, NULL, &dmDist));
        if (dmDist) { PetscCall(DMDestroy(&dmMesh)); dmMesh = dmDist; }

        {
            DM dmGhost = NULL;
            PetscCall(DMPlexConstructGhostCells(dmMesh, "Face Sets", NULL, &dmGhost));
            if (dmGhost) { PetscCall(DMDestroy(&dmMesh)); dmMesh = dmGhost; }
        }

        PetscCall(DMSetBasicAdjacency(dmMesh, PETSC_TRUE, PETSC_TRUE));
        PetscCall(DMPlexGetGeometryFVM(dmMesh, NULL, NULL, &minRadius));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupPrimaryDM() {
        PetscFunctionBeginUser;
        PetscCall(DMClone(dmMesh, &dmQ)); 
        
        PetscFV fvm;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvm));
        PetscCall(PetscFVSetFromOptions(fvm)); 
        PetscCall(PetscFVSetType(fvm, PETSCFVUPWIND)); 
        PetscCall(PetscFVSetNumComponents(fvm, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvm, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_q; ++i) {
            char name[32]; snprintf(name, 32, "Field_%d", i);
            PetscCall(PetscFVSetComponentName(fvm, i, name));
        }
        PetscCall(DMAddField(dmQ, NULL, (PetscObject)fvm));
        PetscCall(DMCreateDS(dmQ));
        PetscCall(DMGetDS(dmQ, &prob));
        PetscCall(PetscDSSetRiemannSolver(prob, 0, RiemannAdapter));

        DMLabel label;
        PetscCall(DMGetLabel(dmQ, "Face Sets", &label));
        auto names = Model<Real>::get_boundary_tags();
        auto ids   = Model<Real>::get_boundary_tag_ids();
        bc_ids_storage.resize(names.size());
        for(size_t i=0; i<names.size(); ++i) {
            PetscInt id_val = std::stoi(ids[i]);
            bc_ids_storage[i] = (PetscInt)i; 
            PetscCall(PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, names[i].c_str(), label, 1, &id_val, 0, 0, NULL, (PetscVoidFn *)BoundaryAdapter, NULL, &bc_ids_storage[i], NULL));
        }
        PetscCall(PetscFVDestroy(&fvm));
        
        PetscCall(DMCreateGlobalVector(dmQ, &X));
        PetscCall(PetscObjectSetName((PetscObject)X, "Solution"));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupAuxiliaryDM() {
        PetscFunctionBeginUser;
        PetscCall(DMClone(dmMesh, &dmAux)); 
        
        PetscFV fvmAux;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmAux));
        PetscCall(PetscFVSetType(fvmAux, PETSCFVUPWIND)); 
        PetscCall(PetscFVSetNumComponents(fvmAux, Model<Real>::n_dof_qaux));
        PetscCall(PetscFVSetSpatialDimension(fvmAux, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_qaux; ++i) {
            char name[32]; snprintf(name, 32, "Aux_%d", i);
            PetscCall(PetscFVSetComponentName(fvmAux, i, name));
        }
        PetscCall(DMAddField(dmAux, NULL, (PetscObject)fvmAux));
        PetscCall(DMCreateDS(dmAux));
        PetscCall(PetscFVDestroy(&fvmAux));

        PetscCall(DMCreateGlobalVector(dmAux, &A));
        PetscCall(PetscObjectSetName((PetscObject)A, "Auxiliary"));
        PetscCall(VecSet(A, 42.0)); 
        PetscCall(DMSetAuxiliaryVec(dmQ, NULL, 0, 0, A));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // --- MERGED OUTPUT DM SETUP ---
    PetscErrorCode SetupOutputDM() {
        PetscFunctionBeginUser;
        PetscCall(DMClone(dmMesh, &dmOut));
        
        // Create a single Section with 2 Fields:
        // Field 0: Solution (n_dof_q)
        // Field 1: Auxiliary (n_dof_qaux)
        
        // 1. Solution Field
        PetscFV fvmQ;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmQ));
        PetscCall(PetscFVSetType(fvmQ, PETSCFVUPWIND)); 
        PetscCall(PetscFVSetNumComponents(fvmQ, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvmQ, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_q; ++i) {
            char name[32]; snprintf(name, 32, "Q_%d", i); // Prefix to avoid confusion
            PetscCall(PetscFVSetComponentName(fvmQ, i, name));
        }
        
        // 2. Aux Field
        PetscFV fvmAux;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmAux));
        PetscCall(PetscFVSetType(fvmAux, PETSCFVUPWIND)); 
        PetscCall(PetscFVSetNumComponents(fvmAux, Model<Real>::n_dof_qaux));
        PetscCall(PetscFVSetSpatialDimension(fvmAux, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_qaux; ++i) {
            char name[32]; snprintf(name, 32, "Aux_%d", i);
            PetscCall(PetscFVSetComponentName(fvmAux, i, name));
        }

        // Add both fields to dmOut
        PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmQ));
        PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmAux));
        
        // This creates the combined section (interleaved by default per cell)
        PetscCall(DMCreateDS(dmOut));
        
        PetscCall(PetscFVDestroy(&fvmQ));
        PetscCall(PetscFVDestroy(&fvmAux));

        // Create the combined vector
        PetscCall(DMCreateGlobalVector(dmOut, &X_out));
        PetscCall(PetscObjectSetName((PetscObject)X_out, "CombinedState"));
        
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // -------------------------------------------------------------------------
    // MERGING & OUTPUT
    // -------------------------------------------------------------------------
    static PetscErrorCode MonitorSeries(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx) {
        FVMSolver* solver = static_cast<FVMSolver*>(ctx);
        
        if (solver->rank == 0) {
            if (solver->time_series.empty() || solver->time_series.back().step != step) {
                solver->time_series.push_back({step, time});
            }
            std::ofstream f("output.vtu.series");
            if (f.is_open()) {
                f << "{\n  \"file-series-version\" : \"1.0\",\n  \"files\" : [\n";
                for (size_t i = 0; i < solver->time_series.size(); ++i) {
                    f << "    { \"name\" : \"output-" << std::setfill('0') << std::setw(3) << solver->time_series[i].step << ".vtu\", \"time\" : " << std::scientific << std::setprecision(6) << solver->time_series[i].time << " }";
                    if (i < solver->time_series.size() - 1) f << ",";
                    f << "\n";
                }
                f << "  ]\n}\n";
                f.close();
            }
        }

        // --- MERGE DATA ---
        // Copy data from X (dmQ) and A (dmAux) into X_out (dmOut)
        // Since dmQ, dmAux, and dmOut share the topology and cell numbering, 
        // we can iterate over cells and copy directly.
        
        // 1. Get Local Vectors (to handle ghosts if needed, though Global is fine for output)
        //    Using Global access here for simplicity as layout matches.
        const PetscScalar *x_ptr, *a_ptr;
        PetscScalar *out_ptr;
        
        // Note: Using 'u' (current solution) instead of solver->X to catch latest TS state
        PetscCall(VecGetArrayRead(u, &x_ptr));
        PetscCall(VecGetArrayRead(solver->A, &a_ptr));
        PetscCall(VecGetArray(solver->X_out, &out_ptr));
        
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(solver->dmMesh, 0, &cStart, &cEnd)); 
        
        // We need the local section offsets to copy correctly
        // However, since all are FVM cell-centered, they are just contiguous blocks per cell.
        // dmQ stride = n_dof_q
        // dmAux stride = n_dof_qaux
        // dmOut stride = n_dof_q + n_dof_qaux
        
        PetscInt nQ = Model<Real>::n_dof_q;
        PetscInt nAux = Model<Real>::n_dof_qaux;
        
        // We iterate over the local owned cells in the vector
        // VecGetArray returns the local portion of the global vector.
        // The size of this array corresponds to the local number of cells * DOFs.
        
        PetscInt localSize;
        PetscCall(VecGetLocalSize(u, &localSize));
        PetscInt nCells = localSize / nQ; // Number of owned cells
        
        for (PetscInt c = 0; c < nCells; ++c) {
            // Source Offsets
            PetscInt idx_q = c * nQ;
            PetscInt idx_a = c * nAux;
            
            // Dest Offset (Interleaved: Field0 then Field1 per cell)
            PetscInt idx_out = c * (nQ + nAux);
            
            // Copy Q
            for(int i=0; i<nQ; ++i) {
                out_ptr[idx_out + i] = x_ptr[idx_q + i];
            }
            // Copy Aux
            for(int i=0; i<nAux; ++i) {
                out_ptr[idx_out + nQ + i] = a_ptr[idx_a + i];
            }
        }

        PetscCall(VecRestoreArrayRead(u, &x_ptr));
        PetscCall(VecRestoreArrayRead(solver->A, &a_ptr));
        PetscCall(VecRestoreArray(solver->X_out, &out_ptr));

        // --- WRITE FILE ---
        char filename[64];
        snprintf(filename, 64, "output-%03d.vtu", step);
        PetscViewer viewer;
        PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
        
        // Write the Combined Vector using the Combined DM
        PetscCall(VecView(solver->X_out, viewer));
        
        PetscCall(PetscViewerDestroy(&viewer));
        return 0;
    }

    // -------------------------------------------------------------------------
    // STANDARD HELPERS
    // -------------------------------------------------------------------------
    static PetscErrorCode PreStepWrapper(TS ts) {
        PetscFunctionBeginUser;
        void *ctx;
        PetscCall(TSGetApplicationContext(ts, &ctx));
        FVMSolver* solver = static_cast<FVMSolver*>(ctx);
        PetscReal time;
        PetscCall(TSGetTime(ts, &time));
        PetscCall(solver->UpdateBoundaryGhosts(time));
        PetscCall(solver->ComputeTimeStep(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode UpdateBoundaryGhosts(PetscReal time) {
        PetscFunctionBeginUser;
        Vec locX;
        PetscCall(DMGetLocalVector(dmQ, &locX));
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, locX));
        PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, locX));
        PetscCall(DMPlexTSComputeBoundary(dmQ, time, locX, NULL, NULL));
        PetscCall(DMRestoreLocalVector(dmQ, &locX));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode ComputeTimeStep(TS ts) {
        PetscFunctionBeginUser;
        Vec X_local;
        PetscCall(DMGetLocalVector(dmQ, &X_local));
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_local));
        PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_local));
        const PetscScalar *x_ptr;
        PetscCall(VecGetArrayRead(X_local, &x_ptr));
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd)); 
        Real max_eigen_local = 0.0;
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0}; 
        std::vector<Real> lam(Model<Real>::n_dof_q); 
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const Real* Q_cell = &x_ptr[c * Model<Real>::n_dof_q];
            if (Q_cell[0] < 1e-6) continue; 
            for (int d = 0; d < Model<Real>::dimension; ++d) {
                Real n[3] = {0.0, 0.0, 0.0}; n[d] = 1.0;
                Model<Real>::eigenvalues(Q_cell, Qaux, n, lam.data());
                for(Real val : lam) max_eigen_local = std::max(max_eigen_local, std::abs(val));
            }
        }
        PetscCall(VecRestoreArrayRead(X_local, &x_ptr));
        PetscCall(DMRestoreLocalVector(dmQ, &X_local));
        Real max_eigen_global;
        PetscCallMPI(MPI_Allreduce(&max_eigen_local, &max_eigen_global, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));
        Real dt = (max_eigen_global > 1e-12) ? cfl * minRadius / max_eigen_global : 1e-4; 
        PetscCall(TSSetTimeStep(ts, dt));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupTimeStepping() {
        PetscFunctionBeginUser;
        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
        PetscCall(TSSetDM(ts, dmQ));
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(TSSetType(ts, TSSSP)); 
        PetscCall(TSSetMaxTime(ts, 1.0));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
        PetscCall(TSSetPreStep(ts, PreStepWrapper));
        PetscCall(TSMonitorSet(ts, MonitorSeries, this, NULL)); 
        PetscCall(DMTSSetBoundaryLocal(dmQ, DMPlexTSComputeBoundary, NULL));
        PetscCall(DMTSSetRHSFunctionLocal(dmQ, DMPlexTSComputeRHSFunctionFVM, NULL));
        PetscCall(TSSetFromOptions(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupInitialConditions() {
        PetscFunctionBeginUser;
        PetscCall(DMCreateGlobalVector(dmQ, &X));
        PetscCall(PetscObjectSetName((PetscObject)X, "Solution"));
        PetscErrorCode (*ic_funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*);
        ic_funcs[0] = InitialCondition;
        void* ctxs[1] = {NULL};
        PetscCall(DMProjectFunction(dmQ, 0.0, ic_funcs, ctxs, INSERT_ALL_VALUES, X));
        PetscCall(TSSetSolution(ts, X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode InitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
        Real x_dam = 5.0; 
        for(int i=0; i<Model<Real>::n_dof_q; ++i) u[i] = 0.0;
        if (x[0] <= x_dam) u[0] = 2.0; 
        else               u[0] = 1.0;
        return PETSC_SUCCESS;
    }

    static void RiemannAdapter(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscInt numConstants, const PetscScalar constants[], PetscScalar *flux, void *ctx) {
        Real Qaux_L[Model<Real>::n_dof_qaux] = {0.0};
        Real Qaux_R[Model<Real>::n_dof_qaux] = {0.0};
        Real area = 0.0;
        for(int d=0; d<dim; ++d) area += n[d]*n[d];
        area = std::sqrt(area);
        Real n_hat[3] = {0.0, 0.0, 0.0};
        if (area > 1e-14) { for(int d=0; d<dim; ++d) n_hat[d] = n[d] / area; } else { for(int d=0; d<dim; ++d) flux[d] = 0.0; return; }
        Real flux_per_area[Model<Real>::n_dof_q];
        Numerics<Real>::numerical_flux(xL, xR, Qaux_L, Qaux_R, (const Real*)n_hat, flux_per_area);
        for(int i=0; i<Model<Real>::n_dof_q; ++i) flux[i] = flux_per_area[i] * area;
    }

    static PetscErrorCode BoundaryAdapter(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx) {
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