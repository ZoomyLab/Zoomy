static char help[] = "Simplified 1st Order FVM Solver (Zoomy-Core)\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscfv.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

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
    std::vector<PetscInt> bc_ids_storage; 
    
    // Series Output Data
    struct StepData {
        PetscInt step;
        PetscReal time;
    };
    std::vector<StepData> time_series;

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
        PetscCall(SetupDiscretization()); 
        PetscCall(SetupTimeStepping());
        PetscCall(SetupInitialConditions());

        // Pre-calculate ghosts to avoid vacuum state at t=0
        PetscCall(UpdateBoundaryGhosts(0.0));

        PetscCall(TSSolve(ts, X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    // -------------------------------------------------------------------------
    // 1. MESH SETUP
    // -------------------------------------------------------------------------
    PetscErrorCode SetupMesh() {
        PetscFunctionBeginUser;
        
        PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
        PetscCall(DMSetType(dm, DMPLEX));
        PetscCall(DMSetFromOptions(dm)); 

        // 1. Distribute Mesh (MPI Overlap)
        DM dmDist = NULL;
        PetscInt overlap = 1; 
        PetscCall(DMPlexDistribute(dm, overlap, NULL, &dmDist));
        if (dmDist) {
            PetscCall(DMDestroy(&dm));
            dm = dmDist; 
        }

        // 2. Construct Physical Ghosts (Required for Riemann BC)
        {
            DM dmGhost = NULL;
            PetscCall(DMPlexConstructGhostCells(dm, "Face Sets", NULL, &dmGhost));
            if (dmGhost) {
                PetscCall(DMDestroy(&dm));
                dm = dmGhost;
            }
        }

        // 3. Finalize Adjacency (Cone + Closure)
        PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_TRUE));

        PetscCall(DMPlexGetGeometryFVM(dm, NULL, NULL, &minRadius));
        
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // -------------------------------------------------------------------------
    // 2. DISCRETIZATION
    // -------------------------------------------------------------------------
    PetscErrorCode SetupDiscretization() {
        PetscFunctionBeginUser;
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
        
        PetscCall(DMAddField(dm, NULL, (PetscObject)fvm));
        PetscCall(DMCreateDS(dm));
        PetscCall(DMGetDS(dm, &prob));
        PetscCall(PetscDSSetRiemannSolver(prob, 0, RiemannAdapter));

        DMLabel label;
        PetscCall(DMGetLabel(dm, "Face Sets", &label));
        auto names = Model<Real>::get_boundary_tags();
        auto ids   = Model<Real>::get_boundary_tag_ids();
        bc_ids_storage.resize(names.size());

        for(size_t i=0; i<names.size(); ++i) {
            PetscInt id_val = std::stoi(ids[i]);
            bc_ids_storage[i] = (PetscInt)i; 
            PetscCall(PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, 
                                         names[i].c_str(), 
                                         label, 
                                         1, &id_val, 
                                         0, 0, NULL, 
                                         (PetscVoidFn *)BoundaryAdapter, 
                                         NULL, &bc_ids_storage[i], NULL));
        }
        PetscCall(PetscFVDestroy(&fvm));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // -------------------------------------------------------------------------
    // 3. TIME STEPPING & OUTPUT
    // -------------------------------------------------------------------------
    static PetscErrorCode PreStepWrapper(TS ts) {
        PetscFunctionBeginUser;
        void *ctx;
        PetscCall(TSGetApplicationContext(ts, &ctx));
        FVMSolver* solver = static_cast<FVMSolver*>(ctx);
        
        // Update ghosts & Calc dt
        PetscReal time;
        PetscCall(TSGetTime(ts, &time));
        PetscCall(solver->UpdateBoundaryGhosts(time));
        PetscCall(solver->ComputeTimeStep(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // Monitor to write output.vtu.series file
    static PetscErrorCode MonitorSeries(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx) {
        FVMSolver* solver = static_cast<FVMSolver*>(ctx);
        if (solver->rank == 0) {
            solver->time_series.push_back({step, time});
            
            // Write to output.vtu.series (Standard Paraview JSON format for VTU)
            std::ofstream f("output.vtu.series");
            if (f.is_open()) {
                f << "{\n";
                f << "  \"file-series-version\" : \"1.0\",\n";
                f << "  \"files\" : [\n";
                for (size_t i = 0; i < solver->time_series.size(); ++i) {
                    f << "    { \"name\" : \"output-" 
                      << std::setfill('0') << std::setw(3) << solver->time_series[i].step 
                      << ".vtu\", \"time\" : " << std::scientific << std::setprecision(6) << solver->time_series[i].time << " }";
                    if (i < solver->time_series.size() - 1) f << ",";
                    f << "\n";
                }
                f << "  ]\n";
                f << "}\n";
                f.close();
            }
        }
        return 0;
    }

    PetscErrorCode UpdateBoundaryGhosts(PetscReal time) {
        PetscFunctionBeginUser;
        Vec locX;
        PetscCall(DMGetLocalVector(dm, &locX));
        PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX));
        PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX));
        
        PetscCall(DMPlexTSComputeBoundary(dm, time, locX, NULL, NULL));
        
        PetscCall(DMRestoreLocalVector(dm, &locX));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode ComputeTimeStep(TS ts) {
        PetscFunctionBeginUser;
        
        Vec X_local;
        PetscCall(DMGetLocalVector(dm, &X_local));
        PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, X_local));
        PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, X_local));

        const PetscScalar *x_ptr;
        PetscCall(VecGetArrayRead(X_local, &x_ptr));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd)); 

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
        PetscCall(DMRestoreLocalVector(dm, &X_local));

        Real max_eigen_global;
        PetscCallMPI(MPI_Allreduce(&max_eigen_local, &max_eigen_global, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));

        Real dt;
        if (max_eigen_global > 1e-12) dt = cfl * minRadius / max_eigen_global;
        else dt = 1e-4; 

        PetscCall(TSSetTimeStep(ts, dt));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupTimeStepping() {
        PetscFunctionBeginUser;
        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
        PetscCall(TSSetDM(ts, dm));
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(TSSetType(ts, TSSSP)); 
        PetscCall(TSSetMaxTime(ts, 1.0));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
        
        PetscCall(TSSetPreStep(ts, PreStepWrapper));
        
        // Attach the Series Monitor
        PetscCall(TSMonitorSet(ts, MonitorSeries, this, NULL)); 
        
        PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL));
        PetscCall(DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, NULL));

        PetscCall(TSSetFromOptions(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // -------------------------------------------------------------------------
    // 4. HELPERS
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
        if (x[0] <= x_dam) u[0] = 2.0; 
        else               u[0] = 1.0;
        return PETSC_SUCCESS;
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