#ifndef VIRTUALSOLVER_HPP
#define VIRTUALSOLVER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

// --- CORE PETSC HEADERS ---
#include <petscdmplex.h>
#include <petscts.h>
#include <petscfv.h>
#include <petscds.h>
#include <petscviewerhdf5.h> 

#include "PetscHelpers.hpp"
#include "Numerics.H"
#include "Settings.hpp"

class VirtualSolver {
protected:
    Settings settings;
    
    // Core DMs
    DM dmMesh; DM dmQ; DM dmAux; DM dmOut;
    
    // Core Solver Objects
    TS ts; Vec X; Vec A; Vec X_out;
    
    // Config
    PetscMPIInt rank;
    PetscReal cfl;
    PetscReal minRadius;
    std::vector<PetscInt> bc_ids_storage; 
    
    struct StepData { PetscInt step; PetscReal time; };
    std::vector<StepData> time_series;

public:
    VirtualSolver() : rank(0), cfl(0.5), minRadius(0.0) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dmMesh = NULL; dmQ = NULL; dmAux = NULL; dmOut = NULL;
        ts = NULL; X = NULL; A = NULL; X_out = NULL;
    }

    virtual ~VirtualSolver() {
        if (X)      VecDestroy(&X);
        if (A)      VecDestroy(&A);
        if (X_out)  VecDestroy(&X_out);
        if (ts)     TSDestroy(&ts);
        if (dmQ)    DMDestroy(&dmQ);
        if (dmAux)  DMDestroy(&dmAux);
        if (dmOut)  DMDestroy(&dmOut);
        if (dmMesh) DMDestroy(&dmMesh);
    }

    virtual PetscErrorCode Run(int argc, char **argv) {
        PetscFunctionBeginUser;
        // 1. Load Settings
        char settings_path[PETSC_MAX_PATH_LEN] = "settings.json";
        PetscCall(PetscOptionsGetString(NULL, NULL, "-settings", settings_path, PETSC_MAX_PATH_LEN, NULL));
        settings = Settings::from_json(settings_path);

        PetscCall(SetupArchitecture(1));

        // 2. Handle Initial Condition / Restart
        PetscReal time = 0.0;
        if (settings.io.restart) {
            PetscCall(LoadH5(settings.io.restart_file, &time));
        } else if (!settings.io.initial_condition_file.empty()) {
            PetscReal dummy_t;
            PetscCall(LoadH5(settings.io.initial_condition_file, &dummy_t));
            time = 0.0; 
        } else {
            PetscCall(SetupInitialConditions()); 
        }

        // 3. Snapshot Logic
        PetscReal dt_snap = settings.solver.t_end / settings.io.snapshots;
        PetscReal next_snap = time + dt_snap;
        PetscInt snapshot_idx = (PetscInt)(time / dt_snap);
        PetscInt step = 0;

        PetscCall(UpdateBoundaryGhosts(time));
        PetscCall(WriteH5(snapshot_idx++, time));

        while (time < settings.solver.t_end) {
            PetscReal dt = ComputeTimeStep();
            
            // Limit dt to hit the exact snapshot time
            dt = std::min(dt, settings.solver.t_end - time);
            if (next_snap > time) dt = std::min(dt, next_snap - time);

            PetscCall(TakeOneStep(time, dt, step));

            time += dt;
            step++;

            if (step % 10 == 0 && rank == 0) {
                 PetscPrintf(PETSC_COMM_WORLD, "Step %3d | Time %.5g | dt %.5g\n", step, (double)time, (double)dt);
            }

            if (time >= next_snap - 1e-9) {
                PetscCall(WriteH5(snapshot_idx++, time)); 
                next_snap += dt_snap;
            }
            PetscCall(UpdateBoundaryGhosts(time));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    virtual PetscErrorCode TakeOneStep(PetscReal time, PetscReal dt, PetscInt step) = 0;

protected:
    void SetSolverOrder(int order) {
        PetscFV fvm;
        DMGetField(dmQ, 0, NULL, (PetscObject*)&fvm);
        if (order >= 2) {
            PetscFVSetType(fvm, PETSCFVLEASTSQUARES);
        } else {
            PetscFVSetType(fvm, PETSCFVUPWIND);
        }
    }

    PetscErrorCode WriteH5(PetscInt snapshot_idx, PetscReal time) {
        PetscFunctionBeginUser;
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/%s-%03d.h5", 
                 settings.io.directory.c_str(), settings.io.filename.c_str(), snapshot_idx);
        
        PetscViewer viewer;
        PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
        PetscCall(PetscViewerHDF5PushGroup(viewer, "/"));
        PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "Time", PETSC_REAL, &time));
        
        PetscCall(PetscObjectSetName((PetscObject)X, "Solution"));
        PetscCall(VecView(X, viewer));
        if (A) {
            PetscCall(PetscObjectSetName((PetscObject)A, "Auxiliary"));
            PetscCall(VecView(A, viewer));
        }
        PetscCall(PetscViewerDestroy(&viewer));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode LoadH5(const std::string& path, PetscReal* time) {
        PetscFunctionBeginUser;
        PetscViewer viewer;
        PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, path.c_str(), FILE_MODE_READ, &viewer));
        PetscCall(PetscViewerHDF5PushGroup(viewer, "/"));
        PetscCall(PetscViewerHDF5ReadAttribute(viewer, NULL, "Time", PETSC_REAL, NULL, time));
        PetscCall(PetscObjectSetName((PetscObject)X, "Solution"));
        PetscCall(VecLoad(X, viewer));
        if (A) {
            PetscCall(PetscObjectSetName((PetscObject)A, "Auxiliary"));
            PetscCall(VecLoad(A, viewer));
        }
        PetscCall(PetscViewerDestroy(&viewer));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupArchitecture(PetscInt overlap = 1) {
        PetscFunctionBeginUser;
        PetscCall(SetupBaseMesh(overlap));
        PetscCall(SetupPrimaryDM());
        PetscCall(SetupAuxiliaryDM());
        PetscCall(SetupOutputDM()); 
        PetscCall(SetupTimeStepping());
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    
    PetscErrorCode SetupBaseMesh(PetscInt overlap) {
        PetscFunctionBeginUser;
        // Use settings.io.mesh_path
        if (!settings.io.mesh_path.empty()) {
             PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, settings.io.mesh_path.c_str(), "zoomy_mesh", PETSC_TRUE, &dmMesh));
        } else {
             PetscCall(DMCreate(PETSC_COMM_WORLD, &dmMesh));
             PetscCall(DMSetType(dmMesh, DMPLEX));
             PetscCall(DMSetFromOptions(dmMesh)); 
        }

        DM dmDist = NULL;
        PetscCall(DMPlexDistribute(dmMesh, overlap, NULL, &dmDist));
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

    // [SetupPrimaryDM, SetupAuxiliaryDM, SetupOutputDM, SetupTimeStepping, SetupInitialConditions, UpdateBoundaryGhosts, ComputeTimeStep remain unchanged]
    // ... (Keep the implementations from your previous file) ...
    PetscReal ComputeTimeStep() {
        // ... (Same as your uploaded file) ...
        Vec X_local;
        PetscCall(DMGetLocalVector(dmQ, &X_local));
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_local));
        PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_local));
        const PetscScalar *x_ptr;
        PetscCall(VecGetArrayRead(X_local, &x_ptr));
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd)); 
        Real max_eigen_global = 0.0;
        Real max_eigen_local = 0.0;
        Real Qaux[Model<Real>::n_dof_qaux] = {0.0}; 
        Real res[1]; 
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const Real* Q_cell = &x_ptr[c * Model<Real>::n_dof_q];
            if (Q_cell[0] < 1e-6) continue; 
            for (int d = 0; d < Model<Real>::dimension; ++d) {
                Real n[3] = {0.0, 0.0, 0.0}; 
                n[d] = 1.0;
                Numerics<Real>::local_max_abs_eigenvalue(Q_cell, Qaux, n, res);
                if (res[0] > max_eigen_local) max_eigen_local = res[0];
            }
        }
        PetscCall(VecRestoreArrayRead(X_local, &x_ptr));
        PetscCall(DMRestoreLocalVector(dmQ, &X_local));
        PetscCallMPI(MPI_Allreduce(&max_eigen_local, &max_eigen_global, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));
        if (max_eigen_global > 1e-12) return cfl * minRadius / max_eigen_global;
        return 1e-4; 
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
        PetscDS prob;
        PetscCall(DMGetDS(dmQ, &prob));
        PetscCall(PetscDSSetRiemannSolver(prob, 0, Zoomy::RiemannAdapter));
        DMLabel label;
        PetscCall(DMGetLabel(dmQ, "Face Sets", &label));
        auto names = Model<Real>::get_boundary_tags();
        auto ids   = Model<Real>::get_boundary_tag_ids();
        bc_ids_storage.resize(names.size());
        for(size_t i=0; i<names.size(); ++i) {
            PetscInt id_val = std::stoi(ids[i]);
            bc_ids_storage[i] = (PetscInt)i; 
            PetscCall(PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, names[i].c_str(), label, 1, &id_val, 0, 0, NULL, (PetscVoidFn *)Zoomy::BoundaryAdapter, NULL, &bc_ids_storage[i], NULL));
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
        PetscCall(VecSet(A, 1.0)); 
        PetscCall(DMSetAuxiliaryVec(dmQ, NULL, 0, 0, A));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupOutputDM() {
        PetscFunctionBeginUser;
        PetscCall(DMClone(dmMesh, &dmOut));
        PetscFV fvmQ, fvmAux;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmQ));
        PetscCall(PetscFVSetType(fvmQ, PETSCFVUPWIND)); 
        PetscCall(PetscFVSetNumComponents(fvmQ, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvmQ, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_q; ++i) { char name[32]; snprintf(name, 32, "Q_%d", i); PetscCall(PetscFVSetComponentName(fvmQ, i, name)); }
        
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmAux));
        PetscCall(PetscFVSetType(fvmAux, PETSCFVUPWIND)); 
        PetscCall(PetscFVSetNumComponents(fvmAux, Model<Real>::n_dof_qaux));
        PetscCall(PetscFVSetSpatialDimension(fvmAux, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_qaux; ++i) { char name[32]; snprintf(name, 32, "Aux_%d", i); PetscCall(PetscFVSetComponentName(fvmAux, i, name)); }

        PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmQ));
        PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmAux));
        PetscCall(DMCreateDS(dmOut));
        PetscCall(PetscFVDestroy(&fvmQ));
        PetscCall(PetscFVDestroy(&fvmAux));
        PetscCall(DMCreateGlobalVector(dmOut, &X_out));
        PetscCall(PetscObjectSetName((PetscObject)X_out, "CombinedState"));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupTimeStepping() {
        PetscFunctionBeginUser;
        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
        PetscCall(TSSetDM(ts, dmQ));
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(TSSetType(ts, TSSSP)); 
        PetscCall(DMTSSetBoundaryLocal(dmQ, DMPlexTSComputeBoundary, NULL));
        PetscCall(DMTSSetRHSFunctionLocal(dmQ, DMPlexTSComputeRHSFunctionFVM, NULL));
        PetscCall(TSSetFromOptions(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupInitialConditions() {
        PetscFunctionBeginUser;
        PetscErrorCode (*ic_funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*);
        ic_funcs[0] = Zoomy::InitialCondition;
        void* ctxs[1] = {NULL};
        PetscCall(DMProjectFunction(dmQ, 0.0, ic_funcs, ctxs, INSERT_ALL_VALUES, X));
        PetscCall(TSSetSolution(ts, X));
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
};
#endif