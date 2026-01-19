#ifndef VIRTUALSOLVER_HPP
#define VIRTUALSOLVER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>

// --- CORE PETSC HEADERS ---
#include <petscdmplex.h>
#include <petscts.h>
#include <petscfv.h>
#include <petscds.h>
#include <petscviewerhdf5.h> 

#include "PetscHelpers.hpp"
#include "Numerics.H"
#include "Settings.hpp"

// Forward declaration
template <typename T>
class Model;

namespace fs = std::filesystem;

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
    
    // Memory Safety
    std::vector<PetscInt> bc_tag_ids;      
    std::vector<PetscInt> bc_ctx_indices;  
    
    // Time Series Metadata
    struct SnapInfo { 
        std::string filename; 
        PetscReal time; 
    };
    std::vector<SnapInfo> snapshots_log;

public:
    VirtualSolver() : rank(0), cfl(0.5), minRadius(0.0) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dmMesh = NULL; dmQ = NULL; dmAux = NULL; dmOut = NULL;
        ts = NULL; X = NULL; A = NULL; X_out = NULL;
    }

    VirtualSolver(const VirtualSolver&) = delete;
    VirtualSolver& operator=(const VirtualSolver&) = delete;

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
        char settings_path[PETSC_MAX_PATH_LEN] = "settings.json";
        PetscCall(PetscOptionsGetString(NULL, NULL, "-settings", settings_path, PETSC_MAX_PATH_LEN, NULL));
        settings = Settings::from_json(settings_path);

        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Snapshot Logic: %s\n", settings.io.snapshot_logic.c_str());
            fs::path outDir(settings.io.directory);
            if (!fs::exists(outDir)) {
                fs::create_directories(outDir);
            } else if (settings.io.clean_directory) {
                PetscPrintf(PETSC_COMM_SELF, "[INFO] Cleaning output directory: %s\n", settings.io.directory.c_str());
                for (const auto& entry : fs::directory_iterator(outDir)) {
                    fs::remove_all(entry.path());
                }
            }
        }
        PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

        PetscCall(SetupArchitecture(1));
        PetscCall(SetupInitialConditions()); 

        PetscReal time = 0.0;
        PetscReal dt_snap = settings.solver.t_end / settings.io.snapshots;
        PetscReal next_snap = time + dt_snap;
        PetscInt snapshot_idx = 0;
        PetscInt step = 0;

        // Vector for interpolation (Size matches X, not X_out)
        Vec X_interp = NULL;
        Vec X_old = NULL;
        
        if (settings.io.snapshot_logic == "interpolate") {
            PetscCall(VecDuplicate(X, &X_old));
            PetscCall(VecDuplicate(X, &X_interp));
        }

        PetscCall(UpdateBoundaryGhosts(time));
        PetscCall(WriteVTK(X, snapshot_idx++, time));

        if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "Starting Time Loop...\n");

        while (time < settings.solver.t_end) {
            PetscReal dt = ComputeTimeStep();
            
            if (settings.io.snapshot_logic == "snap") {
                if (next_snap > time) dt = std::min(dt, next_snap - time);
            }
            dt = std::min(dt, settings.solver.t_end - time);

            // Save state if interpolating
            if (X_old) PetscCall(VecCopy(X, X_old));
            PetscReal t_old = time;

            PetscCall(TakeOneStep(time, dt));
            time += dt;
            step++;

            if (step % 10 == 0 && rank == 0) {
                 PetscPrintf(PETSC_COMM_WORLD, "Step %3d | Time %.5g | dt %.5g\n", step, (double)time, (double)dt);
            }

            if (time >= next_snap - 1e-9) {
                if (settings.io.snapshot_logic == "snap") {
                    PetscCall(WriteVTK(X, snapshot_idx++, time)); 
                    next_snap += dt_snap;
                }
                else if (settings.io.snapshot_logic == "loose") {
                    PetscCall(WriteVTK(X, snapshot_idx++, time));
                    while(next_snap <= time) next_snap += dt_snap;
                }
                else if (settings.io.snapshot_logic == "interpolate") {
                    while (next_snap <= time) {
                        PetscReal alpha = (next_snap - t_old) / (time - t_old);
                        
                        // 1. Interpolate X -> X_interp (Safe because sizes match)
                        PetscCall(VecCopy(X, X_interp)); 
                        PetscCall(VecAXPBY(X_interp, 1.0 - alpha, alpha, X_old));
                        
                        // 2. Write X_interp (PackState will handle merging with A)
                        PetscCall(WriteVTK(X_interp, snapshot_idx++, next_snap));
                        
                        next_snap += dt_snap;
                    }
                }
            }
            PetscCall(UpdateBoundaryGhosts(time));
        }

        if (X_old) PetscCall(VecDestroy(&X_old));
        if (X_interp) PetscCall(VecDestroy(&X_interp));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    virtual PetscErrorCode TakeOneStep(PetscReal time, PetscReal dt) = 0;

protected:
    virtual PetscErrorCode SetupInitialConditions() {
        PetscFunctionBeginUser;
        PetscErrorCode (*ic_func)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*);
        ic_func = Model<Real>::InitialCondition;
        void* ctx = NULL;
        PetscCall(DMProjectFunction(dmQ, 0.0, &ic_func, &ctx, INSERT_ALL_VALUES, X));
        PetscCall(TSSetSolution(ts, X));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    void SetSolverOrder(int order) {
        PetscFV fvm;
        DMGetField(dmQ, 0, NULL, (PetscObject*)&fvm);
        if (order >= 2) PetscFVSetType(fvm, PETSCFVLEASTSQUARES);
        else PetscFVSetType(fvm, PETSCFVUPWIND);
    }

    // --- HELPER: PACK DATA INTO ONE VECTOR ---
    PetscErrorCode PackState(Vec X_in, Vec A_in, Vec X_target) {
        PetscFunctionBeginUser;
        const PetscScalar *arr_x, *arr_a;
        PetscScalar *arr_out;
        
        PetscCall(VecGetArrayRead(X_in, &arr_x));
        PetscCall(VecGetArrayRead(A_in, &arr_a));
        PetscCall(VecGetArray(X_target, &arr_out));
        
        PetscInt n_q = Model<Real>::n_dof_q;
        PetscInt n_aux = Model<Real>::n_dof_qaux;
        PetscInt n_tot = n_q + n_aux;
        
        // Assuming cell-by-cell interleaving for dmOut
        // We get local size of cells. Since all DMs are clones, local sizes (in terms of cells) match.
        PetscInt size;
        PetscCall(VecGetLocalSize(X_in, &size));
        PetscInt n_cells = size / n_q;
        
        for (PetscInt c = 0; c < n_cells; ++c) {
            // Copy Solution (Field 0)
            for (int i = 0; i < n_q; ++i) {
                arr_out[c * n_tot + i] = arr_x[c * n_q + i];
            }
            // Copy Aux (Field 1)
            for (int i = 0; i < n_aux; ++i) {
                arr_out[c * n_tot + n_q + i] = arr_a[c * n_aux + i];
            }
        }

        PetscCall(VecRestoreArrayRead(X_in, &arr_x));
        PetscCall(VecRestoreArrayRead(A_in, &arr_a));
        PetscCall(VecRestoreArray(X_target, &arr_out));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // --- WRITE VTK ---
    PetscErrorCode WriteVTK(Vec solutionVec, PetscInt snapshot_idx, PetscReal time) {
        PetscFunctionBeginUser;
        char basename[256];
        snprintf(basename, sizeof(basename), "%s-%03d.vtu", settings.io.filename.c_str(), snapshot_idx);
        std::string fullPath = settings.io.directory + "/" + std::string(basename);

        // 1. Pack Solution and Aux into X_out (the combined vector)
        PetscCall(PackState(solutionVec, A, X_out));

        // 2. Write X_out (It contains everything)
        PetscViewer viewer;
        PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, fullPath.c_str(), FILE_MODE_WRITE, &viewer));
        PetscCall(PetscObjectSetName((PetscObject)X_out, "State"));
        PetscCall(VecView(X_out, viewer));
        PetscCall(PetscViewerDestroy(&viewer));

        if (rank == 0) {
            snapshots_log.push_back({std::string(basename), time});
            UpdateSeriesFile();
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    void UpdateSeriesFile() {
        std::string seriesPath = settings.io.directory + "/" + settings.io.filename + ".vtu.series";
        std::ofstream f(seriesPath);
        if (f.is_open()) {
            f << "{\n  \"file-series-version\" : \"1.0\",\n  \"files\" : [\n";
            for (size_t i = 0; i < snapshots_log.size(); ++i) {
                f << "    { \"name\" : \"" << snapshots_log[i].filename << "\", \"time\" : " 
                  << std::scientific << std::setprecision(6) << snapshots_log[i].time << " }";
                if (i < snapshots_log.size() - 1) f << ",";
                f << "\n";
            }
            f << "  ]\n}\n";
            f.close();
        }
    }

    PetscErrorCode SetupArchitecture(PetscInt overlap = 1) {
        PetscFunctionBeginUser;
        PetscCall(SetupBaseMesh(overlap));
        PetscCall(SetupPrimaryDM());
        PetscCall(SetupAuxiliaryDM());
        PetscCall(SetupOutputDM()); // <--- Re-enabled
        PetscCall(SetupTimeStepping());
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    
    PetscErrorCode SetupBaseMesh(PetscInt overlap) {
        PetscFunctionBeginUser;
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

    PetscErrorCode SetupPrimaryDM() {
        PetscFunctionBeginUser;
        PetscCall(DMClone(dmMesh, &dmQ)); 
        PetscFV fvm;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvm));
        PetscCall(PetscFVSetFromOptions(fvm)); 
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
        bc_tag_ids.resize(names.size());
        bc_ctx_indices.resize(names.size());
        for(size_t i=0; i<names.size(); ++i) {
            bc_tag_ids[i] = std::stoi(ids[i]);
            bc_ctx_indices[i] = (PetscInt)i;
            PetscCall(PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, names[i].c_str(), label, 
                                         1, &bc_tag_ids[i], 
                                         0, 0, NULL, 
                                         (PetscVoidFn *)Zoomy::BoundaryAdapter, NULL, 
                                         &bc_ctx_indices[i], NULL)); 
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
        
        // Field 0: Solution
        PetscFV fvmQ;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmQ));
        PetscCall(PetscFVSetNumComponents(fvmQ, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvmQ, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_q; ++i) { char name[32]; snprintf(name, 32, "Q_%d", i); PetscCall(PetscFVSetComponentName(fvmQ, i, name)); }
        
        // Field 1: Aux
        PetscFV fvmAux;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmAux));
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
        PetscCall(TSSetMaxSteps(ts, 2000000000)); 
        PetscCall(TSSetMaxTime(ts, 1.0e20)); 
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP)); 
        PetscCall(TSSetFromOptions(ts));
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

    PetscReal ComputeTimeStep() {
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
};
#endif