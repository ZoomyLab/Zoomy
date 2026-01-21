#ifndef VIRTUALSOLVER_HPP
#define VIRTUALSOLVER_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map> 

#include <petscdmplex.h>
#include <petscts.h>
#include <petscfv.h>
#include <petscds.h>

#include "PetscHelpers.hpp"
#include "Numerics.H"
#include "Settings.hpp"
#include "Model.H"      
#include "IOManager.hpp"
#include "MeshConfigLoader.hpp" // New Include

class VirtualSolver {
protected:
    Settings settings;
    IOManager* io = nullptr;
    
    DM dmMesh; DM dmQ; DM dmAux; DM dmOut;
    TS ts; Vec X; Vec A; Vec X_out;
    
    PetscMPIInt rank;
    PetscReal minRadius;
    
    std::vector<PetscInt> bc_tag_ids;      
    std::vector<PetscInt> bc_ctx_indices;  

public:
    VirtualSolver() : rank(0), minRadius(0.0) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dmMesh = NULL; dmQ = NULL; dmAux = NULL; dmOut = NULL;
        ts = NULL; X = NULL; A = NULL; X_out = NULL;
    }

    VirtualSolver(const VirtualSolver&) = delete;
    VirtualSolver& operator=(const VirtualSolver&) = delete;

    virtual ~VirtualSolver() {
        if (io) delete io;
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
        PetscCall(Initialize(argc, argv));
        PetscCall(Solve());
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    virtual PetscErrorCode TakeOneStep(PetscReal time, PetscReal dt) = 0;

protected:
    PetscErrorCode Initialize(int argc, char **argv) {
        PetscFunctionBeginUser;
        char settings_path[PETSC_MAX_PATH_LEN] = "settings.json";
        PetscCall(PetscOptionsGetString(NULL, NULL, "-settings", settings_path, PETSC_MAX_PATH_LEN, NULL));
        settings = Settings::from_json(settings_path);

        io = new IOManager(settings, rank);
        io->PrepareDirectory();

        if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[INFO] Snapshot Logic: %s\n", settings.io.snapshot_logic.c_str());

        PetscCall(SetupArchitecture(1));

        PetscCall(SetupInitialConditions()); 
        PetscCall(CheckStateValidity(X)); 

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode Solve() {
        PetscFunctionBeginUser;
        PetscReal time = 0.0;
        PetscInt step = 0;

        Vec X_interp = NULL; 
        Vec X_old = NULL;
        if (settings.io.snapshot_logic == "interpolate") {
            PetscCall(VecDuplicate(X, &X_old));
            PetscCall(VecDuplicate(X, &X_interp));
        }

        PetscCall(UpdateBoundaryGhosts(time));
        PetscCall(PackState(X, A, X_out));
        io->WriteVTK(X_out, time);
        
        if (settings.io.snapshot_logic != "loose") io->AdvanceSnapshot();

        if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "[INFO] Starting Time Loop...\n");

        while (time < settings.solver.t_end) {
            PetscReal dt = ComputeTimeStep();
            
            PetscReal limit = io->GetDtLimit(time);
            if (limit < dt) dt = limit;
            
            if (time + dt > settings.solver.t_end) dt = settings.solver.t_end - time;

            PetscReal t_old = time;
            if (X_old) PetscCall(VecCopy(X, X_old));

            PetscCall(TakeOneStep(time, dt));
            time += dt;
            step++;

            if (step % 10 == 0 && rank == 0) {
                 PetscPrintf(PETSC_COMM_WORLD, "Step %3d | Time %.5g | dt %.5g\n", step, (double)time, (double)dt);
            }

            if (settings.io.snapshot_logic == "interpolate") {
                while (io->GetNextSnapTime() <= time) {
                    PetscReal target_t = io->GetNextSnapTime();
                    PetscReal alpha = (target_t - t_old) / (time - t_old);
                    
                    PetscCall(VecCopy(X, X_interp)); 
                    PetscCall(VecAXPBY(X_interp, 1.0 - alpha, alpha, X_old));
                    
                    PetscCall(PackState(X_interp, A, X_out));
                    io->WriteVTK(X_out, target_t);
                    io->AdvanceSnapshot();
                }
            } else {
                if (io->ShouldWrite(time)) {
                    PetscCall(PackState(X, A, X_out));
                    io->WriteVTK(X_out, time);
                    
                    if (settings.io.snapshot_logic == "loose") {
                        while(io->GetNextSnapTime() <= time) io->AdvanceSnapshot();
                    } else {
                        io->AdvanceSnapshot();
                    }
                }
            }
            PetscCall(UpdateBoundaryGhosts(time));
        }

        if (X_old) PetscCall(VecDestroy(&X_old));
        if (X_interp) PetscCall(VecDestroy(&X_interp));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode CheckStateValidity(Vec V) {
        PetscFunctionBeginUser;
        PetscReal minVal, maxVal;
        PetscCall(VecMax(V, NULL, &maxVal));
        PetscCall(VecMin(V, NULL, &minVal));
        
        if (std::isnan(minVal) || std::isnan(maxVal) || std::isinf(minVal) || std::isinf(maxVal)) {
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FP, "FATAL: State vector contains NaN or Inf after initialization!");
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

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
        
        PetscInt size;
        PetscCall(VecGetLocalSize(X_in, &size));
        PetscInt n_cells = size / n_q;
        
        for (PetscInt c = 0; c < n_cells; ++c) {
            for (int i = 0; i < n_q; ++i)   arr_out[c * n_tot + i] = arr_x[c * n_q + i];
            for (int i = 0; i < n_aux; ++i) arr_out[c * n_tot + n_q + i] = arr_a[c * n_aux + i];
        }

        PetscCall(VecRestoreArrayRead(X_in, &arr_x));
        PetscCall(VecRestoreArrayRead(A_in, &arr_a));
        PetscCall(VecRestoreArray(X_target, &arr_out));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

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
        if (settings.solver.reconstruction_order >= 2) PetscFVSetType(fvm, PETSCFVLEASTSQUARES);
        else PetscFVSetType(fvm, PETSCFVUPWIND);
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
        
        PetscCall(DMSetBasicAdjacency(dmMesh, settings.solver.use_deep_adjacency ? PETSC_TRUE : PETSC_FALSE, PETSC_FALSE));
        PetscCall(DMPlexGetGeometryFVM(dmMesh, NULL, NULL, &minRadius));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // --- REFACTORED: Infer IDs from Mesh File ---
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
        PetscCall(DMGetLabel(dmQ, settings.io.mesh_label.c_str(), &label));
        if (!label) {
            PetscPrintf(PETSC_COMM_WORLD, "[CRITICAL] Configured mesh label '%s' not found!\n", settings.io.mesh_label.c_str());
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Mesh Label not found.");
        }

        // 1. Get User Expected Names
        auto names = Model<Real>::get_boundary_tags();
        bc_tag_ids.resize(names.size());
        bc_ctx_indices.resize(names.size());

        // 2. Rank 0 Reads and Maps
        if (rank == 0) {
            try {
                // Load map from .msh file
                std::map<std::string, int> mesh_map = MeshConfigLoader::loadBoundaryMapping(settings.io.mesh_path);
                
                // Match required names to mesh IDs
                for(size_t i=0; i<names.size(); ++i) {
                    if (mesh_map.find(names[i]) == mesh_map.end()) {
                        // Error formatting
                        std::string found_tags = "";
                        for(const auto& pair : mesh_map) found_tags += "'" + pair.first + "' ";
                        
                        PetscPrintf(PETSC_COMM_SELF, "\n[ERROR] Model requires boundary '%s', but it was not found in '%s'.\n", names[i].c_str(), settings.io.mesh_path.c_str());
                        PetscPrintf(PETSC_COMM_SELF, "[ERROR] Available physical names in mesh: [ %s]\n", found_tags.c_str());
                        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing boundary tag in mesh file.");
                    }
                    bc_tag_ids[i] = mesh_map[names[i]];
                }
            } catch (const std::exception& e) {
                PetscPrintf(PETSC_COMM_SELF, "[ERROR] Mesh Config Loader failed: %s\n", e.what());
                SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Failed to load mesh boundary mapping.");
            }
        }

        // 3. Broadcast Result to all Ranks
        // bc_tag_ids is vector of PetscInt. Check if PetscInt is int or long long.
        // PETSc usually defines MPIU_INT appropriately, but safely we use MPI_Bcast.
        PetscCallMPI(MPI_Bcast(bc_tag_ids.data(), bc_tag_ids.size(), MPIU_INT, 0, PETSC_COMM_WORLD));

        // 4. Register with PETSc
        for(size_t i=0; i<names.size(); ++i) {
            bc_ctx_indices[i] = (PetscInt)i; // Context index matches order in Model
            
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
        
        PetscFV fvmQ;
        PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmQ));
        PetscCall(PetscFVSetNumComponents(fvmQ, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvmQ, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_q; ++i) { char name[32]; snprintf(name, 32, "Q_%d", i); PetscCall(PetscFVSetComponentName(fvmQ, i, name)); }
        
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
        
        if (max_eigen_global <= settings.solver.min_dt) { 
             char err[256];
             snprintf(err, 256, "System has zero/tiny wave speed (max_eig=%g). Aborting.", max_eigen_global);
             SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FP, "%s", err);
        }
        
        return settings.solver.cfl * minRadius / max_eigen_global;
    }
};
#endif