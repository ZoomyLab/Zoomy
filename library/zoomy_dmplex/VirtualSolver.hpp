#ifndef VIRTUALSOLVER_HPP
#define VIRTUALSOLVER_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <map> 
#include <iostream>

#include <petscdmplex.h>
#include <petscts.h>
#include <petscfv.h>
#include <petscds.h>

#include "Numerics.H"
#include "Settings.hpp"
#include "Model.H"      
#include "IOManager.hpp"
#include "MeshConfigLoader.hpp"

using Real = PetscReal;

class VirtualSolver {
protected:
    Settings settings;
    IOManager* io = nullptr;
    
    DM dmMesh; DM dmQ; DM dmAux; DM dmOut;
    TS ts; Vec X; Vec A; Vec X_out;
    
    PetscMPIInt rank;
    PetscReal minRadius;
    std::vector<PetscReal> parameters; 

    std::vector<PetscInt> bc_tag_ids;      
    std::vector<PetscInt> bc_ctx_indices; 

public:
    VirtualSolver() : rank(0), minRadius(0.0) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dmMesh = NULL; dmQ = NULL; dmAux = NULL; dmOut = NULL;
        ts = NULL; X = NULL; A = NULL; X_out = NULL;
        parameters = {9.81, 0.0, 1e-6};
    }

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

    virtual PetscErrorCode ComputeRHS(PetscReal time, Vec X_global, Vec F_global) = 0;
    virtual PetscErrorCode UpdateState() = 0;

    static void BoundaryFunc(PetscReal time, const PetscReal *c, const PetscReal *n, 
                             const PetscScalar *xI, const PetscScalar *xG, void *ctx) {
    }

protected:
    PetscErrorCode Initialize(int argc, char **argv) {
        char settings_path[PETSC_MAX_PATH_LEN] = "settings.json";
        PetscCall(PetscOptionsGetString(NULL, NULL, "-settings", settings_path, PETSC_MAX_PATH_LEN, NULL));
        settings = Settings::from_json(settings_path);

        io = new IOManager(settings, rank);
        io->PrepareDirectory();

        PetscCall(SetupArchitecture(1));
        PetscCall(SetupInitialConditions()); 
        PetscCall(SetupAuxiliaryConditions()); 
        PetscCall(UpdateState()); 
        PetscCall(CheckStateValidity(X));
        return PETSC_SUCCESS;
    }

    PetscErrorCode Solve() {
        PetscReal time = 0.0;
        PetscInt step = 0;

        Vec X_interp = NULL, X_old = NULL;
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

            Vec F; 
            PetscCall(VecDuplicate(X, &F));
            PetscCall(ComputeRHS(time, X, F));
            PetscCall(VecAXPY(X, dt, F));
            PetscCall(VecDestroy(&F));

            PetscCall(UpdateState());
            
            time += dt; 
            step++;
            
            if (rank == 0 && step % 10 == 0) PetscPrintf(PETSC_COMM_WORLD, "Step %d Time %.4f dt %.4e\n", step, (double)time, (double)dt);
            
            if (io->ShouldWrite(time)) {
                PetscCall(HandleOutput(time, t_old, X_interp, X_old));
            }
            PetscCall(UpdateBoundaryGhosts(time));
        }
        if (X_old) PetscCall(VecDestroy(&X_old));
        if (X_interp) PetscCall(VecDestroy(&X_interp));
        return PETSC_SUCCESS;
    }

    virtual PetscErrorCode SetupInitialConditions() {
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        PetscScalar *x_arr;
        PetscCall(VecGetArray(X, &x_arr));
        
        // Correct Geometry Handling
        Vec cellGeom;
        PetscCall(DMPlexComputeGeometryFVM(dmQ, &cellGeom, NULL));
        const PetscScalar *geom_ptr;
        PetscCall(VecGetArrayRead(cellGeom, &geom_ptr));
        
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscFVCellGeom *cg;
            PetscCall(DMPlexPointLocalRead(dmQ, c, geom_ptr, &cg));
            auto q_init = Model<Real>::initial_condition(cg->centroid, parameters.data());
            for(int i=0; i<Model<Real>::n_dof_q; ++i) {
                x_arr[(c-cStart) * Model<Real>::n_dof_q + i] = q_init[i];
            }
        }

        PetscCall(VecRestoreArrayRead(cellGeom, &geom_ptr));
        PetscCall(VecDestroy(&cellGeom));
        PetscCall(VecRestoreArray(X, &x_arr));
        PetscCall(TSSetSolution(ts, X));
        return PETSC_SUCCESS;
    }

    virtual PetscErrorCode SetupAuxiliaryConditions() {
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmAux, 0, &cStart, &cEnd));
        
        PetscScalar *a_arr;
        PetscCall(VecGetArray(A, &a_arr));

        Vec cellGeom;
        PetscCall(DMPlexComputeGeometryFVM(dmAux, &cellGeom, NULL));
        const PetscScalar *geom_ptr;
        PetscCall(VecGetArrayRead(cellGeom, &geom_ptr));

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscFVCellGeom *cg;
            PetscCall(DMPlexPointLocalRead(dmAux, c, geom_ptr, &cg));
            auto aux_init = Model<Real>::initial_aux_condition(cg->centroid, parameters.data());
            for(int i=0; i<Model<Real>::n_dof_qaux; ++i) {
                a_arr[(c-cStart) * Model<Real>::n_dof_qaux + i] = aux_init[i];
            }
        }

        PetscCall(VecRestoreArrayRead(cellGeom, &geom_ptr));
        PetscCall(VecDestroy(&cellGeom));
        PetscCall(VecRestoreArray(A, &a_arr));
        return PETSC_SUCCESS;
    }

    PetscErrorCode UpdateBoundaryGhosts(PetscReal time) {
        Vec locX;
        PetscCall(DMGetLocalVector(dmQ, &locX));
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, locX));
        PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, locX));
        PetscCall(DMPlexTSComputeBoundary(dmQ, time, locX, NULL, NULL));
        PetscCall(DMRestoreLocalVector(dmQ, &locX));
        return PETSC_SUCCESS;
    }

    PetscReal ComputeTimeStep() {
        Vec X_loc, A_loc;
        const PetscScalar *x, *a;
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(VecGetArrayRead(X_loc, &x));
        PetscCall(VecGetArrayRead(A_loc, &a));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        Real max_eig = 0.0;
        for(PetscInt c = cStart; c < cEnd; ++c) {
            const Real* qc = &x[c * Model<Real>::n_dof_q];
            const Real* ac = &a[c * Model<Real>::n_dof_qaux];
            if(qc[0] < 1e-6) continue;
            for(int d=0; d<Model<Real>::dimension; ++d) {
                Real n[3] = {0}; n[d] = 1.0;
                auto res = Numerics<Real>::local_max_abs_eigenvalue(qc, ac, parameters.data(), n);
                if(res[0] > max_eig) max_eig = res[0];
            }
        }
        
        PetscCall(VecRestoreArrayRead(X_loc, &x));
        PetscCall(VecRestoreArrayRead(A_loc, &a));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        
        PetscReal glob_eig;
        MPI_Allreduce(&max_eig, &glob_eig, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD);
        
        if (glob_eig < 1e-9) return 1e-3; 
        return settings.solver.cfl * minRadius / glob_eig;
    }

    PetscErrorCode PackState(Vec X_in, Vec A_in, Vec X_target) {
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
        return PETSC_SUCCESS;
    }

    PetscErrorCode HandleOutput(PetscReal time, PetscReal t_old, Vec X_interp, Vec X_old) {
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
            PetscCall(PackState(X, A, X_out));
            io->WriteVTK(X_out, time);
            if (settings.io.snapshot_logic == "loose") {
                while(io->GetNextSnapTime() <= time) io->AdvanceSnapshot();
            } else {
                io->AdvanceSnapshot();
            }
        }
        return PETSC_SUCCESS;
    }

    PetscErrorCode CheckStateValidity(Vec V) {
        PetscReal minVal, maxVal;
        PetscCall(VecMax(V, NULL, &maxVal));
        PetscCall(VecMin(V, NULL, &minVal));
        if (std::isnan(minVal) || std::isnan(maxVal) || std::isinf(minVal) || std::isinf(maxVal)) {
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FP, "FATAL: State vector contains NaN or Inf!");
        }
        return PETSC_SUCCESS;
    }

    PetscErrorCode SetupArchitecture(PetscInt overlap = 1) {
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

        PetscCall(DMClone(dmMesh, &dmQ)); 
        PetscFV fvm; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvm));
        PetscCall(PetscFVSetNumComponents(fvm, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvm, Model<Real>::dimension));
        for(int i=0; i<Model<Real>::n_dof_q; ++i) {
            char name[32]; snprintf(name, 32, "Field_%d", i);
            PetscCall(PetscFVSetComponentName(fvm, i, name));
        }
        PetscCall(DMAddField(dmQ, NULL, (PetscObject)fvm));
        PetscCall(DMCreateDS(dmQ));
        
        PetscDS prob; PetscCall(DMGetDS(dmQ, &prob));
        DMLabel label; PetscCall(DMGetLabel(dmQ, settings.io.mesh_label.c_str(), &label));
        auto names = Model<Real>::get_boundary_tags();
        bc_tag_ids.resize(names.size()); bc_ctx_indices.resize(names.size());
        
        if (rank == 0) {
            try {
                std::map<std::string, int> mesh_map = MeshConfigLoader::loadBoundaryMapping(settings.io.mesh_path);
                for(size_t i=0; i<names.size(); ++i) bc_tag_ids[i] = mesh_map[names[i]];
            } catch(...) {}
        }
        PetscCallMPI(MPI_Bcast(bc_tag_ids.data(), bc_tag_ids.size(), MPIU_INT, 0, PETSC_COMM_WORLD));

        for(size_t i=0; i<names.size(); ++i) {
            bc_ctx_indices[i] = (PetscInt)i; 
            PetscCall(PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, names[i].c_str(), label, 1, &bc_tag_ids[i], 0, 0, NULL, (PetscVoidFn *)VirtualSolver::BoundaryFunc, NULL, &bc_ctx_indices[i], NULL)); 
        }
        
        PetscCall(PetscFVDestroy(&fvm));
        PetscCall(DMCreateGlobalVector(dmQ, &X));

        PetscCall(DMClone(dmMesh, &dmAux)); 
        PetscFV fvmAux; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmAux));
        PetscCall(PetscFVSetNumComponents(fvmAux, Model<Real>::n_dof_qaux));
        PetscCall(PetscFVSetSpatialDimension(fvmAux, Model<Real>::dimension));
        PetscCall(DMAddField(dmAux, NULL, (PetscObject)fvmAux));
        PetscCall(DMCreateDS(dmAux));
        PetscCall(PetscFVDestroy(&fvmAux));
        PetscCall(DMCreateGlobalVector(dmAux, &A));
        PetscCall(DMSetAuxiliaryVec(dmQ, NULL, 0, 0, A));

        PetscCall(DMClone(dmMesh, &dmOut));
        PetscFV fvmQ; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmQ));
        PetscCall(PetscFVSetNumComponents(fvmQ, Model<Real>::n_dof_q));
        PetscCall(PetscFVSetSpatialDimension(fvmQ, Model<Real>::dimension));
        PetscFV fvmA; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmA));
        PetscCall(PetscFVSetNumComponents(fvmA, Model<Real>::n_dof_qaux));
        PetscCall(PetscFVSetSpatialDimension(fvmA, Model<Real>::dimension));
        PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmQ));
        PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmA));
        PetscCall(DMCreateDS(dmOut));
        PetscCall(PetscFVDestroy(&fvmQ));
        PetscCall(PetscFVDestroy(&fvmA));
        PetscCall(DMCreateGlobalVector(dmOut, &X_out));

        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
        PetscCall(TSSetDM(ts, dmQ));
        PetscCall(TSSetApplicationContext(ts, this));
        return PETSC_SUCCESS;
    }
};
#endif