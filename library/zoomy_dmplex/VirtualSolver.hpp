#ifndef VIRTUALSOLVER_HPP
#define VIRTUALSOLVER_HPP

#include <petscdmplex.h>
#include <petscts.h>
#include <petscfv.h>
#include <petscds.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>

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
    
    DM dmMesh, dmQ, dmAux, dmOut;
    TS ts; 
    Vec X, A, X_out;
    
    PetscMPIInt rank;
    PetscReal minRadius;
    std::vector<PetscReal> parameters; 
    std::map<PetscInt, PetscInt> boundary_map;

public:
    VirtualSolver() : rank(0), minRadius(0.0) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dmMesh = NULL; dmQ = NULL; dmAux = NULL; dmOut = NULL;
        ts = NULL; X = NULL; A = NULL; X_out = NULL;
        parameters = {9.81, 1.0, 1e-6}; 
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
        
        // --- PETSc TS Setup ---
        PetscCall(TSSetApplicationContext(ts, this));
        
        // 1. Register callbacks (RHS, IFunction)
        PetscCall(RegisterCallbacks(ts));

        // 2. Register PostStep for State Updates
        PetscCall(TSSetPostStep(ts, PostStepWrapper));

        // 3. Time Loop Config
        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
        PetscCall(TSSetTimeStep(ts, ComputeTimeStep())); 
        PetscCall(TSSetFromOptions(ts)); 

        // 4. Monitor for I/O
        PetscCall(TSMonitorSet(ts, MonitorWrapper, this, NULL));

        // 5. Solve
        if (rank == 0) std::cout << "[INFO] Starting PETSc TSSolve..." << std::endl;
        PetscCall(TSSolve(ts, X));
        if (rank == 0) std::cout << "[INFO] TSSolve Finished." << std::endl;

        PetscFunctionReturn(PETSC_SUCCESS);
    }

protected:
    virtual PetscErrorCode RegisterCallbacks(TS ts) = 0;
    virtual PetscErrorCode UpdateState(Vec Q_loc, Vec Aux_loc) = 0; 

    // --- PostStep: Physics Constraints & Time Step Control ---
    static PetscErrorCode PostStepWrapper(TS ts) {
        void *ctx; PetscCall(TSGetApplicationContext(ts, &ctx));
        return ((VirtualSolver*)ctx)->PostStep(ts);
    }

    PetscErrorCode PostStep(TS ts) {
        // 1. Update Physics State (Constraints & Aux)
        Vec X_curr; PetscCall(TSGetSolution(ts, &X_curr));

        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_curr, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X_curr, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        // Perform the Physics Update
        PetscCall(UpdateState(X_loc, A_loc));

        // Scatter back
        PetscCall(DMLocalToGlobalBegin(dmQ, X_loc, INSERT_VALUES, X_curr));
        PetscCall(DMLocalToGlobalEnd(dmQ, X_loc, INSERT_VALUES, X_curr));
        PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A));
        PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));

        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));

        // 2. Compute Next Time Step
        PetscReal dt_next = ComputeTimeStep();
        PetscCall(TSSetTimeStep(ts, dt_next));

        return PETSC_SUCCESS;
    }

    // --- Monitor: I/O Only (Read-Only) ---
    static PetscErrorCode MonitorWrapper(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
        return ((VirtualSolver*)ctx)->Monitor(ts, step, time, X);
    }

    PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal time, Vec X_curr) {
        if (io->ShouldWrite(time)) {
            PetscCall(PackState(X_curr, A, X_out));
            io->WriteVTK(X_out, time);
            io->AdvanceSnapshot();
        }
        
        if (rank == 0 && step % 10 == 0) {
            PetscReal dt; TSGetTimeStep(ts, &dt);
            PetscPrintf(PETSC_COMM_WORLD, "Step %d Time %.4f dt %.4e\n", step, (double)time, (double)dt);
        }
        return PETSC_SUCCESS;
    }

    PetscErrorCode Initialize(int argc, char **argv) {
        char settings_path[PETSC_MAX_PATH_LEN] = "settings.json";
        PetscCall(PetscOptionsGetString(NULL, NULL, "-settings", settings_path, PETSC_MAX_PATH_LEN, NULL));
        settings = Settings::from_json(settings_path);
        io = new IOManager(settings, rank);
        io->PrepareDirectory();
        PetscCall(SetupArchitecture(1));
        PetscCall(SetupInitialConditions()); 
        PetscCall(SetupAuxiliaryConditions()); 
        
        // Initial state sync
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc)); PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(UpdateState(X_loc, A_loc));
        PetscCall(DMLocalToGlobalBegin(dmQ, X_loc, INSERT_VALUES, X)); PetscCall(DMLocalToGlobalEnd(dmQ, X_loc, INSERT_VALUES, X));
        PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A)); PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        
        return PETSC_SUCCESS;
    }

    // [SetupArchitecture, SetupInitialConditions, ComputeTimeStep, PackState wrappers... identical to previous]
    // (Included for complete file replacement)
    static PetscErrorCode ICWrapper(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) { Real* params = (Real*)ctx; auto res = Model<Real>::initial_condition(x, params); for(int i=0; i<Nf; ++i) u[i] = res[i]; return PETSC_SUCCESS; }
    static PetscErrorCode AuxWrapper(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) { Real* params = (Real*)ctx; auto res = Model<Real>::initial_aux_condition(x, params); for(int i=0; i<Nf; ++i) u[i] = res[i]; return PETSC_SUCCESS; }
    virtual PetscErrorCode SetupInitialConditions() { PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*) = { ICWrapper }; void* ctxs[1] = { parameters.data() }; PetscCall(DMProjectFunction(dmQ, 0.0, funcs, ctxs, INSERT_ALL_VALUES, X)); PetscCall(TSSetSolution(ts, X)); return PETSC_SUCCESS; }
    virtual PetscErrorCode SetupAuxiliaryConditions() { PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*) = { AuxWrapper }; void* ctxs[1] = { parameters.data() }; PetscCall(DMProjectFunction(dmAux, 0.0, funcs, ctxs, INSERT_ALL_VALUES, A)); return PETSC_SUCCESS; }
    PetscReal ComputeTimeStep() {
        Vec X_loc, A_loc; const PetscScalar *x, *a;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc)); PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(VecGetArrayRead(X_loc, &x)); PetscCall(VecGetArrayRead(A_loc, &a));
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        Real max_eig = 0.0;
        for(PetscInt c = cStart; c < cEnd; ++c) {
            const Real *qc, *ac;
            PetscCall(DMPlexPointLocalRead(dmQ, c, x, &qc)); PetscCall(DMPlexPointLocalRead(dmAux, c, a, &ac));
            if (!qc || !ac) continue;
            for(int d=0; d<Model<Real>::dimension; ++d) {
                Real n[3] = {0}; n[d] = 1.0;
                auto res = Numerics<Real>::local_max_abs_eigenvalue(qc, ac, parameters.data(), n);
                if(res[0] > max_eig) max_eig = res[0];
            }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x)); PetscCall(VecRestoreArrayRead(A_loc, &a));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscReal glob_eig; MPI_Allreduce(&max_eig, &glob_eig, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD);
        if (glob_eig < 1e-14) return 1e-3;
        return settings.solver.cfl * minRadius / glob_eig;
    }
    PetscErrorCode PackState(Vec X_in, Vec A_in, Vec X_target) {
        const PetscScalar *arr_x, *arr_a; PetscScalar *arr_out;
        PetscCall(VecGetArrayRead(X_in, &arr_x)); PetscCall(VecGetArrayRead(A_in, &arr_a)); PetscCall(VecGetArray(X_target, &arr_out));
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar *qx, *qa; PetscScalar *qout;
            PetscCall(DMPlexPointGlobalRead(dmQ, c, arr_x, &qx)); PetscCall(DMPlexPointGlobalRead(dmAux, c, arr_a, &qa)); PetscCall(DMPlexPointGlobalRef(dmOut, c, arr_out, &qout));
            if (qout) {
                int n_q = Model<Real>::n_dof_q, n_aux = Model<Real>::n_dof_qaux;
                if (qx) for (int i = 0; i < n_q; ++i) qout[i] = qx[i];
                if (qa) for (int i = 0; i < n_aux; ++i) qout[n_q + i] = qa[i];
            }
        }
        PetscCall(VecRestoreArrayRead(X_in, &arr_x)); PetscCall(VecRestoreArrayRead(A_in, &arr_a)); PetscCall(VecRestoreArray(X_target, &arr_out));
        return PETSC_SUCCESS;
    }
    PetscErrorCode DebugGeometry(DM dm) { if (rank != 0) return PETSC_SUCCESS; PetscInt cStart, cEnd, fStart, fEnd; PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd)); PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd)); Vec cellGeom, faceGeom; PetscCall(DMPlexGetGeometryFVM(dm, &faceGeom, &cellGeom, NULL)); PetscCall(VecDestroy(&cellGeom)); PetscCall(VecDestroy(&faceGeom)); return PETSC_SUCCESS; }
    PetscErrorCode SetupArchitecture(PetscInt overlap = 1) {
        if (!settings.io.mesh_path.empty()) PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, settings.io.mesh_path.c_str(), "zoomy_mesh", PETSC_TRUE, &dmMesh));
        else { PetscCall(DMCreate(PETSC_COMM_WORLD, &dmMesh)); PetscCall(DMSetType(dmMesh, DMPLEX)); }
        DM dmDist = NULL; PetscCall(DMPlexDistribute(dmMesh, overlap, NULL, &dmDist));
        if (dmDist) { PetscCall(DMDestroy(&dmMesh)); dmMesh = dmDist; }
        PetscCall(DMLocalizeCoordinates(dmMesh));
        PetscCall(DMSetBasicAdjacency(dmMesh, settings.solver.use_deep_adjacency ? PETSC_TRUE : PETSC_FALSE, PETSC_FALSE));
        PetscCall(DMClone(dmMesh, &dmQ)); PetscCall(DMClone(dmMesh, &dmAux)); PetscCall(DMClone(dmMesh, &dmOut));
        PetscFV fvm; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvm)); PetscCall(PetscFVSetNumComponents(fvm, Model<Real>::n_dof_q)); PetscCall(PetscFVSetSpatialDimension(fvm, Model<Real>::dimension)); PetscCall(PetscFVSetType(fvm, PETSCFVUPWIND)); 
        PetscCall(DMAddField(dmQ, NULL, (PetscObject)fvm)); PetscCall(DMCreateDS(dmQ)); PetscCall(PetscFVDestroy(&fvm));
        PetscCall(DMCreateGlobalVector(dmQ, &X));
        PetscFV fvmAux; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmAux)); PetscCall(PetscFVSetNumComponents(fvmAux, Model<Real>::n_dof_qaux)); PetscCall(PetscFVSetSpatialDimension(fvmAux, Model<Real>::dimension)); PetscCall(PetscFVSetType(fvmAux, PETSCFVUPWIND));
        PetscCall(DMAddField(dmAux, NULL, (PetscObject)fvmAux)); PetscCall(DMCreateDS(dmAux)); PetscCall(PetscFVDestroy(&fvmAux));
        PetscCall(DMCreateGlobalVector(dmAux, &A));
        PetscFV fvmQ_out; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmQ_out)); PetscCall(PetscFVSetNumComponents(fvmQ_out, Model<Real>::n_dof_q)); PetscCall(PetscFVSetSpatialDimension(fvmQ_out, Model<Real>::dimension));
        PetscFV fvmA_out; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmA_out)); PetscCall(PetscFVSetNumComponents(fvmA_out, Model<Real>::n_dof_qaux)); PetscCall(PetscFVSetSpatialDimension(fvmA_out, Model<Real>::dimension));
        PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmQ_out)); PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmA_out)); PetscCall(DMCreateDS(dmOut)); PetscCall(PetscFVDestroy(&fvmQ_out)); PetscCall(PetscFVDestroy(&fvmA_out));
        PetscCall(DMCreateGlobalVector(dmOut, &X_out));
        auto names = Model<Real>::get_boundary_tags();
        std::map<std::string, std::pair<int, int>> mesh_map; bool mismatch = false;
        if (rank == 0) { try { mesh_map = MeshConfigLoader::loadBoundaryMapping(settings.io.mesh_path); int boundary_dim = Model<Real>::dimension - 1; 
                for (const auto& name : names) { if (mesh_map.find(name) == mesh_map.end() || mesh_map[name].first != boundary_dim) mismatch = true; }
            } catch(...) { mismatch = true; } }
        PetscCallMPI(MPI_Bcast(&mismatch, 1, MPI_C_BOOL, 0, PETSC_COMM_WORLD));
        if (mismatch) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "GMSH boundaries must exactly match model tags.");
        for(size_t i=0; i<names.size(); ++i) { PetscInt tag_id = -1; if (rank == 0) tag_id = mesh_map[names[i]].second;
            PetscCallMPI(MPI_Bcast(&tag_id, 1, MPIU_INT, 0, PETSC_COMM_WORLD)); boundary_map[tag_id] = (PetscInt)i; }
        PetscCall(DMPlexGetGeometryFVM(dmMesh, NULL, NULL, &minRadius));
        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts)); PetscCall(TSSetDM(ts, dmQ));
        return PETSC_SUCCESS;
    }
};
#endif