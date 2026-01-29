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
#include <memory>

#include "Numerics.H"
#include "Settings.hpp"
#include "Model.H"      
#include "IOManager.hpp"
#include "MeshConfigLoader.hpp"

using Real = PetscReal;

// --- GLOBAL DEFINITIONS ---

enum GradientMethod { GREEN_GAUSS, LEAST_SQUARES };
enum ReconstructionType { PCM, LINEAR }; 

using FluxKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_q> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*, const PetscScalar*, 
    const PetscScalar*, const PetscScalar*);

using SourceKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_q> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*);

using JacobianKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_q * Model<Real>::n_dof_q> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*);
using JacobianAuxKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_q * Model<Real>::n_dof_qaux> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*);
using JacobianAuxQKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_qaux * Model<Real>::n_dof_q> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*);

using NonConservativeFluxKernelPtr = SimpleArray<PetscScalar, 2 * Model<Real>::n_dof_q> (*)(
    const PetscScalar* qL, const PetscScalar* qR, const PetscScalar* aL, const PetscScalar* aR, 
    const PetscScalar* p, const PetscScalar* n);

// ------------------------------------------------

class VirtualSolver {
protected:
    Settings settings;
    IOManager* io = nullptr;
    
    // DMs
    DM dmMesh;
    DM dmQ; 
    DM dmAux; 
    DM dmOut;
    DM dmGrad; 

    TS ts; 
    Vec X; 
    Vec X_old; 
    Vec A; 
    Vec G;     
    Vec X_out;
    
    PetscMPIInt rank;
    PetscReal minRadius;
    std::vector<PetscReal> parameters; 
    std::map<PetscInt, PetscInt> boundary_map;

public:
    VirtualSolver() : rank(0), minRadius(0.0) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dmMesh = NULL; dmQ = NULL; dmAux = NULL; dmOut = NULL; dmGrad = NULL;
        ts = NULL; X = NULL; X_old = NULL; A = NULL; G = NULL; X_out = NULL;
        parameters = Model<Real>::default_parameters();
    }

    virtual ~VirtualSolver() {
        if (io) delete io;
        if (X)      VecDestroy(&X);
        if (X_old)  VecDestroy(&X_old);
        if (A)      VecDestroy(&A);
        if (G)      VecDestroy(&G);
        if (X_out)  VecDestroy(&X_out);
        if (ts)     TSDestroy(&ts);
        if (dmQ)    DMDestroy(&dmQ);
        if (dmAux)  DMDestroy(&dmAux);
        if (dmOut)  DMDestroy(&dmOut);
        if (dmGrad) DMDestroy(&dmGrad);
        if (dmMesh) DMDestroy(&dmMesh);
    }

    PetscErrorCode CreateVector(DM dm, Vec* v) {
        return DMCreateGlobalVector(dm, v);
    }

    virtual PetscErrorCode Run(int argc, char **argv) {
        PetscFunctionBeginUser;
        PetscCall(Initialize(argc, argv));
        
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(RegisterCallbacks(ts));
        
        PetscCall(TSSetPreStep(ts, PreStepWrapper));
        PetscCall(TSSetPostStep(ts, PostStepWrapper));
        
        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
        PetscCall(TSSetTimeStep(ts, ComputeTimeStep())); 
        PetscCall(TSSetFromOptions(ts)); 
        PetscCall(TSMonitorSet(ts, MonitorWrapper, this, NULL));
        
        if (rank == 0) std::cout << "[INFO] Starting PETSc TSSolve..." << std::endl;
        PetscCall(TSSolve(ts, X));
        if (rank == 0) std::cout << "[INFO] TSSolve Finished." << std::endl;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

protected:
    virtual PetscErrorCode RegisterCallbacks(TS ts) = 0;
    virtual PetscErrorCode UpdateState(Vec Q_loc, Vec Aux_loc) = 0; 

    static PetscErrorCode PreStepWrapper(TS ts) {
        void *ctx; PetscCall(TSGetApplicationContext(ts, &ctx));
        return ((VirtualSolver*)ctx)->PreStep(ts);
    }
    virtual PetscErrorCode PreStep(TS ts) {
        if (X_old) PetscCall(VecCopy(X, X_old)); 
        return PETSC_SUCCESS;
    }

    static PetscErrorCode PostStepWrapper(TS ts) {
        void *ctx; PetscCall(TSGetApplicationContext(ts, &ctx));
        return ((VirtualSolver*)ctx)->PostStep(ts);
    }
    virtual PetscErrorCode PostStep(TS ts) {
        Vec X_curr; PetscCall(TSGetSolution(ts, &X_curr));
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_curr, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X_curr, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        PetscCall(UpdateState(X_loc, A_loc)); 

        PetscCall(DMLocalToGlobalBegin(dmQ, X_loc, INSERT_VALUES, X_curr));
        PetscCall(DMLocalToGlobalEnd(dmQ, X_loc, INSERT_VALUES, X_curr));
        PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A));
        PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));

        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));

        PetscReal dt_next = ComputeTimeStep();
        PetscCall(TSSetTimeStep(ts, dt_next));
        return PETSC_SUCCESS;
    }

    static PetscErrorCode MonitorWrapper(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
        return ((VirtualSolver*)ctx)->Monitor(ts, step, time, X);
    }

    PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal time, Vec X_curr) {
        if (io->ShouldWrite(time)) {
            PetscCall(PackState(X_curr, A, X_out));
            // Correctly passing dmOut to WriteVTK
            io->WriteVTK(dmOut, X_out, time);
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
        
        auto valid_names = Model<Real>::parameter_names();
        
        for (const auto& [key, val] : settings.model.parameters) {
            bool found = false;
            for (size_t i = 0; i < valid_names.size(); ++i) {
                if (valid_names[i] == key) {
                    this->parameters[i] = val; 
                    if (rank == 0) {
                        std::cout << "[INFO] Overriding parameter '" << key << "' = " << val << std::endl;
                    }
                    found = true;
                    break;
                }
            }
            if (!found && rank == 0) {
                std::cout << "[WARNING] Settings contained parameter '" << key 
                          << "' which is NOT defined in the Model. Ignoring." << std::endl;
            }
        }
        
        io = new IOManager(settings, rank);
        io->PrepareDirectory();
        PetscCall(SetupArchitecture(2));
        PetscCall(SetupInitialConditions()); 
        PetscCall(SetupAuxiliaryConditions()); 
        
        if (X_old) PetscCall(VecCopy(X, X_old));
        return PETSC_SUCCESS;
    }

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

    PetscErrorCode SetupArchitecture(PetscInt overlap = 1) {
        if (!settings.io.mesh_path.empty()) PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, settings.io.mesh_path.c_str(), "zoomy_mesh", PETSC_TRUE, &dmMesh));
        else { PetscCall(DMCreate(PETSC_COMM_WORLD, &dmMesh)); PetscCall(DMSetType(dmMesh, DMPLEX)); }
        
        auto names = Model<Real>::get_boundary_tags();
        std::map<std::string, std::pair<int, int>> mesh_map; 
        bool mismatch = false;
        bool map_empty = false;

        if (rank == 0) { 
            try { 
                mesh_map = MeshConfigLoader::loadBoundaryMapping(settings.io.mesh_path); 
                map_empty = mesh_map.empty();
            } catch(...) { 
                map_empty = true;
            }
            
            int boundary_dim = Model<Real>::dimension - 1; 
            for (const auto& name : names) { 
                if (name == "default") continue; 
                if (mesh_map.find(name) == mesh_map.end() || mesh_map[name].first != boundary_dim) mismatch = true; 
            }
        }
        
        PetscCallMPI(MPI_Bcast(&mismatch, 1, MPI_C_BOOL, 0, PETSC_COMM_WORLD));
        PetscCallMPI(MPI_Bcast(&map_empty, 1, MPI_C_BOOL, 0, PETSC_COMM_WORLD));

        if (mismatch) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "GMSH boundaries must exactly match model tags (excluding 'default').");

        // --- FIX: Enable Natural Ordering Mapping ---
        // This is crucial for HDF5 loading to work correctly in parallel!
        PetscCall(DMSetUseNatural(dmMesh, PETSC_TRUE));
        // --------------------------------------------

        DM dmDist = NULL; PetscCall(DMPlexDistribute(dmMesh, overlap, NULL, &dmDist));
        if (dmDist) { PetscCall(DMDestroy(&dmMesh)); dmMesh = dmDist; }
        
        if (map_empty) {
            DMLabel label;
            PetscCall(DMGetLabel(dmMesh, "Face Sets", &label));
            if (!label) {
                PetscCall(DMCreateLabel(dmMesh, "Face Sets"));
                PetscCall(DMGetLabel(dmMesh, "Face Sets", &label));
            }
            PetscCall(DMPlexMarkBoundaryFaces(dmMesh, 1, label));
        }

        PetscCall(DMLocalizeCoordinates(dmMesh));
        PetscCall(DMSetBasicAdjacency(dmMesh, settings.solver.use_deep_adjacency ? PETSC_TRUE : PETSC_FALSE, PETSC_FALSE));
        
        PetscCall(DMClone(dmMesh, &dmQ)); 
        PetscCall(DMClone(dmMesh, &dmAux)); 
        PetscCall(DMClone(dmMesh, &dmOut));
        PetscCall(DMClone(dmMesh, &dmGrad));

        PetscFV fvm; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvm)); PetscCall(PetscFVSetNumComponents(fvm, Model<Real>::n_dof_q)); PetscCall(PetscFVSetSpatialDimension(fvm, Model<Real>::dimension)); PetscCall(PetscFVSetType(fvm, PETSCFVUPWIND)); 
        PetscCall(DMAddField(dmQ, NULL, (PetscObject)fvm)); PetscCall(DMCreateDS(dmQ)); PetscCall(PetscFVDestroy(&fvm));
        PetscCall(DMCreateGlobalVector(dmQ, &X));
        PetscCall(DMCreateGlobalVector(dmQ, &X_old)); 
        
        PetscFV fvmAux; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmAux)); PetscCall(PetscFVSetNumComponents(fvmAux, Model<Real>::n_dof_qaux)); PetscCall(PetscFVSetSpatialDimension(fvmAux, Model<Real>::dimension)); PetscCall(PetscFVSetType(fvmAux, PETSCFVUPWIND));
        PetscCall(DMAddField(dmAux, NULL, (PetscObject)fvmAux)); PetscCall(DMCreateDS(dmAux)); PetscCall(PetscFVDestroy(&fvmAux));
        PetscCall(DMCreateGlobalVector(dmAux, &A));

        PetscFV fvmGrad; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmGrad));
        PetscCall(PetscFVSetNumComponents(fvmGrad, Model<Real>::n_dof_q * Model<Real>::dimension)); 
        PetscCall(PetscFVSetSpatialDimension(fvmGrad, Model<Real>::dimension));
        PetscCall(PetscFVSetType(fvmGrad, PETSCFVUPWIND)); 
        PetscCall(DMAddField(dmGrad, NULL, (PetscObject)fvmGrad)); 
        PetscCall(DMCreateDS(dmGrad)); 
        PetscCall(PetscFVDestroy(&fvmGrad));
        PetscCall(DMCreateGlobalVector(dmGrad, &G)); 
        
        PetscFV fvmQ_out; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmQ_out)); PetscCall(PetscFVSetNumComponents(fvmQ_out, Model<Real>::n_dof_q)); PetscCall(PetscFVSetSpatialDimension(fvmQ_out, Model<Real>::dimension));
        PetscFV fvmA_out; PetscCall(PetscFVCreate(PETSC_COMM_WORLD, &fvmA_out)); PetscCall(PetscFVSetNumComponents(fvmA_out, Model<Real>::n_dof_qaux)); PetscCall(PetscFVSetSpatialDimension(fvmA_out, Model<Real>::dimension));
        PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmQ_out)); PetscCall(DMAddField(dmOut, NULL, (PetscObject)fvmA_out)); PetscCall(DMCreateDS(dmOut)); PetscCall(PetscFVDestroy(&fvmQ_out)); PetscCall(PetscFVDestroy(&fvmA_out));
        PetscCall(DMCreateGlobalVector(dmOut, &X_out));
        
        for(size_t i=0; i<names.size(); ++i) { 
            PetscInt tag_id = -1; 
            if (rank == 0) {
                if (mesh_map.find(names[i]) != mesh_map.end()) {
                    tag_id = mesh_map[names[i]].second;
                } else if (names[i] == "default") {
                    tag_id = 1; 
                }
            }
            PetscCallMPI(MPI_Bcast(&tag_id, 1, MPIU_INT, 0, PETSC_COMM_WORLD)); 
            if (tag_id != -1) boundary_map[tag_id] = (PetscInt)i; 
        }
        
        PetscCall(DMPlexGetGeometryFVM(dmMesh, NULL, NULL, &minRadius));
        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts)); PetscCall(TSSetDM(ts, dmQ));
        return PETSC_SUCCESS;
    }
};
#endif