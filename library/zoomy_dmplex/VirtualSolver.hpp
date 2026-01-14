#ifndef FVM_SOLVER_HPP
#define FVM_SOLVER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

// --- CORE PETSC HEADERS ---
#include <petscdmplex.h>
#include <petscts.h>
#include <petscfv.h>
#include <petscds.h>
#include <petscviewer.h>

#include "PetscHelpers.hpp"

class VirtualSolver {
protected:
    // Core DMs
    DM          dmMesh;     // Topology
    DM          dmQ;        // Solution Layout
    DM          dmAux;      // Auxiliary Layout
    DM          dmOut;      // Output Layout (Merged)
    
    // Core Solver Objects
    TS          ts;
    Vec         X;          // Solution Vector
    Vec         A;          // Aux Vector
    Vec         X_out;      // Merged Output Buffer
    
    // Config
    PetscMPIInt rank;
    PetscReal   cfl;
    PetscReal   minRadius;
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

    // Main Entry Point
    virtual PetscErrorCode Run(int argc, char **argv) = 0;

protected:
    // --- Setup Routines ---
    PetscErrorCode SetupArchitecture(PetscInt overlap = 1) {
        PetscFunctionBeginUser;
        PetscCall(SetupBaseMesh(overlap));
        PetscCall(SetupPrimaryDM());
        PetscCall(SetupAuxiliaryDM());
        PetscCall(SetupOutputDM());
        PetscCall(SetupTimeStepping());
        PetscCall(SetupInitialConditions());
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupBaseMesh(PetscInt overlap) {
        PetscFunctionBeginUser;
        PetscCall(DMCreate(PETSC_COMM_WORLD, &dmMesh));
        PetscCall(DMSetType(dmMesh, DMPLEX));
        PetscCall(DMSetFromOptions(dmMesh)); 
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

    PetscErrorCode WriteVTU(PetscInt step, PetscReal time) {
        PetscFunctionBeginUser;
        if (rank == 0) {
            if (time_series.empty() || time_series.back().step != step) {
                time_series.push_back({step, time});
            }
            std::ofstream f("output.vtu.series");
            if (f.is_open()) {
                f << "{\n  \"file-series-version\" : \"1.0\",\n  \"files\" : [\n";
                for (size_t i = 0; i < time_series.size(); ++i) {
                    f << "    { \"name\" : \"output-" << std::setfill('0') << std::setw(3) << time_series[i].step << ".vtu\", \"time\" : " << std::scientific << std::setprecision(6) << time_series[i].time << " }";
                    if (i < time_series.size() - 1) f << ",";
                    f << "\n";
                }
                f << "  ]\n}\n";
                f.close();
            }
        }

        const PetscScalar *x_ptr, *a_ptr;
        PetscScalar *out_ptr;
        PetscCall(VecGetArrayRead(X, &x_ptr));
        PetscCall(VecGetArrayRead(A, &a_ptr));
        PetscCall(VecGetArray(X_out, &out_ptr));
        
        PetscInt nQ = Model<Real>::n_dof_q;
        PetscInt nAux = Model<Real>::n_dof_qaux;
        PetscInt localSize;
        PetscCall(VecGetLocalSize(X, &localSize));
        PetscInt nCells = localSize / nQ; 
        
        for (PetscInt c = 0; c < nCells; ++c) {
            PetscInt idx_q = c * nQ;
            PetscInt idx_a = c * nAux;
            PetscInt idx_out = c * (nQ + nAux);
            for(int i=0; i<nQ; ++i) out_ptr[idx_out + i] = x_ptr[idx_q + i];
            for(int i=0; i<nAux; ++i) out_ptr[idx_out + nQ + i] = a_ptr[idx_a + i];
        }

        PetscCall(VecRestoreArrayRead(X, &x_ptr));
        PetscCall(VecRestoreArrayRead(A, &a_ptr));
        PetscCall(VecRestoreArray(X_out, &out_ptr));

        char filename[64];
        snprintf(filename, 64, "output-%03d.vtu", step);
        PetscViewer viewer;
        PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
        PetscCall(VecView(X_out, viewer));
        PetscCall(PetscViewerDestroy(&viewer));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};
#endif