static char help[] = "Robust MOOD Solver (2nd Order + Fallback)\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscfv.h>
#include <petscviewer.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

#include "Model.H"
#include "Numerics.H"

using Real = PetscScalar;

class FVMSolver {
private:
    DM          dmMesh;     // Base Topology
    DM          dmQ;        // Primary DM
    DM          dmAux;      // Auxiliary DM
    DM          dmOut;      // Output DM
    
    TS          ts;
    PetscDS     prob;
    
    Vec         X;          // Current Solution
    Vec         X_old;      // Previous Time Step
    Vec         X_low;      // Low Order Candidate
    Vec         A;          // Aux Vector (Order)
    Vec         X_out;      // Merged Output Vector
    
    PetscMPIInt rank;
    PetscReal   cfl;
    PetscReal   minRadius;
    PetscInt    max_steps;
    
    std::vector<PetscInt> bc_ids_storage; 
    
    struct StepData { PetscInt step; PetscReal time; };
    std::vector<StepData> time_series;

    // -------------------------------------------------------------------------
    // STATIC HELPERS (Moved to Top to fix Declaration Errors)
    // -------------------------------------------------------------------------
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

public:
    FVMSolver() : cfl(0.4), minRadius(0.0), max_steps(100) { 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        dmMesh = NULL; dmQ = NULL; dmAux = NULL; dmOut = NULL;
        ts = NULL; X = NULL; X_old = NULL; X_low = NULL; A = NULL; X_out = NULL; prob = NULL;
    }

    ~FVMSolver() {
        if (X)      VecDestroy(&X);
        if (X_old)  VecDestroy(&X_old);
        if (X_low)  VecDestroy(&X_low);
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
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-ts_max_steps", &max_steps, NULL));

        PetscCall(SetupBaseMesh());
        PetscCall(SetupPrimaryDM());
        PetscCall(SetupAuxiliaryDM());
        PetscCall(SetupOutputDM()); 
        
        PetscCall(SetupTimeStepping());
        PetscCall(SetupInitialConditions());

        // Create Work Vectors
        PetscCall(VecDuplicate(X, &X_old));
        PetscCall(VecDuplicate(X, &X_low));

        PetscCall(UpdateBoundaryGhosts(0.0));

        // --- MANUAL MOOD TIME LOOP ---
        PetscReal time = 0.0;
        PetscInt step = 0;
        
        // Initial Output
        PetscCall(VecSet(A, 2.0)); // Initial condition is perfect 2nd order
        PetscCall(MonitorSeries(step, time, X));

        while (step < max_steps) {
            // 1. Compute Time Step
            PetscReal dt = ComputeTimeStep();
            PetscCall(TSSetTime(ts, time));
            PetscCall(TSSetTimeStep(ts, dt));
            
            // 2. Backup State
            PetscCall(VecCopy(X, X_old));
            
            // 3. High Order Candidate Step (Order 2)
            SetSolverOrder(2);
            PetscCall(TSSetSolution(ts, X)); // TS works on X
            PetscCall(TSStep(ts));           // Advances X from t to t+dt
            
            // 4. Check Validity (DMP/TVD)
            std::vector<PetscInt> bad_cells;
            PetscCall(CheckTVD(X_old, X, bad_cells));
            
            PetscInt n_bad_local = bad_cells.size();
            PetscInt n_bad_global = 0;
            PetscCallMPI(MPI_Allreduce(&n_bad_local, &n_bad_global, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

            // 5. Fallback if needed
            if (n_bad_global > 0) {
                // Compute Low Order Candidate (Order 1)
                // We reset TS to X_old (start of step) and compute into X_low
                PetscCall(VecCopy(X_old, X_low));
                PetscCall(TSSetTime(ts, time));     
                PetscCall(TSSetStepNumber(ts, step));
                PetscCall(TSSetSolution(ts, X_low));
                
                SetSolverOrder(1);
                PetscCall(TSStep(ts)); // Advances X_low
                
                // Blend: Put X_low values into X for bad cells
                PetscCall(BlendSolutions(X, X_low, bad_cells));
                
                // Mark Aux Vector
                PetscCall(VecSet(A, 2.0)); // Reset all to 2
                PetscCall(UpdateAuxVector(A, bad_cells, 1.0)); // Mark bad as 1
                
                // Restore X as the main solution for TS
                PetscCall(TSSetSolution(ts, X));
            } else {
                PetscCall(VecSet(A, 2.0)); // All Good
            }

            // 6. Advance
            time += dt;
            step++;
            
            PetscPrintf(PETSC_COMM_WORLD, "Step %3d | Time %.5g | dt %.5g | Fallback Cells: %d\n", step, (double)time, (double)dt, n_bad_global);
            
            PetscCall(MonitorSeries(step, time, X));
            PetscCall(UpdateBoundaryGhosts(time));
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    // -------------------------------------------------------------------------
    // MOOD LOGIC
    // -------------------------------------------------------------------------
    void SetSolverOrder(int order) {
        PetscFV fvm;
        DMGetField(dmQ, 0, NULL, (PetscObject*)&fvm);
        if (order == 2) {
            PetscFVSetType(fvm, PETSCFVLEASTSQUARES);
            // Unlimited 2nd order (we rely on fallback to handle oscillations)
            PetscLimiter lim;
            PetscLimiterCreate(PETSC_COMM_WORLD, &lim);
            PetscLimiterSetType(lim, PETSCLIMITERNONE);
            PetscFVSetLimiter(fvm, lim);
            PetscLimiterDestroy(&lim);
        } else {
            PetscFVSetType(fvm, PETSCFVUPWIND);
        }
    }

    PetscErrorCode CheckTVD(Vec x_old, Vec x_new, std::vector<PetscInt>& bad_cells) {
        PetscFunctionBeginUser;
        bad_cells.clear();
        
        Vec locX_old, locX_new;
        PetscCall(DMGetLocalVector(dmQ, &locX_old));
        PetscCall(DMGetLocalVector(dmQ, &locX_new));

        PetscCall(DMGlobalToLocalBegin(dmQ, x_old, INSERT_VALUES, locX_old));
        PetscCall(DMGlobalToLocalEnd(dmQ, x_old, INSERT_VALUES, locX_old));
        PetscCall(DMGlobalToLocalBegin(dmQ, x_new, INSERT_VALUES, locX_new));
        PetscCall(DMGlobalToLocalEnd(dmQ, x_new, INSERT_VALUES, locX_new));

        const PetscScalar *old_ptr, *new_ptr;
        PetscCall(VecGetArrayRead(locX_old, &old_ptr));
        PetscCall(VecGetArrayRead(locX_new, &new_ptr));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd)); 
        const Real eps = 1e-10; 

        // Check local Min/Max principle
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt gStart, gEnd;
            PetscCall(DMPlexGetPointGlobal(dmQ, c, &gStart, &gEnd));
            if (gStart < 0) continue; // Skip ghosts

            Real u_old = old_ptr[c * Model<Real>::n_dof_q + 0]; // Check Field 0 (Density)
            Real min_val = u_old;
            Real max_val = u_old;

            const PetscInt *faces;
            PetscInt numFaces;
            PetscCall(DMPlexGetCone(dmQ, c, &faces));
            PetscCall(DMPlexGetConeSize(dmQ, c, &numFaces));
            
            for(int f=0; f<numFaces; ++f) {
                const PetscInt *neighbors;
                PetscInt numNeighbors;
                PetscCall(DMPlexGetSupport(dmQ, faces[f], &neighbors));
                PetscCall(DMPlexGetSupportSize(dmQ, faces[f], &numNeighbors));
                for(int n=0; n<numNeighbors; ++n) {
                    PetscInt nCell = neighbors[n];
                    if (nCell == c) continue;
                    Real u_neigh = old_ptr[nCell * Model<Real>::n_dof_q + 0];
                    if (u_neigh < min_val) min_val = u_neigh;
                    if (u_neigh > max_val) max_val = u_neigh;
                }
            }

            Real u_new = new_ptr[c * Model<Real>::n_dof_q + 0];
            if (u_new < min_val - eps || u_new > max_val + eps) {
                bad_cells.push_back(c);
            }
        }

        PetscCall(VecRestoreArrayRead(locX_old, &old_ptr));
        PetscCall(VecRestoreArrayRead(locX_new, &new_ptr));
        PetscCall(DMRestoreLocalVector(dmQ, &locX_old));
        PetscCall(DMRestoreLocalVector(dmQ, &locX_new));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode BlendSolutions(Vec x_target, Vec x_source, const std::vector<PetscInt>& cells) {
        PetscFunctionBeginUser;
        PetscScalar *t_ptr; 
        const PetscScalar *s_ptr;
        PetscCall(VecGetArray(x_target, &t_ptr));
        PetscCall(VecGetArrayRead(x_source, &s_ptr));
        
        PetscSection section;
        PetscCall(DMGetLocalSection(dmQ, &section));
        
        for (PetscInt c : cells) {
            PetscInt gPoint, gStart;
            PetscCall(DMPlexGetPointGlobal(dmQ, c, &gPoint, &gStart));
            if (gPoint < 0) continue; 
            
            PetscInt off;
            PetscCall(PetscSectionGetOffset(section, c, &off));
            for(int d=0; d<Model<Real>::n_dof_q; ++d) {
                t_ptr[off + d] = s_ptr[off + d];
            }
        }
        PetscCall(VecRestoreArray(x_target, &t_ptr));
        PetscCall(VecRestoreArrayRead(x_source, &s_ptr));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode UpdateAuxVector(Vec A, const std::vector<PetscInt>& cells, PetscScalar val) {
        PetscFunctionBeginUser;
        PetscScalar *a_ptr;
        PetscCall(VecGetArray(A, &a_ptr));
        PetscSection section;
        PetscCall(DMGetLocalSection(dmAux, &section));
        
        for (PetscInt c : cells) {
            PetscInt gPoint, gStart;
            PetscCall(DMPlexGetPointGlobal(dmAux, c, &gPoint, &gStart));
            if (gPoint < 0) continue; 
            
            PetscInt off;
            PetscCall(PetscSectionGetOffset(section, c, &off));
            a_ptr[off] = val;
        }
        PetscCall(VecRestoreArray(A, &a_ptr));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

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
        for(int i=0; i<Model<Real>::n_dof_q; ++i) { char name[32]; snprintf(name, 32, "Field_%d", i); PetscCall(PetscFVSetComponentName(fvm, i, name)); }
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
        for(int i=0; i<Model<Real>::n_dof_qaux; ++i) { char name[32]; snprintf(name, 32, "Aux_%d", i); PetscCall(PetscFVSetComponentName(fvmAux, i, name)); }
        PetscCall(DMAddField(dmAux, NULL, (PetscObject)fvmAux));
        PetscCall(DMCreateDS(dmAux));
        PetscCall(PetscFVDestroy(&fvmAux));
        PetscCall(DMCreateGlobalVector(dmAux, &A));
        PetscCall(PetscObjectSetName((PetscObject)A, "Auxiliary"));
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

    // --- MONITOR: Writes Single File via Merged DM ---
    PetscErrorCode MonitorSeries(PetscInt step, PetscReal time, Vec u) {
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
        PetscCall(VecGetArrayRead(u, &x_ptr));
        PetscCall(VecGetArrayRead(A, &a_ptr));
        PetscCall(VecGetArray(X_out, &out_ptr));
        
        PetscInt nQ = Model<Real>::n_dof_q;
        PetscInt nAux = Model<Real>::n_dof_qaux;
        PetscInt localSize;
        PetscCall(VecGetLocalSize(u, &localSize));
        PetscInt nCells = localSize / nQ; 
        
        for (PetscInt c = 0; c < nCells; ++c) {
            PetscInt idx_q = c * nQ;
            PetscInt idx_a = c * nAux;
            PetscInt idx_out = c * (nQ + nAux);
            for(int i=0; i<nQ; ++i) out_ptr[idx_out + i] = x_ptr[idx_q + i];
            for(int i=0; i<nAux; ++i) out_ptr[idx_out + nQ + i] = a_ptr[idx_a + i];
        }

        PetscCall(VecRestoreArrayRead(u, &x_ptr));
        PetscCall(VecRestoreArrayRead(A, &a_ptr));
        PetscCall(VecRestoreArray(X_out, &out_ptr));

        char filename[64];
        snprintf(filename, 64, "output-%03d.vtu", step);
        PetscViewer viewer;
        PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer));
        PetscCall(VecView(X_out, viewer));
        PetscCall(PetscViewerDestroy(&viewer));
        return 0;
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
        DMGetLocalVector(dmQ, &X_local);
        DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_local);
        DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_local);
        const PetscScalar *x_ptr;
        VecGetArrayRead(X_local, &x_ptr);
        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd); 
        Real max_eigen_local = 0.0;
        Real Qaux_local[Model<Real>::n_dof_qaux] = {0.0}; 
        std::vector<Real> lam(Model<Real>::n_dof_q); 
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const Real* Q_cell = &x_ptr[c * Model<Real>::n_dof_q];
            if (Q_cell[0] < 1e-6) continue; 
            for (int d = 0; d < Model<Real>::dimension; ++d) {
                Real n[3] = {0.0, 0.0, 0.0}; n[d] = 1.0;
                Model<Real>::eigenvalues(Q_cell, Qaux_local, n, lam.data());
                for(Real val : lam) max_eigen_local = std::max(max_eigen_local, std::abs(val));
            }
        }
        VecRestoreArrayRead(X_local, &x_ptr);
        DMRestoreLocalVector(dmQ, &X_local);
        Real max_eigen_global;
        MPI_Allreduce(&max_eigen_local, &max_eigen_global, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD);
        if (max_eigen_global > 1e-12) return cfl * minRadius / max_eigen_global;
        return 1e-4; 
    }

    PetscErrorCode SetupTimeStepping() {
        PetscFunctionBeginUser;
        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
        PetscCall(TSSetDM(ts, dmQ));
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(TSSetType(ts, TSSSP)); 
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
        PetscCall(DMTSSetBoundaryLocal(dmQ, DMPlexTSComputeBoundary, NULL));
        PetscCall(DMTSSetRHSFunctionLocal(dmQ, DMPlexTSComputeRHSFunctionFVM, NULL));
        PetscCall(TSSetFromOptions(ts));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode SetupInitialConditions() {
        PetscFunctionBeginUser;
        PetscErrorCode (*ic_funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void*);
        ic_funcs[0] = InitialCondition;
        void* ctxs[1] = {NULL};
        PetscCall(DMProjectFunction(dmQ, 0.0, ic_funcs, ctxs, INSERT_ALL_VALUES, X));
        PetscCall(TSSetSolution(ts, X));
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