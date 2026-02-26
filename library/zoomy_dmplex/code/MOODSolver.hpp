#ifndef MOOD_SOLVER_HPP
#define MOOD_SOLVER_HPP

#include "ModularSolver.hpp"
#include "SolverStrategies.hpp"
#include <vector>
#include <algorithm>
#include <numeric> 

class MOODSolver : public ModularSolver {
private:
    Vec X_backup;
    Vec X_low;  
    
    // Mask: 0=High Order, 1=Low Order
    std::vector<uint8_t> current_cell_orders; 

    struct CellBounds { PetscReal min_val; PetscReal max_val; };
    std::vector<std::vector<CellBounds>> bounds_cache;

public:
    MOODSolver() : ModularSolver() { X_backup = NULL; X_low = NULL; }

    MOODSolver(std::shared_ptr<SolverStrategy> strat) : ModularSolver() {
        this->SetStrategy(strat);
        X_backup = NULL; 
        X_low = NULL;
    }

    ~MOODSolver() { 
        if(X_backup) VecDestroy(&X_backup); 
        if(X_low) VecDestroy(&X_low);
    }

    // --- Helper to Enforce Model Constraints on Global State ---
    PetscErrorCode EnforcePhysicalConstraints(Vec U_global) {
        PetscFunctionBeginUser;
        Vec U_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &U_loc)); 
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        
        PetscCall(DMGlobalToLocalBegin(dmQ, U_global, INSERT_VALUES, U_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, U_global, INSERT_VALUES, U_loc));
        
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        PetscCall(transport->UpdateState(U_loc, A_loc)); 

        PetscCall(DMLocalToGlobalBegin(dmQ, U_loc, INSERT_VALUES, U_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, U_loc, INSERT_VALUES, U_global));

        PetscCall(DMRestoreLocalVector(dmQ, &U_loc)); 
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(VirtualSolver::Initialize(argc, argv));
        
        SetReconstruction(LINEAR); 
        PetscCall(InitializeComponents()); 

        std::vector<std::string> names = {"b", "h", "u", "v", "w", "p"};
        PetscCall(io->Setup3D(dmQ, 6, names));
        // Use this->LoadInitialCondition() to ensure visibility
        PetscCall(this->LoadInitialCondition());

        if (!strategy) strategy = std::make_shared<SplittingStrategy>();
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(RegisterCallbacks(ts)); 
        
        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
        
        PetscCall(VecDuplicate(X, &X_backup));
        PetscCall(VecDuplicate(X, &X_low));

        PetscReal dt_start = std::max(ComputeTimeStep(), settings.solver.min_dt);
        PetscCall(TSSetTimeStep(ts, dt_start));
        PetscCall(TSSetFromOptions(ts)); 

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        current_cell_orders.assign(cEnd - cStart, 0); 
        
        PetscInt local_n_cells = cEnd - cStart;
        PetscInt global_n_cells = 0;
        MPI_Allreduce(&local_n_cells, &global_n_cells, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

        if (rank == 0) std::cout << "[INFO] Starting MOOD Solver..." << std::endl;

        PetscReal time;
        PetscCall(TSGetTime(ts, &time));
        PetscInt step_num = 0;
        PetscCall(MonitorWrapper(ts, 0, time, X, this));

        while (time < settings.solver.t_end) {
            PetscCall(PrecomputeDMPBounds(X));
            PetscCall(VecCopy(X, X_backup));

            std::fill(current_cell_orders.begin(), current_cell_orders.end(), 0);
            transport->SetCellOrders(current_cell_orders);
            SetOrder(2); 
            PetscCall(TSStep(ts));

            bool needs_rollback = DetectTroubledCells(X);
            
            PetscMPIInt local_rb = needs_rollback ? 1 : 0;
            PetscMPIInt global_rb = 0;
            MPI_Allreduce(&local_rb, &global_rb, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);

            if (global_rb == 1) {
                if (step_num % 10 == 0) {
                    PetscInt local_bad = 0;
                    for(auto val : current_cell_orders) if(val == 1) local_bad++;
                    PetscInt global_bad = 0;
                    MPI_Allreduce(&local_bad, &global_bad, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
                    if (rank == 0) {
                        double percent = 100.0 * (double)global_bad / (double)global_n_cells;
                        if(global_bad > 0) std::cout << "  [MOOD] " << global_bad << " cells (" << percent << "%) replaced." << std::endl;
                    }
                }

                transport->SetCellOrders(current_cell_orders);
                PetscCall(VecCopy(X_backup, X));
                PetscCall(TSSetSolution(ts, X));
                PetscCall(TSSetTime(ts, time)); 
                
                SetOrder(1);
                PetscCall(TSStep(ts));
            }

            // --- Enforce Physical Constraints (Global) ---
            PetscCall(EnforcePhysicalConstraints(X));

            step_num++;
            PetscCall(TSGetTime(ts, &time));
            
            PetscCall(PostStep(ts)); 
            PetscCall(MonitorWrapper(ts, step_num, time, X, this));
        }

        if (rank == 0) std::cout << "[INFO] Finished." << std::endl;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    void SetOrder(int order) {
        if (order == 2 && settings.solver.reconstruction_order >= 2) {
            transport->SetReconstruction(std::make_shared<LinearReconstructor<Real>>());
        } else {
            transport->SetReconstruction(std::make_shared<PCMReconstructor<Real>>());
        }
    }

    PetscErrorCode ApplyReplacement() {
        PetscScalar *x_arr; const PetscScalar *low_arr;
        PetscCall(VecGetArray(X, &x_arr)); PetscCall(VecGetArrayRead(X_low, &low_arr));
        
        PetscInt rstart;
        PetscCall(VecGetOwnershipRange(X, &rstart, NULL));
        PetscSection sGlob;
        PetscCall(DMGetGlobalSection(dmQ, &sGlob));

        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));

        for (PetscInt c = cStart; c < cEnd; ++c) {
            if (current_cell_orders[c - cStart] == 1) {
                PetscInt off;
                PetscCall(PetscSectionGetOffset(sGlob, c, &off)); 
                if (off >= 0) {
                    PetscInt idx = off - rstart; 
                    for (int i = 0; i < Model<Real>::n_dof_q; ++i) {
                        x_arr[idx + i] = low_arr[idx + i];
                    }
                }
            }
        }
        PetscCall(VecRestoreArray(X, &x_arr)); PetscCall(VecRestoreArrayRead(X_low, &low_arr));
        return PETSC_SUCCESS;
    }

    PetscErrorCode PrecomputeDMPBounds(Vec U_curr) {
        PetscFunctionBeginUser;
        Vec X_loc; PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, U_curr, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, U_curr, INSERT_VALUES, X_loc));
        const PetscScalar* x_arr; PetscCall(VecGetArrayRead(X_loc, &x_arr));

        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        bounds_cache.resize(cEnd - cStart);
        
        std::vector<int> check_indices = {1}; 
        if (Model<Real>::n_dof_q > 2) check_indices.push_back(2);
        if (Model<Real>::n_dof_q > 4) check_indices.push_back(4);

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt num_adj = -1; PetscInt *adj = NULL; PetscCall(DMPlexGetAdjacency(dmQ, c, &num_adj, &adj));
            const PetscScalar* qc; PetscCall(DMPlexPointLocalRead(dmQ, c, x_arr, &qc));
            bounds_cache[c - cStart].resize(Model<Real>::n_dof_q);
            
            for(int idx : check_indices) { 
                bounds_cache[c - cStart][idx] = {qc[idx], qc[idx]}; 
            }

            for (int k = 0; k < num_adj; ++k) {
                PetscInt n = adj[k]; if (n < 0) continue;
                const PetscScalar* qn; PetscCall(DMPlexPointLocalRead(dmQ, n, x_arr, &qn));
                for(int idx : check_indices) {
                    if (qn[idx] < bounds_cache[c - cStart][idx].min_val) bounds_cache[c - cStart][idx].min_val = qn[idx];
                    if (qn[idx] > bounds_cache[c - cStart][idx].max_val) bounds_cache[c - cStart][idx].max_val = qn[idx];
                }
            }
            PetscCall(PetscFree(adj));
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_arr)); PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    bool DetectTroubledCells(Vec U_next) {
        Vec X_loc; DMGetLocalVector(dmQ, &X_loc);
        DMGlobalToLocalBegin(dmQ, U_next, INSERT_VALUES, X_loc);
        DMGlobalToLocalEnd(dmQ, U_next, INSERT_VALUES, X_loc);
        const PetscScalar* x_arr; VecGetArrayRead(X_loc, &x_arr);

        PetscInt cStart, cEnd; DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd);
        
        std::vector<int> check_indices = {1}; 
        if (Model<Real>::n_dof_q > 2) check_indices.push_back(2);
        if (Model<Real>::n_dof_q > 4) check_indices.push_back(4);

        const Real eps_rel = 1e-4; 
        const Real eps_abs = 1e-7;

        bool found_new_bad = false;

        for (PetscInt c = cStart; c < cEnd; ++c) {
            if (current_cell_orders[c - cStart] == 1) continue;

            const PetscScalar* q; DMPlexPointLocalRead(dmQ, c, x_arr, &q);
            bool bad = false;

            if (q[1] < 0.0) {
                bad = true; 
            }

            if (!bad) {
                for(int idx : check_indices) {
                    Real min_b = bounds_cache[c - cStart][idx].min_val;
                    Real max_b = bounds_cache[c - cStart][idx].max_val;
                    Real range = max_b - min_b;
                    Real tol = std::max(eps_abs, range * eps_rel);

                    if (q[idx] < min_b - tol || q[idx] > max_b + tol) {
                        bad = true; 
                        break;
                    }
                }
            }

            if (bad) {
                current_cell_orders[c - cStart] = 1; 
                found_new_bad = true;
            }
        }

        VecRestoreArrayRead(X_loc, &x_arr);
        DMRestoreLocalVector(dmQ, &X_loc);
        return found_new_bad;
    }
};
#endif