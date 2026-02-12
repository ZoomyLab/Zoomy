#ifndef MOOD_SOLVER_HPP
#define MOOD_SOLVER_HPP

#include "ModularSolver.hpp"
#include <vector>
#include <algorithm>
#include <numeric> 

class MOODSolver : public ModularSolver {
private:
    Vec X_backup;
    
    // Mask: 0=High Order, 1=Low Order
    std::vector<uint8_t> current_cell_orders; 

    struct CellBounds { PetscReal min_val; PetscReal max_val; };
    std::vector<std::vector<CellBounds>> bounds_cache;

public:
    MOODSolver() : ModularSolver() { X_backup = NULL; }

    MOODSolver(std::shared_ptr<SolverStrategy> strat) : ModularSolver() {
        this->strategy = strat;
        X_backup = NULL;
    }

    ~MOODSolver() { 
        if(X_backup) VecDestroy(&X_backup); 
    }

    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(VirtualSolver::Initialize(argc, argv));
        
        // --- MOOD CONFIGURATION ---
        if (settings.solver.reconstruction_order >= 2) {
            SetLimiters(false); 
        }
        // --------------------------

        PetscCall(InitializeComponents()); 

        int n_dof = Model<Real>::n_dof_q;
        std::vector<std::string> names;
        if (n_dof >= 6) names = {"b", "h", "u", "v", "w", "p"};
        else if (n_dof == 4) names = {"b", "h", "hu", "hv"};
        else names = {"b", "h"}; 
        
        PetscCall(io->Setup3D(dmQ, n_dof, names));
        PetscCall(LoadInitialCondition());

        if (!strategy) strategy = std::make_shared<SplittingStrategy>();
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(RegisterCallbacks(ts)); 
        
        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
        
        if (X_backup) PetscCall(VecDestroy(&X_backup));
        PetscCall(VecDuplicate(X, &X_backup));

        // Initial Time Step
        PetscReal dt_start = ComputeTimeStep();
        dt_start = std::max(dt_start, settings.solver.min_dt);
        PetscCall(TSSetTimeStep(ts, dt_start));
        PetscCall(TSSetFromOptions(ts)); 
        
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        current_cell_orders.assign(cEnd - cStart, 0); 
        
        PetscInt local_n_cells = cEnd - cStart;
        PetscInt global_n_cells = 0;
        MPI_Allreduce(&local_n_cells, &global_n_cells, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

        if (rank == 0) {
            std::cout << "[INFO] Starting MOOD Solver..." << std::endl;
            if (settings.solver.reconstruction_order == 1) {
                std::cout << "[INFO] Running in DEBUG MODE (Order=1). MOOD Rollback is effectively disabled." << std::endl;
            } else {
                std::cout << "[INFO] Strategy: Unlimited Order 2 Candidate -> Rollback to Order 1." << std::endl;
            }
        }

        PetscReal time;
        PetscCall(TSGetTime(ts, &time));
        PetscInt step_num = 0;
        PetscCall(MonitorWrapper(ts, 0, time, X, this));

        while (time < settings.solver.t_end) {
            // 1. Pre-step: Compute bounds and backup solution
            PetscCall(PrecomputeDMPBounds(X));
            PetscCall(VecCopy(X, X_backup));

            std::fill(current_cell_orders.begin(), current_cell_orders.end(), 0);
            transport->SetCellOrders(current_cell_orders);

            // 2. Predictor Step
            PetscCall(TSStep(ts));

            // 3. Detection
            bool needs_rollback = DetectTroubledCells(X);
            
            PetscMPIInt local_rb = needs_rollback ? 1 : 0;
            PetscMPIInt global_rb = 0;
            MPI_Allreduce(&local_rb, &global_rb, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);

            if (global_rb == 1) {
                if (rank == 0 && step_num % 1 == 0) {
                    PetscInt local_bad = 0;
                    for(auto val : current_cell_orders) if(val == 1) local_bad++;
                    PetscInt global_bad = 0;
                    MPI_Allreduce(&local_bad, &global_bad, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
                    
                    double percent = 100.0 * (double)global_bad / (double)global_n_cells;
                    std::cout << "  Step " << step_num << " MOOD Rollback: " 
                              << global_bad << "/" << global_n_cells << " cells (" 
                              << percent << "%) marked." << std::endl;
                }

                // 4. Corrector Step
                transport->SetCellOrders(current_cell_orders);

                // Revert to U^n
                PetscCall(VecCopy(X_backup, X));
                PetscCall(TSSetSolution(ts, X));
                PetscCall(TSSetTime(ts, time)); 
                
                PetscCall(TSStep(ts));
            }

            step_num++;
            PetscCall(TSGetTime(ts, &time));

            // --- CHANGED: Update Auxiliary Variables & Compute Next Time Step ---
            // Calling PostStep(ts) here ensures:
            // 1. Aux variables (A) are updated based on the new X (needed for IO and next step).
            // 2. ComputeTimeStep() is called to calculate the new CFL-based dt.
            // 3. TSSetTimeStep is called to force the next step size (disabling RK adaptivity).
            PetscCall(PostStep(ts)); 
            // --------------------------------------------------------------------

            PetscCall(MonitorWrapper(ts, step_num, time, X, this));
        }

        if (rank == 0) std::cout << "[INFO] Finished." << std::endl;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
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

            // 1. Positivity Check
            if (q[1] < 0.0) {
                bad = true; 
            }

            // 2. Relaxed DMP Check
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