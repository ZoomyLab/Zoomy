#ifndef MOOD_SOLVER_HPP
#define MOOD_SOLVER_HPP

#include "ModularSolver.hpp"
#include <vector>
#include <algorithm>

class MOODSolver : public ModularSolver {
private:
    TS ts_low;    // Isolated TS for the 1st Order Predictor
    Vec X_backup; // Stores U^n
    Vec A_backup; // Stores Aux^n
    Vec X_low;    // Stores U^{n+1}_low (1st Order)

    bool is_predictor = false; // Prevents ts_low from advancing dt or IO
    
    std::vector<uint8_t> current_cell_orders; 
    struct CellBounds { PetscReal min_val; PetscReal max_val; };
    std::vector<std::vector<PetscInt>> neighbor_cache;

public:
    MOODSolver() : ModularSolver() { X_backup = NULL; X_low = NULL; A_backup = NULL; ts_low = NULL; }

    MOODSolver(std::shared_ptr<SolverStrategy> strat) : ModularSolver() {
        this->strategy = strat;
        X_backup = NULL; X_low = NULL; A_backup = NULL; ts_low = NULL;
    }

    ~MOODSolver() { 
        if(X_backup) VecDestroy(&X_backup); 
        if(A_backup) VecDestroy(&A_backup); 
        if(X_low) VecDestroy(&X_low);
        if(ts_low) TSDestroy(&ts_low);
    }

    // Intercept PostStep to safely handle Auxiliary variables for the predictor
    PetscErrorCode PostStep(TS ts_in) override {
        if (is_predictor) {
            // Predictor only updates Aux vars locally for the implicit source solver.
            // It MUST NOT calculate dt or write IO.
            Vec X_curr; TSGetSolution(ts_in, &X_curr);
            Vec X_loc, A_loc;
            DMGetLocalVector(dmQ, &X_loc); DMGetLocalVector(dmAux, &A_loc);
            
            DMGlobalToLocalBegin(dmQ, X_curr, INSERT_VALUES, X_loc); 
            DMGlobalToLocalEnd(dmQ, X_curr, INSERT_VALUES, X_loc);
            DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc); 
            DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc);
            
            transport->UpdateState(X_loc, A_loc);
            
            DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A); 
            DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A);
            
            DMRestoreLocalVector(dmQ, &X_loc); DMRestoreLocalVector(dmAux, &A_loc);
            return PETSC_SUCCESS;
        }
        return ModularSolver::PostStep(ts_in);
    }

    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(VirtualSolver::Initialize(argc, argv));
        
        // Restore Config Logic
        if (settings.solver.reconstruction_order == 2) SetReconstruction(LINEAR); 
        else SetReconstruction(PCM);

        if (settings.solver.reconstruction_order >= 2) {
            SetLimiters(false); 
        }

        PetscCall(InitializeComponents()); 

        if (X_backup) PetscCall(VecDestroy(&X_backup));
        PetscCall(VecDuplicate(X, &X_backup));
        if (A_backup) PetscCall(VecDestroy(&A_backup));
        PetscCall(VecDuplicate(A, &A_backup)); 
        if (X_low) PetscCall(VecDestroy(&X_low));
        PetscCall(VecDuplicate(X, &X_low));

        int n_dof = Model<Real>::n_dof_q;
        std::vector<std::string> names;
        if (n_dof >= 6) names = {"b", "h", "u", "v", "w", "p"};
        else if (n_dof == 4) names = {"b", "h", "hu", "hv"};
        else names = {"b", "h"}; 
        
        PetscCall(io->Setup3D(dmQ, n_dof, names));
        PetscCall(LoadInitialCondition());

        if (!strategy) strategy = std::make_shared<SplittingStrategy>();
        
        // Setup Main TS
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(RegisterCallbacks(ts)); 
        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
        
        // Setup Predictor TS (Isolated state)
        PetscCall(TSCreate(PETSC_COMM_WORLD, &ts_low));
        PetscCall(TSSetDM(ts_low, dmQ));
        TSAdapt adapt_low; PetscCall(TSGetAdapt(ts_low, &adapt_low)); 
        PetscCall(TSAdaptSetType(adapt_low, TSADAPTNONE));
        PetscCall(strategy->SetupTS(ts_low, this));

        PetscCall(TSSetMaxTime(ts_low, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts_low, TS_EXACTFINALTIME_MATCHSTEP));
        PetscCall(TSSetFromOptions(ts_low));
        
        PetscReal dt_start = ComputeTimeStep();
        dt_start = std::max(dt_start, settings.solver.min_dt);
        PetscCall(TSSetTimeStep(ts, dt_start));
        PetscCall(TSSetFromOptions(ts)); 

        // Cache Topology
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        current_cell_orders.assign(cEnd - cStart, 0);
        neighbor_cache.resize(cEnd - cStart);
        
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt num_adj = -1; PetscInt *adj = NULL; 
            PetscCall(DMPlexGetAdjacency(dmQ, c, &num_adj, &adj));
            for (int k = 0; k < num_adj; ++k) {
                if (adj[k] != c && adj[k] >= 0) neighbor_cache[c - cStart].push_back(adj[k]);
            }
            PetscCall(PetscFree(adj));
        }
        
        PetscInt local_n_cells = cEnd - cStart;
        PetscInt global_n_cells = 0;
        MPI_Allreduce(&local_n_cells, &global_n_cells, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

        if (rank == 0) std::cout << "[INFO] Starting Hybrid MOOD Solver (Dual-TS)..." << std::endl;

        PetscReal time;
        PetscCall(TSGetTime(ts, &time));
        PetscInt step_num = 0;
        PetscCall(MonitorWrapper(ts, 0, time, X, this));

        while (time < settings.solver.t_end) {
            PetscReal dt;
            PetscCall(TSGetTimeStep(ts, &dt));

            // ==========================================
            // 0. Backup State
            // ==========================================
            PetscCall(VecCopy(X, X_backup)); 
            PetscCall(VecCopy(A, A_backup)); 

            // ==========================================
            // 1. Compute 1st Order Predictor (ts_low)
            // ==========================================
            PetscCall(VecCopy(X, X_low)); 
            PetscCall(TSSetSolution(ts_low, X_low));
            PetscCall(TSSetTime(ts_low, time));
            PetscCall(TSSetTimeStep(ts_low, dt));
            PetscCall(TSSetStepNumber(ts_low, step_num));

            transport->SetCellOrders(std::vector<uint8_t>(local_n_cells, 1)); // Force PCM
            
            is_predictor = true;
            PetscCall(TSStep(ts_low)); 
            is_predictor = false;

            // Restore Aux variables so Main TS starts clean
            PetscCall(VecCopy(A_backup, A));

            // ==========================================
            // 2. Compute Main Step (ts)
            // ==========================================
            // Force configured spatial order (0 for Linear, 1 for PCM based on settings)
            uint8_t main_order = (settings.solver.reconstruction_order >= 2) ? 0 : 1;
            transport->SetCellOrders(std::vector<uint8_t>(local_n_cells, main_order)); 
            
            PetscCall(TSStep(ts)); 

            // ==========================================
            // 3. Detect & Replace
            // ==========================================
            bool found_trouble = DetectTroubledCells(X);
            
            if (found_trouble) {
                PetscCall(ApplyReplacement());
                if (rank == 0 && step_num % 10 == 0) {
                    PetscInt local_bad = 0;
                    for(auto val : current_cell_orders) if(val == 1) local_bad++;
                    PetscInt global_bad = 0;
                    MPI_Allreduce(&local_bad, &global_bad, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
                    double percent = 100.0 * (double)global_bad / (double)global_n_cells;
                    if(global_bad > 0) std::cout << "  [MOOD] " << global_bad << " cells (" << percent << "%) replaced with 1st order." << std::endl;
                }
            }

            // ==========================================
            // 4. Finalize Step
            // ==========================================
            PetscCall(TSGetTime(ts, &time));
            step_num++;
            PetscCall(PostStep(ts)); // Normal PostStep for high order X updates dt and writes IO
            PetscCall(MonitorWrapper(ts, step_num, time, X, this));
        }

        if (rank == 0) std::cout << "[INFO] Finished." << std::endl;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    PetscErrorCode ApplyReplacement() {
        PetscScalar *x_arr; const PetscScalar *low_arr;
        PetscCall(VecGetArray(X, &x_arr)); PetscCall(VecGetArrayRead(X_low, &low_arr));
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ));

        for (PetscInt c = cStart; c < cEnd; ++c) {
            if (current_cell_orders[c - cStart] == 1) {
                PetscInt off; PetscCall(PetscSectionGetOffset(sQ, c, &off));
                if (off >= 0) {
                    for (int i = 0; i < Model<Real>::n_dof_q; ++i) x_arr[off + i] = low_arr[off + i];
                }
            }
        }
        PetscCall(VecRestoreArray(X, &x_arr)); PetscCall(VecRestoreArrayRead(X_low, &low_arr));
        return PETSC_SUCCESS;
    }

    bool DetectTroubledCells(Vec U_high_glob) {
        Vec X_low_loc, X_high_loc;
        DMGetLocalVector(dmQ, &X_low_loc); DMGetLocalVector(dmQ, &X_high_loc);
        DMGlobalToLocalBegin(dmQ, X_low, INSERT_VALUES, X_low_loc); DMGlobalToLocalEnd(dmQ, X_low, INSERT_VALUES, X_low_loc);
        DMGlobalToLocalBegin(dmQ, U_high_glob, INSERT_VALUES, X_high_loc); DMGlobalToLocalEnd(dmQ, U_high_glob, INSERT_VALUES, X_high_loc);

        const PetscScalar *low_arr, *high_arr; 
        VecGetArrayRead(X_low_loc, &low_arr); VecGetArrayRead(X_high_loc, &high_arr);

        PetscInt cStart, cEnd; DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd);
        std::fill(current_cell_orders.begin(), current_cell_orders.end(), 0);

        std::vector<int> check_indices = {1}; 
        if (Model<Real>::n_dof_q > 2) check_indices.push_back(2); 
        if (Model<Real>::n_dof_q > 4) check_indices.push_back(4); 

        const Real eps_rel = 1e-4; const Real eps_abs = 1e-7;
        bool found_trouble = false;

        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar* q_high; DMPlexPointLocalRead(dmQ, c, high_arr, &q_high);
            const PetscScalar* q_low; DMPlexPointLocalRead(dmQ, c, low_arr, &q_low);
            bool bad = false;

            // Tolerate dry cells with negative numerical noise (prevents 90% replacement false positives)
            if (q_high[1] < 0.) bad = true; 

            if (!bad) {
                std::vector<PetscReal> min_b(Model<Real>::n_dof_q);
                std::vector<PetscReal> max_b(Model<Real>::n_dof_q);
                
                for(int idx : check_indices) { min_b[idx] = q_low[idx]; max_b[idx] = q_low[idx]; }

                const auto& adj = neighbor_cache[c - cStart];
                for (PetscInt n : adj) {
                    const PetscScalar* qn_low; DMPlexPointLocalRead(dmQ, n, low_arr, &qn_low);
                    if (!qn_low) continue; 
                    for(int idx : check_indices) {
                        if (qn_low[idx] < min_b[idx]) min_b[idx] = qn_low[idx];
                        if (qn_low[idx] > max_b[idx]) max_b[idx] = qn_low[idx];
                    }
                }

                for(int idx : check_indices) {
                    Real range = max_b[idx] - min_b[idx];
                    Real tol = std::max(eps_abs, range * eps_rel);
                    if (q_high[idx] < min_b[idx] - tol || q_high[idx] > max_b[idx] + tol) {
                        bad = true; break;
                    }
                }
            }

            if (bad) { current_cell_orders[c - cStart] = 1; found_trouble = true; }
        }

        VecRestoreArrayRead(X_low_loc, &low_arr); VecRestoreArrayRead(X_high_loc, &high_arr);
        DMRestoreLocalVector(dmQ, &X_low_loc); DMRestoreLocalVector(dmQ, &X_high_loc);
        return found_trouble;
    }
};
#endif