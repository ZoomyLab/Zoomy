#ifndef MOOD_SOLVER_HPP
#define MOOD_SOLVER_HPP

#include "ModularSolver.hpp"
#include "SolverStrategies.hpp"

class MOODSolver : public ModularSolver {
private:
    Vec X_backup;
    Vec X_low; 

public:
    // MOODSolver can work with ANY strategy (IMEX, Splitting, etc.)
    // Pass the desired physics strategy in the constructor.
    MOODSolver(std::shared_ptr<SolverStrategy> strat) : ModularSolver() {
        this->SetStrategy(strat);
        X_backup = NULL; 
        X_low = NULL;
    }

    ~MOODSolver() {
        if(X_backup) VecDestroy(&X_backup);
        if(X_low) VecDestroy(&X_low);
    }

    // --- The MOOD Time Loop ---
    PetscErrorCode Run(int argc, char **argv) override {
        PetscCall(VirtualSolver::Initialize(argc, argv));
        
        // 1. Default to High Order (Linear Reconstruction)
        SetReconstruction(LINEAR); 
        PetscCall(InitializeComponents()); 

        // 2. Setup IO & Initial Condition
        std::vector<std::string> names = {"b", "h", "u", "v", "w", "p"};
        PetscCall(io->Setup3D(dmQ, 6, names));
        PetscCall(LoadInitialCondition());

        // 3. Register Strategy Callbacks
        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(RegisterCallbacks(ts)); 
        
        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
        
        // Use a safe starting timestep
        PetscReal dt_start = std::max(ComputeTimeStep(), settings.solver.min_dt);
        PetscCall(TSSetTimeStep(ts, dt_start));
        PetscCall(TSSetFromOptions(ts));
        
        PetscCall(VecDuplicate(X, &X_backup));

        if (rank == 0) std::cout << "[INFO] Starting MOOD Solver..." << std::endl;

        // 4. Manual Time Stepping Loop
        PetscReal t; 
        PetscCall(TSGetTime(ts, &t));
        
        while (t < settings.solver.t_end) {
            // A. Backup State
            PetscCall(VecCopy(X, X_backup));

            // B. Try High-Order Step (2nd Order)
            SetOrder(2); 
            PetscCall(TSStep(ts)); 

            // C. Check Validity
            bool valid = CheckAdmissibility(X); 

            // D. Fallback if invalid
            if (!valid) {
                if(rank == 0) std::cout << "  [MOOD] Cell failure detected at t=" << t << ". Rolling back to 1st order." << std::endl;
                
                // Rollback
                PetscCall(VecCopy(X_backup, X));
                PetscCall(TSSetSolution(ts, X)); 
                
                // Low-Order Step (1st Order - PCM)
                SetOrder(1);
                PetscCall(TSStep(ts));
                
                // Note: We accept the 1st order step globally here for simplicity.
                // A more advanced MOOD would blend locally.
            }
            
            // E. IO and Update Time
            PetscCall(TSGetTime(ts, &t));
            MonitorWrapper(ts, 0, t, X, this); 
        }

        if (rank == 0) std::cout << "[INFO] Finished." << std::endl;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

private:
    // Helper to switch transport reconstruction on the fly
    void SetOrder(int order) {
        if (order == 2) transport->SetReconstruction(std::make_shared<LinearReconstructor<Real>>());
        else transport->SetReconstruction(std::make_shared<PCMReconstructor<Real>>());
    }

    // The Admissibility Detector
    bool CheckAdmissibility(Vec U) {
        // Get Min H
        PetscReal min_h;
        // Assuming H is index 1. 
        // Note: VecStrideMin is global, so this detects failure anywhere in the mesh.
        PetscCall(VecStrideMin(U, 1, NULL, &min_h)); 
        
        // Failure if H < 0
        if (min_h < 0.0) return false;

        // Future: Add Froude number check or DMP check here
        return true; 
    }
};

#endif