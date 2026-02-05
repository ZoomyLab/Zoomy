#ifndef MODULARSOLVER_HPP
#define MODULARSOLVER_HPP

#include <memory>
#include "VirtualSolver.hpp"
#include "TransportStep.hpp"
#include "SourceStep.hpp"
#include "Reconstruction.hpp" 
#include "Gradient.hpp"
#include "MshLoader.hpp" 

// Forward Declaration of Strategy Interface
class SolverStrategy;

class ModularSolver : public VirtualSolver {
public: // Public so Strategies can access components
    std::unique_ptr<TransportStep<Real>> transport;
    std::unique_ptr<SourceStep<Real>> source_solver;
    
    // The Strategy (Logic)
    std::shared_ptr<SolverStrategy> strategy;

    // Configuration
    int config_reconstruction_order = 1;
    GradientMethod config_grad_method = GREEN_GAUSS;
    FluxKernelPtr config_flux_kernel = nullptr; 
    NonConservativeFluxKernelPtr config_noncons_flux_kernel = nullptr;

public:
    ModularSolver() : VirtualSolver() {}
    virtual ~ModularSolver() = default;

    // --- Configuration ---
    void SetStrategy(std::shared_ptr<SolverStrategy> s) { strategy = s; }
    void SetReconstruction(ReconstructionType type) { config_reconstruction_order = (type == LINEAR ? 2 : 1); }
    void SetGradientMethod(GradientMethod method) { config_grad_method = method; }
    void SetFluxKernel(FluxKernelPtr k) { config_flux_kernel = k; }
    void SetNonConsFluxKernel(NonConservativeFluxKernelPtr k) { config_noncons_flux_kernel = k; }

    // --- Component Initialization ---
    PetscErrorCode InitializeComponents() {
        // 1. Transport
        transport = std::make_unique<TransportStep<Real>>(dmQ, dmAux, dmGrad, parameters, boundary_map);
        if (config_flux_kernel) transport->SetFluxKernel(config_flux_kernel);
        if (config_noncons_flux_kernel) transport->SetNonConsFlux(config_noncons_flux_kernel);
        
        // 2. Reconstruction
        if (config_reconstruction_order == 2) {
            transport->SetReconstruction(std::make_shared<LinearReconstructor<Real>>());
            if (config_grad_method == GREEN_GAUSS) transport->SetGradient(std::make_shared<GreenGaussGradient<Real>>());
            else transport->SetGradient(std::make_shared<LeastSquaresGradient<Real>>(1));
        } else {
            transport->SetReconstruction(std::make_shared<PCMReconstructor<Real>>());
        }

        // 3. Source Solver (Lazy init: mostly used by Splitting strategy)
        source_solver = std::make_unique<SourceStep<Real>>(dmQ, dmAux, parameters);
        
        return PETSC_SUCCESS;
    }

    // --- Main Execution Loop ---
    PetscErrorCode Run(int argc, char **argv) override;

    // --- Helper: Update State (Proxy to Transport) ---
    PetscErrorCode UpdateState(Vec Q_loc, Vec Aux_loc) override { 
        return transport->UpdateState(Q_loc, Aux_loc); 
    }

    // --- Helper: Implicit Source Residual Calculation (Used by IMEX/Implicit Strategies) ---
    // F_glob += sign * Source(U)
    PetscErrorCode AddImplicitSourceToResidual(Vec X_glob, Vec F_glob, PetscReal sign) {
        PetscFunctionBeginUser;
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
        PetscCall(DMGetLocalVector(dmAux, &A_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, X_glob, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X_glob, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        // Ensure Aux is consistent with State
        PetscCall(transport->UpdateState(X_loc, A_loc));

        const PetscScalar *x_arr, *a_arr;
        PetscScalar *f_arr;
        PetscCall(VecGetArrayRead(X_loc, &x_arr));
        PetscCall(VecGetArrayRead(A_loc, &a_arr));
        PetscCall(VecGetArray(F_glob, &f_arr));

        PetscInt cStart, cEnd, rstart;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscCall(VecGetOwnershipRange(F_glob, &rstart, NULL));
        
        PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ));
        PetscSection sAux; PetscCall(DMGetLocalSection(dmAux, &sAux));
        PetscSection sGlob; PetscCall(DMGetGlobalSection(dmQ, &sGlob));

        const PetscReal* params_ptr = parameters.data();

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt offQ, offA, offGlob;
            PetscCall(PetscSectionGetOffset(sQ, c, &offQ));
            PetscCall(PetscSectionGetOffset(sAux, c, &offA));
            PetscCall(PetscSectionGetOffset(sGlob, c, &offGlob));

            if (offGlob >= 0) {
                PetscInt idx_glob = offGlob - rstart;
                // Evaluate Source: S(U)
                auto S = Model<Real>::source(&x_arr[offQ], &a_arr[offA], params_ptr);
                for (int i = 0; i < Model<Real>::n_dof_q; ++i) {
                    f_arr[idx_glob + i] += sign * S[i];
                }
            }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_arr));
        PetscCall(VecRestoreArrayRead(A_loc, &a_arr));
        PetscCall(VecRestoreArray(F_glob, &f_arr));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); 
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // --- Helper: Form Source Jacobian (Used by IMEX/Implicit Strategies) ---
    // P = a*I - dSource/dQ
    PetscErrorCode FormSourceJacobian(PetscReal t, Vec X_glob, PetscReal a, Mat P) {
        PetscFunctionBeginUser;
        PetscCall(MatZeroEntries(P));
        
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_glob, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X_glob, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        const PetscScalar *x_arr, *a_arr;
        PetscCall(VecGetArrayRead(X_loc, &x_arr));
        PetscCall(VecGetArrayRead(A_loc, &a_arr));

        PetscInt cStart, cEnd, rstart;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscCall(VecGetOwnershipRange(X_glob, &rstart, NULL));
        
        PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ));
        PetscSection sAux; PetscCall(DMGetLocalSection(dmAux, &sAux));
        PetscSection sGlob; PetscCall(DMGetGlobalSection(dmQ, &sGlob));

        const PetscReal* params_ptr = parameters.data();
        const int n_dof = Model<Real>::n_dof_q;
        const int n_aux = Model<Real>::n_dof_qaux;

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt offQ, offA, offGlob;
            PetscCall(PetscSectionGetOffset(sQ, c, &offQ));
            PetscCall(PetscSectionGetOffset(sAux, c, &offA));
            PetscCall(PetscSectionGetOffset(sGlob, c, &offGlob));

            if (offGlob >= 0) {
                PetscInt idx_glob = offGlob; 
                const PetscScalar* q = &x_arr[offQ];
                const PetscScalar* aux = &a_arr[offA];

                auto dS_dQ = Model<Real>::source_jacobian_wrt_variables(q, aux, params_ptr);
                auto dS_dAux = Model<Real>::source_jacobian_wrt_aux_variables(q, aux, params_ptr);
                auto dAux_dQ = Model<Real>::update_aux_variables_jacobian_wrt_variables(q, aux, params_ptr);

                PetscScalar J_block[n_dof * n_dof];
                for(int i=0; i<n_dof; ++i) {
                    for(int j=0; j<n_dof; ++j) {
                        Real val = (i == j) ? a : 0.0;
                        Real dSource = dS_dQ[i*n_dof + j];
                        // Chain rule: dS/dQ + dS/dAux * dAux/dQ
                        for(int k=0; k<n_aux; ++k) {
                             dSource += dS_dAux[i*n_aux + k] * dAux_dQ[k*n_dof + j];
                        }
                        J_block[i*n_dof + j] = val - dSource; 
                    }
                }
                PetscInt rows[n_dof];
                for(int i=0; i<n_dof; ++i) rows[i] = idx_glob + i;
                PetscCall(MatSetValues(P, n_dof, rows, n_dof, rows, J_block, ADD_VALUES));
            }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_arr));
        PetscCall(VecRestoreArrayRead(A_loc, &a_arr));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    
    // --- Load Helpers ---
    PetscErrorCode LoadInitialCondition() {
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
        PetscCall(DMGetLocalVector(dmAux, &A_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(UpdateState(X_loc, A_loc));

        // Use IO Manager to load file if present
        std::string init_file = settings.io.initial_condition_file;
        if (!init_file.empty()) {
             // (Assuming MshLoader logic from your original code goes here or reused)
             // For brevity, calling the IO load:
             std::vector<PetscInt> mask;
             for(int m : settings.io.initial_condition_mask) mask.push_back((PetscInt)m);
             PetscCall(io->LoadSolution(X, dmQ, init_file, mask));
        }

        PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A)); PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        return PETSC_SUCCESS;
    }

    // Helper static wrapper for IO Monitoring
    static PetscErrorCode MonitorWrapper(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
        ModularSolver *solver = (ModularSolver *)ctx;
        if (solver->io->ShouldWrite(time)) {
            if (solver->rank == 0) std::cout << "Writing snapshot at t=" << time << std::endl;
            PetscCall(solver->io->WriteVTK(solver->dmQ, X, time));
            if (solver->settings.io.write_3d) {
                solver->io->Write3D<Model<Real>>(time, X, solver->A, solver->dmQ, solver->dmAux, solver->parameters);
            }
            solver->io->AdvanceSnapshot();
        }
        if (solver->rank == 0 && step % 10 == 0) {
            PetscReal dt; TSGetTimeStep(ts, &dt);
            PetscPrintf(PETSC_COMM_WORLD, "Step %d Time %.4f dt %.4e\n", step, (double)time, (double)dt);
        }
        return PETSC_SUCCESS;
    }

protected:
    virtual PetscErrorCode RegisterCallbacks(TS ts);
};

// ============================================================================
// INCLUDE STRATEGIES AT THE END TO RESOLVE DEPENDENCIES
// ============================================================================
#include "SolverStrategies.hpp"

// Implementation of Run() is here because it needs SolverStrategy definition
inline PetscErrorCode ModularSolver::Run(int argc, char **argv) {
    PetscFunctionBeginUser;
    PetscCall(VirtualSolver::Initialize(argc, argv)); 
    
    if (settings.solver.reconstruction_order == 2) SetReconstruction(LINEAR); 
    else SetReconstruction(PCM);

    PetscCall(InitializeComponents()); 

    std::vector<std::string> names = {"b", "h", "u", "v", "w", "p"};
    PetscCall(io->Setup3D(dmQ, 6, names)); 
    PetscCall(LoadInitialCondition());

    // --- STRATEGY INJECTION ---
    if (!strategy) {
        // Default to Splitting if none set
        strategy = std::make_shared<SplittingStrategy>();
    }

    PetscCall(TSSetApplicationContext(ts, this));
    PetscCall(RegisterCallbacks(ts)); 
    
    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
    
    PetscReal dt_start = ComputeTimeStep();
    dt_start = std::max(dt_start, settings.solver.min_dt);
    PetscCall(TSSetTimeStep(ts, dt_start)); 
    PetscCall(TSSetFromOptions(ts)); 
    PetscCall(TSMonitorSet(ts, MonitorWrapper, this, NULL));
    
    if (rank == 0) std::cout << "[INFO] Starting ModularSolver..." << std::endl;
    PetscCall(TSSolve(ts, X));
    if (rank == 0) std::cout << "[INFO] Finished." << std::endl;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Implementation of RegisterCallbacks delegating to Strategy
inline PetscErrorCode ModularSolver::RegisterCallbacks(TS ts) {
    if (!strategy) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "No SolverStrategy set!");
    return strategy->SetupTS(ts, this);
}

#endif