#ifndef MODULARSOLVER_HPP
#define MODULARSOLVER_HPP

#include "VirtualSolver.hpp"
#include "TransportStep.hpp"
#include "SourceStep.hpp"

class ModularSolver : public VirtualSolver {
private:
    std::unique_ptr<TransportStep<Real>> transport;
    std::unique_ptr<SourceStep<Real>> source;
    
    // Configuration Storage
    bool config_implicit_source = false;
    int config_reconstruction_order = 1;
    GradientMethod config_grad_method = GREEN_GAUSS;
    FluxKernelPtr config_flux_kernel = Numerics<Real>::numerical_flux;
    NonConservativeFluxKernelPtr config_noncons_flux_kernel = nullptr; // Default: None

public:
    ModularSolver() {}

    // --- Configuration Setters ---
    void SetReconstruction(ReconstructionType type) { config_reconstruction_order = (type == LINEAR ? 2 : 1); }
    void SetGradientMethod(GradientMethod method) { config_grad_method = method; }
    
    void SetFluxKernel(FluxKernelPtr k) { config_flux_kernel = k; }
    void SetNonConsFluxKernel(NonConservativeFluxKernelPtr k) { config_noncons_flux_kernel = k; }
    
    void SetImplicitSource(bool enable) { config_implicit_source = enable; }
    void SetExplicitSource(bool enable) { /* Unused placeholder */ }
    void SetLimiter(bool enable) { /* Implied by order 2 */ }
    void SetLeastSquaresOrder(int order) { /* Stored if needed */ }

    PetscErrorCode InitializeComponents() {
        transport = std::make_unique<TransportStep<Real>>(dmQ, dmAux, dmGrad, parameters, boundary_map);
        
        // Pass Kernels to Transport
        transport->SetFluxKernel(config_flux_kernel);
        transport->SetNonConsFlux(config_noncons_flux_kernel);
        
        // Configure Reconstruction / Gradient
        if (config_reconstruction_order == 2) {
            transport->SetReconstruction(std::make_shared<LinearReconstructor<Real>>());
            if (config_grad_method == GREEN_GAUSS) 
                transport->SetGradient(std::make_shared<GreenGaussGradient<Real>>());
            else 
                transport->SetGradient(std::make_shared<LeastSquaresGradient<Real>>(1));
        } else {
            transport->SetReconstruction(std::make_shared<PCMReconstructor<Real>>());
        }

        // Configure Source
        if (config_implicit_source) {
            source = std::make_unique<SourceStep<Real>>(dmQ, dmAux, parameters);
        }
        return PETSC_SUCCESS;
    }

    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(VirtualSolver::Initialize(argc, argv)); 
        
        // 1. Build Components
        PetscCall(InitializeComponents()); 

        // 2. Explicitly Initialize State + Aux
        {
            Vec X_loc, A_loc;
            PetscCall(DMGetLocalVector(dmQ, &X_loc)); PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
            PetscCall(DMGetLocalVector(dmAux, &A_loc)); PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
            
            PetscCall(UpdateState(X_loc, A_loc));
            
            PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A)); PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));
            PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        }

        PetscCall(TSSetApplicationContext(ts, this));
        PetscCall(RegisterCallbacks(ts)); 
        PetscCall(TSSetTime(ts, 0.0));
        PetscCall(TSSetMaxTime(ts, settings.solver.t_end));
        PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
        PetscCall(TSSetTimeStep(ts, ComputeTimeStep())); 
        PetscCall(TSSetFromOptions(ts)); 
        PetscCall(TSMonitorSet(ts, MonitorWrapper, this, NULL));
        
        if (rank == 0) std::cout << "[INFO] Starting Solver..." << std::endl;
        PetscCall(TSSolve(ts, X));
        if (rank == 0) std::cout << "[INFO] Finished." << std::endl;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

protected:
    PetscErrorCode RegisterCallbacks(TS ts) override {
        PetscCall(TSSetRHSFunction(ts, NULL, RHSWrapper, this));
        if (config_implicit_source) PetscCall(TSSetPostStep(ts, SplittingWrapper));
        else PetscCall(TSSetPostStep(ts, PostStepWrapper)); 
        PetscCall(TSSetType(ts, TSSSP)); 
        return PETSC_SUCCESS;
    }

    static PetscErrorCode RHSWrapper(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
        return ((ModularSolver*)ctx)->transport->FormRHS(t, X, F);
    }

    static PetscErrorCode SplittingWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        ModularSolver* solver = (ModularSolver*)ctx;
        Vec X_curr; TSGetSolution(ts, &X_curr);
        solver->CheckPositivity(X_curr);
        if (solver->source) { PetscReal dt; TSGetTimeStep(ts, &dt); solver->source->Solve(dt, X_curr); }
        return solver->PostStep(ts);
    }

    PetscErrorCode CheckPositivity(Vec V) {
        PetscScalar *x; PetscCall(VecGetArray(V, &x));
        PetscInt pStart, pEnd; PetscCall(DMPlexGetChart(dmQ, &pStart, &pEnd));
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        for (PetscInt c = pStart; c < pEnd; ++c) {
             PetscInt g_idx; PetscCall(DMPlexGetPointGlobal(dmQ, c, &g_idx, NULL));
             if (g_idx < 0) continue; 
             PetscScalar *q; PetscCall(DMPlexPointLocalRef(dmQ, c, x, &q));
             if (q && q[1] < 1e-8) { q[1] = 1e-8; q[2] = 0.0; q[3] = 0.0; }
        }
        PetscCall(VecRestoreArray(V, &x));
        return PETSC_SUCCESS;
    }

    PetscErrorCode UpdateState(Vec Q_loc, Vec Aux_loc) override {
        return transport->UpdateState(Q_loc, Aux_loc);
    }
};
#endif