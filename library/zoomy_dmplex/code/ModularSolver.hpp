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
public:
    std::unique_ptr<TransportStep<Real>> transport;
    std::unique_ptr<SourceStep<Real>> source_solver;
    
    std::shared_ptr<SolverStrategy> strategy;

    int config_reconstruction_order = 1;
    GradientMethod config_grad_method = GREEN_GAUSS;
    FluxKernelPtr config_flux_kernel = nullptr; 
    NonConservativeFluxKernelPtr config_noncons_flux_kernel = nullptr;

public:
    ModularSolver() : VirtualSolver() {}
    virtual ~ModularSolver() = default;

    void SetStrategy(std::shared_ptr<SolverStrategy> s) { strategy = s; }
    void SetReconstruction(ReconstructionType type) { config_reconstruction_order = (type == LINEAR ? 2 : 1); }
    void SetGradientMethod(GradientMethod method) { config_grad_method = method; }
    void SetFluxKernel(FluxKernelPtr k) { config_flux_kernel = k; }
    void SetNonConsFluxKernel(NonConservativeFluxKernelPtr k) { config_noncons_flux_kernel = k; }

    // --- Component Initialization ---
    PetscErrorCode InitializeComponents() {
        // 1. Create Transport Step (Now including dmGradAux)
        transport = std::make_unique<TransportStep<Real>>(dmQ, dmAux, dmGrad, dmGradAux, parameters, boundary_map);
        
        if (config_flux_kernel) transport->SetFluxKernel(config_flux_kernel);
        if (config_noncons_flux_kernel) transport->SetNonConsFlux(config_noncons_flux_kernel);
        
        // 2. Configure Reconstruction & Gradients
        if (config_reconstruction_order == 2) {
            // A. State Reconstruction
            transport->SetReconstruction(std::make_shared<LinearReconstructor<Real>>());
            auto grad = std::make_shared<GreenGaussGradient<Real>>();
            grad->SetBCFunction(Model<Real>::boundary_conditions); 
            transport->SetGradient(grad);

            // B. Auxiliary Reconstruction (Matches State Order)
            transport->SetAuxReconstruction(std::make_shared<LinearReconstructor<Real>>());
            auto gradAux = std::make_shared<GreenGaussGradient<Real>>();
            gradAux->SetBCFunction(Model<Real>::aux_boundary_conditions); 
            transport->SetAuxGradient(gradAux);

        } else {
            // 1st Order (PCM)
            transport->SetReconstruction(std::make_shared<PCMReconstructor<Real>>());
            transport->SetAuxReconstruction(std::make_shared<PCMReconstructor<Real>>());
        }

        // 3. Create Source Solver
        source_solver = std::make_unique<SourceStep<Real>>(dmQ, dmAux, parameters);
        return PETSC_SUCCESS;
    }

    PetscErrorCode Run(int argc, char **argv) override;

    PetscErrorCode UpdateState(Vec Q_loc, Vec Aux_loc) override { 
        return transport->UpdateState(Q_loc, Aux_loc); 
    }

    PetscErrorCode AddImplicitSourceToResidual(Vec X_glob, Vec F_glob, PetscReal sign) {
        PetscFunctionBeginUser;
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
        PetscCall(DMGetLocalVector(dmAux, &A_loc)); 
        
        PetscCall(DMGlobalToLocalBegin(dmQ, X_glob, INSERT_VALUES, X_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, X_glob, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        // Ensure state is consistent for Source calculation
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
        
        // Note: UpdateState likely called in Residual eval before Jacobian, but good practice if needed.
        // But for pure Jacobian assembly we assume X is state.

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
        
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        
        PetscCall(UpdateState(X_loc, A_loc));
        
        std::string init_file = settings.io.initial_condition_file;
        if (!init_file.empty()) {
            if (init_file.find(".msh") != std::string::npos) {
                if (rank == 0) std::cout << "[INFO] Detected .msh file. Using spatial interpolation loader." << std::endl;
                MshLoader loader; loader.Load(init_file);
                
                PetscScalar *q_arr, *aux_arr;
                PetscCall(VecGetArray(X_loc, &q_arr)); 
                PetscCall(VecGetArray(A_loc, &aux_arr));
                
                PetscInt loc_size_Q, loc_size_A;
                PetscCall(VecGetLocalSize(X_loc, &loc_size_Q)); 
                PetscCall(VecGetLocalSize(A_loc, &loc_size_A));
                
                PetscInt cStart, cEnd; 
                PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
                Vec cellGeom; 
                PetscCall(DMPlexGetGeometryFVM(dmQ, NULL, &cellGeom, NULL));
                const PetscScalar *cGeom_ptr; 
                PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));
                
                DM dmCell; PetscCall(VecGetDM(cellGeom, &dmCell)); 
                PetscSection secCell; PetscCall(DMGetLocalSection(dmCell, &secCell));
                PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ)); 
                PetscSection sAux; PetscCall(DMGetLocalSection(dmAux, &sAux));
                
                const int n_dof = Model<Real>::n_dof_q; 
                const int n_dof_aux = Model<Real>::n_dof_qaux;
                const PetscReal* params_ptr = parameters.data();

                for(PetscInt c=cStart; c<cEnd; ++c) {
                    PetscInt offG; PetscCall(PetscSectionGetOffset(secCell, c, &offG));
                    const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cGeom_ptr[offG];
                    double x = cg->centroid[0]; double y = cg->centroid[1];
                    double B = loader.Interpolate("B", x, y); 
                    double H = loader.Interpolate("H", x, y); 
                    double U = loader.Interpolate("U", x, y); 
                    double V = loader.Interpolate("V", x, y);
                    
                    PetscInt offQ; PetscCall(PetscSectionGetOffset(sQ, c, &offQ)); 
                    PetscInt offAux; PetscCall(PetscSectionGetOffset(sAux, c, &offAux));
                    
                    if (offQ >= 0 && (offQ + n_dof) <= loc_size_Q) {
                        if (0 < n_dof) q_arr[offQ + 0] = B; 
                        if (1 < n_dof) q_arr[offQ + 1] = H; 
                        if (2 < n_dof) q_arr[offQ + 2] = H * U;
                        if (n_dof == 4) { 
                            if (3 < n_dof) q_arr[offQ + 3] = H * V; 
                        } else { 
                            if (4 < n_dof) q_arr[offQ + 4] = H * V; 
                        }
                    }
                    // Update Aux from interpolated State
                    if (offQ >= 0 && (offQ + n_dof) <= loc_size_Q && offAux >= 0 && (offAux + n_dof_aux) <= loc_size_A) {
                        Model<Real>::update_aux_variables(&q_arr[offQ], &aux_arr[offAux], params_ptr);
                    }
                }
                PetscCall(VecRestoreArray(X_loc, &q_arr)); 
                PetscCall(VecRestoreArray(A_loc, &aux_arr)); 
                PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr));
                PetscCall(DMLocalToGlobalBegin(dmQ, X_loc, INSERT_VALUES, X)); 
                PetscCall(DMLocalToGlobalEnd(dmQ, X_loc, INSERT_VALUES, X));
            } else {
                std::vector<PetscInt> mask; 
                for(int m : settings.io.initial_condition_mask) mask.push_back((PetscInt)m);
                PetscCall(io->LoadSolution(X, dmQ, init_file, mask));
                PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc)); 
                PetscCall(DMGlobalToLocalEnd(dmQ, X_loc, INSERT_VALUES, X_loc));
                PetscCall(UpdateState(X_loc, A_loc));
            }
        }
        PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A)); 
        PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); 
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        return PETSC_SUCCESS;
    }

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

#include "SolverStrategies.hpp"

inline PetscErrorCode ModularSolver::Run(int argc, char **argv) {
    PetscFunctionBeginUser;
    PetscCall(VirtualSolver::Initialize(argc, argv)); 
    
    // Set Reconstruction Type from Settings
    if (settings.solver.reconstruction_order == 2) SetReconstruction(LINEAR); 
    else SetReconstruction(PCM);
    
    PetscCall(InitializeComponents()); 
    
    // Detect Output Fields (Dynamic check)
    int n_dof = Model<Real>::n_dof_q;
    std::vector<std::string> names;
    if (n_dof >= 6) names = {"b", "h", "u", "v", "w", "p"};
    else if (n_dof == 4) names = {"b", "h", "hu", "hv"};
    else names = {"b", "h"};

    PetscCall(io->Setup3D(dmQ, n_dof, names)); 
    PetscCall(LoadInitialCondition());
    
    if (!strategy) { strategy = std::make_shared<SplittingStrategy>(); }
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

inline PetscErrorCode ModularSolver::RegisterCallbacks(TS ts) {
    if (!strategy) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "No SolverStrategy set!");
    return strategy->SetupTS(ts, this);
}
#endif