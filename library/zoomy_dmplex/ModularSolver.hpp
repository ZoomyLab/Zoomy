#ifndef MODULARSOLVER_HPP
#define MODULARSOLVER_HPP

#include <memory>
#include "VirtualSolver.hpp"
#include "TransportStep.hpp"
#include "SourceStep.hpp"
#include "Reconstruction.hpp" 
#include "Gradient.hpp"
#include "MshLoader.hpp" 

class ModularSolver : public VirtualSolver {
private:
    std::unique_ptr<TransportStep<Real>> transport;
    std::unique_ptr<SourceStep<Real>> source;
    
    bool config_implicit_source = false;
    int config_reconstruction_order = 1;
    GradientMethod config_grad_method = GREEN_GAUSS;
    FluxKernelPtr config_flux_kernel = nullptr; 
    NonConservativeFluxKernelPtr config_noncons_flux_kernel = nullptr;

public:
    ModularSolver() {}

    void SetReconstruction(ReconstructionType type) { config_reconstruction_order = (type == LINEAR ? 2 : 1); }
    void SetGradientMethod(GradientMethod method) { config_grad_method = method; }
    void SetFluxKernel(FluxKernelPtr k) { config_flux_kernel = k; }
    void SetNonConsFluxKernel(NonConservativeFluxKernelPtr k) { config_noncons_flux_kernel = k; }
    void SetImplicitSource(bool enable) { config_implicit_source = enable; }
    void SetExplicitSource(bool enable) { }
    void SetLimiter(bool enable) { }
    void SetLeastSquaresOrder(int order) { }

    PetscErrorCode InitializeComponents() {
        transport = std::make_unique<TransportStep<Real>>(dmQ, dmAux, dmGrad, parameters, boundary_map);
        if (config_flux_kernel) transport->SetFluxKernel(config_flux_kernel);
        if (config_noncons_flux_kernel) transport->SetNonConsFlux(config_noncons_flux_kernel);
        
        if (config_reconstruction_order == 2) {
            transport->SetReconstruction(std::make_shared<LinearReconstructor<Real>>());
            if (config_grad_method == GREEN_GAUSS) transport->SetGradient(std::make_shared<GreenGaussGradient<Real>>());
            else transport->SetGradient(std::make_shared<LeastSquaresGradient<Real>>(1));
        } else {
            transport->SetReconstruction(std::make_shared<PCMReconstructor<Real>>());
        }

        if (config_implicit_source) {
            source = std::make_unique<SourceStep<Real>>(dmQ, dmAux, parameters);
        }
        return PETSC_SUCCESS;
    }

    PetscErrorCode Run(int argc, char **argv) override {
        PetscFunctionBeginUser;
        PetscCall(VirtualSolver::Initialize(argc, argv)); 
        
        if (settings.solver.reconstruction_order == 2) SetReconstruction(LINEAR); 
        else SetReconstruction(PCM);

        PetscCall(InitializeComponents()); 

        std::vector<std::string> names = {"b", "h", "u", "v", "w", "p"};
        PetscCall(io->Setup3D(dmQ, 6, names)); 

        {
            Vec X_loc, A_loc;
            PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
            PetscCall(DMGetLocalVector(dmAux, &A_loc)); 
            
            PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
            PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
            
            PetscCall(UpdateState(X_loc, A_loc));
            
            std::string init_file = settings.io.initial_condition_file;
            if (!init_file.empty()) {
                if (init_file.find(".msh") != std::string::npos) {
                    if (rank == 0) std::cout << "[INFO] Detected .msh file. Using spatial interpolation loader." << std::endl;
                    
                    MshLoader loader;
                    loader.Load(init_file);

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
                        double x = cg->centroid[0];
                        double y = cg->centroid[1];

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

                        if (offQ >= 0 && (offQ + n_dof) <= loc_size_Q &&
                            offAux >= 0 && (offAux + n_dof_aux) <= loc_size_A) 
                        {
                            Model<Real>::update_aux_variables(&q_arr[offQ], &aux_arr[offAux], params_ptr);
                        }
                    }

                    PetscCall(VecRestoreArray(X_loc, &q_arr));
                    PetscCall(VecRestoreArray(A_loc, &aux_arr));
                    PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr));
                    
                    PetscCall(DMLocalToGlobalBegin(dmQ, X_loc, INSERT_VALUES, X)); 
                    PetscCall(DMLocalToGlobalEnd(dmQ, X_loc, INSERT_VALUES, X));
                }
                else {
                    PetscCall(DMLocalToGlobalBegin(dmQ, X_loc, INSERT_VALUES, X)); 
                    PetscCall(DMLocalToGlobalEnd(dmQ, X_loc, INSERT_VALUES, X));
                    
                    std::vector<PetscInt> mask;
                    for(int m : settings.io.initial_condition_mask) mask.push_back((PetscInt)m);
                    PetscCall(io->LoadSolution(X, dmQ, init_file, mask));
                    
                    PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc)); 
                    PetscCall(DMGlobalToLocalEnd(dmQ, X_loc, INSERT_VALUES, X_loc));

                    PetscCall(UpdateState(X_loc, A_loc));
                }
            }
            
            PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A)); PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));
            
            {
                PetscReal min_val[2], max_val[2];
                PetscCall(VecStrideMin(X, 0, NULL, &min_val[0]));
                PetscCall(VecStrideMax(X, 0, NULL, &max_val[0]));
                PetscCall(VecStrideMin(X, 1, NULL, &min_val[1]));
                PetscCall(VecStrideMax(X, 1, NULL, &max_val[1]));
            
            }

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
        if (config_implicit_source) {
            PetscCall(TSSetPostStep(ts, SplittingWrapper));
        } else {
            PetscCall(TSSetPostStep(ts, PostStepWrapper)); 
        }
        PetscCall(TSSetType(ts, TSSSP)); 
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

    static PetscErrorCode RHSWrapper(TS ts, PetscReal t, Vec X, Vec F, void* ctx) {
        return ((ModularSolver*)ctx)->transport->FormRHS(t, X, F);
    }

    static PetscErrorCode SplittingWrapper(TS ts) {
        void* ctx; TSGetApplicationContext(ts, &ctx);
        ModularSolver* solver = (ModularSolver*)ctx;
        Vec X_curr; TSGetSolution(ts, &X_curr);
        solver->CheckPositivity(X_curr);
        if (solver->source) { 
            PetscReal dt; TSGetTimeStep(ts, &dt); 
            PetscCall(solver->source->Solve(dt, X_curr, solver->A)); 
        }
        return solver->PostStep(ts);
    }

    PetscErrorCode CheckPositivity(Vec V) { return PETSC_SUCCESS; }
    PetscErrorCode UpdateState(Vec Q_loc, Vec Aux_loc) override { return transport->UpdateState(Q_loc, Aux_loc); }
};
#endif