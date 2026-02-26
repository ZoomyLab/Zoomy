#ifndef MODULARSOLVER_HPP
#define MODULARSOLVER_HPP

#include <memory>
#include "VirtualSolver.hpp"
#include "TransportStep.hpp"
#include "SourceStep.hpp"
#include "Reconstruction.hpp" 
#include "Gradient.hpp"
#include "MshLoader.hpp" 

class SolverStrategy;

class ModularSolver : public VirtualSolver {
public:
    std::unique_ptr<TransportStep<Real>> transport;
    std::unique_ptr<SourceStep<Real>> source_solver;
    
    std::shared_ptr<SolverStrategy> strategy;

    int config_reconstruction_order = 1;
    bool config_use_limiters = true; 
    
    GradientMethod config_grad_method = GREEN_GAUSS;
    FluxKernelPtr config_flux_kernel = nullptr; 
    NonConservativeFluxKernelPtr config_noncons_flux_kernel = nullptr;

public:
    ModularSolver() : VirtualSolver() {}
    virtual ~ModularSolver() = default;

    void SetStrategy(std::shared_ptr<SolverStrategy> s) { strategy = s; }
    void SetReconstruction(ReconstructionType type) { config_reconstruction_order = (type == LINEAR ? 2 : 1); }
    void SetLimiters(bool active) { config_use_limiters = active; } 
    void SetGradientMethod(GradientMethod method) { config_grad_method = method; }
    void SetFluxKernel(FluxKernelPtr k) { config_flux_kernel = k; }
    void SetNonConsFluxKernel(NonConservativeFluxKernelPtr k) { config_noncons_flux_kernel = k; }

    PetscErrorCode InitializeComponents() {
        transport = std::make_unique<TransportStep<Real>>(dmQ, dmAux, dmGrad, parameters, boundary_map);
        
        if (config_flux_kernel) transport->SetFluxKernel(config_flux_kernel);
        if (config_noncons_flux_kernel) transport->SetNonConsFlux(config_noncons_flux_kernel);
        
        if (config_reconstruction_order == 2) {
            transport->SetReconstruction(std::make_shared<LinearReconstructor<Real>>(Model<Real>::n_dof_q, config_use_limiters));
            auto grad = std::make_shared<GreenGaussGradient<Real>>();
            grad->SetBCFunction([](int idx, const Real* p, const Real* c, const Real* n, const Real* x, Real t, Real dx, Real* out) {
                auto res = Model<Real>::boundary_conditions(idx, p, c, n, x, t, dx);
                for(int i=0; i<Model<Real>::n_dof_q; ++i) out[i] = res[i];
            });
            transport->SetGradient(grad);
            transport->SetAuxReconstruction(std::make_shared<PCMReconstructor<Real>>(Model<Real>::n_dof_qaux));
        } else {
            transport->SetReconstruction(std::make_shared<PCMReconstructor<Real>>(Model<Real>::n_dof_q));
            transport->SetAuxReconstruction(std::make_shared<PCMReconstructor<Real>>(Model<Real>::n_dof_qaux));
        }

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
        PetscCall(DMGlobalToLocalBegin(dmQ, X_glob, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X_glob, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        PetscCall(transport->UpdateState(X_loc, A_loc));

        const PetscScalar *x_arr, *a_arr; PetscScalar *f_arr;
        PetscCall(VecGetArrayRead(X_loc, &x_arr)); PetscCall(VecGetArrayRead(A_loc, &a_arr)); PetscCall(VecGetArray(F_glob, &f_arr));

        PetscInt cStart, cEnd, rstart;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscCall(VecGetOwnershipRange(F_glob, &rstart, NULL));
        PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ));
        PetscSection sAux; PetscCall(DMGetLocalSection(dmAux, &sAux));
        PetscSection sGlob; PetscCall(DMGetGlobalSection(dmQ, &sGlob));
        const PetscReal* params_ptr = parameters.data();

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt offQ, offA, offGlob;
            PetscCall(PetscSectionGetOffset(sQ, c, &offQ)); PetscCall(PetscSectionGetOffset(sAux, c, &offA)); PetscCall(PetscSectionGetOffset(sGlob, c, &offGlob));
            if (offGlob >= 0) {
                PetscInt idx_glob = offGlob - rstart;
                // Check for dry cell to avoid singular source terms
                if (x_arr[offQ + 1] > 1e-6) {
                    auto S = Model<Real>::source(&x_arr[offQ], &a_arr[offA], params_ptr);
                    for (int i = 0; i < Model<Real>::n_dof_q; ++i) f_arr[idx_glob + i] += sign * S[i];
                }
            }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_arr)); PetscCall(VecRestoreArrayRead(A_loc, &a_arr)); PetscCall(VecRestoreArray(F_glob, &f_arr));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    void ComputeFaceJacobian_FD(
        const PetscScalar* qL, const PetscScalar* qR, 
        const PetscScalar* aL, const PetscScalar* aR,
        const PetscReal* params, const PetscScalar* n,
        PetscScalar* J_LL, PetscScalar* J_LR, PetscScalar* J_RL, PetscScalar* J_RR) 
    {
        const int n_dof = Model<Real>::n_dof_q;
        const int n_aux = Model<Real>::n_dof_qaux;
        const Real eps = 1e-7;

        auto F_base = config_flux_kernel(qL, qR, aL, aR, params, n);
        SimpleArray<Real, 2 * Model<Real>::n_dof_q> NC_base;
        bool use_nc = (config_noncons_flux_kernel != nullptr);
        if (use_nc) NC_base = config_noncons_flux_kernel(qL, qR, aL, aR, params, n);

        for (int j = 0; j < n_dof; ++j) {
            Real qL_p[n_dof]; for(int k=0; k<n_dof; ++k) qL_p[k] = qL[k];
            Real aL_p[n_aux]; for(int k=0; k<n_aux; ++k) aL_p[k] = aL[k];
            qL_p[j] += eps;
            if (n_aux > 0) {
                auto res_a = Model<Real>::update_aux_variables(qL_p, aL_p, params);
                for(int k=0; k<n_aux; ++k) aL_p[k] = res_a[k];
            }
            auto F_p = config_flux_kernel(qL_p, qR, aL_p, aR, params, n);
            for (int i = 0; i < n_dof; ++i) {
                Real dFlux = (F_p[i] - F_base[i]) / eps;
                J_LL[i*n_dof + j] += dFlux; J_RL[i*n_dof + j] -= dFlux;
                if (use_nc) {
                    auto NC_p = config_noncons_flux_kernel(qL_p, qR, aL_p, aR, params, n);
                    J_LL[i*n_dof + j] += (NC_p[n_dof + i] - NC_base[n_dof + i]) / eps; 
                    J_RL[i*n_dof + j] += (NC_p[i] - NC_base[i]) / eps; 
                }
            }
        }

        for (int j = 0; j < n_dof; ++j) {
            Real qR_p[n_dof]; for(int k=0; k<n_dof; ++k) qR_p[k] = qR[k];
            Real aR_p[n_aux]; for(int k=0; k<n_aux; ++k) aR_p[k] = aR[k];
            qR_p[j] += eps;
            if (n_aux > 0) {
                auto res_a = Model<Real>::update_aux_variables(qR_p, aR_p, params);
                for(int k=0; k<n_aux; ++k) aR_p[k] = res_a[k];
            }
            auto F_p = config_flux_kernel(qL, qR_p, aL, aR_p, params, n);
            for (int i = 0; i < n_dof; ++i) {
                Real dFlux = (F_p[i] - F_base[i]) / eps;
                J_LR[i*n_dof + j] += dFlux; J_RR[i*n_dof + j] -= dFlux;
                if (use_nc) {
                    auto NC_p = config_noncons_flux_kernel(qL, qR_p, aL, aR_p, params, n);
                    J_LR[i*n_dof + j] += (NC_p[n_dof + i] - NC_base[n_dof + i]) / eps; 
                    J_RR[i*n_dof + j] += (NC_p[i] - NC_base[i]) / eps; 
                }
            }
        }
    }

    void ComputeBoundaryJacobian_FD(
        const PetscScalar* qL, const PetscScalar* aL, 
        const PetscReal* params, const PetscScalar* n, 
        const PetscScalar* centroid, PetscReal time, int bc_idx,
        PetscScalar* J_LL) 
    {
        const int n_dof = Model<Real>::n_dof_q;
        const int n_aux = Model<Real>::n_dof_qaux;
        const Real eps = 1e-7;

        auto qR_base = Model<Real>::boundary_conditions(bc_idx, qL, aL, n, centroid, time, 0.0);
        PetscScalar aR_base[n_aux];
        if (n_aux > 0) {
             auto res = Model<Real>::update_aux_variables(qR_base.data, aL, params); 
             for(int i=0; i<n_aux; ++i) aR_base[i] = res[i];
        }
        
        auto F_base = config_flux_kernel(qL, qR_base.data, aL, aR_base, params, n);
        SimpleArray<Real, 2 * Model<Real>::n_dof_q> NC_base;
        bool use_nc = (config_noncons_flux_kernel != nullptr);
        if (use_nc) NC_base = config_noncons_flux_kernel(qL, qR_base.data, aL, aR_base, params, n);

        for (int j = 0; j < n_dof; ++j) {
            Real qL_p[n_dof]; for(int k=0; k<n_dof; ++k) qL_p[k] = qL[k];
            Real aL_p[n_aux]; for(int k=0; k<n_aux; ++k) aL_p[k] = aL[k];
            
            qL_p[j] += eps;
            if (n_aux > 0) {
                auto res_a = Model<Real>::update_aux_variables(qL_p, aL_p, params);
                for(int k=0; k<n_aux; ++k) aL_p[k] = res_a[k];
            }

            auto qR_p = Model<Real>::boundary_conditions(bc_idx, qL_p, aL_p, n, centroid, time, 0.0);
            PetscScalar aR_p[n_aux];
            if (n_aux > 0) {
                auto res_ar = Model<Real>::update_aux_variables(qR_p.data, aL_p, params);
                for(int k=0; k<n_aux; ++k) aR_p[k] = res_ar[k];
            }

            auto F_p = config_flux_kernel(qL_p, qR_p.data, aL_p, aR_p, params, n);
            
            for (int i = 0; i < n_dof; ++i) {
                Real dFlux = (F_p[i] - F_base[i]) / eps;
                J_LL[i*n_dof + j] += dFlux; 
                if (use_nc) {
                    auto NC_p = config_noncons_flux_kernel(qL_p, qR_p.data, aL_p, aR_p, params, n);
                    J_LL[i*n_dof + j] += (NC_p[n_dof + i] - NC_base[n_dof + i]) / eps; 
                }
            }
        }
    }

    PetscErrorCode FormImplicitJacobian(PetscReal t, Vec X_glob, PetscReal a, Mat P) {
        PetscFunctionBeginUser;
        PetscCall(MatZeroEntries(P));

        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_glob, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X_glob, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(transport->UpdateState(X_loc, A_loc));

        const PetscScalar *x_ptr, *a_ptr;
        PetscCall(VecGetArrayRead(X_loc, &x_ptr)); PetscCall(VecGetArrayRead(A_loc, &a_ptr));

        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscInt fStart, fEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 1, &fStart, &fEnd));
        PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ));
        PetscSection sAux; PetscCall(DMGetLocalSection(dmAux, &sAux));
        PetscSection sGlob; PetscCall(DMGetGlobalSection(dmQ, &sGlob));
        
        Vec faceGeom, cellGeom; 
        PetscCall(DMPlexGetGeometryFVM(dmQ, &faceGeom, &cellGeom, NULL));
        const PetscScalar *fGeom_ptr, *cGeom_ptr;
        PetscCall(VecGetArrayRead(faceGeom, &fGeom_ptr)); PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));
        DM dmFace; PetscCall(VecGetDM(faceGeom, &dmFace)); PetscSection secFace; PetscCall(DMGetLocalSection(dmFace, &secFace));
        DM dmCell; PetscCall(VecGetDM(cellGeom, &dmCell)); PetscSection secCell; PetscCall(DMGetLocalSection(dmCell, &secCell));
        DMLabel label; PetscCall(DMGetLabel(dmQ, "Face Sets", &label));

        const int n_dof = Model<Real>::n_dof_q; const int n_aux = Model<Real>::n_dof_qaux;
        const int dim = Model<Real>::dimension;
        const PetscReal* params_ptr = parameters.data();
        
        // A. Source Terms (Diagonal)
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt offQ, offA, offGlob;
            PetscCall(PetscSectionGetOffset(sQ, c, &offQ)); PetscCall(PetscSectionGetOffset(sAux, c, &offA)); PetscCall(PetscSectionGetOffset(sGlob, c, &offGlob));
            if (offGlob >= 0) {
                const PetscScalar* qc = &x_ptr[offQ];
                
                // --- RELAXATION FOR DRY CELLS ---
                // If h is small, friction term singularity makes Jacobian explode (Inf/NaN).
                // We detect this and set the block to Identity*a (Source = 0 assumption).
                if (qc[1] < 1e-4) {
                    PetscScalar J_block[n_dof * n_dof] = {0};
                    for(int i=0; i<n_dof; ++i) J_block[i*n_dof + i] = a; // Diag = a
                    PetscInt rows[n_dof]; for(int i=0; i<n_dof; ++i) rows[i] = offGlob + i;
                    PetscCall(MatSetValues(P, n_dof, rows, n_dof, rows, J_block, ADD_VALUES));
                } 
                else {
                    auto dS_dQ = Model<Real>::source_jacobian_wrt_variables(qc, &a_ptr[offA], params_ptr);
                    auto dS_dAux = Model<Real>::source_jacobian_wrt_aux_variables(qc, &a_ptr[offA], params_ptr);
                    auto dAux_dQ = Model<Real>::update_aux_variables_jacobian_wrt_variables(qc, &a_ptr[offA], params_ptr);
                    
                    PetscScalar J_block[n_dof * n_dof];
                    for(int i=0; i<n_dof; ++i) {
                        for(int j=0; j<n_dof; ++j) {
                            Real val = (i == j) ? a : 0.0;
                            Real dSource = dS_dQ[i*n_dof + j];
                            for(int k=0; k<n_aux; ++k) dSource += dS_dAux[i*n_aux + k] * dAux_dQ[k*n_dof + j];
                            J_block[i*n_dof + j] = val - dSource;
                        }
                    }
                    PetscInt rows[n_dof]; for(int i=0; i<n_dof; ++i) rows[i] = offGlob + i;
                    PetscCall(MatSetValues(P, n_dof, rows, n_dof, rows, J_block, ADD_VALUES));
                }
            }
        }

        // B. Transport Terms
        for (PetscInt f = fStart; f < fEnd; ++f) {
            PetscInt off; PetscCall(PetscSectionGetOffset(secFace, f, &off)); 
            const PetscFVFaceGeom *fg = (const PetscFVFaceGeom*)&fGeom_ptr[off];
            PetscScalar n_hat[3] = {0}; PetscReal area = 0; for(int d=0; d<dim; ++d) area += fg->normal[d]*fg->normal[d]; area = std::sqrt(area);
            if (area <= 1e-15) continue;
            for(int d=0; d<dim; ++d) n_hat[d] = fg->normal[d]/area;

            PetscInt num_cells; const PetscInt *cells; 
            PetscCall(DMPlexGetSupportSize(dmQ, f, &num_cells)); PetscCall(DMPlexGetSupport(dmQ, f, &cells));

            if (num_cells == 2) {
                PetscInt cL = cells[0]; PetscInt cR = cells[1];
                PetscInt offL_g, offR_g;
                PetscCall(PetscSectionGetOffset(sGlob, cL, &offL_g)); PetscCall(PetscSectionGetOffset(sGlob, cR, &offR_g));

                if (offL_g < 0 && offR_g < 0) continue;

                PetscInt offL, offR, offAL, offAR;
                PetscCall(PetscSectionGetOffset(sQ, cL, &offL)); PetscCall(PetscSectionGetOffset(sQ, cR, &offR));
                PetscCall(PetscSectionGetOffset(sAux, cL, &offAL)); PetscCall(PetscSectionGetOffset(sAux, cR, &offAR));

                const PetscScalar *qL = &x_ptr[offL]; const PetscScalar *qR = &x_ptr[offR];
                const PetscScalar *aL = &a_ptr[offAL]; const PetscScalar *aR = &a_ptr[offAR];

                PetscScalar J_LL[n_dof*n_dof] = {0}, J_LR[n_dof*n_dof] = {0};
                PetscScalar J_RL[n_dof*n_dof] = {0}, J_RR[n_dof*n_dof] = {0};

                ComputeFaceJacobian_FD(qL, qR, aL, aR, params_ptr, n_hat, J_LL, J_LR, J_RL, J_RR);

                PetscInt offCellL, offCellR;
                PetscCall(PetscSectionGetOffset(secCell, cL, &offCellL)); PetscCall(PetscSectionGetOffset(secCell, cR, &offCellR));
                const PetscFVCellGeom *cgL = (const PetscFVCellGeom*)&cGeom_ptr[offCellL];
                const PetscFVCellGeom *cgR = (const PetscFVCellGeom*)&cGeom_ptr[offCellR];
                
                Real factorL = area / cgL->volume; Real factorR = area / cgR->volume;
                for(int i=0; i<n_dof*n_dof; ++i) {
                    J_LL[i] *= factorL; J_LR[i] *= factorL;
                    J_RL[i] *= factorR; J_RR[i] *= factorR;
                }

                PetscInt rowsL[n_dof], rowsR[n_dof];
                bool hasL = (offL_g >= 0), hasR = (offR_g >= 0);
                if (hasL) for(int i=0; i<n_dof; ++i) rowsL[i] = offL_g + i;
                if (hasR) for(int i=0; i<n_dof; ++i) rowsR[i] = offR_g + i;

                if (hasL) {
                    PetscCall(MatSetValues(P, n_dof, rowsL, n_dof, rowsL, J_LL, ADD_VALUES));
                    if (hasR) PetscCall(MatSetValues(P, n_dof, rowsL, n_dof, rowsR, J_LR, ADD_VALUES));
                }
                if (hasR) {
                    if (hasL) PetscCall(MatSetValues(P, n_dof, rowsR, n_dof, rowsL, J_RL, ADD_VALUES));
                    PetscCall(MatSetValues(P, n_dof, rowsR, n_dof, rowsR, J_RR, ADD_VALUES));
                }
            } else if (num_cells == 1) {
                PetscInt tag_id; PetscCall(DMLabelGetValue(label, f, &tag_id));
                if (boundary_map.count(tag_id)) {
                    PetscInt bc_idx = boundary_map.at(tag_id);
                    PetscInt cL = cells[0];
                    PetscInt offL_g; PetscCall(PetscSectionGetOffset(sGlob, cL, &offL_g));
                    
                    if (offL_g >= 0) {
                        PetscInt offL, offAL;
                        PetscCall(PetscSectionGetOffset(sQ, cL, &offL)); 
                        PetscCall(PetscSectionGetOffset(sAux, cL, &offAL));
                        const PetscScalar *qL = &x_ptr[offL]; 
                        const PetscScalar *aL = &a_ptr[offAL];

                        PetscScalar J_LL[n_dof*n_dof] = {0};
                        ComputeBoundaryJacobian_FD(qL, aL, params_ptr, n_hat, fg->centroid, t, bc_idx, J_LL);

                        PetscInt offCellL; PetscCall(PetscSectionGetOffset(secCell, cL, &offCellL));
                        const PetscFVCellGeom *cgL = (const PetscFVCellGeom*)&cGeom_ptr[offCellL];
                        Real factorL = area / cgL->volume;
                        
                        for(int i=0; i<n_dof*n_dof; ++i) J_LL[i] *= factorL;

                        PetscInt rowsL[n_dof];
                        for(int i=0; i<n_dof; ++i) rowsL[i] = offL_g + i;
                        PetscCall(MatSetValues(P, n_dof, rowsL, n_dof, rowsL, J_LL, ADD_VALUES));
                    }
                }
            }
        }

        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr)); PetscCall(VecRestoreArrayRead(A_loc, &a_ptr));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(VecRestoreArrayRead(faceGeom, &fGeom_ptr)); PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr));
        
        PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY)); 
        PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    
    // KEEP Source-Only Jacobian for IMEX (compatibility)
    PetscErrorCode FormSourceJacobian(PetscReal t, Vec X_glob, PetscReal a, Mat P) {
        // [Existing Code matches what you uploaded, kept for compatibility]
        // Same as fully implicit but without Transport part
        PetscFunctionBeginUser;
        PetscCall(MatZeroEntries(P));
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_glob, INSERT_VALUES, X_loc)); PetscCall(DMGlobalToLocalEnd(dmQ, X_glob, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc)); PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(transport->UpdateState(X_loc, A_loc));
        const PetscScalar *x_arr, *a_arr;
        PetscCall(VecGetArrayRead(X_loc, &x_arr)); PetscCall(VecGetArrayRead(A_loc, &a_arr));
        PetscInt cStart, cEnd, rstart;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscCall(VecGetOwnershipRange(X_glob, &rstart, NULL));
        PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ));
        PetscSection sAux; PetscCall(DMGetLocalSection(dmAux, &sAux));
        PetscSection sGlob; PetscCall(DMGetGlobalSection(dmQ, &sGlob));
        const PetscReal* params_ptr = parameters.data();
        const int n_dof = Model<Real>::n_dof_q; const int n_aux = Model<Real>::n_dof_qaux;
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt offQ, offA, offGlob;
            PetscCall(PetscSectionGetOffset(sQ, c, &offQ)); PetscCall(PetscSectionGetOffset(sAux, c, &offA)); PetscCall(PetscSectionGetOffset(sGlob, c, &offGlob));
            if (offGlob >= 0) {
                // RELAXATION also here for consistency
                if (x_arr[offQ + 1] < 1e-4) {
                    PetscScalar J_block[n_dof * n_dof] = {0};
                    for(int i=0; i<n_dof; ++i) J_block[i*n_dof + i] = a; 
                    PetscInt rows[n_dof]; for(int i=0; i<n_dof; ++i) rows[i] = offGlob + i;
                    PetscCall(MatSetValues(P, n_dof, rows, n_dof, rows, J_block, ADD_VALUES));
                } else {
                    auto dS_dQ = Model<Real>::source_jacobian_wrt_variables(&x_arr[offQ], &a_arr[offA], params_ptr);
                    auto dS_dAux = Model<Real>::source_jacobian_wrt_aux_variables(&x_arr[offQ], &a_arr[offA], params_ptr);
                    auto dAux_dQ = Model<Real>::update_aux_variables_jacobian_wrt_variables(&x_arr[offQ], &a_arr[offA], params_ptr);
                    PetscScalar J_block[n_dof * n_dof];
                    for(int i=0; i<n_dof; ++i) {
                        for(int j=0; j<n_dof; ++j) {
                            Real val = (i == j) ? a : 0.0;
                            Real dSource = dS_dQ[i*n_dof + j];
                            for(int k=0; k<n_aux; ++k) dSource += dS_dAux[i*n_aux + k] * dAux_dQ[k*n_dof + j];
                            J_block[i*n_dof + j] = val - dSource; 
                        }
                    }
                    PetscInt rows[n_dof]; for(int i=0; i<n_dof; ++i) rows[i] = offGlob + i;
                    PetscCall(MatSetValues(P, n_dof, rows, n_dof, rows, J_block, ADD_VALUES));
                }
            }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_arr)); PetscCall(VecRestoreArrayRead(A_loc, &a_arr));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};
#endif