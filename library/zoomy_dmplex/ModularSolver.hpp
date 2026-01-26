#ifndef MODULARSOLVER_HPP
#define MODULARSOLVER_HPP

#include "VirtualSolver.hpp"
#include <functional>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

// --- Definitions for GPU-Ready Building Blocks ---

using FluxKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_q> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*, const PetscScalar*, 
    const PetscScalar*, const PetscScalar*);

using SourceKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_q> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*);

using JacobianKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_q * Model<Real>::n_dof_q> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*);
using JacobianAuxKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_q * Model<Real>::n_dof_qaux> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*);
using JacobianAuxQKernelPtr = SimpleArray<PetscScalar, Model<Real>::n_dof_qaux * Model<Real>::n_dof_q> (*)(
    const PetscScalar*, const PetscScalar*, const PetscScalar*);

enum ReconstructionType { PCM, LINEAR, MOOD };

struct ReconstructionStrategy {
    ReconstructionType type = PCM;
    void Reconstruct(const PetscScalar* qC, const PetscScalar* qN, const PetscScalar* gradC, const PetscScalar* r_vec,
                     PetscScalar* q_face_L, PetscScalar* q_face_R) const {
        for(int i=0; i<Model<Real>::n_dof_q; ++i) {
            q_face_L[i] = qC[i];
            q_face_R[i] = qN[i];
        }
    }
};

class ModularSolver : public VirtualSolver {
protected:
    FluxKernelPtr flux_kernel;
    SourceKernelPtr source_kernel;
    
    JacobianKernelPtr jac_q;
    JacobianAuxKernelPtr jac_aux;
    JacobianAuxQKernelPtr jac_aux_q;

    ReconstructionStrategy reconstructor;
    
    bool has_explicit_source = false;
    bool has_implicit_source = false;

public:
    ModularSolver() {
        flux_kernel = Numerics<Real>::numerical_flux;
        source_kernel = Model<Real>::source;
        jac_q = Model<Real>::source_jacobian_wrt_variables;
        jac_aux = Model<Real>::source_jacobian_wrt_aux_variables;
        jac_aux_q = Model<Real>::update_aux_variables_jacobian_wrt_variables;
    }

    void SetFluxKernel(FluxKernelPtr k) { flux_kernel = k; }
    void SetExplicitSource(bool enable) { has_explicit_source = enable; }
    void SetImplicitSource(bool enable) { has_implicit_source = enable; }
    void SetReconstruction(ReconstructionType type) { reconstructor.type = type; }

    PetscErrorCode RegisterCallbacks(TS ts) override {
        // Explicit Part
        PetscCall(TSSetRHSFunction(ts, NULL, FormRHSFunctionWrapper, this));
        
        // Implicit Part
        if (has_implicit_source) {
            PetscCall(TSSetIFunction(ts, NULL, FormIFunctionWrapper, this));
            PetscCall(TSSetIJacobian(ts, NULL, NULL, FormIJacobianWrapper, this));
            PetscCall(TSSetType(ts, TSARKIMEX)); 
            
            // To see convergence, run with: ./solver_cpu -snes_monitor
        } else {
            PetscCall(TSSetType(ts, TSEULER));
        }
        return PETSC_SUCCESS;
    }

    // --- Helper to Update Local Aux (With Domain Checks) ---
    PetscErrorCode UpdateLocalAux(PetscScalar* x_ptr, PetscScalar* a_ptr, PetscInt cStart, PetscInt cEnd) {
        for(PetscInt c=cStart; c<cEnd; ++c) {
            PetscScalar *q, *a;
            DMPlexPointLocalRef(dmQ, c, x_ptr, &q);
            DMPlexPointLocalRef(dmAux, c, a_ptr, &a);
            
            if (q && a) {
                // Domain Check: Negative Depth
                if (q[1] < 0.0) {
                    return PETSC_ERR_ARG_OUTOFRANGE; 
                }
                auto res_a = Model<Real>::update_aux_variables(q, a, parameters.data());
                for(int i=0; i<Model<Real>::n_dof_qaux; ++i) a[i] = res_a[i];
            }
        }
        return PETSC_SUCCESS;
    }

    // --- 1. Explicit RHS ---
    PetscErrorCode FormRHSFunction(PetscReal time, Vec X_global, Vec F_global) {
        Vec X_loc, A_loc, cellGeom, faceGeom;
        
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(VecZeroEntries(F_global)); 

        const PetscScalar *x_ptr, *a_ptr; 
        PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscCall(VecGetArrayRead(A_loc, &a_ptr));

        PetscCall(DMPlexGetGeometryFVM(dmMesh, &faceGeom, &cellGeom, NULL));
        const PetscScalar *cGeom_ptr, *fGeom_ptr;
        PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));
        PetscCall(VecGetArrayRead(faceGeom, &fGeom_ptr));
        
        Vec F_loc;
        PetscCall(DMGetLocalVector(dmQ, &F_loc));
        PetscCall(VecZeroEntries(F_loc));
        PetscScalar *f_ptr;
        PetscCall(VecGetArray(F_loc, &f_ptr));

        DM dmFace; PetscCall(VecGetDM(faceGeom, &dmFace));
        PetscSection secFace; PetscCall(DMGetLocalSection(dmFace, &secFace));
        PetscCall(ComputeFaceFluxes(time, x_ptr, a_ptr, f_ptr, fGeom_ptr, secFace));

        DM dmCell; PetscCall(VecGetDM(cellGeom, &dmCell));
        PetscSection secCell; PetscCall(DMGetLocalSection(dmCell, &secCell));
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
             PetscInt off; PetscCall(PetscSectionGetOffset(secCell, c, &off));
             if (off < 0) continue; 
             const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cGeom_ptr[off];
             PetscScalar *f_cell;
             PetscCall(DMPlexPointLocalRef(dmQ, c, f_ptr, &f_cell));
             if (f_cell && cg->volume > 1e-15) { 
                 for(int i=0; i<Model<Real>::n_dof_q; ++i) f_cell[i] /= cg->volume; 
             }
        }

        if (has_explicit_source) {
            PetscCall(ComputeCellSources(x_ptr, a_ptr, f_ptr, 1.0));
        }

        PetscCall(VecRestoreArray(F_loc, &f_ptr));
        PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr));
        PetscCall(VecRestoreArrayRead(faceGeom, &fGeom_ptr));
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr));
        PetscCall(VecRestoreArrayRead(A_loc, &a_ptr));

        PetscCall(DMLocalToGlobalBegin(dmQ, F_loc, ADD_VALUES, F_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, F_loc, ADD_VALUES, F_global));

        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(DMRestoreLocalVector(dmQ, &F_loc));
        return PETSC_SUCCESS;
    }

    // --- 2. Implicit IFunction ---
    PetscErrorCode FormIFunction(PetscReal time, Vec X, Vec X_t, Vec F) {
        PetscCall(VecCopy(X_t, F)); 

        if (has_implicit_source) {
            Vec X_loc, A_loc, S_loc;
            PetscCall(DMGetLocalVector(dmQ, &X_loc));
            PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc));
            PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
            
            // Get AUX (Global values as base)
            PetscCall(DMGetLocalVector(dmAux, &A_loc));
            PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
            PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
            
            PetscCall(DMGetLocalVector(dmQ, &S_loc));
            PetscCall(VecZeroEntries(S_loc));

            PetscScalar *x_ptr, *a_ptr, *s_ptr;
            PetscCall(VecGetArray(X_loc, &x_ptr)); 
            PetscCall(VecGetArray(A_loc, &a_ptr)); 
            PetscCall(VecGetArray(S_loc, &s_ptr));

            PetscInt cStart, cEnd;
            PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));

            // CRITICAL: Check Domain Validity (h < 0?)
            PetscErrorCode ierr = UpdateLocalAux(x_ptr, a_ptr, cStart, cEnd);
            if (ierr == PETSC_ERR_ARG_OUTOFRANGE) {
                SNES snes; TSGetSNES(ts, &snes);
                SNESSetFunctionDomainError(snes);
                VecRestoreArray(X_loc, &x_ptr); VecRestoreArray(A_loc, &a_ptr); VecRestoreArray(S_loc, &s_ptr);
                DMRestoreLocalVector(dmQ, &X_loc); DMRestoreLocalVector(dmAux, &A_loc); DMRestoreLocalVector(dmQ, &S_loc);
                return PETSC_SUCCESS; 
            }

            PetscCall(ComputeCellSources(x_ptr, a_ptr, s_ptr, 1.0));

            PetscCall(VecRestoreArray(X_loc, &x_ptr));
            PetscCall(VecRestoreArray(A_loc, &a_ptr));
            PetscCall(VecRestoreArray(S_loc, &s_ptr));

            Vec S_global;
            PetscCall(VecDuplicate(F, &S_global));
            PetscCall(VecZeroEntries(S_global));
            PetscCall(DMLocalToGlobalBegin(dmQ, S_loc, ADD_VALUES, S_global));
            PetscCall(DMLocalToGlobalEnd(dmQ, S_loc, ADD_VALUES, S_global));
            
            // F = X_t - S
            PetscCall(VecAXPY(F, -1.0, S_global));

            // Debug Print (Manual Monitor)
            static int call_count = 0;
            if (this->rank == 0 && call_count % 500 == 0) { 
                PetscReal fnorm; VecNorm(F, NORM_2, &fnorm);
                if (std::isnan(fnorm) || std::isinf(fnorm)) {
                    std::cerr << "[ERROR] FormIFunction: Residual is NaN/Inf!" << std::endl;
                    SNES snes; TSGetSNES(ts, &snes);
                    SNESSetFunctionDomainError(snes);
                } else if (fnorm > 1e15) {
                     std::cout << "[DEBUG] High Residual: " << fnorm << std::endl;
                }
            }
            call_count++;

            PetscCall(VecDestroy(&S_global));
            PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
            PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
            PetscCall(DMRestoreLocalVector(dmQ, &S_loc));
        }
        return PETSC_SUCCESS;
    }

    // --- 3. Implicit Jacobian ---
    PetscErrorCode FormIJacobian(PetscReal time, Vec X, Vec X_t, PetscReal shift, Mat J, Mat Jpre) {
        if (!has_implicit_source) {
            PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
            return PETSC_SUCCESS;
        }

        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc)); // CORRECTED: X, not X_global
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        PetscScalar *x_ptr, *a_ptr;
        PetscCall(VecGetArray(X_loc, &x_ptr));
        PetscCall(VecGetArray(A_loc, &a_ptr));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        UpdateLocalAux(x_ptr, a_ptr, cStart, cEnd);
        
        PetscSection section;
        PetscCall(DMGetGlobalSection(dmQ, &section));

        constexpr int N = Model<Real>::n_dof_q;
        constexpr int N_aux = Model<Real>::n_dof_qaux;
        
        std::vector<PetscScalar> values(N * N);
        std::vector<PetscInt> indices(N);

        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar *q, *aux;
            DMPlexPointLocalRead(dmQ, c, x_ptr, &q);
            DMPlexPointLocalRead(dmAux, c, a_ptr, &aux);
            
            if (q && aux) {
                auto J_Q = jac_q(q, aux, parameters.data());       
                auto J_Aux = jac_aux(q, aux, parameters.data());   
                auto J_Aux_Q = jac_aux_q(q, aux, parameters.data()); 

                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        double chain_term = 0.0;
                        for (int k = 0; k < N_aux; ++k) chain_term += J_Aux[i * N_aux + k] * J_Aux_Q[k * N + j];
                        double dS_dQ_total = J_Q[i * N + j] + chain_term;
                        
                        if (std::isnan(dS_dQ_total)) dS_dQ_total = 0.0; 

                        values[i * N + j] = (i == j ? shift : 0.0) - dS_dQ_total;
                    }
                }

                PetscInt goff;
                PetscCall(PetscSectionGetOffset(section, c, &goff));
                if (goff >= 0) { 
                    for(int i=0; i<N; ++i) indices[i] = goff + i;
                    PetscCall(MatSetValues(Jpre, N, indices.data(), N, indices.data(), values.data(), INSERT_VALUES));
                }
            }
        }

        PetscCall(VecRestoreArray(X_loc, &x_ptr));
        PetscCall(VecRestoreArray(A_loc, &a_ptr));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));

        PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
        if (J != Jpre) {
            PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
        }
        return PETSC_SUCCESS; 
    }

    PetscErrorCode UpdateState(Vec Q_loc, Vec Aux_loc) override {
        PetscScalar *q_ptr, *a_ptr;
        PetscCall(VecGetArray(Q_loc, &q_ptr));
        PetscCall(VecGetArray(Aux_loc, &a_ptr));
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd)); 
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
            PetscScalar *q, *a;
            DMPlexPointLocalRef(dmQ, c, q_ptr, &q);
            DMPlexPointLocalRef(dmAux, c, a_ptr, &a);
            if (q && a) {
                auto res = Model<Real>::update_variables(q, a, parameters.data());
                for(int i=0; i<Model<Real>::n_dof_q; ++i) q[i] = res[i];
                auto res_a = Model<Real>::update_aux_variables(q, a, parameters.data());
                for(int i=0; i<Model<Real>::n_dof_qaux; ++i) a[i] = res_a[i];
            }
        }

        PetscCall(VecRestoreArray(Q_loc, &q_ptr));
        PetscCall(VecRestoreArray(Aux_loc, &a_ptr));
        return PETSC_SUCCESS;
    }

protected:
    PetscErrorCode ComputeFaceFluxes(PetscReal time, const PetscScalar* x, const PetscScalar* a, PetscScalar* f_loc, 
                                     const PetscScalar* fGeom_ptr, PetscSection secFace) {
        PetscInt fStart, fEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 1, &fStart, &fEnd));
        PetscInt dim = Model<Real>::dimension;
        DMLabel label; PetscCall(DMGetLabel(dmQ, "Face Sets", &label));

        for (PetscInt f = fStart; f < fEnd; ++f) {
            PetscInt off; PetscCall(PetscSectionGetOffset(secFace, f, &off));
            if (off < 0) continue;
            const PetscFVFaceGeom *fg = (const PetscFVFaceGeom*)&fGeom_ptr[off];
            PetscScalar n_hat[3] = {0}; PetscReal area = 0;
            for(int d=0; d<dim; ++d) area += fg->normal[d]*fg->normal[d];
            area = std::sqrt(area);
            if(area <= 1e-15) continue;
            for(int d=0; d<dim; ++d) n_hat[d] = fg->normal[d] / area;

            const PetscInt *cells; PetscInt num_cells;
            PetscCall(DMPlexGetSupportSize(dmQ, f, &num_cells));
            PetscCall(DMPlexGetSupport(dmQ, f, &cells));

            if (num_cells == 2) {
                const PetscScalar *qL_cell, *qR_cell;
                PetscCall(DMPlexPointLocalRead(dmQ, cells[0], x, &qL_cell));
                PetscCall(DMPlexPointLocalRead(dmQ, cells[1], x, &qR_cell));
                
                PetscScalar qL_face[Model<Real>::n_dof_q], qR_face[Model<Real>::n_dof_q];
                reconstructor.Reconstruct(qL_cell, qR_cell, NULL, NULL, qL_face, qR_face);

                const PetscScalar *aL, *aR;
                PetscCall(DMPlexPointLocalRead(dmAux, cells[0], a, &aL));
                PetscCall(DMPlexPointLocalRead(dmAux, cells[1], a, &aR));

                auto fr = flux_kernel(qL_face, qR_face, aL, aR, parameters.data(), n_hat);
                
                PetscScalar *fL, *fR;
                PetscCall(DMPlexPointLocalRef(dmQ, cells[0], f_loc, &fL));
                PetscCall(DMPlexPointLocalRef(dmQ, cells[1], f_loc, &fR));
                for(int i=0; i<Model<Real>::n_dof_q; ++i) {
                    if (fL) fL[i] -= fr[i] * area; 
                    if (fR) fR[i] += fr[i] * area; 
                }
            } else if (num_cells == 1) {
                PetscInt tag_id;
                PetscCall(DMLabelGetValue(label, f, &tag_id));
                if (boundary_map.count(tag_id)) {
                    PetscInt bc_idx = boundary_map[tag_id];
                    const PetscScalar *qL, *aL;
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[0], x, &qL));
                    PetscCall(DMPlexPointLocalRead(dmAux, cells[0], a, &aL));
                    
                    auto qR_arr = Model<Real>::boundary_conditions(bc_idx, qL, aL, n_hat, fg->centroid, time, 0.0);
                    auto fr = flux_kernel(qL, qR_arr.data, aL, aL, parameters.data(), n_hat);
                    PetscScalar *fL;
                    PetscCall(DMPlexPointLocalRef(dmQ, cells[0], f_loc, &fL));
                    if (fL) { for(int i=0; i<Model<Real>::n_dof_q; ++i) fL[i] -= fr[i] * area; }
                }
            }
        }
        return PETSC_SUCCESS;
    }

    PetscErrorCode ComputeCellSources(const PetscScalar* x, const PetscScalar* a, PetscScalar* f_loc, PetscReal scale) {
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar *q, *aux; PetscScalar *f;
            PetscCall(DMPlexPointLocalRead(dmQ, c, x, &q));
            PetscCall(DMPlexPointLocalRead(dmAux, c, a, &aux));
            PetscCall(DMPlexPointLocalRef(dmQ, c, f_loc, &f));
            if (q && aux && f) {
                auto src = source_kernel(q, aux, parameters.data());
                for(int i=0; i<Model<Real>::n_dof_q; ++i) f[i] += src[i] * scale;
            }
        }
        return PETSC_SUCCESS;
    }

    static PetscErrorCode FormRHSFunctionWrapper(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
        return ((ModularSolver*)ctx)->FormRHSFunction(t, X, F);
    }
    static PetscErrorCode FormIFunctionWrapper(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, void *ctx) {
        return ((ModularSolver*)ctx)->FormIFunction(t, X, X_t, F);
    }
    static PetscErrorCode FormIJacobianWrapper(TS ts, PetscReal t, Vec X, Vec X_t, PetscReal a, Mat J, Mat Jpre, void *ctx) {
        return ((ModularSolver*)ctx)->FormIJacobian(t, X, X_t, a, J, Jpre);
    }
};
#endif