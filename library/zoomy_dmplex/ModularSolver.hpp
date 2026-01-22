#ifndef MODULARSOLVER_HPP
#define MODULARSOLVER_HPP

#include "VirtualSolver.hpp"
#include <functional>

struct FaceState {
    SimpleArray<PetscScalar, Model<Real>::n_dof_q> qL;
    SimpleArray<PetscScalar, Model<Real>::n_dof_q> qR;
    SimpleArray<PetscScalar, Model<Real>::n_dof_qaux> auxL;
    SimpleArray<PetscScalar, Model<Real>::n_dof_qaux> auxR;
};

struct FluxResult {
    SimpleArray<PetscScalar, Model<Real>::n_dof_q> fluxL; 
    SimpleArray<PetscScalar, Model<Real>::n_dof_q> fluxR; 
};

using ReconstructionFunc = std::function<FaceState(
    PetscInt cL, PetscInt cR, 
    const PetscScalar* global_X, const PetscScalar* global_Aux, 
    PetscInt n_dof_q, PetscInt n_dof_aux)>;

using FluxKernelFunc = std::function<FluxResult(
    const PetscScalar* qL, const PetscScalar* qR, 
    const PetscScalar* auxL, const PetscScalar* auxR, 
    const PetscScalar* p, const PetscScalar* n)>;

using SourceKernelFunc = std::function<SimpleArray<PetscScalar, Model<Real>::n_dof_q>(
    const PetscScalar* q, const PetscScalar* aux, const PetscScalar* p)>;


class ModularSolver : public VirtualSolver {
protected:
    ReconstructionFunc reconstructor;
    std::vector<FluxKernelFunc> flux_kernels;
    std::vector<SourceKernelFunc> source_kernels;

public:
    ModularSolver() {
        reconstructor = [](PetscInt cL, PetscInt cR, const PetscScalar* x, const PetscScalar* a, int nq, int na) -> FaceState {
            FaceState fs;
            for(int i=0; i<nq; ++i) fs.qL[i] = x[cL*nq + i];
            for(int i=0; i<nq; ++i) fs.qR[i] = x[cR*nq + i];
            for(int i=0; i<na; ++i) fs.auxL[i] = a[cL*na + i];
            for(int i=0; i<na; ++i) fs.auxR[i] = a[cR*na + i];
            return fs;
        };
    }

    void SetReconstruction(ReconstructionFunc func) { reconstructor = func; }

    template<typename F>
    void AddConservativeFlux(F func) {
        flux_kernels.push_back([func](const PetscScalar* qL, const PetscScalar* qR, 
                                      const PetscScalar* aL, const PetscScalar* aR, 
                                      const PetscScalar* p, const PetscScalar* n) -> FluxResult {
            auto f = func(qL, qR, aL, aR, p, n);
            FluxResult res;
            for(int i=0; i<Model<Real>::n_dof_q; ++i) {
                res.fluxL[i] = f[i];  
                res.fluxR[i] = -f[i]; 
            }
            return res;
        });
    }

    template<typename F>
    void AddSource(F func) { source_kernels.push_back(func); }

    PetscErrorCode ComputeRHS(PetscReal time, Vec X_global, Vec F_global) override {
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        Vec F_loc;
        PetscCall(DMGetLocalVector(dmQ, &F_loc));
        PetscCall(VecSet(F_loc, 0.0));

        const PetscScalar *x_ptr, *a_ptr;
        PetscScalar *f_ptr;
        PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscCall(VecGetArrayRead(A_loc, &a_ptr));
        PetscCall(VecGetArray(F_loc, &f_ptr));

        PetscCall(ComputeFaceFluxes(x_ptr, a_ptr, f_ptr));
        PetscCall(ComputeCellSources(x_ptr, a_ptr, f_ptr));

        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr));
        PetscCall(VecRestoreArrayRead(A_loc, &a_ptr));
        PetscCall(VecRestoreArray(F_loc, &f_ptr));

        Vec cellGeom;
        PetscCall(DMPlexComputeGeometryFVM(dmQ, &cellGeom, NULL));
        const PetscScalar *geom_ptr;
        PetscCall(VecGetArrayRead(cellGeom, &geom_ptr));
        PetscCall(VecGetArray(F_loc, &f_ptr));
        
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        for(PetscInt c=cStart; c<cEnd; ++c) {
             PetscFVCellGeom *cg;
             PetscCall(DMPlexPointLocalRead(dmQ, c, geom_ptr, &cg));
             for(int i=0; i<Model<Real>::n_dof_q; ++i) f_ptr[(c-cStart)*Model<Real>::n_dof_q+i] /= cg->volume;
        }
        
        PetscCall(VecRestoreArray(F_loc, &f_ptr));
        PetscCall(VecRestoreArrayRead(cellGeom, &geom_ptr));
        PetscCall(VecDestroy(&cellGeom));

        PetscCall(VecSet(F_global, 0.0));
        PetscCall(DMLocalToGlobalBegin(dmQ, F_loc, ADD_VALUES, F_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, F_loc, ADD_VALUES, F_global));

        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(DMRestoreLocalVector(dmQ, &F_loc));
        return PETSC_SUCCESS;
    }

    PetscErrorCode UpdateState() override {
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X, INSERT_VALUES, X_loc));
        
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));

        PetscScalar *x_ptr, *a_ptr;
        PetscCall(VecGetArray(X_loc, &x_ptr));
        PetscCall(VecGetArray(A_loc, &a_ptr));
        
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd)); 
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
            Real* q = &x_ptr[c * Model<Real>::n_dof_q];
            Real* a = &a_ptr[c * Model<Real>::n_dof_qaux];
            
            auto res = Numerics<Real>::overwrite_q_qaux(q, a, parameters.data());
            
            for(int i=0; i<Model<Real>::n_dof_q; ++i) q[i] = res[i];
            for(int i=0; i<Model<Real>::n_dof_qaux; ++i) a[i] = res[Model<Real>::n_dof_q + i];
        }
        
        PetscCall(VecRestoreArray(X_loc, &x_ptr));
        PetscCall(VecRestoreArray(A_loc, &a_ptr));
        
        PetscCall(DMLocalToGlobalBegin(dmQ, X_loc, INSERT_VALUES, X));
        PetscCall(DMLocalToGlobalEnd(dmQ, X_loc, INSERT_VALUES, X));
        PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A));
        PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A));
        
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        return PETSC_SUCCESS;
    }

protected:
    PetscErrorCode ComputeFaceFluxes(const PetscScalar* x, const PetscScalar* a, PetscScalar* f_loc) {
        PetscInt fStart, fEnd;
        PetscCall(DMPlexGetHeightStratum(dmMesh, 1, &fStart, &fEnd));

        Vec faceGeom;
        PetscCall(DMPlexComputeGeometryFVM(dmMesh, NULL, &faceGeom));
        const PetscScalar* geom_ptr;
        PetscCall(VecGetArrayRead(faceGeom, &geom_ptr));
        
        PetscInt dim = Model<Real>::dimension;

        for (PetscInt f = fStart; f < fEnd; ++f) {
            const PetscInt *cells; 
            PetscInt num_cells;
            PetscCall(DMPlexGetSupportSize(dmMesh, f, &num_cells));
            PetscCall(DMPlexGetSupport(dmMesh, f, &cells));

            if (num_cells != 2) continue; 

            PetscInt cL = cells[0];
            PetscInt cR = cells[1];

            FaceState state = reconstructor(cL, cR, x, a, Model<Real>::n_dof_q, Model<Real>::n_dof_qaux);

            PetscFVFaceGeom *fg;
            PetscCall(DMPlexPointLocalRead(dmMesh, f, geom_ptr, &fg));
            
            PetscScalar n_hat[3] = {0};
            PetscReal area_sq = 0;
            for(int d=0; d<dim; ++d) { 
                n_hat[d] = fg->normal[d]; 
                area_sq += n_hat[d]*n_hat[d]; 
            }
            PetscReal area = std::sqrt(area_sq);
            
            if(area > 0) { for(int d=0; d<dim; ++d) n_hat[d] /= area; }

            for (auto& kernel : flux_kernels) {
                FluxResult fr = kernel(state.qL.data, state.qR.data, state.auxL.data, state.auxR.data, parameters.data(), n_hat);
                
                for(int i=0; i<Model<Real>::n_dof_q; ++i) {
                    f_loc[cL * Model<Real>::n_dof_q + i] -= fr.fluxL[i] * area;
                    f_loc[cR * Model<Real>::n_dof_q + i] -= fr.fluxR[i] * area;
                }
            }
        }
        PetscCall(VecRestoreArrayRead(faceGeom, &geom_ptr));
        PetscCall(VecDestroy(&faceGeom));
        return PETSC_SUCCESS;
    }

    PetscErrorCode ComputeCellSources(const PetscScalar* x, const PetscScalar* a, PetscScalar* f_loc) {
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmMesh, 0, &cStart, &cEnd));

        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar* q = &x[c * Model<Real>::n_dof_q];
            const PetscScalar* aux = &a[c * Model<Real>::n_dof_qaux];

            for (auto& kernel : source_kernels) {
                auto src = kernel(q, aux, parameters.data());
                for(int i=0; i<Model<Real>::n_dof_q; ++i) {
                    f_loc[c * Model<Real>::n_dof_q + i] += src[i];
                }
            }
        }
        return PETSC_SUCCESS;
    }
};
#endif