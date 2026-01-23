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

using FluxKernelFunc = std::function<FluxResult(
    const PetscScalar* qL, const PetscScalar* qR, 
    const PetscScalar* auxL, const PetscScalar* auxR, 
    const PetscScalar* p, const PetscScalar* n)>;

using SourceKernelFunc = std::function<SimpleArray<PetscScalar, Model<Real>::n_dof_q>(
    const PetscScalar* q, const PetscScalar* aux, const PetscScalar* p)>;

class ModularSolver : public VirtualSolver {
protected:
    std::vector<FluxKernelFunc> flux_kernels;
    std::vector<SourceKernelFunc> source_kernels;

public:
    ModularSolver() {}

    template<typename F>
    void AddConservativeFlux(F func) {
        flux_kernels.push_back([func](const PetscScalar* qL, const PetscScalar* qR, 
                                      const PetscScalar* aL, const PetscScalar* aR, 
                                      const PetscScalar* p, const PetscScalar* n) -> FluxResult {
            auto f = func(qL, qR, aL, aR, p, n);
            FluxResult res;
            for(int i=0; i<Model<Real>::n_dof_q; ++i) { res.fluxL[i] = f[i]; res.fluxR[i] = -f[i]; }
            return res;
        });
    }

    template<typename F>
    void AddSource(F func) { source_kernels.push_back(func); }

    PetscErrorCode ComputeRHS(PetscReal time, Vec X_global, Vec F_global) override {
        Vec X_loc, A_loc, F_loc, cellGeom, faceGeom;
        
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A, INSERT_VALUES, A_loc));
        PetscCall(DMGetLocalVector(dmQ, &F_loc));
        PetscCall(VecSet(F_loc, 0.0));

        const PetscScalar *x_ptr, *a_ptr; PetscScalar *f_ptr;
        PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscCall(VecGetArrayRead(A_loc, &a_ptr));
        PetscCall(VecGetArray(F_loc, &f_ptr));

        // Use dmMesh for geometry (cached)
        // Correct Order: Face, Cell
        PetscCall(DMPlexGetGeometryFVM(dmMesh, &faceGeom, &cellGeom, NULL));
        
        DM dmCell, dmFace;
        PetscSection secCell, secFace;
        PetscCall(VecGetDM(cellGeom, &dmCell));
        PetscCall(VecGetDM(faceGeom, &dmFace));
        PetscCall(DMGetLocalSection(dmCell, &secCell));
        PetscCall(DMGetLocalSection(dmFace, &secFace));

        const PetscScalar *cGeom_ptr, *fGeom_ptr;
        PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));
        PetscCall(VecGetArrayRead(faceGeom, &fGeom_ptr));

        PetscCall(ComputeFaceFluxes(time, x_ptr, a_ptr, f_ptr, fGeom_ptr, secFace));
        PetscCall(ComputeCellSources(x_ptr, a_ptr, f_ptr));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
             PetscInt off;
             PetscCall(PetscSectionGetOffset(secCell, c, &off));
             if (off < 0) continue; 

             const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cGeom_ptr[off];
             if (cg->volume <= 1e-15) continue;

             PetscScalar *f_cell;
             PetscCall(DMPlexPointLocalRef(dmQ, c, f_ptr, &f_cell));
             if (f_cell) { for(int i=0; i<Model<Real>::n_dof_q; ++i) f_cell[i] /= cg->volume; }
        }

        PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr));
        PetscCall(VecRestoreArrayRead(faceGeom, &fGeom_ptr));
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr));
        PetscCall(VecRestoreArrayRead(A_loc, &a_ptr));
        PetscCall(VecRestoreArray(F_loc, &f_ptr));

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
            PetscScalar *q, *a;
            PetscCall(DMPlexPointLocalRef(dmQ, c, x_ptr, &q));
            PetscCall(DMPlexPointLocalRef(dmAux, c, a_ptr, &a));
            if (q && a) {
                auto res = Numerics<Real>::overwrite_q_qaux(q, a, parameters.data());
                for(int i=0; i<Model<Real>::n_dof_q; ++i) q[i] = res[i];
                for(int i=0; i<Model<Real>::n_dof_qaux; ++i) a[i] = res[Model<Real>::n_dof_q + i];
            }
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
    PetscErrorCode ComputeFaceFluxes(PetscReal time, const PetscScalar* x, const PetscScalar* a, PetscScalar* f_loc, 
                                     const PetscScalar* fGeom_ptr, PetscSection secFace) {
        PetscInt fStart, fEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 1, &fStart, &fEnd));
        PetscInt dim = Model<Real>::dimension;
        DMLabel label; PetscCall(DMGetLabel(dmQ, "Face Sets", &label));

        static bool warned = false;

        for (PetscInt f = fStart; f < fEnd; ++f) {
            PetscInt off;
            PetscCall(PetscSectionGetOffset(secFace, f, &off));
            if (off < 0) continue; 

            const PetscFVFaceGeom *fg = (const PetscFVFaceGeom*)&fGeom_ptr[off];
            
            // --- ON-THE-FLY NORMALIZATION ---
            PetscScalar n_hat[3] = {0}; PetscReal area = 0;
            for(int d=0; d<dim; ++d) area += fg->normal[d]*fg->normal[d];
            area = std::sqrt(area);
            
            if(area <= 1e-15) continue; // Skip degenerate faces

            for(int d=0; d<dim; ++d) n_hat[d] = fg->normal[d] / area;

            const PetscInt *cells; PetscInt num_cells;
            PetscCall(DMPlexGetSupportSize(dmQ, f, &num_cells));
            PetscCall(DMPlexGetSupport(dmQ, f, &cells));

            if (num_cells == 2) {
                const PetscScalar *qL, *qR, *aL, *aR;
                PetscCall(DMPlexPointLocalRead(dmQ, cells[0], x, &qL));
                PetscCall(DMPlexPointLocalRead(dmQ, cells[1], x, &qR));
                PetscCall(DMPlexPointLocalRead(dmAux, cells[0], a, &aL));
                PetscCall(DMPlexPointLocalRead(dmAux, cells[1], a, &aR));

                for (auto& kernel : flux_kernels) {
                    FluxResult fr = kernel(qL, qR, aL, aR, parameters.data(), n_hat);
                    PetscScalar *fL, *fR;
                    PetscCall(DMPlexPointLocalRef(dmQ, cells[0], f_loc, &fL));
                    PetscCall(DMPlexPointLocalRef(dmQ, cells[1], f_loc, &fR));
                    // Independent updates for safety
                    if (fL) { for(int i=0; i<Model<Real>::n_dof_q; ++i) fL[i] -= fr.fluxL[i] * area; }
                    if (fR) { for(int i=0; i<Model<Real>::n_dof_q; ++i) fR[i] -= fr.fluxR[i] * area; }
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
                    for (auto& kernel : flux_kernels) {
                        FluxResult fr = kernel(qL, qR_arr.data, aL, aL, parameters.data(), n_hat);
                        PetscScalar *fL;
                        PetscCall(DMPlexPointLocalRef(dmQ, cells[0], f_loc, &fL));
                        if (fL) { for(int i=0; i<Model<Real>::n_dof_q; ++i) fL[i] -= fr.fluxL[i] * area; }
                    }
                } else if (!warned && this->rank == 0) {
                    std::cerr << "[WARN] Boundary Face " << f << " (Tag " << tag_id << ") ignored! Not in map." << std::endl;
                    warned = true;
                }
            }
        }
        return PETSC_SUCCESS;
    }

    PetscErrorCode ComputeCellSources(const PetscScalar* x, const PetscScalar* a, PetscScalar* f_loc) {
        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar *q, *aux; PetscScalar *f;
            PetscCall(DMPlexPointLocalRead(dmQ, c, x, &q));
            PetscCall(DMPlexPointLocalRead(dmAux, c, a, &aux));
            PetscCall(DMPlexPointLocalRef(dmQ, c, f_loc, &f));
            if (q && aux && f) {
                for (auto& kernel : source_kernels) {
                    auto src = kernel(q, aux, parameters.data());
                    for(int i=0; i<Model<Real>::n_dof_q; ++i) f[i] += src[i];
                }
            }
        }
        return PETSC_SUCCESS;
    }
};
#endif