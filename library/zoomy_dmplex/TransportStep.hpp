#ifndef TRANSPORTSTEP_HPP
#define TRANSPORTSTEP_HPP

#include "VirtualSolver.hpp"
#include "Reconstruction.hpp"
#include "Gradient.hpp"

template <typename T>
class TransportStep {
private:
    DM dmQ, dmAux, dmGrad;
    std::shared_ptr<Reconstructor<T>> reconstructor;
    std::shared_ptr<GradientCalculator<T>> gradient;
    std::vector<T> parameters;
    std::map<PetscInt, PetscInt> boundary_map;
    
    // Internal vectors for limiting and topology caching
    Vec V_min, V_max; 
    Vec V_min_loc, V_max_loc; 
    
    // Cached Topology for Limiter (Min/Max neighbors)
    std::vector<PetscInt> neigh_offsets;
    std::vector<PetscInt> neigh_list;
    bool topology_setup = false;

    // Kernels
    FluxKernelPtr cons_flux_kernel;
    NonConservativeFluxKernelPtr noncons_flux_kernel;
    SourceKernelPtr source_kernel;

public:
    TransportStep(DM q, DM aux, DM grad, std::vector<T> params, std::map<PetscInt, PetscInt> bcs) 
        : dmQ(q), dmAux(aux), dmGrad(grad), parameters(params), boundary_map(bcs) {
        
        V_min = NULL; V_max = NULL; V_min_loc = NULL; V_max_loc = NULL;
        
        // Defaults
        reconstructor = std::make_shared<PCMReconstructor<T>>();
        gradient = nullptr;
        
        cons_flux_kernel = Numerics<T>::numerical_flux;
        noncons_flux_kernel = nullptr; 
        source_kernel = Model<T>::source;
    }

    ~TransportStep() {
        if (V_min) VecDestroy(&V_min);
        if (V_max) VecDestroy(&V_max);
        if (V_min_loc) VecDestroy(&V_min_loc);
        if (V_max_loc) VecDestroy(&V_max_loc);
    }

    // --- CONFIGURATION SETTERS ---
    void SetReconstruction(std::shared_ptr<Reconstructor<T>> r) { reconstructor = r; }
    void SetGradient(std::shared_ptr<GradientCalculator<T>> g) { gradient = g; }
    void SetNonConsFlux(NonConservativeFluxKernelPtr k) { noncons_flux_kernel = k; }
    
    // *** FIX: This was missing in the previous version ***
    void SetFluxKernel(FluxKernelPtr k) { cons_flux_kernel = k; }

    // --- Topology Setup for Limiter ---
    PetscErrorCode SetupTopology() {
        if (topology_setup) return PETSC_SUCCESS;
        
        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        neigh_offsets.clear(); neigh_list.clear();
        neigh_offsets.push_back(0);
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
            PetscInt num_adj = -1; PetscInt *adj = NULL; 
            PetscCall(DMPlexGetAdjacency(dmQ, c, &num_adj, &adj));
            
            for(int k=0; k<num_adj; ++k) {
                PetscInt n = adj[k]; 
                if (n == c || n < 0) continue; 
                neigh_list.push_back(n);
            }
            PetscCall(PetscFree(adj));
            neigh_offsets.push_back(neigh_list.size());
        }
        
        // Allocate Limiter Vectors
        PetscCall(DMCreateGlobalVector(dmQ, &V_min));
        PetscCall(DMCreateGlobalVector(dmQ, &V_max));
        PetscCall(DMCreateLocalVector(dmQ, &V_min_loc));
        PetscCall(DMCreateLocalVector(dmQ, &V_max_loc));
        
        topology_setup = true;
        return PETSC_SUCCESS;
    }

    // --- Neighbor Min/Max Update for Limiter ---
    PetscErrorCode UpdateNeighborBounds(Vec X_loc) {
        if (!topology_setup) PetscCall(SetupTopology());
        
        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        const PetscScalar *x_ptr; 
        PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        
        PetscInt pStart, pEnd;
        PetscCall(DMPlexGetChart(dmQ, &pStart, &pEnd));
        
        for (PetscInt c = pStart; c < pEnd; ++c) {
            if (c < cStart || c >= cEnd) continue; // Only owned cells iterate (ghosts filled via scatter)
            
            PetscInt g_point; PetscCall(DMPlexGetPointGlobal(dmQ, c, &g_point, NULL));
            if (g_point < 0) continue;

            const PetscScalar *q; 
            PetscCall(DMPlexPointLocalRead(dmQ, c, x_ptr, &q));
            
            PetscScalar q_min[Model<T>::n_dof_q];
            PetscScalar q_max[Model<T>::n_dof_q];
            
            for(int i=0; i<Model<T>::n_dof_q; ++i) { q_min[i] = q[i]; q_max[i] = q[i]; }
            
            int idx_start = neigh_offsets[c - cStart]; 
            int idx_end = neigh_offsets[c - cStart + 1];
            
            for(int k=idx_start; k<idx_end; ++k) {
                PetscInt n = neigh_list[k]; 
                const PetscScalar *qn; 
                PetscCall(DMPlexPointLocalRead(dmQ, n, x_ptr, &qn));
                for(int i=0; i<Model<T>::n_dof_q; ++i) {
                    if (qn[i] < q_min[i]) q_min[i] = qn[i];
                    if (qn[i] > q_max[i]) q_max[i] = qn[i];
                }
            }
            PetscCall(DMPlexVecSetClosure(dmQ, NULL, V_min, c, q_min, INSERT_VALUES));
            PetscCall(DMPlexVecSetClosure(dmQ, NULL, V_max, c, q_max, INSERT_VALUES));
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr));

        // Sync Ghosts for bounds
        PetscCall(VecAssemblyBegin(V_min)); PetscCall(VecAssemblyEnd(V_min));
        PetscCall(VecAssemblyBegin(V_max)); PetscCall(VecAssemblyEnd(V_max));
        
        PetscCall(DMGlobalToLocalBegin(dmQ, V_min, INSERT_VALUES, V_min_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, V_min, INSERT_VALUES, V_min_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, V_max, INSERT_VALUES, V_max_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, V_max, INSERT_VALUES, V_max_loc));
        
        return PETSC_SUCCESS;
    }

    // --- Main RHS Function ---
    PetscErrorCode FormRHS(PetscReal time, Vec X_global, Vec F_global) {
        PetscCall(VecZeroEntries(F_global));

        // 1. Prepare Local State
        Vec X_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        
        // 2. Prepare Local Aux
        // We generate Aux locally from X_loc to ensure consistency
        Vec A_loc;
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(UpdateLocalAux(X_loc, A_loc)); 

        // 3. Limiter Bounds (only needed if gradient is active/2nd order)
        if (gradient) {
            PetscCall(UpdateNeighborBounds(X_loc));
        }

        // 4. Compute Gradient (if needed)
        Vec G_global = NULL;
        Vec G_loc = NULL;
        PetscScalar *g_ptr = NULL;
        
        if (gradient) {
             PetscCall(DMCreateGlobalVector(dmGrad, &G_global));
             PetscCall(gradient->Compute(dmQ, X_global, dmGrad, G_global, boundary_map));
             
             PetscCall(DMGetLocalVector(dmGrad, &G_loc));
             PetscCall(DMGlobalToLocalBegin(dmGrad, G_global, INSERT_VALUES, G_loc));
             PetscCall(DMGlobalToLocalEnd(dmGrad, G_global, INSERT_VALUES, G_loc));
             PetscCall(VecGetArray(G_loc, &g_ptr));
        }

        // 5. Flux Computation
        Vec F_loc; 
        PetscCall(DMGetLocalVector(dmQ, &F_loc)); 
        PetscCall(VecZeroEntries(F_loc));

        PetscCall(ComputeFluxes(time, X_loc, A_loc, g_ptr, F_loc));

        // 6. Explicit Source
        PetscCall(ComputeExplicitSource(X_loc, A_loc, F_loc));

        // 7. Assembly
        PetscCall(DMLocalToGlobalBegin(dmQ, F_loc, ADD_VALUES, F_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, F_loc, ADD_VALUES, F_global));

        // Cleanup
        if (G_loc) { PetscCall(VecRestoreArray(G_loc, &g_ptr)); PetscCall(DMRestoreLocalVector(dmGrad, &G_loc)); }
        if (G_global) PetscCall(VecDestroy(&G_global));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(DMRestoreLocalVector(dmQ, &F_loc));

        return PETSC_SUCCESS;
    }

    PetscErrorCode ComputeFluxes(PetscReal time, Vec X_loc, Vec A_loc, const PetscScalar* g_ptr, Vec F_loc) {
        // Access Data
        const PetscScalar *x_ptr, *a_ptr;
        PetscScalar *f_ptr;
        PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscCall(VecGetArrayRead(A_loc, &a_ptr));
        PetscCall(VecGetArray(F_loc, &f_ptr));

        // Geometry
        Vec cellGeom, faceGeom;
        PetscCall(DMPlexGetGeometryFVM(dmQ, &faceGeom, &cellGeom, NULL));
        const PetscScalar *fGeom_ptr, *cGeom_ptr;
        PetscCall(VecGetArrayRead(faceGeom, &fGeom_ptr));
        PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));

        DM dmFace; PetscCall(VecGetDM(faceGeom, &dmFace));
        PetscSection secFace; PetscCall(DMGetLocalSection(dmFace, &secFace));
        DM dmCell; PetscCall(VecGetDM(cellGeom, &dmCell));
        PetscSection secCell; PetscCall(DMGetLocalSection(dmCell, &secCell));

        // Limiter Bounds Access
        const PetscScalar *min_ptr = NULL, *max_ptr = NULL;
        if (V_min_loc) {
            PetscCall(VecGetArrayRead(V_min_loc, &min_ptr));
            PetscCall(VecGetArrayRead(V_max_loc, &max_ptr));
        }

        PetscInt fStart, fEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 1, &fStart, &fEnd));
        DMLabel label; PetscCall(DMGetLabel(dmQ, "Face Sets", &label));
        PetscInt dim = Model<T>::dimension;

        for (PetscInt f = fStart; f < fEnd; ++f) {
            PetscInt off; PetscCall(PetscSectionGetOffset(secFace, f, &off));
            const PetscFVFaceGeom *fg = (const PetscFVFaceGeom*)&fGeom_ptr[off];
            
            // Normal and Area
            PetscScalar n_hat[3] = {0}; 
            PetscReal area = 0; 
            for(int d=0; d<dim; ++d) area += fg->normal[d]*fg->normal[d]; 
            area = std::sqrt(area);
            if(area <= 1e-15) continue; 
            for(int d=0; d<dim; ++d) n_hat[d] = fg->normal[d] / area;

            const PetscInt *cells; PetscInt num_cells; 
            PetscCall(DMPlexGetSupportSize(dmQ, f, &num_cells)); 
            PetscCall(DMPlexGetSupport(dmQ, f, &cells));

            if (num_cells == 2) {
                // Internal Face
                const PetscScalar *qL_cell, *qR_cell;
                const PetscScalar *gL_cell = NULL, *gR_cell = NULL;
                
                PetscCall(DMPlexPointLocalRead(dmQ, cells[0], x_ptr, &qL_cell));
                PetscCall(DMPlexPointLocalRead(dmQ, cells[1], x_ptr, &qR_cell));
                
                if (g_ptr) {
                    PetscCall(DMPlexPointLocalRead(dmGrad, cells[0], g_ptr, &gL_cell));
                    PetscCall(DMPlexPointLocalRead(dmGrad, cells[1], g_ptr, &gR_cell));
                }

                // Bounds
                const PetscScalar *minL=NULL, *maxL=NULL, *minR=NULL, *maxR=NULL;
                if (min_ptr) {
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[0], min_ptr, &minL));
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[0], max_ptr, &maxL));
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[1], min_ptr, &minR));
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[1], max_ptr, &maxR));
                }

                // Centroids
                PetscInt offL, offR;
                PetscCall(PetscSectionGetOffset(secCell, cells[0], &offL));
                PetscCall(PetscSectionGetOffset(secCell, cells[1], &offR));
                const PetscFVCellGeom *cgL = (const PetscFVCellGeom*)&cGeom_ptr[offL];
                const PetscFVCellGeom *cgR = (const PetscFVCellGeom*)&cGeom_ptr[offR];

                // Reconstruct State
                PetscScalar qL_face[Model<T>::n_dof_q], qR_face[Model<T>::n_dof_q];
                reconstructor->Reconstruct(qL_cell, gL_cell, cgL->centroid, fg->centroid, minL, maxL, qL_face);
                reconstructor->Reconstruct(qR_cell, gR_cell, cgR->centroid, fg->centroid, minR, maxR, qR_face);

                // Update Aux at Face (Consistency Fix)
                PetscScalar aL_face[Model<T>::n_dof_qaux], aR_face[Model<T>::n_dof_qaux];
                auto res_aL = Model<T>::update_aux_variables(qL_face, nullptr, parameters.data());
                for(int i=0; i<Model<T>::n_dof_qaux; ++i) aL_face[i] = res_aL[i];
                auto res_aR = Model<T>::update_aux_variables(qR_face, nullptr, parameters.data());
                for(int i=0; i<Model<T>::n_dof_qaux; ++i) aR_face[i] = res_aR[i];

                // Fluxes
                auto flux = cons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);
                
                // Add Non-Conservative
                if (noncons_flux_kernel) {
                    auto nc_L = noncons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);
                    PetscScalar n_neg[3]; for(int d=0; d<dim; ++d) n_neg[d] = -n_hat[d];
                    auto nc_R = noncons_flux_kernel(qR_face, qL_face, aR_face, aL_face, parameters.data(), n_neg);
                    
                    PetscScalar *fL, *fR;
                    PetscCall(DMPlexPointLocalRef(dmQ, cells[0], f_ptr, &fL));
                    PetscCall(DMPlexPointLocalRef(dmQ, cells[1], f_ptr, &fR));
                    // Note: Non-cons usually adds to residual: - D_minus
                    for(int i=0; i<Model<T>::n_dof_q; ++i) {
                        fL[i] -= nc_L[i] * area;
                        fR[i] -= nc_R[i] * area;
                    }
                }

                // Accumulate Conservative Flux (-div F)
                PetscScalar *fL, *fR;
                PetscCall(DMPlexPointLocalRef(dmQ, cells[0], f_ptr, &fL));
                PetscCall(DMPlexPointLocalRef(dmQ, cells[1], f_ptr, &fR));
                for(int i=0; i<Model<T>::n_dof_q; ++i) { 
                    if (fL) fL[i] -= flux[i] * area; 
                    if (fR) fR[i] += flux[i] * area; 
                }

            } else if (num_cells == 1) {
                // Boundary
                PetscInt tag_id; PetscCall(DMLabelGetValue(label, f, &tag_id));
                if (boundary_map.count(tag_id)) {
                    PetscInt bc_idx = boundary_map.at(tag_id);
                    const PetscScalar *qL_cell, *aL_cell;
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[0], x_ptr, &qL_cell));
                    PetscCall(DMPlexPointLocalRead(dmAux, cells[0], a_ptr, &aL_cell));
                    
                    auto q_bc = Model<T>::boundary_conditions(bc_idx, qL_cell, aL_cell, n_hat, fg->centroid, time, 0.0);
                    
                    // Aux for ghost
                    PetscScalar a_bc[Model<T>::n_dof_qaux];
                    auto res_a = Model<T>::update_aux_variables(q_bc.data, nullptr, parameters.data());
                    for(int i=0; i<Model<T>::n_dof_qaux; ++i) a_bc[i] = res_a[i];

                    auto flux = cons_flux_kernel(qL_cell, q_bc.data, aL_cell, a_bc, parameters.data(), n_hat);
                    
                    PetscScalar *fL;
                    PetscCall(DMPlexPointLocalRef(dmQ, cells[0], f_ptr, &fL));
                    if (fL) { for(int i=0; i<Model<T>::n_dof_q; ++i) fL[i] -= flux[i] * area; }
                }
            }
        }

        // Divide by Volume
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        for(PetscInt c=cStart; c<cEnd; ++c) {
             PetscInt off; PetscCall(PetscSectionGetOffset(secCell, c, &off)); 
             const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cGeom_ptr[off];
             PetscScalar *f_cell; PetscCall(DMPlexPointLocalRef(dmQ, c, f_ptr, &f_cell));
             if (f_cell && cg->volume > 1e-15) { 
                 for(int i=0; i<Model<T>::n_dof_q; ++i) f_cell[i] /= cg->volume; 
             }
        }

        // Restore
        if (V_min_loc) {
            PetscCall(VecRestoreArrayRead(V_min_loc, &min_ptr));
            PetscCall(VecRestoreArrayRead(V_max_loc, &max_ptr));
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr));
        PetscCall(VecRestoreArrayRead(A_loc, &a_ptr));
        PetscCall(VecRestoreArray(F_loc, &f_ptr));
        PetscCall(VecRestoreArrayRead(faceGeom, &fGeom_ptr));
        PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr));

        return PETSC_SUCCESS;
    }

    PetscErrorCode ComputeExplicitSource(Vec X_loc, Vec A_loc, Vec F_loc) {
        const PetscScalar *x_ptr, *a_ptr; PetscScalar *f_ptr;
        PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscCall(VecGetArrayRead(A_loc, &a_ptr));
        PetscCall(VecGetArray(F_loc, &f_ptr));
        
        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar *q, *aux; PetscScalar *f;
            PetscCall(DMPlexPointLocalRead(dmQ, c, x_ptr, &q));
            PetscCall(DMPlexPointLocalRead(dmAux, c, a_ptr, &aux));
            PetscCall(DMPlexPointLocalRef(dmQ, c, f_ptr, &f));
            
            if (q && aux && f) {
                // Currently assuming source_kernel is handled by SourceStep (splitting)
                // If you have a purely explicit source term, add it here.
            }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr));
        PetscCall(VecRestoreArrayRead(A_loc, &a_ptr));
        PetscCall(VecRestoreArray(F_loc, &f_ptr));
        return PETSC_SUCCESS;
    }

    PetscErrorCode UpdateLocalAux(Vec X_loc, Vec A_loc) {
        const PetscScalar *x_ptr; PetscScalar *a_ptr;
        PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscCall(VecGetArray(A_loc, &a_ptr));
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        for(PetscInt c=cStart; c<cEnd; ++c) {
            const PetscScalar *q; PetscScalar *a;
            PetscCall(DMPlexPointLocalRead(dmQ, c, x_ptr, &q));
            PetscCall(DMPlexPointLocalRef(dmAux, c, a_ptr, &a));
            if (q && a) {
                auto res = Model<T>::update_aux_variables(q, a, parameters.data());
                for(int i=0; i<Model<T>::n_dof_qaux; ++i) a[i] = res[i];
            }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr));
        PetscCall(VecRestoreArray(A_loc, &a_ptr));
        return PETSC_SUCCESS;
    }
};
#endif