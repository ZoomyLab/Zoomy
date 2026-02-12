#ifndef TRANSPORTSTEP_HPP
#define TRANSPORTSTEP_HPP

#include "VirtualSolver.hpp"
#include "Reconstruction.hpp"
#include "Gradient.hpp"

template <typename T>
class TransportStep {
private:
    DM dmQ, dmAux, dmGrad, dmGradAux;
    std::shared_ptr<Reconstructor<T>> reconstructor;
    std::shared_ptr<Reconstructor<T>> reconstructorAux; 
    std::shared_ptr<GradientCalculator<T>> gradient;
    std::shared_ptr<GradientCalculator<T>> gradientAux; 
    
    std::vector<T> parameters;
    std::map<PetscInt, PetscInt> boundary_map;
    
    Vec V_min_loc, V_max_loc; 
    std::vector<PetscInt> neigh_offsets, neigh_list;
    bool topology_setup = false;

    // MOOD: 0=High, 1=Low
    std::vector<uint8_t> cell_orders;

    FluxKernelPtr cons_flux_kernel;
    NonConservativeFluxKernelPtr noncons_flux_kernel;
    SourceKernelPtr source_kernel;

    bool IsOwned(DM dm, PetscInt p) {
        PetscInt g_idx; 
        DMPlexGetPointGlobal(dm, p, &g_idx, NULL); 
        return (g_idx >= 0);
    }

public:
    TransportStep(DM q, DM aux, DM grad, DM gradAux, std::vector<T> params, std::map<PetscInt, PetscInt> bcs) 
        : dmQ(q), dmAux(aux), dmGrad(grad), dmGradAux(gradAux), parameters(params), boundary_map(bcs) {
        V_min_loc = NULL; V_max_loc = NULL;
        reconstructor = std::make_shared<PCMReconstructor<T>>();
        reconstructorAux = std::make_shared<PCMReconstructor<T>>();
        gradient = nullptr;
        gradientAux = nullptr;
        cons_flux_kernel = nullptr; 
        noncons_flux_kernel = nullptr; 
        source_kernel = Model<T>::source;
    }

    ~TransportStep() {
        if (V_min_loc) VecDestroy(&V_min_loc);
        if (V_max_loc) VecDestroy(&V_max_loc);
    }

    void SetReconstruction(std::shared_ptr<Reconstructor<T>> r) { reconstructor = r; }
    void SetAuxReconstruction(std::shared_ptr<Reconstructor<T>> r) { reconstructorAux = r; }
    void SetGradient(std::shared_ptr<GradientCalculator<T>> g) { gradient = g; }
    void SetAuxGradient(std::shared_ptr<GradientCalculator<T>> g) { gradientAux = g; }
    void SetNonConsFlux(NonConservativeFluxKernelPtr k) { noncons_flux_kernel = k; }
    void SetFluxKernel(FluxKernelPtr k) { cons_flux_kernel = k; }
    void SetCellOrders(const std::vector<uint8_t>& orders) { cell_orders = orders; }

    PetscErrorCode SetupTopology() {
        if (topology_setup) return PETSC_SUCCESS;
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        neigh_offsets.clear(); neigh_list.clear(); neigh_offsets.push_back(0);
        for(PetscInt c=cStart; c<cEnd; ++c) {
            PetscInt num_adj = -1; PetscInt *adj = NULL; PetscCall(DMPlexGetAdjacency(dmQ, c, &num_adj, &adj));
            for(int k=0; k<num_adj; ++k) { PetscInt n = adj[k]; if (n == c || n < 0) continue; neigh_list.push_back(n); }
            PetscCall(PetscFree(adj)); neigh_offsets.push_back(neigh_list.size());
        }
        PetscCall(DMCreateLocalVector(dmQ, &V_min_loc)); PetscCall(DMCreateLocalVector(dmQ, &V_max_loc));
        topology_setup = true; return PETSC_SUCCESS;
    }

    PetscErrorCode UpdateNeighborBounds(Vec X_loc) {
        if (!topology_setup) PetscCall(SetupTopology());
        const PetscScalar *x_ptr; PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscScalar *min_ptr, *max_ptr; PetscCall(VecGetArray(V_min_loc, &min_ptr)); PetscCall(VecGetArray(V_max_loc, &max_ptr));
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar *q; PetscCall(DMPlexPointLocalRead(dmQ, c, x_ptr, &q));
            PetscScalar q_min[Model<T>::n_dof_q], q_max[Model<T>::n_dof_q];
            for(int i=0; i<Model<T>::n_dof_q; ++i) { q_min[i] = q[i]; q_max[i] = q[i]; }
            int idx_start = neigh_offsets[c - cStart]; int idx_end = neigh_offsets[c - cStart + 1];
            for(int k=idx_start; k<idx_end; ++k) {
                PetscInt n = neigh_list[k]; const PetscScalar *qn; PetscCall(DMPlexPointLocalRead(dmQ, n, x_ptr, &qn));
                for(int i=0; i<Model<T>::n_dof_q; ++i) { if (qn[i] < q_min[i]) q_min[i] = qn[i]; if (qn[i] > q_max[i]) q_max[i] = qn[i]; }
            }
            PetscScalar *d_min, *d_max;
            PetscCall(DMPlexPointLocalRef(dmQ, c, min_ptr, &d_min)); PetscCall(DMPlexPointLocalRef(dmQ, c, max_ptr, &d_max));
            for(int i=0; i<Model<T>::n_dof_q; ++i) { d_min[i] = q_min[i]; d_max[i] = q_max[i]; }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr)); PetscCall(VecRestoreArray(V_min_loc, &min_ptr)); PetscCall(VecRestoreArray(V_max_loc, &max_ptr));
        return PETSC_SUCCESS;
    }

    // Updates Aux variables based on State variables. Called at the start of every RK stage.
    PetscErrorCode UpdateState(Vec X_loc, Vec A_loc) {
        PetscScalar *x_ptr, *a_ptr;
        PetscCall(VecGetArray(X_loc, &x_ptr)); PetscCall(VecGetArray(A_loc, &a_ptr));
        PetscInt size_q, size_a;
        PetscCall(VecGetLocalSize(X_loc, &size_q)); PetscCall(VecGetLocalSize(A_loc, &size_a));
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscSection sQ, sAux; PetscCall(DMGetLocalSection(dmQ, &sQ)); PetscCall(DMGetLocalSection(dmAux, &sAux));
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
            PetscInt offQ, offA; PetscCall(PetscSectionGetOffset(sQ, c, &offQ)); PetscCall(PetscSectionGetOffset(sAux, c, &offA));
            if (offQ >= 0 && (offQ + Model<T>::n_dof_q) <= size_q && offA >= 0 && (offA + Model<T>::n_dof_qaux) <= size_a) {
                PetscScalar *q = &x_ptr[offQ]; PetscScalar *a = &a_ptr[offA];
                // Update State (e.g. for EOS consistency if needed, usually Identity)
                auto res_q = Model<T>::update_variables(q, a, parameters.data()); 
                for(int i=0; i<Model<T>::n_dof_q; ++i) q[i] = res_q[i];
                // Update Aux (Computed from State)
                auto res_a = Model<T>::update_aux_variables(q, a, parameters.data()); 
                for(int i=0; i<Model<T>::n_dof_qaux; ++i) a[i] = res_a[i];
            }
        }
        PetscCall(VecRestoreArray(X_loc, &x_ptr)); PetscCall(VecRestoreArray(A_loc, &a_ptr));
        return PETSC_SUCCESS;
    }

    PetscErrorCode FormRHS(PetscReal time, Vec X_global, Vec F_global) {
        PetscCall(VecZeroEntries(F_global));
        
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc)); 
        
        // 1. UPDATE STEP (Consistent physical state for this RK stage)
        PetscCall(UpdateState(X_loc, A_loc)); 
        
        // 2. Compute Bounds (if needed for limiters)
        if (gradient) {
            PetscCall(UpdateNeighborBounds(X_loc));
        }

        Vec G_global = NULL; Vec G_loc = NULL; PetscScalar *g_ptr = NULL;
        Vec G_aux_global = NULL; Vec G_aux_loc = NULL; PetscScalar *g_aux_ptr = NULL;

        // 3. Compute Gradients for State (Q)
        //    Pass A_loc as context so Grad(Q) can use Aux vars if needed (e.g. for BCs)
        if (gradient) {
             PetscCall(DMCreateGlobalVector(dmGrad, &G_global));
             PetscCall(gradient->Compute(dmQ, X_global, dmGrad, G_global, boundary_map, dmAux, A_loc));
             PetscCall(DMGetLocalVector(dmGrad, &G_loc));
             PetscCall(DMGlobalToLocalBegin(dmGrad, G_global, INSERT_VALUES, G_loc));
             PetscCall(DMGlobalToLocalEnd(dmGrad, G_global, INSERT_VALUES, G_loc));
             PetscCall(VecGetArray(G_loc, &g_ptr));
        }

        // 4. Compute Gradients for Aux (A)
        //    Pass X_loc as context so Grad(A) can use Q vars if needed
        if (gradientAux) {
             PetscCall(DMCreateGlobalVector(dmGradAux, &G_aux_global));
             
             // Create a Global A vector to pass to Gradient Compute (which handles ghost exchange)
             Vec A_global; 
             PetscCall(DMCreateGlobalVector(dmAux, &A_global));
             PetscCall(DMLocalToGlobalBegin(dmAux, A_loc, INSERT_VALUES, A_global));
             PetscCall(DMLocalToGlobalEnd(dmAux, A_loc, INSERT_VALUES, A_global));

             PetscCall(gradientAux->Compute(dmAux, A_global, dmGradAux, G_aux_global, boundary_map, dmQ, X_loc));
             
             PetscCall(DMGetLocalVector(dmGradAux, &G_aux_loc));
             PetscCall(DMGlobalToLocalBegin(dmGradAux, G_aux_global, INSERT_VALUES, G_aux_loc));
             PetscCall(DMGlobalToLocalEnd(dmGradAux, G_aux_global, INSERT_VALUES, G_aux_loc));
             PetscCall(VecGetArray(G_aux_loc, &g_aux_ptr));
             PetscCall(VecDestroy(&A_global));
        }

        Vec F_loc; 
        PetscCall(DMGetLocalVector(dmQ, &F_loc)); 
        PetscCall(VecZeroEntries(F_loc));
        
        // 5. Compute Fluxes (Physical Step)
        PetscCall(ComputeFluxes(time, X_loc, A_loc, g_ptr, g_aux_ptr, F_loc));

        // 6. Add Explicit Sources
        PetscCall(ComputeExplicitSource(X_loc, A_loc, F_loc));

        // 7. Assembly
        PetscCall(DMLocalToGlobalBegin(dmQ, F_loc, ADD_VALUES, F_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, F_loc, ADD_VALUES, F_global));

        // Cleanup
        if (G_loc) { PetscCall(VecRestoreArray(G_loc, &g_ptr)); PetscCall(DMRestoreLocalVector(dmGrad, &G_loc)); }
        if (G_global) PetscCall(VecDestroy(&G_global));
        
        if (G_aux_loc) { PetscCall(VecRestoreArray(G_aux_loc, &g_aux_ptr)); PetscCall(DMRestoreLocalVector(dmGradAux, &G_aux_loc)); }
        if (G_aux_global) PetscCall(VecDestroy(&G_aux_global));

        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); 
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(DMRestoreLocalVector(dmQ, &F_loc));
        
        return PETSC_SUCCESS;
    }

    PetscErrorCode ComputeFluxes(PetscReal time, Vec X_loc, Vec A_loc, const PetscScalar* g_ptr, const PetscScalar* g_aux_ptr, Vec F_loc) {
        const PetscScalar *x_ptr, *a_ptr; PetscScalar *f_ptr;
        PetscCall(VecGetArrayRead(X_loc, &x_ptr)); 
        PetscCall(VecGetArrayRead(A_loc, &a_ptr)); 
        PetscCall(VecGetArray(F_loc, &f_ptr));
        
        PetscInt size_f; 
        PetscCall(VecGetLocalSize(F_loc, &size_f));

        Vec cellGeom, faceGeom; 
        PetscCall(DMPlexGetGeometryFVM(dmQ, &faceGeom, &cellGeom, NULL));
        const PetscScalar *fGeom_ptr, *cGeom_ptr;
        PetscCall(VecGetArrayRead(faceGeom, &fGeom_ptr)); 
        PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));
        
        DM dmFace; PetscCall(VecGetDM(faceGeom, &dmFace)); 
        PetscSection secFace; PetscCall(DMGetLocalSection(dmFace, &secFace));
        DM dmCell; PetscCall(VecGetDM(cellGeom, &dmCell)); 
        PetscSection secCell; PetscCall(DMGetLocalSection(dmCell, &secCell));
        PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ));

        const PetscScalar *min_ptr = NULL, *max_ptr = NULL;
        if (V_min_loc) { 
            PetscCall(VecGetArrayRead(V_min_loc, &min_ptr)); 
            PetscCall(VecGetArrayRead(V_max_loc, &max_ptr)); 
        }

        PetscInt fStart, fEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 1, &fStart, &fEnd));
        DMLabel label; 
        PetscCall(DMGetLabel(dmQ, "Face Sets", &label));
        PetscInt dim = Model<T>::dimension;
        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));

        for (PetscInt f = fStart; f < fEnd; ++f) {
            if (!IsOwned(dmQ, f)) continue;
            
            PetscInt off; PetscCall(PetscSectionGetOffset(secFace, f, &off)); 
            const PetscFVFaceGeom *fg = (const PetscFVFaceGeom*)&fGeom_ptr[off];
            
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
                // --- Internal Face ---
                const PetscScalar *qL_cell, *qR_cell; 
                const PetscScalar *gL_cell = NULL, *gR_cell = NULL;
                const PetscScalar *aL_cell, *aR_cell; 
                const PetscScalar *gaL_cell = NULL, *gaR_cell = NULL;

                PetscCall(DMPlexPointLocalRead(dmQ, cells[0], x_ptr, &qL_cell)); 
                PetscCall(DMPlexPointLocalRead(dmQ, cells[1], x_ptr, &qR_cell));
                PetscCall(DMPlexPointLocalRead(dmAux, cells[0], a_ptr, &aL_cell)); 
                PetscCall(DMPlexPointLocalRead(dmAux, cells[1], a_ptr, &aR_cell));

                if (g_ptr) { 
                    PetscCall(DMPlexPointLocalRead(dmGrad, cells[0], g_ptr, &gL_cell)); 
                    PetscCall(DMPlexPointLocalRead(dmGrad, cells[1], g_ptr, &gR_cell)); 
                }
                if (g_aux_ptr) { 
                    PetscCall(DMPlexPointLocalRead(dmGradAux, cells[0], g_aux_ptr, &gaL_cell)); 
                    PetscCall(DMPlexPointLocalRead(dmGradAux, cells[1], g_aux_ptr, &gaR_cell)); 
                }

                // MOOD Masking
                if (!cell_orders.empty()) {
                    if (cells[0] >= cStart && cells[0] < cEnd && cell_orders[cells[0] - cStart] == 1) { gL_cell = nullptr; gaL_cell = nullptr; }
                    if (cells[1] >= cStart && cells[1] < cEnd && cell_orders[cells[1] - cStart] == 1) { gR_cell = nullptr; gaR_cell = nullptr; }
                }

                PetscInt offL, offR; 
                PetscCall(PetscSectionGetOffset(secCell, cells[0], &offL)); 
                PetscCall(PetscSectionGetOffset(secCell, cells[1], &offR));
                const PetscFVCellGeom *cgL = (const PetscFVCellGeom*)&cGeom_ptr[offL]; 
                const PetscFVCellGeom *cgR = (const PetscFVCellGeom*)&cGeom_ptr[offR];
                
                // Reconstruct State (Q)
                PetscScalar qL_face[Model<T>::n_dof_q], qR_face[Model<T>::n_dof_q];
                const PetscScalar *minL=NULL, *maxL=NULL, *minR=NULL, *maxR=NULL;
                if (min_ptr) { 
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[0], min_ptr, &minL)); 
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[0], max_ptr, &maxL)); 
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[1], min_ptr, &minR)); 
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[1], max_ptr, &maxR)); 
                }
                reconstructor->Reconstruct(qL_cell, gL_cell, cgL->centroid, fg->centroid, minL, maxL, qL_face);
                reconstructor->Reconstruct(qR_cell, gR_cell, cgR->centroid, fg->centroid, minR, maxR, qR_face);
                
                // Reconstruct Aux (A)
                PetscScalar aL_face[Model<T>::n_dof_qaux], aR_face[Model<T>::n_dof_qaux];
                reconstructorAux->Reconstruct(aL_cell, gaL_cell, cgL->centroid, fg->centroid, nullptr, nullptr, aL_face);
                reconstructorAux->Reconstruct(aR_cell, gaR_cell, cgR->centroid, fg->centroid, nullptr, nullptr, aR_face);

                // Update Aux on Faces based on Q_face and A_face (Consistency)
                auto res_aL = Model<T>::update_aux_variables(qL_face, aL_face, parameters.data());
                for(int i=0; i<Model<T>::n_dof_qaux; ++i) aL_face[i] = res_aL[i];
                auto res_aR = Model<T>::update_aux_variables(qR_face, aR_face, parameters.data());
                for(int i=0; i<Model<T>::n_dof_qaux; ++i) aR_face[i] = res_aR[i];

                SimpleArray<T, Model<T>::n_dof_q> flux; 
                for(int i=0; i<Model<T>::n_dof_q; ++i) flux[i] = 0.0;
                
                if (cons_flux_kernel) flux = cons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);

                PetscInt offFL, offFR; 
                PetscCall(PetscSectionGetOffset(sQ, cells[0], &offFL)); 
                PetscCall(PetscSectionGetOffset(sQ, cells[1], &offFR));
                PetscScalar *fL = (offFL >= 0 && offFL + Model<T>::n_dof_q <= size_f) ? &f_ptr[offFL] : nullptr;
                PetscScalar *fR = (offFR >= 0 && offFR + Model<T>::n_dof_q <= size_f) ? &f_ptr[offFR] : nullptr;

                if (noncons_flux_kernel) {
                    auto nc_stacked = noncons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);
                    auto* nc_into_right = nc_stacked.data; 
                    auto* nc_into_left = nc_stacked.data + Model<T>::n_dof_q;
                    for(int i=0; i<Model<T>::n_dof_q; ++i) { 
                        if (fL) fL[i] -= nc_into_left[i] * area; 
                        if (fR) fR[i] -= nc_into_right[i] * area; 
                    }
                }
                for(int i=0; i<Model<T>::n_dof_q; ++i) { 
                    if (fL) fL[i] -= flux[i] * area; 
                    if (fR) fR[i] += flux[i] * area; 
                }

            } else if (num_cells == 1) {
                // --- Boundary Face ---
                PetscInt tag_id; 
                PetscCall(DMLabelGetValue(label, f, &tag_id));
                
                if (boundary_map.count(tag_id)) {
                    PetscInt bc_idx = boundary_map.at(tag_id);
                    PetscInt cL = cells[0];
                    const PetscScalar *qL_cell, *gL_cell = NULL; 
                    const PetscScalar *aL_cell, *gaL_cell = NULL;
                    
                    PetscCall(DMPlexPointLocalRead(dmQ, cL, x_ptr, &qL_cell));
                    PetscCall(DMPlexPointLocalRead(dmAux, cL, a_ptr, &aL_cell));
                    
                    if (g_ptr) { PetscCall(DMPlexPointLocalRead(dmGrad, cL, g_ptr, &gL_cell)); }
                    if (g_aux_ptr) { PetscCall(DMPlexPointLocalRead(dmGradAux, cL, g_aux_ptr, &gaL_cell)); }

                    if (!cell_orders.empty()) { 
                        if (cL >= cStart && cL < cEnd && cell_orders[cL - cStart] == 1) { gL_cell = nullptr; gaL_cell = nullptr; } 
                    }
                    
                    PetscInt offL; PetscCall(PetscSectionGetOffset(secCell, cL, &offL));
                    const PetscFVCellGeom *cgL = (const PetscFVCellGeom*)&cGeom_ptr[offL];
                    
                    // Reconstruct State Left
                    PetscScalar qL_face[Model<T>::n_dof_q];
                    const PetscScalar *minL=NULL, *maxL=NULL;
                    if (min_ptr) { 
                        PetscCall(DMPlexPointLocalRead(dmQ, cL, min_ptr, &minL)); 
                        PetscCall(DMPlexPointLocalRead(dmQ, cL, max_ptr, &maxL)); 
                    }
                    reconstructor->Reconstruct(qL_cell, gL_cell, cgL->centroid, fg->centroid, minL, maxL, qL_face);

                    // Reconstruct Aux Left
                    PetscScalar aL_face[Model<T>::n_dof_qaux];
                    reconstructorAux->Reconstruct(aL_cell, gaL_cell, cgL->centroid, fg->centroid, nullptr, nullptr, aL_face);

                    // Update Left Aux Consistency
                    auto res_aL = Model<T>::update_aux_variables(qL_face, aL_face, parameters.data());
                    for(int i=0; i<Model<T>::n_dof_qaux; ++i) aL_face[i] = res_aL[i];

                    // Compute Boundary Values
                    auto qR_arr = Model<T>::boundary_conditions(bc_idx, qL_face, aL_face, n_hat, fg->centroid, time, 0.0);
                    PetscScalar *qR_face = qR_arr.data;
                    auto aR_arr = Model<T>::aux_boundary_conditions(bc_idx, qL_face, aL_face, n_hat, fg->centroid, time, 0.0);
                    PetscScalar *aR_face = aR_arr.data;

                    SimpleArray<T, Model<T>::n_dof_q> flux; 
                    for(int i=0; i<Model<T>::n_dof_q; ++i) flux[i] = 0.0;
                    if (cons_flux_kernel) flux = cons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);

                    PetscInt offFL; PetscCall(PetscSectionGetOffset(sQ, cL, &offFL));
                    PetscScalar *fL = (offFL >= 0 && offFL + Model<T>::n_dof_q <= size_f) ? &f_ptr[offFL] : nullptr;

                    if (noncons_flux_kernel) {
                        auto nc_stacked = noncons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);
                        auto* nc_into_left  = nc_stacked.data + Model<T>::n_dof_q; 
                        if (fL) { for(int i=0; i<Model<T>::n_dof_q; ++i) fL[i] -= nc_into_left[i] * area; }
                    }
                    if (fL) { for(int i=0; i<Model<T>::n_dof_q; ++i) fL[i] -= flux[i] * area; }
                }
            }
        }
        
        // Volume Normalization
        for(PetscInt c=cStart; c<cEnd; ++c) {
             PetscInt off; PetscCall(PetscSectionGetOffset(secCell, c, &off)); 
             const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cGeom_ptr[off];
             PetscInt offF; PetscCall(PetscSectionGetOffset(sQ, c, &offF));
             PetscScalar *f_cell = (offF >= 0 && offF + Model<T>::n_dof_q <= size_f) ? &f_ptr[offF] : nullptr;
             
             if (f_cell && cg->volume > 1e-15) { 
                 for(int i=0; i<Model<T>::n_dof_q; ++i) f_cell[i] /= cg->volume; 
             }
        }
        
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

    PetscErrorCode ComputeExplicitSource(Vec X_loc, Vec A_loc, Vec F_loc) { return PETSC_SUCCESS; }
};
#endif