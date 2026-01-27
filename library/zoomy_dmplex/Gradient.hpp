#ifndef GRADIENT_HPP
#define GRADIENT_HPP

#include <petsc.h>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include "Model.H"

/**
 * @brief Abstract base class for gradient computation strategies.
 */
template <typename T>
class GradientCalculator {
public:
    virtual ~GradientCalculator() = default;

    /**
     * @brief Setup any pre-computed data structures (e.g., LS weights).
     * @param dm The DM object (providing topology and geometry).
     */
    virtual PetscErrorCode Setup(DM dm) { return PETSC_SUCCESS; }

    /**
     * @brief Compute gradients for the state vector X.
     * * @param dmQ The DM associated with the state Q.
     * @param X_global The global state vector.
     * @param dmGrad The DM associated with the gradient (usually vector field).
     * @param G_global The output global gradient vector.
     * @param boundary_map Map of DMLabel tag_ids to boundary indices.
     */
    virtual PetscErrorCode Compute(DM dmQ, Vec X_global, DM dmGrad, Vec G_global, 
                                   const std::map<PetscInt, PetscInt>& boundary_map) = 0;
};

/**
 * @brief Green-Gauss Gradient Reconstruction.
 * * Computes gradient via face flux accumulation:
 * \int \nabla \phi dV = \oint \phi \mathbf{n} dA
 * * Includes fixes for:
 * 1. Parallel race conditions (calculates owned cells fully).
 * 2. Boundary faces (uses BCs to close the integral).
 */
template <typename T>
class GreenGaussGradient : public GradientCalculator<T> {
public:
    PetscErrorCode Compute(DM dmQ, Vec X_global, DM dmGrad, Vec G_global, 
                           const std::map<PetscInt, PetscInt>& boundary_map) override {
        // Zero the output first to ensure clean accumulation slate
        PetscCall(VecZeroEntries(G_global));

        // Get Local Vectors
        Vec X_loc; 
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        
        Vec G_loc; 
        PetscCall(DMGetLocalVector(dmGrad, &G_loc)); 
        PetscCall(VecZeroEntries(G_loc));

        // Access Data
        const PetscScalar *x_ptr; 
        PetscCall(VecGetArrayRead(X_loc, &x_ptr)); 
        PetscScalar *g_ptr; 
        PetscCall(VecGetArray(G_loc, &g_ptr));

        // Geometry
        Vec cellGeom, faceGeom; 
        PetscCall(DMPlexGetGeometryFVM(dmQ, &faceGeom, &cellGeom, NULL));
        const PetscScalar *fGeom_ptr, *cGeom_ptr; 
        PetscCall(VecGetArrayRead(faceGeom, &fGeom_ptr)); 
        PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));

        // Sections and Topology
        DM dmFace; 
        PetscCall(VecGetDM(faceGeom, &dmFace)); 
        PetscSection secFace; 
        PetscCall(DMGetLocalSection(dmFace, &secFace));
        
        PetscInt fStart, fEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 1, &fStart, &fEnd));
        
        PetscInt dim = Model<T>::dimension;
        PetscSection secIn; 
        PetscCall(DMGetLocalSection(dmQ, &secIn)); 
        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscInt n_comp; 
        PetscCall(PetscSectionGetDof(secIn, cStart, &n_comp));
        
        DMLabel label; 
        PetscCall(DMGetLabel(dmQ, "Face Sets", &label));

        // --- 1. Loop over Faces: Accumulate Fluxes ---
        for (PetscInt f = fStart; f < fEnd; ++f) {
            PetscInt off; 
            PetscCall(PetscSectionGetOffset(secFace, f, &off)); 
            const PetscFVFaceGeom *fg = (const PetscFVFaceGeom*)&fGeom_ptr[off];
            
            const PetscInt *cells; 
            PetscInt num_cells; 
            PetscCall(DMPlexGetSupportSize(dmQ, f, &num_cells)); 
            PetscCall(DMPlexGetSupport(dmQ, f, &cells));
            
            if (num_cells == 2) {
                // Internal Face
                PetscInt gL_idx, gR_idx;
                PetscCall(DMPlexGetPointGlobal(dmQ, cells[0], &gL_idx, NULL));
                PetscCall(DMPlexGetPointGlobal(dmQ, cells[1], &gR_idx, NULL));
                bool ownedL = (gL_idx >= 0);
                bool ownedR = (gR_idx >= 0);

                const PetscScalar *qL, *qR; 
                PetscCall(DMPlexPointLocalRead(dmQ, cells[0], x_ptr, &qL)); 
                PetscCall(DMPlexPointLocalRead(dmQ, cells[1], x_ptr, &qR));
                
                PetscScalar *gL, *gR; 
                PetscCall(DMPlexPointLocalRef(dmGrad, cells[0], g_ptr, &gL)); 
                PetscCall(DMPlexPointLocalRef(dmGrad, cells[1], g_ptr, &gR));
                
                for(int i=0; i<n_comp; ++i) {
                    PetscScalar face_val = 0.5 * (qL[i] + qR[i]);
                    for(int d=0; d<dim; ++d) { 
                        PetscScalar val = face_val * fg->normal[d]; 
                        // Only accumulate if the cell is owned by this rank
                        if(gL && ownedL) gL[i*dim + d] += val; 
                        if(gR && ownedR) gR[i*dim + d] -= val; 
                    }
                }
            } else if (num_cells == 1) {
                // Boundary Face
                PetscInt tag_id; 
                PetscCall(DMLabelGetValue(label, f, &tag_id));
                if (boundary_map.count(tag_id)) {
                    PetscInt bc_idx = boundary_map.at(tag_id);
                    PetscInt c = cells[0];
                    PetscInt g_idx; 
                    PetscCall(DMPlexGetPointGlobal(dmQ, c, &g_idx, NULL));
                    
                    if (g_idx >= 0) { // Only if owned
                        const PetscScalar *qL; 
                        PetscCall(DMPlexPointLocalRead(dmQ, c, x_ptr, &qL));
                        
                        PetscScalar n_hat[3];
                        PetscReal area = 0; 
                        for(int d=0; d<dim; ++d) area += fg->normal[d]*fg->normal[d]; 
                        area = std::sqrt(area);
                        for(int d=0; d<dim; ++d) n_hat[d] = fg->normal[d]/area;
                        
                        // Use BC function to get state at boundary face
                        // Passing nullptr for Aux as it might not be needed for simple BCs, or extend if necessary
                        auto q_bc = Model<T>::boundary_conditions(bc_idx, qL, nullptr, n_hat, fg->centroid, 0.0, 0.0);
                        
                        PetscScalar *gL; 
                        PetscCall(DMPlexPointLocalRef(dmGrad, c, g_ptr, &gL));
                        if (gL) {
                            for(int i=0; i<n_comp; ++i) {
                                for(int d=0; d<dim; ++d) {
                                    gL[i*dim + d] += q_bc[i] * fg->normal[d];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // --- 2. Loop over Cells: Divide by Volume ---
        DM dmCell; 
        PetscCall(VecGetDM(cellGeom, &dmCell)); 
        PetscSection secCell; 
        PetscCall(DMGetLocalSection(dmCell, &secCell));
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
            PetscInt g_idx; 
            PetscCall(DMPlexGetPointGlobal(dmQ, c, &g_idx, NULL));
            if (g_idx < 0) continue; // Skip ghost cells

            PetscInt off; 
            PetscCall(PetscSectionGetOffset(secCell, c, &off)); 
            const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cGeom_ptr[off];
            
            if (cg->volume > 1e-15) { 
                PetscScalar *gc; 
                PetscCall(DMPlexPointLocalRef(dmGrad, c, g_ptr, &gc)); 
                if (gc) { 
                    for(int k=0; k < n_comp * dim; ++k) gc[k] /= cg->volume; 
                } 
            }
        }
        
        // Restore arrays
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr)); 
        PetscCall(VecRestoreArray(G_loc, &g_ptr));
        PetscCall(VecRestoreArrayRead(faceGeom, &fGeom_ptr)); 
        PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr));
        
        // --- 3. Synchronization ---
        // Use INSERT_VALUES because we computed the FULL gradient for owned cells locally.
        PetscCall(DMLocalToGlobalBegin(dmGrad, G_loc, INSERT_VALUES, G_global)); 
        PetscCall(DMLocalToGlobalEnd(dmGrad, G_loc, INSERT_VALUES, G_global));
        
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); 
        PetscCall(DMRestoreLocalVector(dmGrad, &G_loc));
        return PETSC_SUCCESS;
    }
};

/**
 * @brief Least Squares Gradient Reconstruction.
 * * Reconstructs gradient by minimizing error over neighbor stencil.
 * Supports weighted least squares.
 */
template <typename T>
class LeastSquaresGradient : public GradientCalculator<T> {
private:
    int order;
    bool initialized = false;
    
    // Pre-computed weights and topology
    std::vector<PetscInt> ls_offsets;
    std::vector<PetscInt> ls_neighbors;
    std::vector<PetscReal> ls_weights;

    // Helper: Cholesky decomposition for small SPD systems (Normal Equations)
    bool SolveSmallSystemSPD(int N, std::vector<double>& A, std::vector<double>& b) {
        std::vector<double> L(N * N, 0.0);
        // Cholesky Decomposition
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                for (int k = 0; k < j; k++) sum += L[i * N + k] * L[j * N + k];
                
                if (i == j) {
                    double val = A[i * N + i] - sum;
                    if (val <= 0) return false; // Not SPD
                    L[i * N + j] = std::sqrt(val);
                } else {
                    L[i * N + j] = (A[i * N + j] - sum) / L[j * N + j];
                }
            }
        }
        // Forward Substitution
        for (int i = 0; i < N; i++) {
            double sum = 0;
            for (int k = 0; k < i; k++) sum += L[i * N + k] * b[k];
            b[i] = (b[i] - sum) / L[i * N + i];
        }
        // Backward Substitution
        for (int i = N - 1; i >= 0; i--) {
            double sum = 0;
            for (int k = i + 1; k < N; k++) sum += L[k * N + i] * b[k];
            b[i] = (b[i] - sum) / L[i * N + i];
        }
        return true;
    }

public:
    LeastSquaresGradient(int order = 1) : order(order), initialized(false) {}

    PetscErrorCode Setup(DM dm) override {
        if (initialized) return PETSC_SUCCESS;

        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
        PetscInt dim = Model<T>::dimension;
        
        Vec cellGeom; 
        PetscCall(DMPlexGetGeometryFVM(dm, NULL, &cellGeom, NULL));
        const PetscScalar *cGeom_ptr; 
        PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));
        
        DM dmCell; 
        PetscCall(VecGetDM(cellGeom, &dmCell)); 
        PetscSection secCell; 
        PetscCall(DMGetLocalSection(dmCell, &secCell));
        
        ls_offsets.clear(); 
        ls_neighbors.clear(); 
        ls_weights.clear(); 
        ls_offsets.push_back(0);
        
        int n_basis = dim; 
        if (order == 2) n_basis = (dim == 2) ? 5 : 9; // Full quadratic basis size

        for(PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt num_adj = -1; 
            PetscInt *adj = NULL; 
            PetscCall(DMPlexGetAdjacency(dm, c, &num_adj, &adj));
            
            std::vector<PetscInt> valid_neighs; 
            std::vector<std::vector<double>> basis_vecs;
            
            PetscInt off_c; 
            PetscCall(PetscSectionGetOffset(secCell, c, &off_c)); 
            const PetscFVCellGeom *cg_c = (const PetscFVCellGeom*)&cGeom_ptr[off_c];
            
            for(int k=0; k<num_adj; ++k) {
                PetscInt n = adj[k]; 
                if (n == c || n < 0) continue; 
                
                PetscInt off_n; 
                PetscCall(PetscSectionGetOffset(secCell, n, &off_n)); 
                if (off_n < 0) continue; // Should not happen with localized coordinates
                
                const PetscFVCellGeom *cg_n = (const PetscFVCellGeom*)&cGeom_ptr[off_n];
                
                valid_neighs.push_back(n);
                
                double dx = cg_n->centroid[0] - cg_c->centroid[0]; 
                double dy = cg_n->centroid[1] - cg_c->centroid[1]; 
                double dz = (dim == 3) ? (cg_n->centroid[2] - cg_c->centroid[2]) : 0.0;
                
                std::vector<double> b; 
                b.push_back(dx); 
                b.push_back(dy); 
                if (dim == 3) b.push_back(dz);
                
                if (order == 2) { 
                    b.push_back(0.5*dx*dx); 
                    b.push_back(0.5*dy*dy); 
                    if (dim == 3) b.push_back(0.5*dz*dz); 
                    b.push_back(dx*dy); 
                    if (dim == 3) { b.push_back(dy*dz); b.push_back(dz*dx); } 
                }
                basis_vecs.push_back(b);
            }
            PetscCall(PetscFree(adj));
            
            // Build Normal Equations: (A^T A) w = A^T e_d
            int N = n_basis; 
            std::vector<double> ATA(N * N, 0.0);
            for(const auto& b : basis_vecs) { 
                for(int i=0; i<N; ++i) { 
                    for(int j=0; j<N; ++j) ATA[i*N + j] += b[i] * b[j]; 
                } 
            }
            
            // For each neighbor, we calculate the weight vector
            for(size_t k=0; k<valid_neighs.size(); ++k) {
                ls_neighbors.push_back(valid_neighs[k]);
                
                // RHS is basically the basis vector for this neighbor (A^T column)
                // We want to solve for weights such that sum(w_k * du_k) ~ grad u * dx
                std::vector<double> rhs = basis_vecs[k]; 
                std::vector<double> sys = ATA; // Copy for solver
                
                bool ok = SolveSmallSystemSPD(N, sys, rhs);
                
                if (ok) { 
                    for(double w : rhs) ls_weights.push_back(w); 
                } else { 
                    // Fallback (e.g. 0 weights) if system ill-conditioned
                    for(int i=0; i<N; ++i) ls_weights.push_back(0.0); 
                }
            }
            ls_offsets.push_back(ls_neighbors.size());
        }
        
        PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr)); 
        initialized = true; 
        return PETSC_SUCCESS;
    }

    PetscErrorCode Compute(DM dmQ, Vec X_global, DM dmGrad, Vec G_global, 
                           const std::map<PetscInt, PetscInt>& boundary_map) override {
        if (!initialized) PetscCall(Setup(dmQ));
        
        Vec X_loc; 
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        
        Vec G_loc; 
        PetscCall(DMGetLocalVector(dmGrad, &G_loc)); 
        PetscCall(VecZeroEntries(G_loc));
        
        const PetscScalar *x_ptr; 
        PetscCall(VecGetArrayRead(X_loc, &x_ptr)); 
        PetscScalar *g_ptr; 
        PetscCall(VecGetArray(G_loc, &g_ptr));
        
        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd)); 
        PetscInt dim = Model<T>::dimension;
        
        PetscSection sec; 
        PetscCall(DMGetLocalSection(dmQ, &sec)); 
        PetscInt n_comp; 
        PetscCall(PetscSectionGetDof(sec, cStart, &n_comp));
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
            // --- CRITICAL CHECK: Skip Ghost Cells ---
            PetscInt g_idx; 
            PetscCall(DMPlexGetPointGlobal(dmQ, c, &g_idx, NULL));
            if (g_idx < 0) continue; 

            PetscScalar *gc; 
            PetscCall(DMPlexPointLocalRef(dmGrad, c, g_ptr, &gc)); 
            const PetscScalar *qc; 
            PetscCall(DMPlexPointLocalRead(dmQ, c, x_ptr, &qc));
            
            if (gc && qc) {
                int idx_start = ls_offsets[c - cStart]; 
                int idx_end = ls_offsets[c - cStart + 1]; 
                int n_basis = (order == 2) ? ((dim==2)?5:9) : dim;
                
                for(int k=idx_start; k<idx_end; ++k) {
                    PetscInt n = ls_neighbors[k]; 
                    const PetscScalar *qn; 
                    PetscCall(DMPlexPointLocalRead(dmQ, n, x_ptr, &qn));
                    
                    const double *w = &ls_weights[k * n_basis];
                    
                    for(int i=0; i<n_comp; ++i) { 
                        PetscScalar du = qn[i] - qc[i]; 
                        for(int d=0; d<dim; ++d) {
                            // w[d] contains the coefficient for the d-th derivative
                            gc[i*dim + d] += w[d] * du; 
                        }
                    }
                }
            }
        }
        
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr)); 
        PetscCall(VecRestoreArray(G_loc, &g_ptr));
        
        // --- Communication: Use INSERT_VALUES ---
        PetscCall(DMLocalToGlobalBegin(dmGrad, G_loc, INSERT_VALUES, G_global)); 
        PetscCall(DMLocalToGlobalEnd(dmGrad, G_loc, INSERT_VALUES, G_global));
        
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); 
        PetscCall(DMRestoreLocalVector(dmGrad, &G_loc));
        return PETSC_SUCCESS;
    }
};

#endif // GRADIENT_HPP