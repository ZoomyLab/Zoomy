#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "VirtualSolver.hpp"
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

// ============================================================================
//  Limiter Interface
// ============================================================================
template <typename T>
class Limiter {
public:
    virtual ~Limiter() = default;

    /**
     * Compute limiting factors (alphas) for the reconstruction.
     * * @param q_cell           The cell average state Q_i
     * @param q_face_unlimited The unlimited reconstructed state at the face (Q_i + \nabla Q \cdot r)
     * @param q_min            Min bounds in neighborhood (for TVD/Vacuum)
     * @param q_max            Max bounds in neighborhood (for TVD)
     * @param alphas_out       Output array [n_dof]. Implementations should WRITE the proposed alpha
     * (0.0 = PCM, 1.0 = Unlimited) into this array.
     */
    virtual void ComputeAlphas(const T* q_cell, const T* q_face_unlimited, 
                               const T* q_min, const T* q_max, T* alphas_out) = 0;
};

// ============================================================================
//  Concrete Limiters
// ============================================================================

// 1. Vacuum Limiter: Forces 1st Order (PCM) near wet/dry fronts
template <typename T>
class VacuumLimiter : public Limiter<T> {
    T tol;
public:
    VacuumLimiter(T tolerance = 1e-2) : tol(tolerance) {}

    void ComputeAlphas(const T* q_cell, const T*, const T* q_min, const T*, T* alphas_out) override {
        // Default: No limiting (1.0)
        for(int i=0; i<Model<T>::n_dof_q; ++i) alphas_out[i] = 1.0;

        // Condition: If cell or ANY neighbor is effectively dry (h < tol)
        // We assume Index 1 is depth (h) based on your model
        if (q_min && q_min[1] < tol) {
            // Force alpha = 0.0 (Revert to PCM) for ALL components to preserve stability
            for(int i=0; i<Model<T>::n_dof_q; ++i) alphas_out[i] = 0.0;
        }
    }
};

// 2. Positivity Limiter: Ensures depth h >= 0
template <typename T>
class PositivityLimiter : public Limiter<T> {
    T tol;
public:
    PositivityLimiter(T tolerance = 1e-9) : tol(tolerance) {}

    void ComputeAlphas(const T* q_cell, const T* q_face_unlimited, const T*, const T*, T* alphas_out) override {
        for(int i=0; i<Model<T>::n_dof_q; ++i) alphas_out[i] = 1.0;

        T h_f = q_face_unlimited[1]; // Depth at face
        if (h_f < tol) {
            T h_c = q_cell[1];       // Depth at cell center
            T theta = 1.0;
            
            // Calculate scaling to exactly hit 'tol' instead of going negative
            if (std::abs(h_f - h_c) > 1e-12) {
                theta = (tol - h_c) / (h_f - h_c);
            }
            theta = std::max((T)0.0, std::min((T)1.0, theta));
            
            // Scale ALL components by this theta to preserve vector consistency
            for(int i=0; i<Model<T>::n_dof_q; ++i) alphas_out[i] = theta;
        }
    }
};

// 3. TVD Limiter (Barth-Jespersen)
template <typename T>
class TVDLimiter : public Limiter<T> {
public:
    void ComputeAlphas(const T* q_cell, const T* q_face_unlimited, const T* q_min, const T* q_max, T* alphas_out) override {
        for(int i=0; i<Model<T>::n_dof_q; ++i) alphas_out[i] = 1.0;

        // TVD requires neighborhood bounds
        if (!q_min || !q_max) return;

        for (int i=0; i<Model<T>::n_dof_q; ++i) {
            T dq = q_face_unlimited[i] - q_cell[i];
            
            // If gradient is effectively zero, no limiting needed
            if (std::abs(dq) < 1e-12) continue;

            T alpha = 1.0;
            if (dq > 0) {
                // Overshoot check
                if (q_max[i] > q_cell[i]) 
                    alpha = std::min((T)1.0, (q_max[i] - q_cell[i]) / dq);
                else 
                    alpha = 0.0; // Max is smaller than Cell (violation), clamp
            } else {
                // Undershoot check
                if (q_min[i] < q_cell[i]) 
                    alpha = std::min((T)1.0, (q_min[i] - q_cell[i]) / dq);
                else 
                    alpha = 0.0;
            }
            alphas_out[i] = alpha;
        }
    }
};

// ============================================================================
//  Reconstructors
// ============================================================================

template <typename T>
class Reconstructor {
public:
    virtual ~Reconstructor() = default;
    virtual void Reconstruct(const T* q_cell, const T* grad_cell, const T* cell_centroid, const T* face_centroid,
                             const T* q_min, const T* q_max, T* q_face_out) = 0;
};

template <typename T>
class PCMReconstructor : public Reconstructor<T> {
public:
    void Reconstruct(const T* q_cell, const T*, const T*, const T*, const T*, const T*, T* q_face_out) override {
        for(int i=0; i<Model<T>::n_dof_q; ++i) q_face_out[i] = q_cell[i];
    }
};

template <typename T>
class LinearReconstructor : public Reconstructor<T> {
private:
    // Stack of independent limiters
    std::vector<std::unique_ptr<Limiter<T>>> limiters;

public:
    LinearReconstructor() {
        // Default Configuration: Stack them in order of "Criticality"
        // 1. Vacuum (Strictest safety)
        limiters.push_back(std::make_unique<VacuumLimiter<T>>());
        // 2. Positivity (Physical validity)
        // limiters.push_back(std::make_unique<PositivityLimiter<T>>());
        // 3. TVD (Oscillation control)
        limiters.push_back(std::make_unique<TVDLimiter<T>>());
    }

    // Allow custom configuration if needed
    void ClearLimiters() { limiters.clear(); }
    void AddLimiter(std::unique_ptr<Limiter<T>> l) { limiters.push_back(std::move(l)); }

    void Reconstruct(const T* q_cell, const T* grad_cell, const T* cell_centroid, const T* face_centroid,
                     const T* q_min, const T* q_max, T* q_face_out) override {
        
        // 1. Compute Unlimited Update (Delta)
        T dx[3];
        for(int d=0; d<Model<T>::dimension; ++d) dx[d] = face_centroid[d] - cell_centroid[d];
        
        T q_face_unlimited[Model<T>::n_dof_q];
        T delta[Model<T>::n_dof_q];

        for(int i=0; i<Model<T>::n_dof_q; ++i) {
            delta[i] = 0.0;
            for(int d=0; d<Model<T>::dimension; ++d) {
                delta[i] += grad_cell[i * Model<T>::dimension + d] * dx[d];
            }
            q_face_unlimited[i] = q_cell[i] + delta[i];
        }

        // 2. Initialize Final Alphas to 1.0 (Unlimited)
        T final_alphas[Model<T>::n_dof_q];
        for(int i=0; i<Model<T>::n_dof_q; ++i) final_alphas[i] = 1.0;

        // 3. Stack Limiters (Intersection of constraints)
        T temp_alphas[Model<T>::n_dof_q];
        for (const auto& limiter : limiters) {
            limiter->ComputeAlphas(q_cell, q_face_unlimited, q_min, q_max, temp_alphas);
            
            // Take the MINIMUM alpha for each component
            for(int i=0; i<Model<T>::n_dof_q; ++i) {
                if (temp_alphas[i] < final_alphas[i]) {
                    final_alphas[i] = temp_alphas[i];
                }
            }
        }

        // 4. Apply Final Reconstruction
        for(int i=0; i<Model<T>::n_dof_q; ++i) {
            q_face_out[i] = q_cell[i] + final_alphas[i] * delta[i];
        }
    }
};

#endif