#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "VirtualSolver.hpp"
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

template <typename T>
class Limiter {
public:
    virtual ~Limiter() = default;
    // Added n_comp to support variable component sizes
    virtual void ComputeAlphas(const T* q_cell, const T* q_face_unlimited, 
                               const T* q_min, const T* q_max, int n_comp, T* alphas_out) = 0;
};

// 1. Vacuum Limiter
template <typename T>
class VacuumLimiter : public Limiter<T> {
    T tol;
public:
    VacuumLimiter(T tolerance = 1e-2) : tol(tolerance) {}
    void ComputeAlphas(const T* q_cell, const T*, const T* q_min, const T*, int n_comp, T* alphas_out) override {
        for(int i=0; i<n_comp; ++i) alphas_out[i] = 1.0;
        // Assuming index 1 is always height/depth if it exists and n_comp > 1
        if (n_comp > 1 && q_min && q_min[1] < tol) {
            for(int i=0; i<n_comp; ++i) alphas_out[i] = 0.0;
        }
    }
};

// 2. Positivity Limiter
template <typename T>
class PositivityLimiter : public Limiter<T> {
    T tol;
public:
    PositivityLimiter(T tolerance = 1e-9) : tol(tolerance) {}
    void ComputeAlphas(const T* q_cell, const T* q_face_unlimited, const T*, const T*, int n_comp, T* alphas_out) override {
        for(int i=0; i<n_comp; ++i) alphas_out[i] = 1.0;
        
        if (n_comp > 1) {
            T h_f = q_face_unlimited[1]; 
            if (h_f < tol) {
                T h_c = q_cell[1];       
                T theta = 1.0;
                if (std::abs(h_f - h_c) > 1e-12) theta = (tol - h_c) / (h_f - h_c);
                theta = std::max((T)0.0, std::min((T)1.0, theta));
                for(int i=0; i<n_comp; ++i) alphas_out[i] = theta;
            }
        }
    }
};

// 3. TVD Limiter
template <typename T>
class TVDLimiter : public Limiter<T> {
public:
    void ComputeAlphas(const T* q_cell, const T* q_face_unlimited, const T* q_min, const T* q_max, int n_comp, T* alphas_out) override {
        for(int i=0; i<n_comp; ++i) alphas_out[i] = 1.0;
        if (!q_min || !q_max) return; // Cannot limit without bounds
        for (int i=0; i<n_comp; ++i) {
            T dq = q_face_unlimited[i] - q_cell[i];
            if (std::abs(dq) < 1e-12) continue;
            T alpha = 1.0;
            if (dq > 0) {
                if (q_max[i] > q_cell[i]) alpha = std::min((T)1.0, (q_max[i] - q_cell[i]) / dq);
                else alpha = 0.0; 
            } else {
                if (q_min[i] < q_cell[i]) alpha = std::min((T)1.0, (q_min[i] - q_cell[i]) / dq);
                else alpha = 0.0;
            }
            alphas_out[i] = alpha;
        }
    }
};

template <typename T>
class Reconstructor {
protected:
    int n_comp; 
public:
    // Default to n_dof_q but allow override
    Reconstructor(int n = Model<T>::n_dof_q) : n_comp(n) {}
    virtual ~Reconstructor() = default;
    virtual void Reconstruct(const T* q_cell, const T* grad_cell, const T* cell_centroid, const T* face_centroid,
                             const T* q_min, const T* q_max, T* q_face_out) = 0;
};

template <typename T>
class PCMReconstructor : public Reconstructor<T> {
public:
    PCMReconstructor(int n = Model<T>::n_dof_q) : Reconstructor<T>(n) {}
    void Reconstruct(const T* q_cell, const T*, const T*, const T*, const T*, const T*, T* q_face_out) override {
        for(int i=0; i<this->n_comp; ++i) q_face_out[i] = q_cell[i];
    }
};

template <typename T>
class LinearReconstructor : public Reconstructor<T> {
private:
    std::vector<std::unique_ptr<Limiter<T>>> limiters;

public:
    // Constructor specifying components (explicit)
    LinearReconstructor(int n, bool use_limiters = true) : Reconstructor<T>(n) {
        if (use_limiters) {
            limiters.push_back(std::make_unique<VacuumLimiter<T>>());
            limiters.push_back(std::make_unique<TVDLimiter<T>>());
        }
    }

    // Default constructor (defaults to Model<T>::n_dof_q)
    LinearReconstructor(bool use_limiters = true) : Reconstructor<T>(Model<T>::n_dof_q) {
        if (use_limiters) {
            limiters.push_back(std::make_unique<VacuumLimiter<T>>());
            limiters.push_back(std::make_unique<TVDLimiter<T>>());
        }
    }

    void Reconstruct(const T* q_cell, const T* grad_cell, const T* cell_centroid, const T* face_centroid,
                     const T* q_min, const T* q_max, T* q_face_out) override {
        
        T dx[3];
        for(int d=0; d<Model<T>::dimension; ++d) dx[d] = face_centroid[d] - cell_centroid[d];
        
        // We assume Model<T>::n_dof_q is the maximum number of components we handle efficiently on stack.
        // If n_comp > n_dof_q, this is unsafe, but standard usage implies n_comp <= n_dof_q.
        T q_face_unlimited[Model<T>::n_dof_q];
        T delta[Model<T>::n_dof_q];

        for(int i=0; i<this->n_comp; ++i) {
            delta[i] = 0.0;
            if (grad_cell) {
                for(int d=0; d<Model<T>::dimension; ++d) {
                    delta[i] += grad_cell[i * Model<T>::dimension + d] * dx[d];
                }
            }
            q_face_unlimited[i] = q_cell[i] + delta[i];
        }

        T final_alphas[Model<T>::n_dof_q];
        for(int i=0; i<this->n_comp; ++i) final_alphas[i] = 1.0;

        T temp_alphas[Model<T>::n_dof_q];
        for (const auto& limiter : limiters) {
            limiter->ComputeAlphas(q_cell, q_face_unlimited, q_min, q_max, this->n_comp, temp_alphas);
            for(int i=0; i<this->n_comp; ++i) {
                if (temp_alphas[i] < final_alphas[i]) final_alphas[i] = temp_alphas[i];
            }
        }

        auto alpha = 1.;
        for(int i=0; i<this->n_comp; ++i) {
            if (alpha > final_alphas[i]) alpha = final_alphas[i];
        }

        for(int i=0; i<this->n_comp; ++i) {
            q_face_out[i] = q_cell[i] + alpha * delta[i];
        }
    }
};
#endif