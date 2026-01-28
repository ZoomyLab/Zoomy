#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "VirtualSolver.hpp" 

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
public:
    void Reconstruct(const T* q_cell, const T* grad_cell, const T* cell_centroid, const T* face_centroid,
                     const T* q_min, const T* q_max, T* q_face_out) override {
        T dx[3];
        for(int d=0; d<Model<T>::dimension; ++d) dx[d] = face_centroid[d] - cell_centroid[d];
        
        for(int i=0; i<Model<T>::n_dof_q; ++i) {
            T delta = 0.0;
            for(int d=0; d<Model<T>::dimension; ++d) delta += grad_cell[i * Model<T>::dimension + d] * dx[d];
            q_face_out[i] = q_cell[i] + delta;
        }

        // 1. Positivity Limiter
        if (q_face_out[1] < 1e-9) {
            T h_c = q_cell[1]; T h_f = q_face_out[1];
            T theta = 1.0;
            if (std::abs(h_f - h_c) > 1e-12) theta = (1e-9 - h_c) / (h_f - h_c);
            theta = std::max((T)0.0, std::min((T)1.0, theta));
            for(int i=0; i<Model<T>::n_dof_q; ++i) q_face_out[i] = q_cell[i] + theta * (q_face_out[i] - q_cell[i]);
        }
        
        // 2. TVD Limiter
        if (q_min && q_max) {
            for (int i=0; i<Model<T>::n_dof_q; ++i) {
                T dq = q_face_out[i] - q_cell[i];
                if (std::abs(dq) < 1e-12) continue;
                T alpha = 1.0;
                if (dq > 0) {
                    if (q_max[i] > q_cell[i]) alpha = std::min((T)1.0, (q_max[i] - q_cell[i]) / dq); else alpha = 0.0;
                } else {
                    if (q_min[i] < q_cell[i]) alpha = std::min((T)1.0, (q_min[i] - q_cell[i]) / dq); else alpha = 0.0;
                }
                if (alpha < 1.0) q_face_out[i] = q_cell[i] + alpha * dq;
            }
        }
    }
};
#endif