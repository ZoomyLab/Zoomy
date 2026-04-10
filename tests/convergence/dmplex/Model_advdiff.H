#pragma once
#include <cmath>
#include <array>

#ifndef ZOOMY_SIMPLE_ARRAY
#define ZOOMY_SIMPLE_ARRAY
template <typename T, int N>
struct SimpleArray {
    T data[N];
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
    T* begin() { return data; }
    const T* begin() const { return data; }
    T* end() { return data + N; }
    const T* end() const { return data + N; }
};
#endif

#include <vector>
#include <string>

#ifdef __CUDACC__
#define PORTABLE_FN __host__ __device__
#else
#define PORTABLE_FN
#endif

/**
 * 2D Scalar Advection-Diffusion:  u_t + a . grad(u) = kappa * laplacian(u)
 *
 * Parameters: ax, ay, sigma, kappa  (defaults: 1.0, 1.0, 0.15, 0.01)
 * Single DOF: Q[0] = u
 *
 * Exact solution (Gaussian hump with diffusive spreading):
 *   u(x,y,t) = [sigma^2 / (sigma^2 + 2*kappa*t)]
 *            * exp(-((x-0.25-ax*t)^2+(y-0.25-ay*t)^2) / (2*(sigma^2+2*kappa*t)))
 *
 * has_diffusion = true activates diffusive_flux in TransportStep.hpp
 */
template <typename T>
struct Model {
    static constexpr int n_dof_q    = 1;
    static constexpr int n_dof_qaux = 0;
    static constexpr int n_parameters = 4;
    static constexpr int dimension  = 2;
    static constexpr int n_dof_gradQ = 2;   // n_dof_q * dimension
    static constexpr bool has_diffusion = true;
    static constexpr bool has_free_surface = false;
    static constexpr int idx_b = -1;
    static constexpr int idx_h = -1;
    static constexpr int n_boundary_tags = 1;

    static const std::vector<std::string> get_boundary_tags() { return { "wall" }; }
    static const std::vector<std::string> parameter_names() { return { "ax", "ay", "sigma", "kappa" }; }
    static const std::vector<T> default_parameters() { return { 1.0, 1.0, 0.15, 0.01 }; }

    PORTABLE_FN static inline SimpleArray<T, 1> flux(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    // Diffusive flux: kappa * (gradQ . n_hat)
    // TransportStep projects the gradient onto the face normal before calling this,
    // so gradQ[i] = d(Q_i)/dn (the normal derivative for DOF i).
    // For scalar advection-diffusion: diff_flux = kappa * du/dn
    PORTABLE_FN static inline SimpleArray<T, 1> diffusive_flux(
        const T* Q, const T* Qaux, const T* gradQ, const T* p)
    {
        T kappa = p[3];
        SimpleArray<T, 1> res;
        // gradQ[0] = du/dn (normal derivative, projected by TransportStep)
        res[0] = kappa * gradQ[0];
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> dflux(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> hydrostatic_pressure(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> nonconservative_matrix(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> quasilinear_matrix(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> eigenvalues(
        const T* Q, const T* Qaux, const T* p, const T* n)
    {
        SimpleArray<T, 1> res;
        res[0] = p[0] * n[0] + p[1] * n[1];
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> left_eigenvectors(
        const T* Q, const T* Qaux, const T* p, const T* n)
    {
        SimpleArray<T, 1> res;
        res[0] = 1;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> right_eigenvectors(
        const T* Q, const T* Qaux, const T* p, const T* n)
    {
        SimpleArray<T, 1> res;
        res[0] = 1;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> source(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> source_jacobian_wrt_variables(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 0> source_jacobian_wrt_aux_variables(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 0> res;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 2> project_2d_to_3d(
        const T* X, const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 2> res;
        res[0] = 0; res[1] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> project_3d_to_2d(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> residual(
        const T time, const T* X, const T dX,
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 0;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> interpolate(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = Q[0];
        return res;
    }

    // IC: Gaussian hump centered at (0.25, 0.25)
    PORTABLE_FN static inline SimpleArray<T, 1> initial_condition(
        const T* X, const T* p)
    {
        T sigma = p[2];
        T dx = X[0] - 0.25;
        T dy = X[1] - 0.25;
        SimpleArray<T, 1> res;
        res[0] = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 0> initial_aux_condition(
        const T* X, const T* p)
    {
        SimpleArray<T, 0> res;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> update_variables(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = Q[0];
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 0> update_aux_variables(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 0> res;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> update_variables_jacobian_wrt_variables(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 1> res;
        res[0] = 1;
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 0> update_aux_variables_jacobian_wrt_variables(
        const T* Q, const T* Qaux, const T* p)
    {
        SimpleArray<T, 0> res;
        return res;
    }

    // Wall BC: copy interior state
    PORTABLE_FN static inline SimpleArray<T, 1> boundary_conditions(
        const int bc_idx, const T* Q, const T* Qaux,
        const T* n, const T* X, const T time, const T dX)
    {
        SimpleArray<T, 1> res;
        res[0] = Q[0];
        return res;
    }

    PORTABLE_FN static inline SimpleArray<T, 1> aux_boundary_conditions(
        const int bc_idx, const T* Q, const T* Qaux,
        const T* n, const T* X, const T time, const T dX)
    {
        SimpleArray<T, 1> res;
        res[0] = Q[0];
        return res;
    }
};
