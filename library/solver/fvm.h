#include<Kokkos_Core.hpp>

#include "define.h"
#include "model.h"
#include "boundary_conditions.h"
#include "mesh.h"
#include "numerical_flux.h"
#include "nonconservative_flux.h"
#include "ode_source.h"

#ifndef FVM_H
#define FVM_H

KOKKOS_INLINE_FUNCTION void numerical_flux(Model& model, const realArr qi, const realArr qj, const realArr qauxi, const realArr qauxj, const realArr param, const realArr normal, realArr& F)
{
    rusanov(qi, qj, qauxi, qauxj, param, normal, model, F);
}

KOKKOS_INLINE_FUNCTION void nonconservative_flux(Model& model, const realArr qi, const realArr qj, const realArr qauxi, const realArr qauxj, const realArr param, const realArr normal, realArr& NC)
{
    segmentpath(qi, qj, qauxi, qauxj, param, normal, model, NC);
}

void fvm_semidiscrete_split_step(const realArr2 Q, const realArr2 Qaux, const realArr param, const intArr2 element_neighbor_index_tuples, Model& model, const Mesh& mesh, const BoundaryConditions& boundary_conditions, realArr2& out)
{

    //TODO passing num_flux and nc_flux as lambdas with non-zero capture list does not work. Maybe I should use a class instead?
//     auto numerical_flux =  [model] __device__ (const realArr qi, const realArr qj, const realArr qauxi, const realArr qauxj, const realArr param, const realArr normal, realArr& F) -> void
//     {
//         rusanov(qi, qj, qauxi, qauxj, param, normal, model, F);
//     };

//    auto nonconservative_flux = [model] __device__  (const realArr& qi, const realArr& qj, const realArr& qauxi, const realArr& qauxj, const realArr& param, const realArr& normal, realArr& NC) -> void
//    {
//        segmentpath(qi, qj, qauxi, qauxj, param, normal, model, NC);
//    };
    const int dim = mesh.dimension;
    const int n_fields = Q.extent(1);
    const int n_aux_fields = Qaux.extent(1);
    const int n_elements = Q.extent(0);
    const int n_inner_faces = element_neighbor_index_tuples.extent(0);

    realArr eigenvalues = realArr("eigenvalues", n_fields);
    double max_abs_ev = -1.;

    out = realArr2("out", n_elements, n_fields);
    for (int i = 0; i < n_elements; ++i)
    {
        for (int j = 0; j < n_fields; ++j)
        {
            out(i, j) = 0.;
        }
    }


    // for (int i = 0; i < n_inner_faces; ++i)
    Kokkos::parallel_for("Space Solution Operator: inner edges", n_inner_faces, KOKKOS_LAMBDA (const int i)
    {
        int element;
        int i_neighbor;
        int face;
        int neighbor;
        realArr F = realArr("F", n_fields);
        realArr NC = realArr("NC", n_fields);
        realArr qi = realArr("qi", n_fields);
        realArr qj = realArr("qj", n_fields);
        realArr qauxi = realArr("qauxi", n_aux_fields);
        realArr qauxj = realArr("qauxj", n_aux_fields);
        realArr normal = realArr("normal", dim);
        // realArr normal;
        // realArr qi;
        // realArr qj;
        // realArr qauxi;
        // realArr qauxj;
        // realArr F = realArr("F", n_fields);
        // realArr NC = realArr("NC", n_fields);
        
        element = element_neighbor_index_tuples(i, 0);
        i_neighbor = element_neighbor_index_tuples(i, 1);
        neighbor = mesh.element_neighbors(element, i_neighbor);
        face = mesh.element_neighbors_face_index(element, i_neighbor);
        qi = get_element2(Q, element);
        qj = get_element2(Q, neighbor);
        qauxi = get_element2(Qaux, element);
        qauxj = get_element2(Qaux, neighbor);
        normal = get_normal(mesh.element_face_normals, element, face);

        // model.eigenvalues(qi, qauxi, param, normal, eigenvalues);
        // max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));
        // model.eigenvalues(qj, qauxj, param, normal, eigenvalues);
        // max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));

        // numerical_flux(model, qi, qj, qauxi, qauxj, param, normal, F);
        // nonconservative_flux(model, qi, qj, qauxi, qauxj, param, normal, NC);

        for (int i = 0; i < n_fields; ++i)
        {
            out(element, i) -= (F(i) + NC(i)) * mesh.element_face_areas(element, face) / mesh.element_volume(element);
            out(neighbor, i) += (F(i) - NC(i)) * mesh.element_face_areas(element, face) / mesh.element_volume(neighbor);
        }
    });


    // for (int i = 0; i < mesh.n_boundary_elements; ++i)
    Kokkos::parallel_for("Space Solution Operator: boundary edges", mesh.n_boundary_elements, KOKKOS_LAMBDA (const int i)
    {
        int element;
        int face;
        int neighbor;
        // realArr normal;
        // realArr qi;
        // realArr qj;
        // realArr qauxi
        // realArr qauxj;
        realArr F = realArr("F", n_fields);
        realArr NC = realArr("NC", n_fields);
        realArr qi = realArr("qi", n_fields);
        realArr qj = realArr("qj", n_fields);
        realArr qauxi = realArr("qauxi", n_aux_fields);
        realArr qauxj = realArr("qauxj", n_aux_fields);
        realArr normal = realArr("normal", dim);
        element = mesh.boundary_face_corresponding_element(i);
        face = mesh.boundary_face_element_face_index(i);
        neighbor = mesh.element_neighbors(element, face);

        normal = get_normal(mesh.element_face_normals, element, face);

        qi = get_element2(Q, element);
        qauxi = get_element2(Qaux, element);
        int i_boundary_function_index = mesh.boundary_function_index(i);
        realArr q_req = get_element2(Q, mesh.boundary_function_required_element(i));
        realArr qaux_req = get_element2(Q, mesh.boundary_function_required_element(i));

        boundary_conditions.apply(i_boundary_function_index, q_req, qaux_req, param, normal, qj);

        //TODO this should probably also be done using a dedicated boundary condition...
        qauxj = qauxi;

        // model.eigenvalues(qi, qauxi, param, normal, eigenvalues);
        // max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));
        // model.eigenvalues(qj, qauxj, param, normal, eigenvalues);
        // max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));

        // numerical_flux(model, qi, qj, qauxi, qauxj, param, normal, F);
        // nonconservative_flux(model, qi, qj, qauxi, qauxj, param, normal, NC);

        for (int i = 0; i < n_fields; ++i)
        {
            out(element, i) -= (F(i) + NC(i)) * mesh.element_face_areas(element, face) / mesh.element_volume(element);
        }
    });

}


class FluxSolutionOperator
{
    // TODO this should not hold the mesh or any other data!
    private:
        Model model;
        const BoundaryConditions boundary_conditions;
        const Mesh mesh;
        intArr2 element_neighbor_index_tuples;
        void (*method)(const realArr2, const realArr2, const realArr, const intArr2, Model&, const Mesh&, const BoundaryConditions&, realArr2&);

    public:
        FluxSolutionOperator(Model model, const BoundaryConditions boundary_conditions, const Mesh mesh, intArr2 element_neighbor_index_tuples, std::string method): model(model), boundary_conditions(boundary_conditions), mesh(mesh), element_neighbor_index_tuples(element_neighbor_index_tuples)
        {
            if (method == "fvm_semidiscrete_split_step")
            {
                this->method = fvm_semidiscrete_split_step;
            }
            else
            {
                throw std::invalid_argument("Invalid space solution method");
            }
        }
        ~FluxSolutionOperator() = default;

        void evaluate(const realArr2 Q, const realArr2 Qaux, const realArr param, realArr2& out)
        {
            this->method(Q, Qaux, param, this->element_neighbor_index_tuples, this->model, this->mesh, this->boundary_conditions, out);
        }
};

class SourceSolutionOperator 
{
    private:
        Model model;

    public:
        SourceSolutionOperator(Model model): model(model){}
        ~SourceSolutionOperator() = default;

        void evaluate(const realArr2 Q, const realArr2 Qaux, const realArr param, double dt, realArr2& out)
        {
            const int n_elements = Q.extent(0);
            const int n_fields = Q.extent(1);
            // for (int i = 0; i < n_elements; ++i)
            Kokkos::parallel_for("Source Solution Operator", n_elements, KOKKOS_LAMBDA (const int i)
            {
                realArr s = realArr("s", n_fields);
                // model.source(get_element2(Q, i), get_element2(Qaux, i), param, s);
                // integrator.evaluate(get_element2(Q, i), get_element2(Qaux, i), param, dt, s);
                SOURCE_INTEGRATOR(model, get_element2(Q, i), get_element2(Qaux, i), param, dt, s);
                for (int j = 0; j < n_fields; ++j)
                {
                    out(i, j) = s(j);
                }

            });
        }
};

#endif // FVM_H
