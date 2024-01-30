#include "define.h"
#include "model.h"
#include "mesh.h"
#include "numerical_flux.h"
#include "nonconservative_flux.h"

#ifndef SPACE_SOLUTION_OPERATOR_H
#define SPACE_SOLUTION_OPERATOR_H

void fvm_semidiscrete_split_step(const realArr2 Q, const realArr2 Qaux, const realArr param, const intArr2 faces, Model& model, Mesh& mesh,realArr2& out)
{

    //TODO passing num_flux and nc_flux as lambdas with non-zero capture list does not work. Maybe I should use a class instead?
    auto numerical_flux = [&model](const realArr qi, const realArr qj, const realArr qauxi, const realArr qauxj, const realArr param, const realArr normal, realArr& F) -> void
    {
        rusanov(qi, qj, qauxi, qauxj, param, normal, model, F);
    };

    auto nonconservative_flux = [&model](const realArr& qi, const realArr& qj, const realArr& qauxi, const realArr& qauxj, const realArr& param, const realArr& normal, realArr& NC) -> void
    {
        segmentpath(qi, qj, qauxi, qauxj, param, normal, model, NC);
    };
    const int dim = mesh.dimension;
    const int n_fields = Q.extent(1);
    const int n_elements = Q.extent(0);
    int element;
    int face;
    int neighbor;
    realArr eigenvalues = realArr("eigenvalues", n_fields);
    double max_abs_ev = -1.;
    realArr normal;
    realArr qi;
    realArr qj;
    realArr qauxi;
    realArr qauxj;
    realArr F = realArr("F", n_fields);
    realArr NC = realArr("NC", n_fields);
    out = realArr2("out", n_elements, n_fields);


    for (int i = 0; i < n_elements; ++i)
    {
        element = faces(i, 0);
        face = faces(i, 1);
        neighbor = mesh.element_neighbors(element, face);

        qi = get_element2(Q, element);
        qj = get_element2(Q, neighbor);
        qauxi = get_element2(Qaux, element);
        qauxj = get_element2(Qaux, neighbor);
        normal = get_element_and_face(mesh.element_face_normals, element, face);

        // model.eigenvalues(qi, qauxi, param, normal, eigenvalues);
        // max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));
        // model.eigenvalues(qj, qauxj, param, normal, eigenvalues);
        // max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));

        numerical_flux(qi, qj, qauxi, qauxj, param, normal, F);
        nonconservative_flux(qi, qj, qauxi, qauxj, param, normal, NC);

        for (int i = 0; i < n_fields; ++i)
        {
            out(element, i) -= (F(i) + NC(i)) * mesh.element_face_areas(element, face) / mesh.element_volume(element);
            out(neighbor, i) += (F(i) - NC(i)) * mesh.element_face_areas(element, face) / mesh.element_volume(neighbor);
        }

    }
}

class OdeOperator
{
    public:
        virtual void evaluate(const realArr2 Q, const realArr2 Qaux, const realArr param, realArr2 &out) = 0;
        virtual ~OdeOperator() = default;
};

class SpaceSolutionOperator : public OdeOperator
{
    private:
        Model model;
        Mesh mesh;
        intArr2 faces;
        void (*method)(const realArr2, const realArr2, const realArr, const intArr2, Model&, Mesh&, realArr2&);

    public:
        SpaceSolutionOperator(Model model, Mesh mesh, intArr2 faces, std::string method): model(model), mesh(mesh), faces(faces)
        {
            if (method == "fvm_semidiscrete_split_step")
            {
                this->method = fvm_semidiscrete_split_step;
            }
            else
            {
                this->method = fvm_semidiscrete_split_step;
            }
        }
        ~SpaceSolutionOperator() = default;

        void evaluate(const realArr2 Q, const realArr2 Qaux, const realArr param, realArr2& out)
        {
            this->method(Q, Qaux, param, this->faces, this->model, this->mesh, out);
        }
};

#endif // SPACE_SOLUTION_OPERATOR_H