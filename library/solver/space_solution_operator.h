#include "define.h"
#include "model.h"
#include "mesh.h"

void fvm_semidiscrete_split_step(const realArr2 Q, const realArr2 Qaux, const realArr param, const intArr2 faces, Model& model, Mesh& mesh, realArr2& out)
{
    int element;
    int face´
    int neighbor;
    realArr eigenvalues = realArr("eigenvalues", Q.extent(1));
    double max_abs_ev = -1.;
    realArr normal;
    realArr qi;
    realArr qj;
    realArr qauxi;
    realArr qauxj;
    realArr F = realArr("F", Q.extent(1));
    realArr NC = realArr("NC", Q.extent(1));;


    for (int i = 0; i < faces.extent(0); ++i)
    {
        element = faces(i, 0);
        face = faces(i, 1);
        neighbor = mesh.element_neighbors(element, face);
        model.eigenvalues(get_element2(Q, element), get_element2(Q, element), param, , eigenvalues);
        max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));
        normal = get_element_and_face(mesh.element_face_normals, element, face);
        qi = get_element2(Q, element);
        qj = get_element2(Q, neighbor);
        qauxi = get_element2(Qaux, element);
        qauxj = get_element2(Qaux, neighbor);

        // numerical_flux(qi, qj, qauxi, qauxj, param, normal, model, F);
        // nonconservative_flux(qi, qj, qauxi, qauxj, param, normal, model, NC);

        

    }
}