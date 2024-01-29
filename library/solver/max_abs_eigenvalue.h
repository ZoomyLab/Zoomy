#include "define.h"
#include "model.h"
#include "mesh.h"

#include <cmath>

double max_abs(realArr arr) 
{
    double max_abs = abs(arr(0));
    for (int i = 1; i < arr.extent(0); ++i)
    {
        max_abs = max(max_abs, abs(arr(i)));
    }
    return max_abs;
}

double max_abs_eigenvalue(realArr2 Q, realArr2 Qaux, realArr param, const realArr2 faces,  Model& model, Mesh& mesh)
{
    realArr eigenvalues = realArr("eigenvalues", Q.extent(1));
    int i = 0;
    int element = faces(0, 0);
    int face = faces(0, 1);

    model.eigenvalues(get_element2(Q, element), get_element2(Q, element), param, get_element_and_face(mesh.element_face_normals,element, face), eigenvalues);
    double max_abs_ev = max_abs(eigenvalues);
    for (int i = 0; i < faces.extent(0); ++i)
    {
        element = faces(i, 0);
        face = faces(i, 1);
        model.eigenvalues(get_element2(Q, element), get_element2(Q, element), param, get_element_and_face(mesh.element_face_normals,element, face), eigenvalues);
        max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));
    }
    return max_abs_ev;
}