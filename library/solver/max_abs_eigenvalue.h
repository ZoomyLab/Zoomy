#include "define.h"
#include "model.h"
#include "mesh.h"

#include <cmath>

#ifndef MAX_ABS_EIGENVALUE_H
#define MAX_ABS_EIGENVALUE_H

double max_abs(const realArr arr) 
{
    double maxabs = abs(arr(0));
    for (int i = 1; i < arr.extent(0); ++i)
    {
        maxabs = max(maxabs, abs(arr(i)));
    }
    return maxabs;
}

double max_abs_eigenvalue(const realArr2 Q, const realArr2 Qaux, const realArr param, const intArr2 faces,  Model& model, const Mesh& mesh)
{
    realArr eigenvalues = realArr("eigenvalues", Q.extent(1));
    int i = 0;
    int element = faces(0, 0);
    int face = faces(0, 1);

    model.eigenvalues(get_element2(Q, element), get_element2(Q, element), param, get_normal(mesh.element_face_normals,element, face), eigenvalues);
    std::cout << param(0) << std::endl;
    std::cout << Q(0,0) << std::endl;
    std::cout << eigenvalues(0) << " " << eigenvalues(1) << " " << eigenvalues(2) << " " << eigenvalues(3) << " " << eigenvalues(4) << " " << eigenvalues(5) << " " << std::endl;
    double max_abs_ev = max_abs(eigenvalues);
    for (int i = 0; i < faces.extent(0); ++i)
    {
        element = faces(i, 0);
        face = faces(i, 1);
        model.eigenvalues(get_element2(Q, element), get_element2(Q, element), param, get_normal(mesh.element_face_normals,element, face), eigenvalues);
        // std::cout << eigenvalues(0) << " " << eigenvalues(1) << " " << eigenvalues(2) << " " << eigenvalues(3) << " " << eigenvalues(4) << " " << eigenvalues(5) << " " << std::endl;
        max_abs_ev = max(max_abs_ev, max_abs(eigenvalues));
    }
    return max_abs_ev;
}

#endif // MAX_ABS_EIGENVALUE_H