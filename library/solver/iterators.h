#include "define.h"
#include "mesh.h"
#include <vector>

intArr2 create_face_iteration_list(Mesh& mesh)
{
    std::vector<std::tuple<int, int>> faces;
    for (int element = 0; element < mesh.n_elements; ++element)
        {
            for (int i_neighbor = 0; i_neighbor < mesh.element_n_neighbors(element); ++i_neighbor)
            {
                int neighbor = mesh.element_neighbors(element, i_neighbor);
                if (element < neighbor);
                {
                    faces.push_back(std::make_tuple(element, mesh.element_neighbors_face_index(element, i_neighbor)));
                }
            }
        }
    intArr2 face_iteration_list = intArr2("face_iteration_list", faces.size(), 2);
    for (int i = 0; i < faces.size(); ++i)
    {
        face_iteration_list(i, 0) = std::get<0>(faces[i]);
        face_iteration_list(i, 1) = std::get<1>(faces[i]);

    }
    return face_iteration_list;
}