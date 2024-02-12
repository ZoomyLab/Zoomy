#include "define.h"
#include "mesh.h"
#include <vector>

intArr2 create_neighbor_index_iteration_list(const Mesh& mesh)
{
    std::vector<std::tuple<int, int>> element_neighbor_tuple;
    for (int element = 0; element < mesh.n_elements; ++element)
        {
            for (int i_neighbor = 0; i_neighbor < mesh.element_n_neighbors(element); ++i_neighbor)
            {
                int neighbor = mesh.element_neighbors(element, i_neighbor);
                if (element < neighbor)
                {
                    element_neighbor_tuple.push_back(std::make_tuple(element, i_neighbor));
                }
            }
        }
    intArr2 face_iteration_list = intArr2("face_iteration_list", element_neighbor_tuple.size(), 2);
    for (int i = 0; i < element_neighbor_tuple.size(); ++i)
    {
        face_iteration_list(i, 0) = std::get<0>(element_neighbor_tuple[i]);
        face_iteration_list(i, 1) = std::get<1>(element_neighbor_tuple[i]);
    }
    return face_iteration_list;
}