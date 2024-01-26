// mesh.h
#ifndef MESH_H
#define MESH_H

#include "hdf5.h"
#include <string>
#include <vector>
#include "helpers_hdf5.h"

class Mesh
{
public:
    int dimension;
    std::string type;
    int n_elements;
    int n_vertices;
    int n_boundary_elements;
    int n_faces_per_element;
    std::vector<std::vector<double>> vertex_coordinates;
    std::vector<std::vector<int>> element_vertices;
    std::vector<std::vector<double>> element_face_areas;
    std::vector<std::vector<double>> element_center;
    std::vector<double> element_volume;
    std::vector<double> element_inradius;
    std::vector<std::vector<std::vector<double>>> element_face_normals;
    std::vector<int> element_n_neighbors;
    std::vector<std::vector<int>> element_neighbors;
    std::vector<std::vector<int>> element_neighbors_face_index;
    std::vector<std::vector<int>> boundary_face_vertices;
    std::vector<int> boundary_face_corresponding_element;
    std::vector<int> boundary_face_element_face_index;
    std::vector<int> boundary_face_tag;
    std::vector<std::string> boundary_tag_names;
    std::vector<int> boundary_function_index;
    std::vector<int> boundary_function_required_element;
    std::vector<std::string> boundary_function_name;

    Mesh(std::string filepath)
    {
	    hid_t file = openHdf5(filepath);
        readInt(file, "dimension", this->dimension);
        readString(file, "type", this->type);
        readInt(file, "n_elements", this->n_elements);
        readInt(file, "n_boundary_elements", this->n_boundary_elements);
        readInt(file, "n_faces_per_element", this->n_faces_per_element);
        readDouble2dArray(file, "vertex_coordinates", this->vertex_coordinates);
        readInt2dArray(file, "element_vertices", this->element_vertices);
        readDouble2dArray(file, "element_face_areas", this->element_face_areas);
        readDouble2dArray(file, "element_center", this->element_center);
        readDoubleArray(file, "element_volume", this->element_volume);
        readDoubleArray(file, "element_inradius", this->element_inradius);
        readDouble3dArray(file, "element_face_normals", this->element_face_normals);
        readIntArray(file, "element_n_neighbors", this->element_n_neighbors);
        readInt2dArray(file, "element_neighbors", this->element_neighbors);
        readInt2dArray(file, "element_neighbors_face_index", this->element_neighbors_face_index);
        readInt2dArray(file, "boundary_face_vertices", this->boundary_face_vertices);
        readIntArray(file, "boundary_face_corresponding_element", this->boundary_face_corresponding_element);
        readIntArray(file, "boundary_face_element_face_index", this->boundary_face_element_face_index);
        readIntArray(file, "boundary_face_tag", this->boundary_face_tag);
        readStringArray(file, "boundary_tag_names", this->boundary_tag_names);
        if (H5Lexists(file, "boundary_function_index", H5P_DEFAULT))
        {
            readIntArray(file, "boundary_function_index", this->boundary_function_index);
            readIntArray(file, "required_elements", this->boundary_function_required_element);
            readStringArray(file, "boundary_function_name", this->boundary_function_name);
        }
        else
        {
            std::cerr << "Boundary Conditions related data not present. You can append the data by using the function - model.boundary_conditions.append_boundary_map_to_mesh_hdf5(settings.output_dir) - in the pyhon API." << std::endl;
        }
        
        H5Fclose(file);
    }
};


#endif // MESH_H