// mesh.h
#ifndef MESH_H
#define MESH_H

#include "hdf5.h"
#include <string>
#include <vector>
#include "fileIO.h"

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

    Mesh(std::string filepath)
    {
	    hid_t file = loadHDF5(filepath);
        readIntFromDataset(file, "dimension", this->dimension);
        readStringFromDataset(file, "type", this->type);
        readIntFromDataset(file, "n_elements", this->n_elements);
        readIntFromDataset(file, "n_boundary_elements", this->n_boundary_elements);
        readIntFromDataset(file, "n_faces_per_element", this->n_faces_per_element);
        readDouble2dArrayFromDataset(file, "vertex_coordinates", this->vertex_coordinates);
        readInt2dArrayFromDataset(file, "element_vertices", this->element_vertices);
        readDouble2dArrayFromDataset(file, "element_face_areas", this->element_face_areas);
        readDouble2dArrayFromDataset(file, "element_center", this->element_center);
        readDoubleArrayFromDataset(file, "element_volume", this->element_volume);
        readDoubleArrayFromDataset(file, "element_inradius", this->element_inradius);
        readDouble3dArrayFromDataset(file, "element_face_normals", this->element_face_normals);
        readIntArrayFromDataset(file, "element_n_neighbors", this->element_n_neighbors);
        readInt2dArrayFromDataset(file, "element_neighbors", this->element_neighbors);
        readInt2dArrayFromDataset(file, "element_neighbors_face_index", this->element_neighbors_face_index);
        readInt2dArrayFromDataset(file, "boundary_face_vertices", this->boundary_face_vertices);
        readIntArrayFromDataset(file, "boundary_face_corresponding_element", this->boundary_face_corresponding_element);
        readIntArrayFromDataset(file, "boundary_face_element_face_index", this->boundary_face_element_face_index);
        readIntArrayFromDataset(file, "boundary_face_tag", this->boundary_face_tag);
        readStringArrayFromDataset(file, "boundary_tag_names", this->boundary_tag_names);
        H5Fclose(file);
    }
};


#endif // MESH_H