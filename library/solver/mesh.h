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
    std::vector<std::vector<double>> element_face_normals;
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
        // readStringFromDataset(file, "name", this->name);
        // readStringFromDataset(file, "output_dir", this->output_dir);
        // readIntFromDataset(file, "output_snapshots", this->output_snapshots);
        // readBoolFromDataset(file, "output_write_all", this->output_write_all);
        // readBoolFromDataset(file, "output_clean_dir", this->output_clean_dir);
        // readBoolFromDataset(file, "truncate_last_time_step", this->truncate_last_time_step);
        H5Fclose(file);
        std::cout << "Mesh dimension: " << this->dimension << std::endl;
    }
};


#endif // MESH_H