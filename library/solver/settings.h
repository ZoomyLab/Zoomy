// Settings.h
#ifndef SETTINGS_H
#define SETTINGS_H

#include "hdf5.h"
#include <string>
#include <vector>

class Settings
{
public:
    std::string name;
    std::vector<float> parameters;
    std::string reconstruction;
    std::string reconstruction_edge;
    std::string numerical_flux;
    std::string compute_dt;
    float time_end;
    bool truncate_last_time_step;
    int output_snapshots;
    bool output_write_all;
    bool output_clean_dir;
    std::string output_dir;

    Settings(std::string filepath)
    {
        // // Open the HDF5 file
        // H5::H5File file(filename, H5F_ACC_RDONLY);

        // // Read the 'name' attribute
        // H5::Attribute attr = file.openAttribute("name");
        // H5::StrType strtype = attr.getStrType();
        // attr.read(strtype, this->name);

        // // Read the 'parameters' dataset
        // H5::DataSet dataset = file.openDataSet("parameters");
        // H5::DataSpace dataspace = dataset.getSpace();
        // hsize_t length = dataspace.getSimpleExtentNpoints();
        // this->parameters.resize(length);
        // dataset.read(this->parameters.data(), H5::PredType::NATIVE_FLOAT);

	    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) 
        {
            std::cerr << "Error opening file: " << filepath << std::endl;
        } 
        else 
        {
            std::cout << "File opened successfully." << std::endl;

		//   hid_t group = H5Gopen(file, "parameter_values", H5P_DEFAULT);
		hid_t dataset = H5Dopen(file, "output_clean_dir", H5P_DEFAULT);
		bool output_clean_dir;
		H5Dread(dataset, H5T_NATIVE_HBOOL, H5S_ALL, H5S_ALL, H5P_DEFAULT, &output_clean_dir);
		std::cout << "output_clean_dir: " << output_clean_dir << std::endl;

        H5Fclose(file);
        }
    }
};

#endif // SETTINGS_H