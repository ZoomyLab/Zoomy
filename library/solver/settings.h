// Settings.h
#ifndef SETTINGS_H
#define SETTINGS_H

#include "hdf5.h"
#include <string>
#include <vector>
#include "fileIO.h"

class Settings
{
public:
    std::string name;
    std::vector<double> parameters;
    // std::string reconstruction;
    // std::string reconstruction_edge;
    // std::string numerical_flux;
    // std::string compute_dt;
    double time_end;
    bool truncate_last_time_step;
    int output_snapshots;
    bool output_write_all;
    bool output_clean_dir;
    std::string output_dir;

    Settings(std::string filepath)
    {
	    hid_t file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) 
        {
            std::cerr << "Error opening file: " << filepath << std::endl;
        } 
        else 
        {
            std::cout << "File opened successfully." << std::endl;

        readStringFromDataset(file, "name", this->name);
        readDoubleArrayFromDataset(file, "parameter_values", this->parameters);
        readDoubleFromDataset(file, "time_end", this->time_end);
        readStringFromDataset(file, "output_dir", this->output_dir);
        readIntFromDataset(file, "output_snapshots", this->output_snapshots);
        readBoolFromDataset(file, "output_write_all", this->output_write_all);
        readBoolFromDataset(file, "output_clean_dir", this->output_clean_dir);
        readBoolFromDataset(file, "truncate_last_time_step", this->truncate_last_time_step);

        H5Fclose(file);
        }
    }
};


#endif // SETTINGS_H