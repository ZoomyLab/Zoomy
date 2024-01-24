// settings.h
#ifndef SETTINGS_H
#define SETTINGS_H

#include "hdf5.h"
#include <string>
#include <vector>
#include "helpers_hdf5.h"

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
    std::string output_dir;
    int output_snapshots;
    bool output_write_all;
    bool output_clean_dir;

    Settings(std::string filepath)
    {
	    hid_t file = openHdf5(filepath);
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
};


#endif // SETTINGS_H