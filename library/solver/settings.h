// settings.h
#ifndef SETTINGS_H
#define SETTINGS_H

#include "hdf5.h"
#include "define.h"
#include <string>
#include "helpers_hdf5.h"

class Settings
{
public:
    std::string name;
    realArr parameters;
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
    std::vector<std::string> callbacks;

    Settings(std::string filepath)
    {
	    hid_t file = openHdf5(filepath);
        readString(file, "name", this->name);
        readDoubleArray(file, "parameter_values", this->parameters);
        readDouble(file, "time_end", this->time_end);
        readString(file, "output_dir", this->output_dir);
        readInt(file, "output_snapshots", this->output_snapshots);
        readBool(file, "output_write_all", this->output_write_all);
        readBool(file, "output_clean_dir", this->output_clean_dir);
        readBool(file, "truncate_last_time_step", this->truncate_last_time_step);
        readStringArray(file, "callbacks", this->callbacks);
        H5Fclose(file);
    }
};


#endif // SETTINGS_H