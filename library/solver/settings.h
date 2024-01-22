// Settings.h
#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>
#include <vector>

class Settings
{
public:
    const std::string name;
    const std::vector<float> parameters;
    const std::string reconstruction;
    const std::string reconstruction_edge;
    const std::string numerical_flux;
    const std::string compute_dt;
    const float time_end;
    const bool truncate_last_time_step;
    const int output_snapshots;
    const bool output_write_all;
    const std::string output_dir;

    Settings(const std::string &filename);
};

#endif // SETTINGS_H