#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

struct SolverSettings {
    double t_end = 1.0;
    double cfl = 0.5;
};

struct IOSettings {
    std::string directory = "output";
    std::string filename = "sol";
    int snapshots = 10;
    std::string snapshot_logic = "snap"; // Default: "snap", "loose", "interpolate"
    bool clean_directory = false;
    std::string mesh_path = "";
    bool restart = false;
    std::string restart_file = "";
    std::string initial_condition_file = "";
};

struct Settings {
    std::string name;
    IOSettings io;
    SolverSettings solver;

    static Settings from_json(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Error: Could not open settings file: " << path << std::endl;
            exit(1);
        }
        json j;
        f >> j;

        Settings s;
        s.name = j.value("name", "Simulation");

        if (j.contains("io")) {
            auto& jio = j["io"];
            s.io.directory = jio.value("directory", "output");
            s.io.filename = jio.value("filename", "sol");
            s.io.snapshots = jio.value("snapshots", 10);
            s.io.snapshot_logic = jio.value("snapshot_logic", "snap"); 
            s.io.clean_directory = jio.value("clean_directory", false);
            s.io.mesh_path = jio.value("mesh_path", "");
            // ... (legacy restart fields if needed)
        }

        if (j.contains("solver")) {
            auto& jsol = j["solver"];
            s.solver.t_end = jsol.value("t_end", 1.0);
            s.solver.cfl = jsol.value("cfl", 0.5);
        }

        return s;
    }
};

#endif