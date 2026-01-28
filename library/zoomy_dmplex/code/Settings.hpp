#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <map> 
#include "nlohmann/json.hpp" 

using json = nlohmann::json;

struct SolverSettings {
    double t_end = 1.0;
    double cfl = 0.5;
    bool use_deep_adjacency = true; 
    int reconstruction_order = 1;   
    double min_dt = 1.0e-12;        
};

struct IOSettings {
    std::string directory = "output";
    std::string filename = "sol";
    int snapshots = 10;
    std::string snapshot_logic = "snap"; 
    bool clean_directory = false;
    std::string mesh_path = "";
    std::string mesh_label = "Face Sets"; 
    bool restart = false;
    std::string restart_file = "";
    std::string initial_condition_file = "";
};

// --- NEW STRUCT ---
struct ModelSettings {
    std::map<std::string, double> parameters; 
};
// ------------------

struct Settings {
    std::string name;
    IOSettings io;
    SolverSettings solver;
    ModelSettings model; // Added member

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
            s.io.mesh_label = jio.value("mesh_label", "Face Sets"); 
        }

        if (j.contains("solver")) {
            auto& jsol = j["solver"];
            s.solver.t_end = jsol.value("t_end", 1.0);
            s.solver.cfl = jsol.value("cfl", 0.5);
            s.solver.use_deep_adjacency = jsol.value("use_deep_adjacency", true); 
            s.solver.reconstruction_order = jsol.value("reconstruction_order", 1); 
            s.solver.min_dt = jsol.value("min_dt", 1.0e-12);
        }

        // --- NEW Parsing Logic ---
        if (j.contains("model")) {
            if (j["model"].contains("parameters")) {
                for (auto& [key, val] : j["model"]["parameters"].items()) {
                    if (val.is_number()) {
                        s.model.parameters[key] = val.get<double>();
                    }
                }
            }
        }
        // -------------------------

        return s;
    }
};

#endif