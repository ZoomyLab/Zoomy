#ifndef SOLVER_SETTINGS_HPP
#define SOLVER_SETTINGS_HPP

#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Settings {
    struct IO {
        // Output Settings
        std::string directory = "output";
        std::string filename = "simulation";
        int snapshots = 10;
        bool clean_directory = false;

        // Input Settings
        std::string mesh_path = "mesh.msh";
        std::string initial_condition_file = "";
        
        // Restart Settings
        bool restart = false;
        std::string restart_file = "";
    } io;

    struct Solver {
        double t_end = 1.0;
        double cfl = 0.5;
    } solver;

    static Settings from_json(const std::string& path) {
        std::ifstream f(path);
        if (!f.good()) return Settings(); 
        
        json data = json::parse(f);
        Settings s;

        if (data.contains("io")) {
            auto io_block = data["io"];
            // Output
            s.io.directory = io_block.value("directory", "output");
            s.io.filename = io_block.value("filename", "simulation");
            s.io.snapshots = io_block.value("snapshots", 10);
            s.io.clean_directory = io_block.value("clean_directory", false);
            
            // Input
            s.io.mesh_path = io_block.value("mesh_path", "mesh.msh");
            s.io.initial_condition_file = io_block.value("initial_condition_file", "");
            
            // Restart
            s.io.restart = io_block.value("restart", false);
            s.io.restart_file = io_block.value("restart_file", "");
        }

        if (data.contains("solver")) {
            auto sol = data["solver"];
            s.solver.t_end = sol.value("t_end", 1.0);
            s.solver.cfl = sol.value("cfl", 0.5);
        }

        return s;
    }
};
#endif