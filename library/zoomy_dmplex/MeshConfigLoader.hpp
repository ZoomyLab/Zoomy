#ifndef MESH_CONFIG_LOADER_H
#define MESH_CONFIG_LOADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <stdexcept>
#include <algorithm>

class MeshConfigLoader {
public:
    /**
     * Parses a GMSH .msh file to extract Physical Names.
     * Returns a map: "Physical Name" -> Tag ID (e.g., "inflow" -> 3002)
     */
    static std::map<std::string, int> loadBoundaryMapping(const std::string& msh_filename) {
        std::ifstream file(msh_filename);
        std::map<std::string, int> mapping;
        std::string line;

        if (!file.is_open()) {
            throw std::runtime_error("MeshConfigLoader: Could not open mesh file: " + msh_filename);
        }

        bool inPhysicalSection = false;

        while (std::getline(file, line)) {
            // 1. Detect start of PhysicalNames section
            if (line.find("$PhysicalNames") != std::string::npos) {
                inPhysicalSection = true;
                if (std::getline(file, line)) { continue; } // Skip the count line
            }

            // 2. Detect end of section
            if (line.find("$EndPhysicalNames") != std::string::npos) {
                break;
            }

            // 3. Parse lines: dimension tag "name"
            if (inPhysicalSection) {
                std::stringstream ss(line);
                int dim, tag;
                
                // Read dimension and tag (e.g., 2 3002)
                if (!(ss >> dim >> tag)) continue;

                // Read the rest of the line as the name
                std::string raw_name;
                std::getline(ss, raw_name); 

                // Extract string between quotes
                size_t first_quote = raw_name.find('\"');
                size_t last_quote = raw_name.rfind('\"');

                if (first_quote != std::string::npos && last_quote != std::string::npos && last_quote > first_quote) {
                    std::string clean_name = raw_name.substr(first_quote + 1, last_quote - first_quote - 1);
                    mapping[clean_name] = tag;
                }
            }
        }
        
        return mapping;
    }
};

#endif // MESH_CONFIG_LOADER_H