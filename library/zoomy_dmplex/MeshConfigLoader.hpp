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
    static std::map<std::string, std::pair<int, int>> loadBoundaryMapping(const std::string& msh_filename) {
        std::ifstream file(msh_filename);
        std::map<std::string, std::pair<int, int>> mapping;
        std::string line;
        if (!file.is_open()) throw std::runtime_error("Could not open mesh file: " + msh_filename);

        bool inPhysicalSection = false;
        while (std::getline(file, line)) {
            if (line.find("$PhysicalNames") != std::string::npos) {
                inPhysicalSection = true;
                std::getline(file, line); continue;
            }
            if (line.find("$EndPhysicalNames") != std::string::npos) break;

            if (inPhysicalSection) {
                std::stringstream ss(line);
                int dim, tag;
                if (!(ss >> dim >> tag)) continue;
                std::string raw_name;
                std::getline(ss, raw_name); 
                size_t first = raw_name.find('\"'), last = raw_name.rfind('\"');
                if (first != std::string::npos && last != std::string::npos) {
                    mapping[raw_name.substr(first + 1, last - first - 1)] = {dim, tag};
                }
            }
        }
        return mapping;
    }
};
#endif