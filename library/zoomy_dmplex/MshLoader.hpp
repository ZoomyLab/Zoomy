#ifndef MSH_LOADER_HPP
#define MSH_LOADER_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>

struct MshNode {
    int id;
    double x, y, z;
};

struct MshData {
    std::string name;
    std::map<int, double> values; // NodeID -> Value
};

class MshLoader {
private:
    std::vector<MshNode> nodes;
    std::map<std::string, MshData> fields;
    bool loaded = false;

public:
    void Load(const std::string& filename) {
        if (loaded) return;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[Error] MshLoader: Could not open " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line == "$Nodes") ReadNodes(file);
            if (line == "$NodeData") ReadNodeData(file);
        }
        loaded = true;
        std::cout << "[Info] MshLoader: Loaded " << nodes.size() << " nodes and " << fields.size() << " data fields from " << filename << std::endl;
    }

    // Interpolate value at (x,y) by averaging the 'k' nearest nodes (k=3 for triangles)
    double Interpolate(const std::string& field_name, double x, double y, int k_nearest = 3) {
        if (fields.find(field_name) == fields.end()) return 0.0;
        const auto& data_map = fields[field_name].values;

        // Simple linear search for nearest neighbors (Optimization: Use KDTree if N > 100k)
        // For N ~ 10k-50k, this is fast enough for initialization (run once).
        
        std::vector<std::pair<double, int>> neighbors; // {dist_sq, node_index}
        neighbors.reserve(nodes.size());

        for (size_t i = 0; i < nodes.size(); ++i) {
            double dx = nodes[i].x - x;
            double dy = nodes[i].y - y;
            double dist_sq = dx*dx + dy*dy;
            neighbors.push_back({dist_sq, (int)i});
        }

        // Partial sort to get top k
        size_t keep = std::min((size_t)k_nearest, neighbors.size());
        std::partial_sort(neighbors.begin(), neighbors.begin() + keep, neighbors.end());

        double num = 0.0;
        double den = 0.0;
        
        for (size_t i = 0; i < keep; ++i) {
            int node_idx = neighbors[i].second;
            int node_id = nodes[node_idx].id;
            double d = std::sqrt(neighbors[i].first);
            
            // Inverse Distance Weighting (IDW)
            double w = 1.0 / (d + 1e-12);
            
            if (data_map.count(node_id)) {
                num += w * data_map.at(node_id);
                den += w;
            }
        }

        return (den > 0.0) ? num / den : 0.0;
    }

    bool HasField(const std::string& name) const {
        return fields.find(name) != fields.end();
    }

private:
    void ReadNodes(std::ifstream& file) {
        std::string line;
        // Simple logic to handle Gmsh 2 and 4 minimally
        // Gmsh 4 has blocks, Gmsh 2 is flat.
        // We rely on standard text structure.
        int num_nodes = 0;
        std::getline(file, line); 
        std::stringstream ss(line); 
        // Gmsh 4 often has multiple nums on first line, Gmsh 2 has just one.
        // Just parsing lines until $EndNodes is safer.
        
        while (std::getline(file, line)) {
            if (line == "$EndNodes") break;
            std::stringstream lss(line);
            int id;
            double x, y, z;
            // Try to parse "id x y z" (Gmsh 2 style)
            // Gmsh 4 blocks might have different headers, but node lines look similar.
            if (lss >> id >> x >> y >> z) {
                nodes.push_back({id, x, y, z});
            }
        }
    }

    void ReadNodeData(std::ifstream& file) {
        std::string line;
        MshData data;
        
        // 1. Tags
        int num_string_tags;
        std::getline(file, line); // Num string tags
        std::stringstream ss(line);
        if (!(ss >> num_string_tags)) return; 

        if (num_string_tags > 0) {
            std::getline(file, line); // Name is usually the first tag
            size_t first = line.find('"');
            size_t last = line.rfind('"');
            if (first != std::string::npos && last != std::string::npos) {
                data.name = line.substr(first + 1, last - first - 1);
            } else {
                data.name = line; // Fallback
            }
        }

        // Skip other tags until we hit values
        // Format: 
        // ...
        // number of values
        // id value
        
        while (std::getline(file, line)) {
            if (line == "$EndNodeData") break;
            std::stringstream lss(line);
            int id; 
            double val;
            // Heuristic: if line has 2 numbers, it's data. If 1, it's count.
            if (lss >> id >> val) {
                data.values[id] = val;
            }
        }
        
        if (!data.name.empty() && !data.values.empty()) {
            fields[data.name] = data;
        }
    }
};

#endif