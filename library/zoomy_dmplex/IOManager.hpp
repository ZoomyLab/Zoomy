#ifndef IOMANAGER_HPP
#define IOMANAGER_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <petscvec.h>
#include <petscviewer.h>
#include "Settings.hpp"

namespace fs = std::filesystem;

class IOManager {
private:
    Settings& settings;
    PetscMPIInt rank;
    
    struct SnapInfo { 
        std::string filename; 
        PetscReal time; 
    };
    std::vector<SnapInfo> snapshots_log;
    
    PetscReal dt_snap;
    PetscReal next_snap;
    PetscInt snapshot_idx;

public:
    IOManager(Settings& s, PetscMPIInt r) : settings(s), rank(r) {
        dt_snap = settings.solver.t_end / settings.io.snapshots;
        next_snap = 0.0; 
        snapshot_idx = 0;
    }

    void PrepareDirectory() {
        if (rank == 0) {
            fs::path outDir(settings.io.directory);
            if (!fs::exists(outDir)) {
                fs::create_directories(outDir);
            } else if (settings.io.clean_directory) {
                PetscPrintf(PETSC_COMM_SELF, "[INFO] Cleaning output directory: %s\n", settings.io.directory.c_str());
                for (const auto& entry : fs::directory_iterator(outDir)) {
                    fs::remove_all(entry.path());
                }
            }
        }
        MPI_Barrier(PETSC_COMM_WORLD);
    }

    // Returns true if the solver should write output at this time
    // Updates next_snap automatically based on logic
    bool ShouldWrite(PetscReal time, bool force = false) {
        if (force) return true;

        // Tolerance to avoid floating point misses
        PetscReal tol = 1e-9 * dt_snap; 

        if (time >= next_snap - tol) {
            return true;
        }
        return false;
    }

    // Handles the logic of advancing the snapshot counter
    // Returns the time we *should* have written at (for interpolation targets)
    void AdvanceSnapshot() {
        if (settings.io.snapshot_logic == "snap") {
            next_snap += dt_snap;
        }
        else if (settings.io.snapshot_logic == "loose") {
            // If we missed multiple, jump ahead
            while(next_snap <= next_snap + dt_snap) next_snap += dt_snap; // Simple advance
        }
        else if (settings.io.snapshot_logic == "interpolate") {
            next_snap += dt_snap;
        }
        else {
             next_snap += dt_snap; // Default
        }
        snapshot_idx++;
    }

    // Calculates the maximum dt allowed to hit the next snapshot exactly (if snapping)
    PetscReal GetDtLimit(PetscReal time) const {
        if (settings.io.snapshot_logic == "snap") {
            if (next_snap > time) return next_snap - time;
        }
        return 1.0e20; // No limit
    }

    PetscReal GetNextSnapTime() const { return next_snap; }
    PetscInt GetSnapIdx() const { return snapshot_idx; }

    // Direct Write
    void WriteVTK(Vec solutionVec, PetscReal time) {
        char basename[256];
        snprintf(basename, sizeof(basename), "%s-%03d.vtu", settings.io.filename.c_str(), snapshot_idx);
        std::string fullPath = settings.io.directory + "/" + std::string(basename);

        PetscViewer viewer;
        PetscViewerVTKOpen(PETSC_COMM_WORLD, fullPath.c_str(), FILE_MODE_WRITE, &viewer);
        PetscObjectSetName((PetscObject)solutionVec, "State");
        VecView(solutionVec, viewer);
        PetscViewerDestroy(&viewer);

        if (rank == 0) {
            snapshots_log.push_back({std::string(basename), time});
            UpdateSeriesFile();
        }
    }

private:
    void UpdateSeriesFile() {
        std::string seriesPath = settings.io.directory + "/" + settings.io.filename + ".vtu.series";
        std::ofstream f(seriesPath);
        if (f.is_open()) {
            f << "{\n  \"file-series-version\" : \"1.0\",\n  \"files\" : [\n";
            for (size_t i = 0; i < snapshots_log.size(); ++i) {
                f << "    { \"name\" : \"" << snapshots_log[i].filename << "\", \"time\" : " 
                  << std::scientific << std::setprecision(6) << snapshots_log[i].time << " }";
                if (i < snapshots_log.size() - 1) f << ",";
                f << "\n";
            }
            f << "  ]\n}\n";
            f.close();
        }
    }
};

#endif