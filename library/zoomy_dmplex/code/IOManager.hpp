#ifndef IOMANAGER_HPP
#define IOMANAGER_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscdmplex.h> 
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

    // --- 3D Output Members ---
    DM dm3D = nullptr;
    Vec X_3D = nullptr;
    bool has_setup_3d = false;

public:
    IOManager(Settings& s, PetscMPIInt r) : settings(s), rank(r) {
        dt_snap = settings.solver.t_end / settings.io.snapshots;
        next_snap = 0.0; 
        snapshot_idx = 0;
    }

    ~IOManager() {
        if (has_setup_3d) {
            if (X_3D) VecDestroy(&X_3D);
            if (dm3D) DMDestroy(&dm3D);
        }
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

    bool ShouldWrite(PetscReal time, bool force = false) {
        if (force) return true;
        PetscReal tol = 1e-9 * dt_snap; 
        if (time >= next_snap - tol) {
            return true;
        }
        return false;
    }

    void AdvanceSnapshot() {
        if (settings.io.snapshot_logic == "snap") {
            next_snap += dt_snap;
        } else if (settings.io.snapshot_logic == "loose") {
            while(next_snap <= next_snap + dt_snap) next_snap += dt_snap; 
        } else if (settings.io.snapshot_logic == "interpolate") {
            next_snap += dt_snap;
        } else {
             next_snap += dt_snap; 
        }
        snapshot_idx++;
    }

    PetscReal GetDtLimit(PetscReal time) const {
        if (settings.io.snapshot_logic == "snap") {
            if (next_snap > time) return next_snap - time;
        }
        return 1.0e20; 
    }

    PetscReal GetNextSnapTime() const { return next_snap; }
    PetscInt GetSnapIdx() const { return snapshot_idx; }

    // Direct Write (Standard 2D)
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

    // --- 3D Setup and Write Logic ---

    PetscErrorCode Setup3D(DM dm2D, PetscInt num_comp_3d) {
        if (!settings.io.write_3d) return PETSC_SUCCESS;
        if (has_setup_3d) return PETSC_SUCCESS;

        PetscFunctionBeginUser;

        // 1. Extrude 2D DM to 3D
        PetscCall(DMPlexExtrude(dm2D, settings.io.n_layers_3d, PETSC_UNLIMITED, PETSC_TRUE, NULL, NULL, NULL, &dm3D));
        PetscCall(PetscObjectSetName((PetscObject)dm3D, "Mesh3D"));

        // 2. Setup Section
        PetscSection section3D;
        PetscCall(DMGetLocalSection(dm3D, &section3D));
        PetscCall(PetscSectionSetNumFields(section3D, 1));
        PetscCall(PetscSectionSetFieldComponents(section3D, 0, num_comp_3d));
        PetscCall(PetscSectionSetFieldName(section3D, 0, "State3D"));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dm3D, 0, &cStart, &cEnd));
        
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscCall(PetscSectionSetDof(section3D, c, num_comp_3d));
            PetscCall(PetscSectionSetFieldDof(section3D, c, 0, num_comp_3d));
        }
        PetscCall(PetscSectionSetUp(section3D));

        // 3. Create Vector
        PetscCall(DMCreateGlobalVector(dm3D, &X_3D));
        PetscCall(PetscObjectSetName((PetscObject)X_3D, "State3D"));

        has_setup_3d = true;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <typename ModelType>
    PetscErrorCode Write3D(PetscReal time, Vec X_2D, Vec A_2D, DM dm2D, DM dmAux, const std::vector<PetscReal>& params) {
        if (!has_setup_3d) return PETSC_SUCCESS;

        PetscFunctionBeginUser;
        
        const PetscScalar *x2d_arr;
        const PetscScalar *a2d_arr;
        PetscScalar *x3d_arr;

        PetscCall(VecGetArrayRead(X_2D, &x2d_arr));
        PetscCall(VecGetArrayRead(A_2D, &a2d_arr));
        PetscCall(VecGetArray(X_3D, &x3d_arr));

        PetscSection s2d, sAux, s3d;
        PetscCall(DMGetLocalSection(dm2D, &s2d));
        PetscCall(DMGetLocalSection(dmAux, &sAux));
        PetscCall(DMGetLocalSection(dm3D, &s3d));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dm2D, 0, &cStart, &cEnd));

        PetscInt n_layers = settings.io.n_layers_3d;
        const PetscReal* params_ptr = params.data();

        for (PetscInt c = cStart; c < cEnd; ++c) {
            // A. Get Offsets
            PetscInt off2d, offAux;
            PetscCall(PetscSectionGetOffset(s2d, c, &off2d));
            PetscCall(PetscSectionGetOffset(sAux, c, &offAux));

            // B. Get Cell Centroid (Coordinates)
            PetscReal centroid[3] = {0.0, 0.0, 0.0};
            PetscCall(DMPlexComputeCellGeometryFVM(dm2D, c, NULL, centroid, NULL));

            // C. Pointers to Q and Qaux
            const PetscScalar *q_ptr = &x2d_arr[off2d];
            const PetscScalar *aux_ptr = &a2d_arr[offAux];

            for (PetscInt k = 0; k < n_layers; ++k) {
                // Construct X argument: {x, y, sigma}
                // sigma goes from 0 to 1 (cell center logic)
                PetscReal sigma = (PetscReal)(k + 0.5) / (PetscReal)n_layers;
                PetscScalar X_input[3] = {centroid[0], centroid[1], sigma};

                // D. Call User Kernel
                // Function sig: project_2d_to_3d(X, Q, Qaux, p)
                auto q_3d_val = ModelType::project_2d_to_3d(X_input, q_ptr, aux_ptr, params_ptr);

                // E. Write to 3D Vector
                PetscInt c_3d = c * n_layers + k; 
                PetscInt off3d;
                PetscCall(PetscSectionGetOffset(s3d, c_3d, &off3d));

                // We assume q_3d_val supports [] operator (SimpleArray, std::array, vector)
                // Use .size() if available, or hardcode 6 if SimpleArray lacks size()
                for (size_t i = 0; i < 6; ++i) { 
                    x3d_arr[off3d + i] = q_3d_val[i];
                }
            }
        }

        PetscCall(VecRestoreArrayRead(X_2D, &x2d_arr));
        PetscCall(VecRestoreArrayRead(A_2D, &a2d_arr));
        PetscCall(VecRestoreArray(X_3D, &x3d_arr));

        // Write VTK
        char basename[256];
        snprintf(basename, sizeof(basename), "%s-%03d.vtu", settings.io.output_3d_name.c_str(), snapshot_idx);
        std::string fullPath = settings.io.directory + "/" + std::string(basename);

        PetscViewer viewer;
        PetscViewerVTKOpen(PETSC_COMM_WORLD, fullPath.c_str(), FILE_MODE_WRITE, &viewer);
        PetscCall(VecView(X_3D, viewer));
        PetscCall(PetscViewerDestroy(&viewer));

        PetscFunctionReturn(PETSC_SUCCESS);
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