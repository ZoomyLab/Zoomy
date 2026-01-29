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
#include <petscfv.h>     
#include <petscds.h>     
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
    std::vector<SnapInfo> snapshots_log_3d; 
    
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

    // --- 3D SETUP FUNCTION ---
    PetscErrorCode Setup3D(DM dm2D, PetscInt num_comp_3d, const std::vector<std::string>& var_names = {}) {
        if (!settings.io.write_3d) return PETSC_SUCCESS;
        if (has_setup_3d) return PETSC_SUCCESS;

        PetscFunctionBeginUser;

        // 1. Extrude 2D DM to a Temporary DM
        DM dmExtruded;
        PetscCall(DMPlexExtrude(dm2D, settings.io.n_layers_3d, PETSC_UNLIMITED, 
                                PETSC_TRUE, PETSC_FALSE, PETSC_FALSE,
                                NULL, NULL, NULL, &dmExtruded));

        // 2. Clone to 'dm3D' to get a clean slate (Removes inherited 2D fields)
        PetscCall(DMClone(dmExtruded, &dm3D));
        PetscCall(PetscObjectSetName((PetscObject)dm3D, "Mesh3D"));

        // 3. Copy Coordinates
        Vec coords;
        DM dmCoords;
        PetscCall(DMGetCoordinateDM(dmExtruded, &dmCoords));
        PetscCall(DMSetCoordinateDM(dm3D, dmCoords)); 
        PetscCall(DMGetCoordinatesLocal(dmExtruded, &coords));
        PetscCall(DMSetCoordinatesLocal(dm3D, coords)); 
        
        PetscCall(DMDestroy(&dmExtruded));

        // 4. Setup Finite Volume (FV) Field
        PetscFV fvm;
        PetscCall(PetscFVCreate(PetscObjectComm((PetscObject)dm3D), &fvm));
        PetscCall(PetscFVSetNumComponents(fvm, num_comp_3d));
        PetscCall(PetscFVSetSpatialDimension(fvm, 3)); 
        PetscCall(PetscFVSetType(fvm, PETSCFVUPWIND)); 
        
        // 5. Apply Custom Component Names
        // These appear as the labels for the vector components (b, h, u...)
        if (!var_names.empty()) {
            for(size_t i=0; i<var_names.size() && (PetscInt)i < num_comp_3d; ++i) {
                PetscCall(PetscFVSetComponentName(fvm, i, var_names[i].c_str()));
            }
        }

        // 6. Finalize DS and Name the Field
        PetscCall(DMAddField(dm3D, NULL, (PetscObject)fvm));
        PetscCall(DMCreateDS(dm3D)); 
        
        // --- FIX: Rename the Field to remove "Field_0" artifact ---
        PetscSection s;
        PetscCall(DMGetLocalSection(dm3D, &s));
        // Setting this to "State" results in "State" (array) and "State.b" (component) in ParaView.
        // If you want it completely hidden, try "" (empty string), though behavior varies by viewer.
        PetscCall(PetscSectionSetFieldName(s, 0, "State")); 
        // ----------------------------------------------------------

        PetscCall(PetscFVDestroy(&fvm));

        // 7. Create Global Vector for Output
        PetscCall(DMCreateGlobalVector(dm3D, &X_3D));
        // Ensure vector name matches or is unrelated (e.g., "Solution")
        PetscCall(PetscObjectSetName((PetscObject)X_3D, "State3D"));

        has_setup_3d = true;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <typename ModelType>
    PetscErrorCode Write3D(PetscReal time, Vec X_global, Vec A_global, DM dm2D, DM dmAux, const std::vector<PetscReal>& params) {
        if (!has_setup_3d) return PETSC_SUCCESS;
        PetscFunctionBeginUser;

        // 1. Get LOCAL vectors (Parallel Safety)
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dm2D, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dm2D, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dm2D, X_global, INSERT_VALUES, X_loc));

        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(DMGlobalToLocalBegin(dmAux, A_global, INSERT_VALUES, A_loc));
        PetscCall(DMGlobalToLocalEnd(dmAux, A_global, INSERT_VALUES, A_loc));

        // 2. Prepare Local 3D Vector
        Vec X_3D_loc;
        PetscCall(DMGetLocalVector(dm3D, &X_3D_loc));
        
        const PetscScalar *x2d_arr, *a2d_arr;
        PetscScalar *x3d_arr;
        PetscCall(VecGetArrayRead(X_loc, &x2d_arr));
        PetscCall(VecGetArrayRead(A_loc, &a2d_arr));
        PetscCall(VecGetArray(X_3D_loc, &x3d_arr));

        PetscSection s2d, sAux, s3d;
        PetscCall(DMGetLocalSection(dm2D, &s2d));
        PetscCall(DMGetLocalSection(dmAux, &sAux));
        PetscCall(DMGetLocalSection(dm3D, &s3d));

        PetscInt cStart, cEnd;
        PetscCall(DMPlexGetHeightStratum(dm2D, 0, &cStart, &cEnd));
        PetscInt n_layers = settings.io.n_layers_3d;
        const PetscReal* params_ptr = params.data();

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt off2d, offAux;
            PetscCall(PetscSectionGetOffset(s2d, c, &off2d));
            PetscCall(PetscSectionGetOffset(sAux, c, &offAux));

            PetscReal centroid[3] = {0.0, 0.0, 0.0};
            PetscCall(DMPlexComputeCellGeometryFVM(dm2D, c, NULL, centroid, NULL));

            const PetscScalar *q_ptr = &x2d_arr[off2d];
            const PetscScalar *aux_ptr = &a2d_arr[offAux];

            for (PetscInt k = 0; k < n_layers; ++k) {
                PetscReal sigma = (PetscReal)(k + 0.5) / (PetscReal)n_layers;
                PetscScalar X_input[3] = {centroid[0], centroid[1], sigma};

                // Kernel Call
                auto q_3d_val = ModelType::project_2d_to_3d(X_input, q_ptr, aux_ptr, params_ptr);

                PetscInt c_3d = c * n_layers + k; 
                PetscInt off3d;
                PetscCall(PetscSectionGetOffset(s3d, c_3d, &off3d));

                for (size_t i = 0; i < 6; ++i) x3d_arr[off3d + i] = q_3d_val[i];
            }
        }

        PetscCall(VecRestoreArrayRead(X_loc, &x2d_arr));
        PetscCall(VecRestoreArrayRead(A_loc, &a2d_arr));
        PetscCall(VecRestoreArray(X_3D_loc, &x3d_arr));
        
        // 4. Scatter to Global Output Vector
        PetscCall(VecZeroEntries(X_3D));
        PetscCall(DMLocalToGlobalBegin(dm3D, X_3D_loc, INSERT_VALUES, X_3D));
        PetscCall(DMLocalToGlobalEnd(dm3D, X_3D_loc, INSERT_VALUES, X_3D));

        PetscCall(DMRestoreLocalVector(dm2D, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(DMRestoreLocalVector(dm3D, &X_3D_loc));

        // 5. Write VTK
        char basename[256];
        snprintf(basename, sizeof(basename), "%s-%03d.vtu", settings.io.output_3d_name.c_str(), snapshot_idx);
        std::string fullPath = settings.io.directory + "/" + std::string(basename);

        PetscViewer viewer;
        PetscViewerVTKOpen(PETSC_COMM_WORLD, fullPath.c_str(), FILE_MODE_WRITE, &viewer);
        PetscCall(VecView(X_3D, viewer));
        PetscCall(PetscViewerDestroy(&viewer));

        if (rank == 0) {
            snapshots_log_3d.push_back({std::string(basename), time});
            UpdateSeriesFile3D();
        }

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

    void UpdateSeriesFile3D() {
        std::string seriesPath = settings.io.directory + "/" + settings.io.output_3d_name + ".vtu.series";
        std::ofstream f(seriesPath);
        if (f.is_open()) {
            f << "{\n  \"file-series-version\" : \"1.0\",\n  \"files\" : [\n";
            for (size_t i = 0; i < snapshots_log_3d.size(); ++i) {
                f << "    { \"name\" : \"" << snapshots_log_3d[i].filename << "\", \"time\" : " 
                  << std::scientific << std::setprecision(6) << snapshots_log_3d[i].time << " }";
                if (i < snapshots_log_3d.size() - 1) f << ",";
                f << "\n";
            }
            f << "  ]\n}\n";
            f.close();
        }
    }
};

#endif