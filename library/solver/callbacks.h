// callbacks.h
#ifndef CALLBACKS_H
#define CALLBACKS_H

#include "define.h"
#include "settings.h"
#include "model.h"
#include "mesh.h"
#include "boundary_conditions.h"
#include "hdf5.h"
#include "helpers_hdf5.h"

class Callback
{
    public:
        virtual void call_loop(const Settings &settings,
            const Mesh &mesh,
            const Model &model,
            const BoundaryConditions &boundary_conditions,
            realArr2 &Q,
            realArr2 &Qaux,
            realArr &parameters,
            double &max_abs_ev,
            double &time,
            double &dt,
            int &iteration) = 0;
        virtual void call_init(const Settings &settings,
            const Mesh &mesh,
            const Model &model,
            const BoundaryConditions &boundary_conditions,
            realArr2 &Q,
            realArr2 &Qaux,
            realArr &parameters,
            double &max_abs_ev,
            double &time,
            double &dt,
            int &iteration)
            {
                return;
            }
        virtual void call_close(const Settings &settings,
            const Mesh &mesh,
            const Model &model,
            const BoundaryConditions &boundary_conditions,
            realArr2 &Q,
            realArr2 &Qaux,
            realArr &parameters,
            double &max_abs_ev,
            double &time,
            double &dt,
            int &iteration)
            {
                return;
            }
};

class LoadOpenfoam : public Callback
{
    public:
    hid_t file_fields;
    void call_init(const Settings &settings,
                   const Mesh &mesh,
                   const Model &model,
                   const BoundaryConditions &boundary_conditions,
                   realArr2 &Q,
                   realArr2 &Qaux,
                   realArr &parameters,
                   double &max_abs_ev,
                   double &time,
                   double &dt,
                   int &iteration) override
    {
        const std::string path_fields = "/home/ingo/Git/SMM/shallow-moments-simulation/openfoam_data/channelflow_coarse/fields_openfoam.hdf5";
        this->file_fields = openHdf5(path_fields);
        return;
    }

    void call_loop(const Settings &settings,
        const Mesh &mesh,
        const Model &model,
        const BoundaryConditions &boundary_conditions,
        realArr2 &Q,
        realArr2 &Qaux,
        realArr &parameters,
        double &max_abs_ev,
        double &time,
        double &dt,
        int &iteration) override
        {
            double next_time = loadFieldFromHdf5(this->file_fields, iteration+1, Q, Qaux, true);
            time = loadFieldFromHdf5(this->file_fields, iteration, Q, Qaux, true);
            dt = 0.1;
            dt = next_time - time;
        }

    void call_close(const Settings &settings,
        const Mesh &mesh,
        const Model &model,
        const BoundaryConditions &boundary_conditions,
        realArr2 &Q,
        realArr2 &Qaux,
        realArr &parameters,
        double &max_abs_ev,
        double &time,
        double &dt,
        int &iteration) override
        {
            H5Fclose(this->file_fields);
            return;
        }
};

class ComputeFoamDeltaDataSet : public Callback
{
    public:
    hid_t file_fields;
    hid_t file_data_input;
    hid_t file_data_output;
    void call_init(const Settings &settings,
                   const Mesh &mesh,
                   const Model &model,
                   const BoundaryConditions &boundary_conditions,
                   realArr2 &Q,
                   realArr2 &Qaux,
                   realArr &parameters,
                   double &max_abs_ev,
                   double &time,
                   double &dt,
                   int &iteration) override
    {
        const std::string path_fields = "/home/ingo/Git/SMM/shallow-moments-simulation/openfoam_data/channelflow_coarse/fields_openfoam.hdf5";
        const std::string path_data_input = "/home/ingo/Git/SMM/shallow-moments-simulation/openfoam_data/channelflow_coarse/closure_dataset_input.hdf5";
        const std::string path_data_output = "/home/ingo/Git/SMM/shallow-moments-simulation/openfoam_data/channelflow_coarse/closure_dataset_output.hdf5";
        this->file_fields= openHdf5(path_fields, "r");
        this->file_data_input= openHdf5(path_data_input, "w");
        this->file_data_output = openHdf5(path_data_output, "w");
        return;
    }

    void call_loop(const Settings &settings,
        const Mesh &mesh,
        const Model &model,
        const BoundaryConditions &boundary_conditions,
        realArr2 &Q,
        realArr2 &Qaux,
        realArr &parameters,
        double &max_abs_ev,
        double &time,
        double &dt,
        int &iteration) override
        {
            realArr2 QIC = realArr2(Q);
            realArr2 Qsol = realArr2(Q);
            realArr2 deltaQ = realArr2(Q);
            realArr2 deltaQaux = realArr2(Q);
            realArr2 Qauxsol = realArr2(Qaux);
            realArr2 QauxIC = realArr2(Qaux);
            double time_ic = loadFieldFromHdf5(this->file_fields, iteration, QIC, QauxIC, true);
            double time_sol = loadFieldFromHdf5(this->file_fields, iteration+1, Qsol, Qauxsol, true);
            for (int i = 0; i < Q.extent(0); ++i)
            {
                for (int j = 0; j < Q.extent(1); ++j)
                {
                    deltaQ(i, j) = Qsol(i, j) - Q(i, j);
                    deltaQaux(i, j) = Qauxsol(i, j) - Qaux(i, j);
                }
            }

            saveFieldToHdf5(this->file_data_input, iteration, time_sol, QIC, QauxIC);
            saveFieldToHdf5(this->file_data_output, iteration, time_sol, deltaQ, deltaQaux);
        }

    void call_close(const Settings &settings,
        const Mesh &mesh,
        const Model &model,
        const BoundaryConditions &boundary_conditions,
        realArr2 &Q,
        realArr2 &Qaux,
        realArr &parameters,
        double &max_abs_ev,
        double &time,
        double &dt,
        int &iteration) override
        {
            H5Fclose(this->file_fields);
            H5Fclose(this->file_data_input);
            H5Fclose(this->file_data_output);
            return;
        }
};


class Callbacks
{
    public:
        std::vector<Callback *> callbacks;

        ~Callbacks()
        {
            for (auto callback : this->callbacks)
            {
                delete callback;
            }
        }

        void register_callbacks(std::vector<std::string> callback_names)
        {
            for (auto name : callback_names)
            {
                if (name == "LoadOpenfoam")
                {
                    this->callbacks.push_back(new LoadOpenfoam());
                }
                else if (name == "ComputeFoamDeltaDataSet")
                {
                    this->callbacks.push_back(new ComputeFoamDeltaDataSet());
                }
            }
        }

        void call_init(const Settings &settings,
            const Mesh &mesh,
            const Model &model,
            const BoundaryConditions &boundary_conditions,
            realArr2 &Q,
            realArr2 &Qaux,
            realArr &parameters,
            double &max_abs_ev,
            double &time,
            double &dt,
            int &iteration)
            {
                for (auto callback : callbacks)
                {
                    callback->call_init(settings, mesh, model, boundary_conditions, Q, Qaux, parameters, max_abs_ev, time, dt, iteration);
                }
            }

        void call_loop(const Settings &settings,
            const Mesh &mesh,
            const Model &model,
            const BoundaryConditions &boundary_conditions,
            realArr2 &Q,
            realArr2 &Qaux,
            realArr &parameters,
            double &max_abs_ev,
            double &time,
            double &dt,
            int &iteration)
            {
                for (auto callback : callbacks)
                {
                    callback->call_loop(settings, mesh, model, boundary_conditions, Q, Qaux, parameters, max_abs_ev, time, dt, iteration);
                }
            }

        void call_close(const Settings &settings,
            const Mesh &mesh,
            const Model &model,
            const BoundaryConditions &boundary_conditions,
            realArr2 &Q,
            realArr2 &Qaux,
            realArr &parameters,
            double &max_abs_ev,
            double &time,
            double &dt,
            int &iteration)
            {
                for (auto callback : callbacks)
                {
                    callback->call_close(settings, mesh, model, boundary_conditions, Q, Qaux, parameters, max_abs_ev, time, dt, iteration);
                }
            }

};

#endif // CALLBACK_H