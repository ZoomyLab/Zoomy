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
            // double next_time = loadFieldFromHdf5(this->file_fields, iteration+1, Q, Qaux, true);
            time = loadFieldFromHdf5(this->file_fields, iteration, Q, Qaux, true);
            dt = 0.1;
            // dt = next_time - time;
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