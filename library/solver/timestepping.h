#include "define.h"
#include "model.h"
#include "mesh.h"

class TimeStepper {
public:
    virtual double get_dt(double max_abs_eigenvalue) = 0; // Pure virtual function
    virtual ~TimeStepper() {} // Virtual destructor
};

class Constant : public TimeStepper {
private:
    double dt; // Time step size
public:
    Constant(double dt) : dt(dt) {}
    double get_dt(double max_abs_eigenvalue)
    {
        return this->dt;
    }
    ~Constant() {}
};

class Adaptive : public TimeStepper {
private:
    double CFL; // Courant–Friedrichs–Lewy condition
    double min_inradius;
public:
    Adaptive(double CFL, Mesh& mesh, Model& model) : CFL(CFL) {
        this->min_inradius = mesh.element_inradius(0);
        for (int i = 0; i < mesh.n_elements; ++i)
        {
            this->min_inradius = min(this->min_inradius, mesh.element_inradius(i));
        }
    }
    double get_dt(double max_abs_eigenvalue)
    {
        return this->CFL * this->min_inradius / max_abs_eigenvalue;
    }
    ~Adaptive() {}
};