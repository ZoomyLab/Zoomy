#include "plotfile_utils.H"
#include "AMReX_MultiFabUtil.H"

#include "constants.H"
#include "model.h"

using namespace amrex;

void
write_plotfiles_2d (const int identifier, const int step, MultiFab const& solution, Geometry const& geom, const Real time)
{
    const std::string& pltfile = amrex::Concatenate(amrex::Concatenate("plt_2d_", identifier),step,5);

    int ncomp = solution.nComp();

    Vector<std::string> var_names;
    for (int n=0; n<ncomp; ++n) {
        var_names.push_back(amrex::Concatenate("q_",n,1));
    }

    WriteSingleLevelPlotfile(pltfile, solution, var_names, geom, time, step);
}

void
write_plotfiles_3d (const int identifier, const int step, MultiFab const& solution, MultiFab const& solution_aux, Geometry const& geom, const Real time)
{
    const std::string& pltfile = amrex::Concatenate(amrex::Concatenate("plt_3d_", identifier),step,5);

    int ifac = 8;
    Real fac = static_cast<Real>(ifac);

    //
    // These must be in the same order as the components in the solution MultiFab solution_3d
    //
    Vector<std::string> var_names;
    var_names.push_back("b");
    var_names.push_back("h");
    var_names.push_back("u");
    var_names.push_back("v");
    var_names.push_back("w");
    var_names.push_back("p");

    BoxArray ba = solution.boxArray();

    // This defines a Geometry object
    Geometry geom_3d;

    //
    // Make the physical domain "fac" times taller in the vertical
    //
    Box domain_3d = geom.Domain();
    domain_3d.setBig(2,ifac);
    //amrex::Print() << "3D Domain " << domain_3d << std::endl;

    RealBox rb = geom.ProbDomain();
    rb.setHi(2,fac*geom.ProbDomain().hi(2));
    //amrex::Print() << "3D ProbSize " << rb << std::endl;

    geom_3d.define(domain_3d, rb, CoordSys::cartesian, geom.isPeriodic());

    int ncomp  = 6;
    int nghost = 0;

    int n_dof = Model::n_dof_q;
    int n_dof_aux = Model::n_dof_qaux;

    AMREX_ALWAYS_ASSERT (ncomp == var_names.size());

    // Here we arbitrarily set the fake 3D domain to have ifac cells in the vertical
    BoxList bl3d = BoxList(ba);
    for (auto& b : bl3d) {
        b.setBig(2,ifac);
    }
    BoxArray ba_3d(std::move(bl3d));
    //amrex::Print() << "3D BoxArray " << ba_3d << std::endl;

    MultiFab solution_3d(ba_3d, solution.DistributionMap(), ncomp, nghost);

    Real z0 = 0.5 * geom_3d.ProbDomain().hi(2);
    //amrex::Print() << "Setting z0 to " << z0 << std::endl; 

    // Loop over boxes
    for (MFIter mfi(solution_3d); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real const>& sol_2d_arr = solution.const_array(mfi);
        const Array4<Real const>& sol_2d_arr_aux = solution_aux.const_array(mfi);
        const Array4<Real      >& sol_3d_arr = solution_3d.array(mfi);


        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real z = (k+0.5); // We have arbitrarily set "dz" = 1 
                    //
            VecQ Q;
            VecQaux Qaux;
            VecQ3d Q3d;
            Vec3 X;

            for (int n=0;n<n_dof; ++n) {
                Q(n, 0) = sol_2d_arr(i, j, 0, n);
            }
            for (int n=0;n<n_dof_aux; ++n) {
                Qaux(n, 0) = sol_2d_arr(i, j, 0, n);
            }
            
            X = makeSmallMatrix<3, 1>({0., 0., z}); 
            for (int n=0;n<ifac; ++n) {
                Q3d = Model::interpolate_3d(Q, Qaux, X);
            }


            for (int n=0;n<6; ++n) {
                sol_3d_arr(i, j, k, n) = Q3d(n, 0);
            }
        });
    } // mfi

    WriteSingleLevelPlotfile(pltfile, solution_3d, var_names, geom_3d, time, step);
}

void write_plotfiles (const int identifier, const int step, MultiFab& solution, MultiFab& solution_aux,Geometry const& geom, const Real time)
{
    // Note that this average from cell centers to nodes assumes that the ghost cells of the solution array
    //      have been filled
    solution.FillBoundary(geom.periodicity());

    //
    // Next we write the plotfile which interprets the results to create a full 3D field 
    // using what we know about the basis functions.  This arbitrarily is set to have 8 cells in the vertical
    //
    //write_plotfiles_2d (identifier ,step, solution, geom, time);

    //
    // Next we write the plotfile which interprets the results to create a full 3D field 
    // using what we know about the basis functions.  This arbitrarily is set to have 8 cells in the vertical
    //
    write_plotfiles_3d (identifier, step, solution, solution_aux,geom, time);
}
