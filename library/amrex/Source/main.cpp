#include <AMReX_ParmParse.H>
#include <AMReX_TimeIntegrator.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParallelReduce.H>
#include <iostream> 

#include "constants.H"
#include "make_rhs.H"
#include "model.h"

using namespace amrex;



// Declare the function here instead of in a separate .H file
void init_solution(const Geometry& geom, MultiFab& solution);
void readRasterIntoComponent(const std::string& filename, const Geometry& geom, MultiFab& solution, int comp);
void write_plotfiles   (const int identifier, const int step, MultiFab & solution, MultiFab & solution_aux, Geometry const& geom, const Real time);

double computeLocalMaxAbsEigenvalue(const VecQ& Q, const VecQaux& Qaux, const Vec2& normal)
{
    VecQ ev = Model::eigenvalues(Q, Qaux, normal);
    amrex::Real sM  = std::abs(ev(0, 0));
    for (int i=0; i<Model::n_dof_q; ++i)
    {
        sM = amrex::max(sM, amrex::max(std::abs(ev(i, 0)), std::abs(ev(i, 0))));
    }
    return sM;
}

void update_q(MultiFab& Q, const MultiFab& Qaux)
{

    for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
    {
        // We will loop over all cells in this box
        const Box& bx = mfi.validbox();

        // These define the pointers we can pass to the GPU
        const Array4<      Real>& Q_arr     = Q.array(mfi);
        const Array4<const Real>& Qaux_arr  = Qaux.array(mfi);

        // evolve
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real h = Q_arr(i,j,k,1);
            h = h > 0 ? h : 0.;
            Real eps = 1e-4;
            Real factor = h / (amrex::max(h, eps));
            Q_arr(i,j,k,1) = h;
            for (int i=2; i<Model::n_dof_q; ++i)
            {
                Q_arr(i,j,k,i) *= factor;
            }
        });
    } // mfi

}


void update_qaux(const MultiFab& Q, MultiFab& Qaux)
{

    for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
    {
        // We will loop over all cells in this box
        const Box& bx = mfi.validbox();

        // These define the pointers we can pass to the GPU
        const Array4<const Real>& Q_arr     = Q.array(mfi);
        const Array4<      Real>& Qaux_arr        = Qaux.array(mfi);

        // evolve
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real h = Q_arr(i,j,k,1);
            h = h > 0 ? h : 0.;
            Real eps = 1e-4;
            Real hinv = 2 / (h+(amrex::max(h, eps)));
            Qaux_arr(i,j,k,0) = hinv;
        });
    } // mfi

}



double computeMaxAbsEigenvalue(const MultiFab& Q, const MultiFab& Qaux)
{
    const Vec2 normal_xp = makeSmallMatrix<2, 1>({+1., 0.});
    const Vec2 normal_xm = makeSmallMatrix<2, 1>({-1., 0.});
    const Vec2 normal_yp = makeSmallMatrix<2, 1>({0., +1.});
    const Vec2 normal_ym = makeSmallMatrix<2, 1>({0., -1.});

    ReduceOps< ReduceOpMax > reduce_op;
    ReduceData<Real> reduce_data(reduce_op);

    for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
    {
        // We will loop over all cells in this box
        const Box& bx = mfi.validbox();

        // These define the pointers we can pass to the GPU
        const Array4<const Real>& Q_arr     = Q.array(mfi);
        const Array4<const Real>& Qaux_arr  = Qaux.array(mfi);

        reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> GpuTuple<Real>
        {
            VecQ q;
            for (int n=0; n<Model::n_dof_q; n++)
            {
                q(n) = Q_arr(i,j,k,n);
            }
            VecQaux qaux;
            for (int n=0; n<Model::n_dof_qaux; n++)
            {
                qaux(n) = Qaux_arr(i,j,k,n);
            }
            const Real ev_xp = computeLocalMaxAbsEigenvalue(q, qaux, normal_xp);
            const Real ev_xm = computeLocalMaxAbsEigenvalue(q, qaux, normal_xm);
            const Real ev_yp = computeLocalMaxAbsEigenvalue(q, qaux, normal_yp);
            const Real ev_ym = computeLocalMaxAbsEigenvalue(q, qaux, normal_ym);
            return {amrex::max(amrex::max(ev_xp, ev_xm), 
                              amrex::max(ev_yp, ev_ym))};
        });
    } // mfi
    auto tuple = reduce_data.value(reduce_op); 
    Real max_abs_ev = amrex::get<0>(tuple);
    amrex::ParallelDescriptor::ReduceRealMax(max_abs_ev);
    return max_abs_ev;
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    /* ---------------------------------------------------------------
    0.  defaults (only where you really want a fall-back) */
    int   n_cell_x = 1,   n_cell_y = 1, n_cell_z = 1;
    int   max_grid_size_x = 32,  max_grid_size_y = 32, max_grid_size_z = 1;
    Real  phy_bb_x0 = 0., phy_bb_y0 = 0., phy_bb_x1 = 1., phy_bb_y1 = 1.;
    Real  plot_dt_interval = 0.1;
    int   test_case = 0, identifier = 0;
    Real  time_end = 1.0, dt = 1.0e-4, CFL = 0.5;
    bool  adapt_dt = false;
    
    int   dem_field = 0, release_field = 1;
    std::string dem_file, release_file;
    
    /* ---------------------------------------------------------------
       1.  &init ------------------------------------------------------*/
    {
        ParmParse pp("init");               // reads the &init section
    
        pp.query("dem_field",     dem_field);       // optional
        pp.query("release_field", release_field);
    }
    
    /* ---------------------------------------------------------------
       2.  &output ----------------------------------------------------*/
    {
        ParmParse pp("output");
    
        pp.query("test_case",        test_case);    // optional
        pp.query("identifier",       identifier);
        pp.query("plot_dt_interval", plot_dt_interval);
    }
    
    /* ---------------------------------------------------------------
       3.  &solver ----------------------------------------------------*/
    {
        ParmParse pp("solver");
    
        pp.query("time_end", time_end);
        pp.query("dt",       dt);
        pp.query("adapt_dt", adapt_dt);
        pp.query("cfl",      CFL);
    }
    
    /* ---------------------------------------------------------------
       4.  &geometry --------------------------------------------------*/
    {
        ParmParse pp("geometry");
    
        pp.query("n_cell_x",  n_cell_x);
        pp.query("n_cell_y",  n_cell_y);
    
        pp.query("phy_bb_x0", phy_bb_x0);
        pp.query("phy_bb_y0", phy_bb_y0);
        pp.query("phy_bb_x1", phy_bb_x1);
        pp.query("phy_bb_y1", phy_bb_y1);
    
        pp.query("dem_file",     dem_file);
        pp.query("release_file", release_file);
    }
    
    /* ---------------------------------------------------------------
       5.  &grid ------------------------------------------------------*/
    {
        ParmParse pp("grid");
    
        pp.query("max_grid_size_x", max_grid_size_x);
        pp.query("max_grid_size_y", max_grid_size_y);
    }


    amrex::Print() << "dem_file = '" << dem_file << "'\n";
    amrex::Print() << "release_file = '" << release_file << "'\n";

    // **********************************
    // SIMULATION SETUP

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity
    BoxArray ba;
    Geometry geom;

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(         0,          0,          0));
    IntVect dom_hi(AMREX_D_DECL(n_cell_x-1, n_cell_y-1, n_cell_z-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than size max_grid_size_x by max_grid_size_y in the x-y plane
    ba.maxSize(IntVect(AMREX_D_DECL(max_grid_size_x,max_grid_size_y,max_grid_size_z)));

    // This defines the physical box, [0,1] in each direction.
    RealBox real_box({AMREX_D_DECL( phy_bb_x0, phy_bb_y0, 0.)},
                     {AMREX_D_DECL( phy_bb_x1, phy_bb_y1, 1.)});

    // periodic in all direction
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    Real cell_size = amrex::min(dx[0], dx[1]);

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    int Ncomp = Model::n_dof_q;
    int n_dof_qaux = Model::n_dof_qaux;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // allocate phi MultiFab
    MultiFab Q(ba, dm, Ncomp, Nghost);
    MultiFab Qaux(ba, dm, n_dof_qaux, Nghost);

    // time = starting time in the simulation
    Real time = 0.0;

    // **********************************
    // INITIALIZE DATA
    // **********************************
    init_solution(geom, Q);
    readRasterIntoComponent(release_file, geom, Q, 1);
    readRasterIntoComponent(dem_file, geom, Q, 0);
    // amrex::Gpu::streamSynchronize();



    int step = 0;
    int iteration = 0;
    Real next_write = 0.;
    if (time >= next_write)
    {
        write_plotfiles(identifier, step,Q, Qaux, geom,time);
        next_write += plot_dt_interval;
        step +=1;
    }

    auto evolve = [&](MultiFab& Q, MultiFab& Qaux, const Real /* time */) 
    {

        // fill periodic ghost cells
        Q.FillBoundary(geom.periodicity());
        Qaux.FillBoundary(geom.periodicity());


        update_q(Q, Qaux);
        update_qaux(Q, Qaux);

        Real max_abs_ev = computeMaxAbsEigenvalue(Q, Qaux);
        if (adapt_dt && iteration > 5) dt = CFL * cell_size / max_abs_ev;
        amrex::Print() << "    dt = " << dt << ", max_abs_ev = " << max_abs_ev << "\n";
        amrex::Print() << "    cell_size = " << cell_size << "\n";



        for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
        {
            // We will loop over all cells in this box
            const Box& bx = mfi.validbox();

            // These define the pointers we can pass to the GPU
            const Array4<      Real>& Q_arr        = Q.array(mfi);
            const Array4<      Real>& Qaux_arr  = Qaux.array(mfi);

            // evolve
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                VecQ dQ = make_rhs(i, j, Q_arr, Qaux_arr, dx[0], dx[1]);
                for (int n=0; n<Ncomp; n++)
                {
                    Q_arr(i,j,k,n) += dt*dQ(n);
                }
            });
        } // mfi
    };

    // Start the timer for the whole loop
    //
    Real evolution_start_time = ParallelDescriptor::second();

    Real new_time;

    while ( time < time_end)
    {

        //
        // Start the timer for each step
        //
        Real step_start_time = ParallelDescriptor::second();

        //
        // Advance to output time
        //
        //
        // Q.ParallelCopy(Qnew);

        evolve(Q, Qaux, time);

        //
        // Set time to evolve to
        //
        new_time = time + dt;
        if (new_time > time_end) new_time = time_end;


        //
        // Stop the timer for each step and compute the difference between stop_time and start_time
        //
        Real step_stop_time = ParallelDescriptor::second() - step_start_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);


        //
        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        //
        if (time >= next_write)
        {
            write_plotfiles(identifier, step,Q, Qaux, geom,time);
            next_write += plot_dt_interval;
            step += 1;
        }

        //
        // Tell the I/O Processor to write out which step we're doing, how long it took,
        //    with what dt and to what final time
        //
        
        amrex::Print() << "Advanced iteration " << iteration << " in " << step_stop_time << " seconds; dt = "
                       << dt << " time = " << new_time << "\n";
        iteration +=1;

        time = new_time;
    }

    //
    // Stop the timer for the whole loop compute the difference between stop_time and start_time
    //
    Real evolution_stop_time = ParallelDescriptor::second() - evolution_start_time;
    ParallelDescriptor::ReduceRealMax(evolution_stop_time);

    //
    // Tell the I/O Processor to write out the total evolution time
    //
    amrex::Print() << "Total evolution time = " << evolution_stop_time << " seconds\n";

    amrex::Finalize();
    return 0;
}
