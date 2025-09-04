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
    VecQ ev = VecQ::Zero(); 
    ev = Model::eigenvalues(Q, Qaux, normal);
    // for (int n=0; n<Model::n_dof_q; ++n)
    // {
    //     ev(n,0) = 0.;
    // }
    // if (Q(idx_h,0) > eps)
    // {
    //     ev = Model::eigenvalues(Q, Qaux, normal);
    // }
    
    // for (int n=0; n<Model::n_dof_q; ++n)
    // {
    //     if (std::isnan(ev(n, 0))) ev(n, 0) = 0.;
    // }

    amrex::Real sM  = std::abs(ev(0, 0));
    for (int i=0; i<Model::n_dof_q; ++i)
    {
        sM = amrex::max(sM, std::abs(ev(i, 0)));
    }
    return sM;
}

void update_q(MultiFab& Q, const MultiFab& Qaux, Real max_velocity)
{

    for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
    {
        // We will loop over all cells in this box
        const Box& bx = mfi.validbox();
        // const Box& gbx = grow(bx,1);               // include 1 ghost


        // These define the pointers we can pass to the GPU
        const Array4<      Real>& Q_arr     = Q.array(mfi);
        const Array4<const Real>& Qaux_arr  = Qaux.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real h = Q_arr(i,j,k,idx_h);
            if (h <= 0)
            {
                h = 0.;
                for (int n=1; n<Model::n_dof_q; ++n)
                {
                    Q_arr(i,j,k,n) = 0.;
                }
            }
            else
            {
                int level = (Model::n_dof_q-2) / Model::dimension-1;
                int offset = level+1;
                if (std::abs(Q_arr(i, j, k, 2)) > max_velocity * h )
                {

                    Q_arr(i,j,k,2) = amrex::min(max_velocity * h, Q_arr(i,j,k,2));
                    Q_arr(i,j,k,2) = amrex::max(-max_velocity * h, Q_arr(i,j,k,2));
                }
                if (std::abs(Q_arr(i, j, k, 2+offset)) > max_velocity * h )
                {
                    Q_arr(i,j,k,2+offset) = amrex::min(max_velocity * h, Q_arr(i,j,k,2+offset));
                    Q_arr(i,j,k,2+offset) = amrex::max(-max_velocity * h, Q_arr(i,j,k,2+offset));
                }
            }

            // for (int n=1; n<Model::n_dof_q; ++n)
            // {
            //     if (std::isnan(Q_arr(i,j,k,n)) || std::isinf(Q_arr(i,j,k,n))) 
            //         {
            //             for (int m=1; m<Model::n_dof_q; ++m)
            //             {
            //                 Q_arr(i,j,k,m) = 0.;
            //             }
            //         }
            // }

            // h = h > 0. ? h : 0.;

            // Real factor = h / (amrex::max(h, eps));
            // Real factor = h > eps? 1. : 0.;
            // factor = 0.;
            // Q_arr(i,j,k,idx_h) = h;
            // for (int n=2; n<Model::n_dof_q; ++n)
            // {
            //     Q_arr(i,j,k,n) *= factor;
            //     // Q_arr(i,j,k,n) = 0.;
            // }
            // if (h < eps)
            // {
            //     for (int n=2; n<Model::n_dof_q; ++n)
            //     {
            //         Q_arr(i,j,k,n) *= factor;
            //         // Q_arr(i,j,k,n) = 0.;
            //     }
            // }
        });
    } // mfi

}


void update_qaux(const MultiFab& Q, MultiFab& Qaux)
{

    for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
    {
        // We will loop over all cells in this box
        const Box& bx = mfi.validbox();
        // const Box& gbx = grow(bx,1);               // include 1 ghost

        // These define the pointers we can pass to the GPU
        const Array4<const Real>& Q_arr     = Q.array(mfi);
        const Array4<      Real>& Qaux_arr        = Qaux.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real h = Q_arr(i,j,k,idx_h);
            h = h > 0 ? h : 0.;
            Real hinv = 2 / (h+(amrex::max(h, eps)));
            // hinv = h < eps ? 0. : hinv;

            Qaux_arr(i,j,k,0) = hinv;
            // if (h < eps)
            // {
            //     Qaux_arr(i,j,k,0) = 0.;
            // }
            // else{
            //     Qaux_arr(i,j,k,0) = 1./h;
            // }
            
        });
    } // mfi

}



double compute_flux_and_save_dt(const MultiFab& Q, MultiFab& Qtmp, const MultiFab& Qaux, Real dt, Real dx, Real dy)
{

    ReduceOps< ReduceOpMin > reduce_op;
    ReduceData<Real> reduce_data(reduce_op);

    for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
    {
        // We will loop over all cells in this box
        const Box& bx = mfi.validbox();

        // These define the pointers we can pass to the GPU
        const Array4<const Real>& Q_arr     = Q.array(mfi);
        const Array4<      Real>& Qtmp_arr     = Qtmp.array(mfi);

        const Array4<const Real>& Qaux_arr  = Qaux.array(mfi);

        reduce_op.eval(bx, reduce_data, [&] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> GpuTuple<Real>
        {
            VecQ dQ = make_flux(i, j, Q_arr, Qaux_arr,
                                dx, dy, dt);
            for (int n=0; n<Model::n_dof_q; ++n)
            {
                Qtmp_arr(i,j,k,n) = dQ(n);
            }

            Real hold = Q_arr(i,j,k,idx_h);
            Real hnew = hold + dt * dQ(idx_h);

            Real s = 1.0;
            if (hnew < 0.0) {
                s = hold / (hold - hnew);
                if (amrex::isnan(hnew) || amrex::isinf(s) || s == 0.) 
                {
                    s = 0.;
                    // amrex::Print() << "hnew " << hnew << "\n";
                    // for (int n=0; n<Model::n_dof_q; ++n)
                    // {
                    //     amrex::Print() << "Q " << n << " " << Q_arr(i,j,k,n) << "\n";
                    // }

                }
            }
            return s;   // one-component tuple
        });
    } // mfi
    auto tuple = reduce_data.value(reduce_op); 
    Real dt_scale = amrex::get<0>(tuple);
    amrex::ParallelDescriptor::ReduceRealMin(dt_scale);
    return dt_scale;
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
                Real max_abs_ev =  {amrex::max(amrex::max(ev_xp, ev_xm), 
                                amrex::max(ev_yp, ev_ym))};
                // if (max_abs_ev > 100)
                // {
                //     amrex::Print() << "Large eigenvalue detected: " << max_abs_ev << "(i,j): " << i << " " << j << "\n";
                // }
                return max_abs_ev;
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
    int   test_case = 0, identifier = 0, warmup_steps = 5;
    Real  time_end = 1.0, dt = 1.0e-4, CFL = 0.5, dtmin=1.0e-7, dtmax=1.e-2;
    bool  adapt_dt = false;
    Real max_velocity = 100.;
    
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
        pp.query("dtmin",       dtmin);
        pp.query("dtmax",       dtmax);
        pp.query("warmup_steps", warmup_steps);
        pp.query("max_velocity", max_velocity);

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
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

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
    MultiFab Qtmp(ba, dm, Ncomp, Nghost);
    MultiFab Qaux(ba, dm, n_dof_qaux, Nghost);


    // time = starting time in the simulation
    Real time = 0.0;

    // **********************************
    // INITIALIZE DATA
    // **********************************
    init_solution(geom, Q);
    init_solution(geom, Qtmp);
    init_solution(geom, Qaux);
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

    auto evolve = [&](MultiFab& Qtmp, MultiFab& Q, MultiFab& Qaux, const Real /* time */) 
    {

        // fill periodic ghost cells
        // Q.FillBoundary(geom.periodicity());
        // Qaux.FillBoundary(geom.periodicity());


        update_q(Q, Qaux, max_velocity);
        update_qaux(Q, Qaux);

        Q.FillBoundary(geom.periodicity());
        Qaux.FillBoundary(geom.periodicity());

        Real max_abs_ev = computeMaxAbsEigenvalue(Q, Qaux);
        if (adapt_dt && iteration > warmup_steps) 
            {
                dt = CFL * cell_size / max_abs_ev;
                dt = amrex::min(dt, dtmax);
                dt = amrex::max(dt, dtmin);
            }



        Real dt_scale = compute_flux_and_save_dt(Q, Qtmp, Qaux, dt, dx[0], dx[1]);
        // Qtmp.FillBoundary(geom.periodicity());

        // finally limit the time step
        dt = std::max(dt * dt_scale, dtmin);
        amrex::Print() << "  Evolve: abs_max_ev: " << max_abs_ev << " dt: " << dt <<  " dt_scale: " << dt_scale << "\n";



        for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
        {
            // We will loop over all cells in this box
            const Box& bx = mfi.validbox();


            // These define the pointers we can pass to the GPU
            const Array4<      Real>& Q_arr        = Q.array(mfi);
            const Array4<      Real>& Qtmp_arr        = Qtmp.array(mfi);
            const Array4<      Real>& Qaux_arr  = Qaux.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                for (int n=0; n<Ncomp; n++)
                {
                    Q_arr(i,j,k,n) += dt* Qtmp_arr(i, j, k, n);
                }
            });
        }
        update_q(Q, Qaux, max_velocity);
        update_qaux(Q, Qaux);
        Q.FillBoundary(geom.periodicity());
        Qaux.FillBoundary(geom.periodicity());
        for ( MFIter mfi(Q); mfi.isValid(); ++mfi )
        {
            // We will loop over all cells in this box
            const Box& bx = mfi.validbox();


            // These define the pointers we can pass to the GPU
            const Array4<      Real>& Q_arr        = Q.array(mfi);
            const Array4<      Real>& Qtmp_arr        = Qtmp.array(mfi);
            const Array4<      Real>& Qaux_arr  = Qaux.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                VecQ Q_chezy = make_rhs_explicit(i, j, Q_arr, Qaux_arr, dx[0], dx[1], dt, max_velocity);
                VecQ dQ = make_rhs(i, j, Q_arr, Qaux_arr, dx[0], dx[1], dt);
                for (int n=0; n<Ncomp; n++)
                {
                    Q_arr(i,j,k,n) = Q_chezy(n);
                    Q_arr(i,j,k,n) = Q_arr(i,j,k,n) + dt*dQ(n);
                }
            });
            // Qtmp.FillBoundary(geom.periodicity());

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

        evolve(Qtmp, Q, Qaux, time);
        // std::swap(Q , Qtmp );

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
        
        amrex::Print() << "Advance: Time: " << time << " s,  " <<  iteration <<  " << iteration  in " << step_stop_time << " seconds; " << "\n";
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
