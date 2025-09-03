#ifndef RIEMANN
#define RIEMANN
#include "AMReX_Array4.H"
#include "model.h"
#include "constants.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
VecQ
price_c (VecQ Qi, VecQ Qj, VecQaux Qauxi, VecQaux Qauxj, Vec2 normal, amrex::Real sign, amrex::Real Vi, amrex::Real Vj, amrex::Real Vij, std::function<MatQ(VecQ, VecQaux)> A, amrex::Real dt)
{
   // Gauss quadrature points shifted from [-1, 1] to [0, 1]
   amrex::Real wi[3] = {0.5*5./9., 0.5*8./9., 0.5*5./9.};
   amrex::Real xi[3] = {0.5*(-sqrt(3./5.)+1), 0.5, 0.5*(sqrt(3./5.)+1)};

   MatQ Aint = MatQ::Zero();
   VecQ q = VecQ::Zero();
   VecQaux qaux = VecQaux::Zero();

   for (int iquad=0; iquad<3; ++iquad){
       q = Qi + xi[iquad] * (Qj-Qi);
       qaux = Qauxi + xi[iquad] * (Qauxj-Qauxi);
       Aint += wi[iquad] * A(q, qaux);
   }

   MatQ Asq = Aint * Aint;
   MatQ I = MatQ::Identity();

   MatQ Am;
   VecQ fluctuation;
   VecQ dQ;
   // Subcell areas
   amrex::Real svi = Vi/4.;
   amrex::Real svj = Vj/4.;
   // TODO this needs to be an input. However, integrator.evolve does not pass this info on
//    amrex::Real dt =0.0001;

   // Approximation of A-|A|
   Am = 0.5 * Aint - (svi * svj)/(svi + svj) * 1./(dt * Vij) * I - 1./4. * (dt * Vij)/(svi + svj) * Asq;
   dQ = Qj-Qi;
   fluctuation = (Am * dQ) * (Vij / Vi);



   return fluctuation;

}
//
//AMREX_GPU_DEVICE
//AMREX_FORCE_INLINE
//VecQ
//uprice (VecQ Qi, VecQ Qj, amrex::Real Vi, amrex::Real Vij, std::function<MatQ(VecQ)> A, std::function<VecQ(VecQ)> eigenvalues)
//{
//    // TODO not working. But I can't find literature besides a (wrong?) citation in Koellermeier's dis.
//    
//    // Gauss quadrature points shifted from [-1, 1] to [0, 1]
//    amrex::Real wi[3] = {0.5*5./9., 0.5*8./9., 0.5*5./9.};
//    amrex::Real xi[3] = {0.5*(-sqrt(3./5.)+1), 0.5, 0.5*(sqrt(3./5.)+1)};
//
//    MatQ Aint{};
//    VecQ q{};
//    for (int iquad=0; iquad<3; ++iquad){
//        q = Qi + xi[iquad] * (Qj-Qi);
//        Aint += wi[iquad] * A(q);
//    }
//
//    MatQ Asq = Aint * Aint;
//    MatQ I = MatQ::Identity();
//
//    MatQ Am;
//    VecQ fluctuation;
//    VecQ dQ;
//
//    VecQ ev_i = eigenvalues(Qi);
//    VecQ ev_j = eigenvalues(Qj);
//    amrex::Real s1  = ev_i(0, 0);
//    amrex::Real sM  = ev_i(0, 0);
//    for (int i=0; i<n_dof; ++i)
//    {
//        s1 = std::min(s1, amrex::min(ev_i(i, 0), ev_j(i, 0)));
//        sM = std::max(sM, amrex::max(ev_i(i, 0), ev_j(i, 0)));
//    }
//    s1 = std::abs(s1);
//    sM = std::abs(sM);
//
//    // Approximation of A-|A|
//    Am = 0.5 * (Aint - (s1 * sM)/(s1 + sM) * I - 1./(s1 + sM) * Asq);
//    dQ = Qj-Qi;
//    fluctuation = (Am * dQ) * (Vij / Vi);
//
//    return fluctuation;
//}


AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
VecQ
rusanov (VecQ Qj, VecQ Qi, VecQaux Qauxj, VecQaux Qauxi, Vec2 normal, amrex::Real sign, amrex::Real Vi, amrex::Real Vj, amrex::Real Vij, std::function<MatQ(VecQ, VecQaux)> A, amrex::Real dt)
{
    VecQ fluctuation = VecQ::Zero();
    int n_dof = Model::n_dof_q;

    // I am a wall from the viewpoint of my neighbor
    // TODO this should become a wall boundary condition
    if (Qi(idx_h,0) < eps && Qi(idx_b,0) > Qj(idx_h,0) + Qj(idx_b,0)) return fluctuation;
    // My neighbor a wall for me
    // TODO this should become a wall boundary condition
    if (Qj(idx_h,0) < eps && Qj(idx_b,0) > Qi(idx_h,0) + Qi(idx_b,0)) return fluctuation;
    // if (Qj(idx_h,0) < eps && Qj(idx_b,0) > Qi(idx_h,0) + Qi(idx_b,0))
    // {
    //     Qj(idx_b, 0) = Qi(idx_b, 0);
    //     Qj(idx_h, 0) = Qi(idx_h, 0);
    //     Qj(2,2) = std::abs(normal(0)) > 0 ? -Qi(2,2) : 0.;
    //     Qj(3,2) = std::abs(normal(1)) > 0 ? -Qi(3,2) : 0.;

    // }
    // My neighbor is dry 
    // if (Qj(idx_h,0) < eps) return fluctuation;
// 
    if ((Qj(idx_h,0) < eps) && (Qi(idx_h, 0) < eps)) return fluctuation;

    
    //Qi = wet_dry_fix(Qi);
    //Qj = wet_dry_fix(Qj);

    // Gauss quadrature points shifted from [-1, 1] to [0, 1]
    amrex::Real wi[3] = {0.5*5./9., 0.5*8./9., 0.5*5./9.};
    amrex::Real xi[3] = {0.5*(-sqrt(3./5.)+1), 0.5, 0.5*(sqrt(3./5.)+1)};

    MatQ Aint = MatQ::Zero();
    VecQ q = VecQ::Zero();
    VecQaux qaux = VecQaux::Zero();
    for (int iquad=0; iquad<3; ++iquad){
        q = Qi + xi[iquad] * (Qj-Qi);
        qaux = Qauxi + xi[iquad] * (Qauxj-Qauxi);
        Aint += wi[iquad] * A(q, qaux);
    }

    // MatQ Asq = Aint * Aint;
    MatQ Asq = Aint;
    MatQ I = MatQ::Identity();
    // I(0,idx_b) = 1.;
    I(idx_b,idx_b) = 0.;

    MatQ Am;
    VecQ dQ;

    VecQ ev_i = VecQ::Zero();
    VecQ ev_j = VecQ::Zero();
    // if (Qi(idx_h,0) > eps)
    // {
    //     ev_i = Model::eigenvalues(Qi, Qauxi, normal);
    // }
    // if (Qj(idx_h,0) > eps) 
    // {
    //     ev_j = Model::eigenvalues(Qj, Qauxj, normal);
    // }
    ev_i = Model::eigenvalues(Qi, Qauxi, normal);
    ev_j = Model::eigenvalues(Qj, Qauxj, normal);

    // for (int n=0; n<n_dof; ++n)
    // {
        
    //     if (std::isnan(ev_i(n, 0))) ev_i(n, 0) = 0.;
    //     if (std::isnan(ev_j(n, 0))) ev_j(n, 0) = 0.;
    // }


    // for (int n=0; n<n_dof; ++n)
    // {
        
    //     if (std::isnan(ev_i(n, 0))) amrex::Print() << "NaN detected in ev_i in rusanov\n";
    //     if (std::isnan(ev_j(n, 0))) amrex::Print() << "NaN detected in ev_j in rusanov\n";
    // }
    amrex::Real sM  = std::abs(ev_i(0, 0));
    for (int i=0; i<n_dof; ++i)
    {
        sM = std::max(sM, amrex::max(std::abs(ev_i(i, 0)), std::abs(ev_j(i, 0))));
    }

    // Approximation of A +/- |A| with |A| the numerical viscosity approx. by max eigenvalue
    // amrex::Real relax = sM > 1000 ? 0. : 1.;
    // sM = amrex::min(sM, amrex::max(Vi/dt, Vj/dt));

    Am = 0.5*(Aint + sign* sM * I);
    dQ = (Qj-Qi);
    fluctuation = (Am * dQ) * (Vij / Vi);

    // bool is_nan = false;
    // for (int n=0; n<n_dof; ++n)
    // {
    //     if (std::isnan(fluctuation(n, 0))) 
    //     {
    //         is_nan = true;
    //     }
    //     if (fluctuation(n, 0) > 1.e6 || fluctuation(n, 0) < -1.e6)
    //     {
    //         amrex::Print() << "Lage fluctuation detected in rusanov\n";   }
    // }

    // if (is_nan)
    // {
    //     amrex::Print() << "=================" << "\n";
    //     amrex::Print() << "Sign " << sign << "\n";
    //     for (int n=0; n<n_dof; ++n)
    //     {
    //         amrex::Print() << "Qi[" << n << "] " << Qi(n,0) << "\n";
    //         amrex::Print() << "Qj[" << n << "] " << Qj(n,0) << "\n";
    //     }
    //     amrex::Print() << "Qauxi[0]" << Qauxi(0,0) << "\n";
    //     amrex::Print() << "Qauxj[0]" << Qauxj(0,0) << "\n";
    //     check_nan(Qi, "Qi in rusanov");
    //     check_nan(Qj, "Qj in rusanov");
    //     check_nan(Qauxi, "Qauxi in rusanov");
    //     check_nan(Qauxj, "Qauxj in rusanov");
    //     check_nan(Aint, "Aint in rusanov");
    //     check_nan(Asq, "Asq in rusanov");
    //     check_nan(Am, "Am in rusanov");
    //     check_nan(dQ, "dQ in rusanov");
    // }

    // if (Qi(idx_h, 0) < eps || Qj(idx_h, 0) < eps)
    // {
    //     for (int n=2; n<n_dof; ++n)
    //     {
    //         fluctuation(n, 0) = 0.;
    //     }
    // }

    return fluctuation;

}

#endif
