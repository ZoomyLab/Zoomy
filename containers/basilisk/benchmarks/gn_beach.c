/**
 * Green-Naghdi Benchmark 2: Solitary wave run-up on beach (1D)
 *
 * Soliton approaching a plane beach with slope 1/19.85.
 * Tests dispersive propagation + shoaling + run-up + wet/dry.
 * Reference: Synolakis (1987), Grilli et al. (1997).
 */
#include "grid/multigrid1D.h"
#include "green-naghdi.h"
#include "to_vtk.h"

double h0 = 1., A = 0.28, L_sol;
double slope = 1./19.85;

double sech2(double x)
{
  double a = 2./(exp(x) + exp(-x));
  return a*a;
}

double soliton(double x, double t)
{
  double c = sqrt(G*(1. + A)*h0), psi = x - c*t;
  double k = sqrt(3.*A*h0)/(2.*h0*sqrt(h0*(1. + A)));
  return A*h0*sech2(k*psi);
}

int main()
{
  double k = sqrt(3.*A/4/cube(h0));
  L_sol = 2./k*acosh(sqrt(1./0.05));
  X0 = -h0/slope - L_sol/2. - L_sol;
  L0 = 6.*L_sol;
  N = 1024;
  G = 1.;
  alpha_d = 1.;
  breaking = 0.4;
  run();
}

event init (i = 0)
{
  double c = sqrt(G*(1. + A)*h0);
  foreach() {
    double eta_val = soliton(x + h0/slope + L_sol/2., t);
    zb[] = max(slope*x, -h0);
    h[] = max(0., eta_val - zb[]);
    u.x[] = c*eta_val/(h0 + eta_val);
  }
  h.name = "Water_Depth";
  u.x.name = "Velocity_X";
  eta.name = "Free_Surface";
  zb.name = "Bathymetry";
}

event friction (i++)
{
  foreach() {
    double a = h[] < dry ? HUGE : 1. + 5e-3*dt*norm(u)/h[];
    foreach_dimension()
      u.x[] /= a;
  }
}

event output (t += 2.5; t <= 65)
{
  output_vtk_series((scalar *){h, eta, zb, u.x}, NULL, t, "gn_beach");
}
