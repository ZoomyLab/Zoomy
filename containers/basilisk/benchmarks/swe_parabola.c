/**
 * SWE Benchmark 4: Parabolic bowl oscillation (1D)
 *
 * Thacker's test case: analytical solution for oscillating flow
 * in a parabolic basin with linear friction.
 * Tests wet/dry fronts and well-balanced topography treatment.
 * Reference: Sampson et al. (2006), Thacker (1981).
 */
#include "grid/multigrid1D.h"
#include "saint-venant.h"
#include "to_vtk.h"

double A = 3000.;
double h0 = 10.;
double tau = 1e-3;
double Bf = 5.;

double Psi(double x, double t)
{
  double p = sqrt(8.*G*h0)/A;
  double s = sqrt(p*p - tau*tau)/2.;
  return A*A*Bf*Bf*exp(-tau*t)/(8.*G*G*h0)*(
    -s*tau*sin(2.*s*t) + (tau*tau/4. - s*s)*cos(2.*s*t))
    - Bf*Bf*exp(-tau*t)/(4.*G)
    - exp(-tau*t/2.)/G*(Bf*s*cos(s*t) + tau*Bf/2.*sin(s*t))*x;
}

int main()
{
  origin(-5000);
  size(10000);
  G = 9.81;
  N = 256;
  DT = 10.;
  run();
}

event init (i = 0)
{
  foreach() {
    zb[] = h0*sq(x/A);
    h[] = max(h0 + Psi(x, 0) - zb[], 0.);
  }
  h.name = "Water_Depth";
  u.x.name = "Velocity_X";
  eta.name = "Free_Surface";
  zb.name = "Bathymetry";
}

event friction (i++)
{
  foreach()
    u.x[] /= 1. + tau*dt;
}

event output (t += 100; t <= 3000)
{
  output_vtk_series((scalar *){h, eta, zb, u.x}, NULL, t, "swe_parabola");
}

event output_analytical (t = 1500)
{
  FILE * fp = fopen("swe_parabola_analytical.csv", "w");
  fprintf(fp, "x,h_numerical,h_analytical,zb,error\n");
  foreach() {
    double h_exact = max(h0 + Psi(x, t) - zb[], 0.);
    fprintf(fp, "%g,%g,%g,%g,%g\n", x, h[], h_exact, zb[], h[] - h_exact);
  }
  fclose(fp);
}
