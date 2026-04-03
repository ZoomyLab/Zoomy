/**
 * Green-Naghdi Benchmark 1: Soliton propagation (1D)
 *
 * Solitary wave on flat bed. Exact analytical soliton solution.
 * Tests dispersive wave propagation and shape preservation.
 * Reference: Serre-Green-Naghdi exact soliton.
 */
#include "grid/multigrid1D.h"
#include "green-naghdi.h"
#include "to_vtk.h"

#define sech(x) (1.0 / cosh(x))
double a_wave = 0.1;

int main()
{
  L0 = 40.;
  X0 = -20.;
  G = 1.;
  N = 512;
  alpha_d = 1.;
  run();
}

event init (i = 0)
{
  double c = sqrt(1. + a_wave);
  foreach() {
    h[] = 1. + a_wave*sq(sech(sqrt(3.*a_wave)/(2.*c)*x));
    u.x[] = c*h[]/(1. + h[]) - c + c/(1. + h[]);
  }
  h.name = "Water_Depth";
  u.x.name = "Velocity_X";
  eta.name = "Free_Surface";
}

event output (t += 0.5; t <= 20)
{
  output_vtk_series((scalar *){h, eta, u.x}, NULL, t, "gn_soliton");
}

event error_check (t = 20)
{
  double c = sqrt(1. + a_wave);
  FILE * fp = fopen("gn_soliton_error.csv", "w");
  fprintf(fp, "x,h_numerical,h_analytical,error\n");
  foreach() {
    double x_shifted = x - c*t;
    double h_exact = 1. + a_wave*sq(sech(sqrt(3.*a_wave)/(2.*c)*x_shifted));
    fprintf(fp, "%g,%g,%g,%g\n", x, h[], h_exact, h[] - h_exact);
  }
  fclose(fp);
}
