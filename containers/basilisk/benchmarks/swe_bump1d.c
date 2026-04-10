/**
 * SWE Benchmark 1: Gaussian bump on flat bed (1D)
 *
 * Radially symmetric Gaussian initial perturbation on flat bed.
 * No topography. Tests pure SWE wave propagation.
 * Reference: Analytical cylindrical wave solution.
 */
#include "grid/cartesian1D.h"
#include "saint-venant.h"
#include "to_vtk.h"

int main()
{
  L0 = 10.;
  X0 = -5.;
  G = 1.;
  N = 500;
  run();
}

event init (i = 0)
{
  foreach() {
    h[] = 0.1 + exp(-200.*x*x);
    u.x[] = 0.;
  }
  h.name = "Water_Depth";
  u.x.name = "Velocity_X";
  eta.name = "Free_Surface";
}

event output (t += 0.1; t <= 2.5)
{
  output_vtk_series((scalar *){h, eta, u.x}, NULL, t, "swe_bump1d");
}
