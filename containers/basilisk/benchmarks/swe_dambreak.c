/**
 * SWE Benchmark 3: Dam break on flat bed (1D)
 *
 * Classical Riemann problem: h_L = 1.0, h_R = 0.1, u = 0.
 * Has exact analytical solution (Ritter solution for dry bed,
 * Stoker solution for wet bed).
 * Tests shock capturing and rarefaction fan resolution.
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
    h[] = x < 0. ? 1.0 : 0.1;
    u.x[] = 0.;
  }
  h.name = "Water_Depth";
  u.x.name = "Velocity_X";
  eta.name = "Free_Surface";
}

event output (t += 0.2; t <= 4.0)
{
  output_vtk_series((scalar *){h, eta, u.x}, NULL, t, "swe_dambreak");
}
