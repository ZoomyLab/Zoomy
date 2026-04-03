/**
 * SWE Benchmark 2: Transcritical flow over a bump (1D)
 *
 * Steady uniform inflow over a cosine-shaped bottom bump.
 * Tests topography source term handling (well-balanced scheme).
 * Flow transitions from subcritical to supercritical over the bump.
 */
#include "grid/cartesian1D.h"
#include "saint-venant.h"
#include "to_vtk.h"

int main()
{
  L0 = 1.;
  X0 = -0.5;
  G = 1.;
  N = 500;
  run();
}

u.n[left] = neumann(0);
u.n[right] = neumann(0);

event init (t = 0)
{
  double a = 0.25, k = pi/0.1;
  foreach() {
    zb[] = a*(cos(k*x) + 1.)*(fabs(x) < 0.1);
    u.x[] = 0.3;
    h[] = 1. - zb[];
  }
  h.name = "Water_Depth";
  u.x.name = "Velocity_X";
  eta.name = "Free_Surface";
  zb.name = "Bathymetry";
}

event output (t += 0.2; t <= 1.8)
{
  output_vtk_series((scalar *){h, eta, zb, u.x}, NULL, t, "swe_transcritical");
}
