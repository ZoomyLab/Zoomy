/**
 * Multi-Layer Benchmark 2: Wind-driven basin (1D, 10 layers)
 *
 * Closed rectangular basin with constant surface wind stress.
 * Develops vertical Ekman-like profile with return flow at depth.
 * Tests: multi-layer vertical viscosity, surface forcing, steady state.
 */
#include "grid/multigrid1D.h"
#include "layered/hydro.h"
#include "layered/implicit.h"
#include "to_vtk.h"

double tau_wind = 0.1;

int main()
{
  L0 = 1000.;
  X0 = 0.;
  G = 9.81;
  N = 64;
  nl = 10;
  nu = 0.01;
  DT = 1.;
  run();
}

u.n[left] = dirichlet(0);
u.n[right] = dirichlet(0);

event init (i = 0)
{
  double h0 = 10.;
  foreach() {
    zb[] = -h0;
    foreach_layer()
      h[] = h0/nl;
  }
}

event wind_forcing (i++)
{
  foreach()
    u.x[] += tau_wind * dt / (h[0,0,nl-1] * 1000.);
}

event output (t += 50; t <= 5000)
{
  char name[80];
  sprintf(name, "ml_wind_%06.0f.csv", t);

  FILE * fp = fopen(name, "w");
  fprintf(fp, "x,z,u,h_layer\n");
  foreach(serial) {
    double z = zb[];
    foreach_layer() {
      z += h[]/2.;
      fprintf(fp, "%g,%g,%g,%g\n", x, z, u.x[], h[]);
      z += h[]/2.;
    }
  }
  fclose(fp);
}

event vtk_out (t += 200; t <= 5000)
{
  output_vtk_series((scalar *){eta, zb}, NULL, t, "ml_wind_vtk");
}
