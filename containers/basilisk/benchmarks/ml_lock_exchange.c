/**
 * Multi-Layer Benchmark 1: Lock exchange (1D, 20 layers)
 *
 * Density-driven gravity current in a rectangular domain.
 * Left half warm (T=0.005), right half cold (T=0).
 * Tests multi-layer advection, vertical structure, and
 * inter-layer exchange.
 * Reference: standard lock-exchange benchmark.
 */
#include "grid/multigrid1D.h"
#include "layered/hydro.h"
#include "layered/implicit.h"
#define rho0 1000.
#define drho(T) ((T))
#include "layered/dr.h"
#include "layered/remap.h"
#include "to_vtk.h"

double nu_H = 0.01;

int main()
{
  size(64e3);
  N = 128;
  nl = 20;
  G = 9.81;
  DT = 60;
  nu = 1e-4;
  cell_lim = mono_limit;
  lambda_b[] = {HUGE};
  run();
}

event init (i = 0)
{
  double h0 = 20;
  foreach() {
    zb[] = -h0;
    foreach_layer() {
      h[] = h0/nl;
      T[] = x < L0/2. ? 0.005 : 0;
    }
  }
}

event viscous_term (i++)
{
  horizontal_diffusion({u.x}, nu_H, dt);
}

event output (t += 1800; t <= 17*3600)
{
  char name[80];
  sprintf(name, "ml_lock_%06.0f", t);

  FILE * fp = fopen(name, "w");
  fprintf(fp, "x,z,u,T\n");
  foreach(serial) {
    double z = zb[];
    foreach_layer() {
      z += h[]/2.;
      fprintf(fp, "%g,%g,%g,%g\n", x, z, u.x[], T[]);
      z += h[]/2.;
    }
  }
  fclose(fp);

  /* Also write layer-averaged VTK for surface layer */
  scalar h_top[], u_top[], T_top[];
  h_top.name = "Layer_Top_h";
  u_top.name = "Layer_Top_u";
  T_top.name = "Layer_Top_T";
  foreach() {
    /* Access top layer values via point indexing */
    h_top[] = h[0,0,nl-1];
    u_top[] = u.x[0,0,nl-1];
    T_top[] = T[0,0,nl-1];
  }
  output_vtk_series((scalar *){h_top, u_top, T_top}, NULL, t, "ml_lock_vtk");
}
