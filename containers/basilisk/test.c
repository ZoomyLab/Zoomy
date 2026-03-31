#include "grid/multigrid1D.h"
#include "green-naghdi.h"
#include "to_vtk.h" // Includes the Docker-baked exporter

#define sech(x) (1.0 / cosh(x))
double a = 0.1;

int main() {
  L0 = 20.;
  X0 = -10.;
  G = 1.;
  N = 256;
  run();
}

event init (i = 0) {
  double c = sqrt(1. + a);
  foreach() {
    double x1 = x;
    h[] = 1. + a*sq(sech(sqrt(3.*a)/(2.*c)*x1));
    u.x[] = c*h[]/(1. + h[]) - c + c/(1. + h[]);
  }
  h.name = "Water_Depth";
  u.x.name = "Velocity";
}

event output (t += 0.1; t <= 10) {
  // One single line handles everything: variables, time, and the base filename
  output_vtk_series((scalar *){h, u.x}, NULL, t, "soliton");
}
