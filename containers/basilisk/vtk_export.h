#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H

// A simple and robust VTK point-cloud exporter for 1D and 2D Basilisk simulations
void output_vtk (scalar * list, vector * vlist, FILE * fp) {
  int n = 0;
  foreach() n++; // Count cells
  
  fprintf (fp, "# vtk DataFile Version 2.0\n");
  fprintf (fp, "Basilisk VTK Export\n");
  fprintf (fp, "ASCII\n");
  fprintf (fp, "DATASET UNSTRUCTURED_GRID\n");
  
  // Write coordinates
  fprintf (fp, "POINTS %d double\n", n);
  foreach() {
    fprintf (fp, "%g %g %g\n", x, y, z);
  }
  
  // Write cells (treating each point as a VTK_VERTEX for maximum compatibility with AMR)
  fprintf (fp, "CELLS %d %d\n", n, n * 2);
  int pt = 0;
  foreach() {
    fprintf (fp, "1 %d\n", pt++);
  }
  
  fprintf (fp, "CELL_TYPES %d\n", n);
  foreach() {
    fprintf (fp, "1\n"); // 1 = VTK_VERTEX
  }
  
  // Write variables
  fprintf (fp, "POINT_DATA %d\n", n);
  
  // Scalars
  for (scalar s in list) {
    fprintf (fp, "SCALARS %s double\n", s.name);
    fprintf (fp, "LOOKUP_TABLE default\n");
    foreach() fprintf (fp, "%g\n", s[]);
  }
  
  // Vectors
  for (vector v in vlist) {
    fprintf (fp, "VECTORS %s double\n", v.x.name); // Using x-component name as vector name
    foreach() fprintf (fp, "%g %g %g\n", v.x[], v.y[], v.z[]);
  }
}

#endif
