#ifndef TO_VTK_H
#define TO_VTK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int vtk_frame = 0;

void output_vtk_series(scalar * list, vector * vlist, double t, const char * base_name) {
  // 1. Write the Legacy VTK file
  char vtk_name[256];
  sprintf(vtk_name, "%s_%04d.vtk", base_name, vtk_frame);
  
  FILE * fp = fopen(vtk_name, "w");
  if (!fp) return;

  int n = 0;
  foreach() n++; 
  
  fprintf(fp, "# vtk DataFile Version 2.0\nBasilisk VTK Export\nASCII\nDATASET UNSTRUCTURED_GRID\n");
  fprintf(fp, "POINTS %d double\n", n);
  foreach() fprintf(fp, "%g %g %g\n", x, y, z);
  
  fprintf(fp, "CELLS %d %d\n", n, n * 2);
  int pt = 0;
  foreach() fprintf(fp, "1 %d\n", pt++);
  
  fprintf(fp, "CELL_TYPES %d\n", n);
  foreach() fprintf(fp, "1\n");
  
  fprintf(fp, "POINT_DATA %d\n", n);
  if (list != NULL) {
    for (scalar s in list) {
      fprintf(fp, "SCALARS %s double\nLOOKUP_TABLE default\n", s.name ? s.name : "scalar");
      foreach() fprintf(fp, "%g\n", s[]);
    }
  }
  if (vlist != NULL) {
    for (vector v in vlist) {
      fprintf(fp, "VECTORS %s double\n", v.x.name ? v.x.name : "vector");
      foreach() fprintf(fp, "%g %g %g\n", v.x[], v.y[], v.z[]);
    }
  }
  fclose(fp);

  // 2. Safely Update the JSON .vtk.series File
  char log_name[256], series_name[256];
  sprintf(log_name, "%s_series.log", base_name);
  sprintf(series_name, "%s.vtk.series", base_name);

  // Append the current frame to a temporary text log WITH A NEWLINE (\n)
  FILE * flog = fopen(log_name, "a");
  fprintf(flog, "{ \"name\" : \"%s\", \"time\" : %g }\n", vtk_name, t);
  fclose(flog);

  // Rebuild the strictly formatted JSON file from the log
  FILE * fs = fopen(series_name, "w");
  fprintf(fs, "{\n  \"file-series-version\" : \"1.0\",\n  \"files\" : [\n");
  
  flog = fopen(log_name, "r");
  char line[512];
  int first = 1;
  while (fgets(line, sizeof(line), flog)) {
      // Strip the newline from the read line for clean JSON formatting
      line[strcspn(line, "\n")] = 0; 
      
      if (!first) fprintf(fs, ",\n");
      fprintf(fs, "    %s", line);
      first = 0;
  }
  fclose(flog);
  fprintf(fs, "\n  ]\n}\n");
  fclose(fs);

  printf("Exported %s at t = %g\n", vtk_name, t);
  vtk_frame++;
}

#endif
