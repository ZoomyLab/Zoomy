/* -*- mode: mode: c++; c-default-style: "linux" -*- */

#ifndef _PARAMS_H_
#define _PARAMS_H_

// Index for variables
#define idH  0
#define idHU 1
#define idHV 2
#define idZ 3
#define idR 4
// Index for variable for IO
#define ioH idH 
#define ioHU idHU
#define ioHV idHV
#define ioZ idZ
#define ioR idR
#define ioHZ 1000
#define ioU 1001
#define ioV 1002


// Some physical constants
#define GRAV 9.81
#define SQRTGRAV 3.132091953

//#define GRAV 9.807
//#define SQRTGRAV 3.13161300291


//Some tolerances
#define TOL4 1e-4
#define TOL5 1e-5
#define TOL6 1e-6
#define TOL8 1e-8
#define TOL8NEG -1e-8
#define TOL9 1e-9
#define TOL9NEG -1e-9
#define TOL12 1e-12
#define TOL12NEG -1e-12
#define TOL14 1e-14
#define TOL14NEG -1e-14
#define TOL15 1e-15
#define TOL15NEG -1e-15

#define ZERO TOL12

//friction model (0-->upwind or 1-->pointwise-centered)
#define POINTWISE_FRICTION 0

#define SERGHEI_FRICTION_MANNING 1
#define SERGHEI_FRICTION_DARCYWEISBACH 2
#define SERGHEI_FRICTION_CHEZY 3
#ifndef SERGHEI_FRICTION_MODEL
#define SERGHEI_FRICTION_MODEL SERGHEI_FRICTION_MANNING
#endif

//min depth from which water is stopped
//#define hmin 0.001

//No Data ThresHold
#define NDTH 9999.0

#define SERGHEI_NAN NAN

// PNETCDF parameters
#define PNETCDF_N_INPUT_VARIABLES 9 // number of variables in an initial input file

//halo cells (overlapping cells between domains for MPI)
#define hc 1

//program options
#ifndef SERGHEI_DEBUG_PARALLEL_DECOMPOSITION
#define SERGHEI_DEBUG_PARALLEL_DECOMPOSITION 0 //debug the subdomains ranks and neighbours
#endif
#ifndef SERGHEI_DEBUG_SUBSURFACE
#define SERGHEI_DEBUG_SUBSURFACE 0
#endif
#ifndef SERGHEI_DEBUG_WORKFLOW
#define SERGHEI_DEBUG_WORKFLOW 0
#endif
#ifndef SERGHEI_DEBUG_BOUNDARY
#define SERGHEI_DEBUG_BOUNDARY 0
#endif
#ifndef SERGHEI_SERGHEI_DEBUG_RAINFALL
#define SERGHEI_SERGHEI_DEBUG_RAINFALL 0
#endif
#ifndef SERGHEI_DEBUG_INFILTRATION
#define SERGHEI_DEBUG_INFILTRATION 0
#endif
#ifndef SERGHEI_DEBUG_KOKKOS_SETUP
#define SERGHEI_DEBUG_KOKKOS_SETUP 0
#endif
#ifndef SERGHEI_DEBUG_DT
#define SERGHEI_DEBUG_DT 0
#endif
#ifndef SERGHEI_DEBUG_MASS_CONS
#define SERGHEI_DEBUG_MASS_CONS 0
#endif
#ifndef SERGHEI_VEGETATION_MODEL
#define SERGHEI_VEGETATION_MODEL 0
#endif
#ifndef SERGHEI_DEBUG_MPI
#define SERGHEI_DEBUG_MPI 0 
#endif
#ifndef SERGHEI_DEBUG_OUTPUT
#define SERGHEI_DEBUG_OUTPUT 0 
#endif
#ifndef SERGHEI_DEBUG_INPUT_NETCDF
#define SERGHEI_DEBUG_INPUT_NETCDF 0
#endif

//colors
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define GRAY   "\033[90m"      /* Gray */
#define BOLD "\033[1m"  /* Bold */ 
//mesagges
#define GSTAR GREEN << "[**] " << RESET
#define GOK   GREEN << "[OK] " << RESET
#define GIO   MAGENTA << "[IO] " << RESET
#define BEXC  BLUE << "[!] " << RESET
#define BDASH  BLUE << "[-] " << RESET
#define YEXC  YELLOW << "[!] " << RESET
#define REXC  RED << "[!] " << RESET
#define RERROR  RED << "[ERROR] " << RESET
#define GGD   GRAY<< "[DEBUG] " << RESET

#define NO_DATA -999;



int const OUT_NETCDF  = 1;
int const OUT_VTK 	 = 2;
int const OUT_BIN 	 = 3;


int const OUTPUT_PRECISION = 6;


#ifndef SERGHEI_MESH_UNIFORM
#define SERGHEI_MESH_UNIFORM 1
#endif

int getIOvarID(std::string vname){
    int varid=-1;
    if(!vname.compare("h")) varid=ioH;
    if(!vname.compare("hu")) varid=ioHV;
    if(!vname.compare("hv")) varid=ioHU;
    if(!vname.compare("z")) varid=ioZ;
    if(!vname.compare("h+z")) varid=ioHZ;
    if(!vname.compare("n")) varid=ioR;
    if(!vname.compare("u")) varid=ioU;
    if(!vname.compare("v")) varid=ioV;
    if(varid<0){
        std::cerr << RERROR << "IO variable " << vname << " not handled by " << __FUNCTION__ << std::endl;

        return -1;
    }
    return varid;
}
#endif
