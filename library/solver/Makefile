#####################################################
## Makefile to build VOLKOS
## Copyright VOLKOS 2023
#####################################################
## IMPORTANT NOTICE
## Make sure you have set the VOLKOSPATH environmental variable to the volkos directory

## USAGE
## Invoke make with your prefered architecture, for example
## make arch=cpu
## make arch=gpu device=Volta70
## If you invoke simply "make", cpu is the default architecture
## For GPU compilation device is mandatory

.PHONY: main

# Retrieve the Git commit hash
GIT_COMMIT := $(shell git describe)

## DEFINE MODEL SETUP
## In this section you can control the compilation of model components
## The defaults settings are hardcoded in VOLKOS
## By uncommenting and defining these flags you can manually control which components are compiled
## You can also pass these definitions from command line, e.g., make VOLKOS_PARTICLE_TRACKING=1
#VOLKOS_SUBSURFACE_MODEL=0
#VOLKOS_PARTICLE_TRACKING=0
#VOLKOS_VEGETATION_MODEL=0
#VOLKOS_TOOLS=0
#VOLKOS_FRICTION_MODEL=1

## NetCDF options
## you can set up VOLKOS_NC_MODE to set the file mode for ncmpi_create in PnetCDF.
## The default is  
## VOLKOS_NC_MODE = NC_CLUBBER
## A useful option for large (> 10^8 cells) problems is 
## VOLKOS_NC_MODE = NC_64BIT_DATA

## VOLKOS_NC_REAL determines if data is written to netcdf files as double or float. The default follows whatever VOLKOS_REAL is. To override you can pass at compilation time:
## VOLKOS_NC_REAL = NC_DOUBLE
## or
## VOLKOS_NC_REAL = NC_FLOAT
## which relies on the netcdf definitions onf NC_DOUBLE and NC_FLOAT.

## Debugging flags
#VOLKOS_DEBUG_PARALLEL_DECOMPOSITION=0
#VOLKOS_DEBUG_WORKFLOW=0
#VOLKOS_DEBUG_KOKKOS_SETUP=0
#VOLKOS_DEBUG_BOUNDARY=0
#VOLKOS_DEBUG_DT=0
#VOLKOS_DEBUG_TOOLS=0
#VOLKOS_DEBUG_MASS_CONS=0
#VOLKOS_DEBUG_INFILTRATION=0
#VOLKOS_DEBUG_MPI=0

## Define library paths
EIGEN_PATH=${VOLKOSPATH}/dependencies/eigen
KOKKOS_PATH=${VOLKOSPATH}/dependencies/kokkos
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
#TODO path difference windows/wsl vs laptop
# PNETCDF_PATH=${VOLKOSPATH}/dependencies/pnetcdf/src
PNETCDF_PATH=${VOLKOSPATH}/dependencies/pnetcdf/src
# HDF5_PATH=${VOLKOSPATH}/dependencies/hdf5/hdf5
#TODO path difference windows/wsl vs laptop
HDF5_PATH=${VOLKOSPATH}/dependencies/hdf5/hdf5
MODEL_PATH=${VOLKOSPATH}/outputs/output_c/c_interface/Model

## by default build for cpu, alternative should be gpu
ifndef ($arch)
	arch = cpu
	archCheck = checkCPU
endif

ifeq ($(VOLKOS_DEBUG_PARALLEL_DECOMPOSITION),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_WORKFLOW),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_KOKKOS_SETUP),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_BOUNDARY),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_DT),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_TOOLS),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_MASS_CONS),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_INFILTRATION),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_OUTPUT),1)
debug=yes
endif
ifeq ($(VOLKOS_DEBUG_MPI),1)
debug=yes
endif

debug=no
profiling=no
DEBUG=no
ifeq ($(DEBUG),yes)
debug=yes
endif
PROFILING=no
ifeq ($(DEBUG),yes)
profiling=yes
endif

# Define binary folders and build executable name
BIN_DIR = ${VOLKOSPATH}/bin
BIN_DIR_UT = $(BIN_DIR)/unitTests
EXE = $(BIN_DIR)/volkos 
UNITTESTS = $(BIN_DIR_UT)/dambreak
dummy_build_folder := $(shell mkdir -p $(BIN_DIR) $(BIN_DIR_UT))

CXXFLAGS = -O3 -I${PNETCDF_PATH}/include -I${HDF5_PATH}/include -DDIMENSION=$(DIMENSION) -DN_BOUNDARY_CONDITIONS=$(N_BOUNDARY_CONDITIONS) -DN_ELEMENTS=$(N_ELEMENTS) -DN_FIELDS=$(N_FIELDS) -DN_FIELDS_AUX=$(N_FIELDS_AUX) -DPATH_SETTINGS=\"$(PATH_SETTINGS)\" -DPATH_MESH=\"$(PATH_MESH)\" -DPATH_FIELDS=\"$(PATH_FIELDS)\" -DTIMESTEPPER=\"$(TIMESTEPPER)\" -DTIMESTEPPER_PARAM=$(TIMESTEPPER_PARAM) -DODE_SPACE=\"$(ODE_SPACE)\" -DODE_SOURCE=\"$(ODE_SOURCE)\"
# CXXFLAGS = -O3  -I${HDF5_PATH}/include
# CXXFLAGS = -O3
MPICXX=`which mpic++`

ifdef VOLKOS_NC_MODE
	VOLKOS_MODEL_FLAGS += -DVOLKOS_NC_MODE=$(VOLKOS_NC_MODE)
endif

ifdef VOLKOS_NC_REAL
	VOLKOS_MODEL_FLAGS += -DVOLKOS_NC_REAL=$(VOLKOS_NC_REAL)
endif

ifdef VOLKOS_WRITE_HZ
	VOLKOS_MODEL_FLAGS += -DVOLKOS_WRITE_HZ=$(VOLKOS_WRITE_HZ)
endif

ifdef VOLKOS_SUBSURFACE_MODEL
	VOLKOS_MODEL_FLAGS += -DVOLKOS_SUBSURFACE_MODEL=$(VOLKOS_SUBSURFACE_MODEL)
endif

ifdef VOLKOS_PARTICLE_TRACKING
	VOLKOS_MODEL_FLAGS += -DVOLKOS_PARTICLE_TRACKING=$(VOLKOS_PARTICLE_TRACKING)
endif
ifdef VOLKOS_VEGETATION_MODEL
	VOLKOS_MODEL_FLAGS += -DVOLKOS_VEGETATION_MODEL=$(VOLKOS_VEGETATION_MODEL)
endif
ifdef VOLKOS_TOOLS
	VOLKOS_MODEL_FLAGS += -DVOLKOS_TOOLS=$(VOLKOS_TOOLS)
endif
ifdef VOLKOS_MAXFLOOD
	VOLKOS_MODEL_FLAGS += -DVOLKOS_MAXFLOOD=$(VOLKOS_MAXFLOOD)
endif
ifdef VOLKOS_WRITE_SUBDOMS
	VOLKOS_MODEL_FLAGS += -DVOLKOS_WRITE_SUBDOMS=$(VOLKOS_WRITE_SUBDOMS)
endif
ifdef VOLKOS_NC_ENABLE_NAN
	VOLKOS_MODEL_FLAGS += -DVOLKOS_NC_ENABLE_NAN=$(VOLKOS_NC_ENABLE_NAN)
endif
ifdef VOLKOS_INPUT_NETCDF
	VOLKOS_MODEL_FLAGS += -DVOLKOS_INPUT_NETCDF=$(VOLKOS_INPUT_NETCDF)
endif
ifdef VOLKOS_MESH
	ifeq ($(VOLKOS_MESH), VOLKOS_MESH_UNIFORM)
		VOLKOS_MODEL_FLAGS += -DVOLKOS_MESH_UNIFORM=1
	else 
		VOLKOS_MODEL_FLAGS += -DVOLKOS_MESH_UNIFORM=0
#		exit("Invalid option for VOLKOS_MESH")
	endif
endif

ifdef VOLKOS_DEBUG_PARALLEL_DECOMPOSITION
	VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_PARALLEL_DECOMPOSITION=$(VOLKOS_DEBUG_PARALLEL_DECOMPOSITION)
endif
ifdef VOLKOS_DEBUG_WORKFLOW
	VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_WORKFLOW=$(VOLKOS_DEBUG_WORKFLOW)
endif
ifdef VOLKOS_DEBUG_KOKKOS_SETUP
	VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_KOKKOS_SETUP=$(VOLKOS_DEBUG_KOKKOS_SETUP)
endif
ifdef VOLKOS_DEBUG_BOUNDARY
	VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_BOUNDARY=$(VOLKOS_DEBUG_BOUNDARY)
endif
ifdef VOLKOS_FRICTION_MODEL
	VOLKOS_MODEL_FLAGS += -DVOLKOS_FRICTION_MODEL=$(VOLKOS_FRICTION_MODEL)
endif
ifdef VOLKOS_DEBUG_DT
	VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_DT=$(VOLKOS_DEBUG_DT)
endif
ifdef VOLKOS_DEBUG_TOOLS
	VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_TOOLS=$(VOLKOS_DEBUG_TOOLS)
endif
ifdef VOLKOS_DEBUG_MASS_CONS
    VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_MASS_CONS=$(VOLKOS_DEBUG_MASS_CONS)
endif
ifdef VOLKOS_DEBUG_INFILTRATION
    VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_INFILTRATION=$(VOLKOS_DEBUG_INFILTRATION)
endif
ifdef VOLKOS_DEBUG_MPI
    VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_MPI=$(VOLKOS_DEBUG_MPI)
endif
ifdef VOLKOS_DEBUG_OUTPUT
		VOLKOS_MODEL_FLAGS += -DVOLKOS_DEBUG_OUTPUT=$(VOLKOS_DEBUG_OUTPUT)
endif

ifdef VOLKOS_REAL
	ifeq ($(VOLKOS_REAL),single)
		VOLKOS_MODEL_FLAGS += -DVOLKOS_REAL=1
	else
		ifeq ($(VOLKOS_REAL),double)
			VOLKOS_MODEL_FLAGS += -DVOLKOS_REAL=2
#		else 
#			exit("Invalid option for VOLKOS_REAL flag. Options are 'single' and 'double'")
		endif
	endif
endif

VOLKOS_MODEL_FLAGS += -DVOLKOS_GIT_VERSION=\"${GIT_COMMIT}\"

ifeq ($(arch),cpu)
	CXX = ${MPICXX}
	CXXFLAGS += -DKOKKOS_ENABLE_CXX17 -DKOKKOS_CXX_STANDARD=c++17 -std=c++17 
	KOKKOS_DEVICES = "OpenMP"
	ifeq ($(profiling),yes)
		CXXFLAGS += -pg -g
		LDPROFILING += -pg
		KOKKOS_OPTIONS = "aggressive_vectorization"
	else
	KOKKOS_OPTIONS = "aggressive_vectorization,disable_profiling"
endif
	ifeq ($(debug),yes)
		CXXFLAGS += -g
	else
		CXXFLAGS += -O3
	endif

endif

ifeq ($(arch),gpu)
	archCheck=checkGPU
	CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper -x cu -ccbin ${MPICXX}
	#CXX = clang
	CXXFLAGS += -DKOKKOS_ENABLE_CXX17 -DKOKKOS_CXX_STANDARD=c++17 -std=c++17 
	KOKKOS_DEVICES = "Cuda"
	KOKKOS_OPTIONS += "disable_profiling"
	KOKKOS_CUDA_OPTIONS += "enable_lambda"
endif

CXXFLAGS += $(VOLKOS_MODEL_FLAGS)
LINK = ${CXX}

LDFLAGS = -L${PNETCDF_PATH}/libs/.libs -lpnetcdf -L${HDF5_PATH}/lib -l:libhdf5.so
# LDFLAGS = -L${HDF5_PATH}/lib -lhdf5
# LDFLAGS = 

# SRC = driver.cpp $(MODEL_PATH)/model_code.c
SRC = driver.cpp $(MODEL_PATH)/model_code.cpp $(MODEL_PATH)/boundary_conditions_code.cpp
SRC_UT = unitTests/dambreak.cpp
# OBJ = $(BIN_DIR)/$(notdir $(SRC:.cpp=.o) $(SRC:.c=.o)) 
OBJ = $(BIN_DIR)/$(notdir $(SRC:.cpp=.o)) 
OBJ_UT = $(BIN_DIR_UT)/$(notdir $(SRC_UT:.cpp=.o))

# what does this do?
DEPFLAGS = -M

# exports all variables to be used by sub-make
export

main: $(archCheck) $(arch)

checkCPU:

checkGPU:
ifndef device
	$(error GPU architecture undefined. Please set e.g., device=Volta70. You can choose from Ampere80, Volta70, Pascal61, Maxwell50, etc... See for more options  https://github.com/kokkos/kokkos/wiki/Compiling#table-43-architecture-variables)
endif
KOKKOS_ARCH = ${device}
# For device keywords see https://github.com/kokkos/kokkos/wiki/Compiling#table-43-architecture-variables



cpu:
	@echo -e "\033[92mBuilding for CPU...\033[0m"
	@make -j4 -f Makefile.cpu

gpu:
	@echo -e "\033[92mBuilding for GPU...\033[0m"
	@make -j4 -f Makefile.gpu



clean:
	rm -f *.gch *.o *.dat 
	rm -rf $(BIN_DIR)

kclean: clean
	rm -f KokkosCore_config.h KokkosCore_config.tmp libkokkos.a
