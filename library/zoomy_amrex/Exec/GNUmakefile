# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= $(ZOOMY_AMREX_HOME)

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
COMP         = gnu
DIM          = 3

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source
INCLUDE_LOCATIONS += $(ZOOMY_AMREX_MODEL)

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
