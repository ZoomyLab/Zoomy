# --- PETSc Discovery ---
PETSC_DIR  ?= $(shell python3 -c "import os, petsc; print(os.path.dirname(petsc.__file__))")
PETSC_ARCH ?= $(shell python3 -c "import petsc4py; print(petsc4py.get_config().get('PETSC_ARCH', ''))")

# Include PETSc plumbing (if these exist, they help; if not, we handle it)
-include ${PETSC_DIR}/lib/petsc/conf/variables

# --- Mode Logic ---
ifeq ($(MODE), ASAN)
    BUILD_OPTS = -fsanitize=address -fno-omit-frame-pointer -g -O1
    LDFLAGS_USER = -fsanitize=address
else ifeq ($(MODE), DEBUG)
    BUILD_OPTS = -g -O0 -DDEBUG
    LDFLAGS_USER = -g
else
    BUILD_OPTS = -O3
    LDFLAGS_USER = 
endif

# --- Compile Flags ---
# Handle both "arch" builds and "flat/pip" builds
ifeq ($(PETSC_ARCH),)
    PETSC_INC = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/include/eigen3
    PETSC_LNK = -L$(PETSC_DIR)/lib -Wl,-rpath,$(PETSC_DIR)/lib
else
    PETSC_INC = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
    PETSC_LNK = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -Wl,-rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib
endif

CXX_FLAGS_USER = $(BUILD_OPTS) -std=c++17 -fPIC

# --- Targets ---
CPU_APP = solver_cpu
CPU_OBJS = obj_cpu/main.o

.PHONY: all CPU clean_all check_env

all: CPU

check_env:
	@echo "PETSC_DIR:  $(PETSC_DIR)"
	@echo "PETSC_ARCH: $(PETSC_ARCH)"
	@echo "Includes:   $(PETSC_INC)"

CPU: $(CPU_APP)
	@echo "CPU build complete."

$(CPU_APP): $(CPU_OBJS)
	@echo "Linking $@..."
	mpicxx $(LDFLAGS_USER) -o $@ $^ $(PETSC_LNK) \
	    -lpetsc -ldmumps -lmumps_common -lpord -lscalapack -lsuperlu_dist \
	    -lpastix -lopenblas -lhdf5_hl -lhdf5 -lparmetis -lmetis -lhwloc -lX11

obj_cpu/%.o: %.cpp
	@mkdir -p obj_cpu
	mpicxx -c $< -o $@ $(PETSC_INC) $(CXX_FLAGS_USER)

clean_all:
	rm -rf obj_cpu $(CPU_APP)
