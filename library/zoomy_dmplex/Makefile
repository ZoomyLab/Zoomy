# --- PETSc Discovery ---
# If PETSC_DIR is not set in the environment (local machine), use Python discovery
ifeq ($(PETSC_DIR),)
    PETSC_DIR  := $(shell python3 -c "import os, petsc; print(os.path.dirname(petsc.__file__))")
    PETSC_ARCH := $(shell python3 -c "import petsc4py; print(petsc4py.get_config().get('PETSC_ARCH', ''))")
    # Local builds often need direct include paths if standard rules aren't available
    PETSC_INC_FALLBACK = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/include/eigen3
endif

# Include PETSc plumbing (standard on clusters)
-include ${PETSC_DIR}/lib/petsc/conf/variables
-include ${PETSC_DIR}/lib/petsc/conf/rules

# --- Configuration ---
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++14 -ccbin mpicxx -Xcompiler -fPIC
CXX_FLAGS_USER = -O3 -std=c++17

# --- ASan Configuration ---
ifdef ASAN
    SANITIZER_FLAGS = -fsanitize=address -fno-omit-frame-pointer -g -O1
    CXX_FLAGS_USER += $(SANITIZER_FLAGS)
    NVCC_FLAGS += -Xcompiler "$(SANITIZER_FLAGS)"
    LDFLAGS_USER = $(SANITIZER_FLAGS)
else
    LDFLAGS_USER =
endif

# Fallback for PETSC_CC_INCLUDES if the include above fails (local pip installs)
ifeq ($(PETSC_CC_INCLUDES),)
    PETSC_CC_INCLUDES = $(PETSC_INC_FALLBACK)
endif

# Targets and Files
CPU_APP = solver_cpu
GPU_APP = solver_gpu
SRCS_CPP = main.cpp
CPU_OBJS = obj_cpu/main.o
GPU_OBJS = obj_gpu/main.o obj_gpu/GPUFirstOrderSolver.o

.PHONY: all CPU GPU clean_all

all: CPU GPU

# --- CPU Build Rules ---
CPU: $(CPU_APP)
	@echo "CPU build complete: ./${CPU_APP}"

$(CPU_APP): $(CPU_OBJS)
	@echo "Linking CPU version..."
	${CLINKER} $(LDFLAGS_USER) -o $@ $^ ${PETSC_LIB}

obj_cpu/%.o: %.cpp
	@mkdir -p obj_cpu
	${CXX} -c $< -o $@ ${PETSC_CC_INCLUDES} ${CXX_FLAGS} ${CXX_FLAGS_USER}

# --- GPU Build Rules ---
GPU: $(GPU_APP)
	@echo "GPU build complete: ./${GPU_APP}"

$(GPU_APP): $(GPU_OBJS)
	@echo "Linking GPU version..."
	${CLINKER} $(LDFLAGS_USER) -o $@ $^ ${PETSC_LIB} -lcudart

obj_gpu/main.o: main.cpp
	@mkdir -p obj_gpu
	${CXX} -c $< -o $@ ${PETSC_CC_INCLUDES} ${CXX_FLAGS} ${CXX_FLAGS_USER} -DENABLE_GPU

obj_gpu/%.o: %.cu
	@mkdir -p obj_gpu
	@echo "Compiling CUDA source $<..."
	$(NVCC) $(NVCC_FLAGS) -I. -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -c $< -o $@

clean_all:
	rm -rf obj_cpu obj_gpu $(CPU_APP) $(GPU_APP) output*.vtu output.vtu.series
