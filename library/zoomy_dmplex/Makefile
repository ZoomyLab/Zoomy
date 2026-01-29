# --- Configuration ---
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++14 -ccbin mpicxx -Xcompiler -fPIC
CXX_FLAGS_USER = -O3 -std=c++17

# --- ASan Configuration ---
# Usage: make CPU ASAN=1
ifdef ASAN
# Use -O1 for better stack traces
SANITIZER_FLAGS = -fsanitize=address -fno-omit-frame-pointer -g -O1
# Append to flags
CXX_FLAGS_USER += $(SANITIZER_FLAGS)
NVCC_FLAGS += -Xcompiler "$(SANITIZER_FLAGS)"
LDFLAGS_USER = $(SANITIZER_FLAGS)
else
LDFLAGS_USER =
endif

# PETSc plumbing
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Final binary names
CPU_APP = solver_cpu
GPU_APP = solver_gpu

# Source files
SRCS_CPP = main.cpp
SRCS_CU  = GPUFirstOrderSolver.cu

# Object files
CPU_OBJS = obj_cpu/main.o
GPU_OBJS = obj_gpu/main.o obj_gpu/GPUFirstOrderSolver.o

.PHONY: all CPU GPU clean_all

# Default
all: CPU GPU

# Targets
CPU: $(CPU_APP)
	@echo "CPU build complete: ./${CPU_APP}"

GPU: $(GPU_APP)
	@echo "GPU build complete: ./${GPU_APP}"

# --- CPU Build Rules ---
$(CPU_APP): $(CPU_OBJS)
	@echo "Linking CPU version..."
	${CLINKER} $(LDFLAGS_USER) -o $@ $^ ${PETSC_LIB}

obj_cpu/%.o: %.cpp
	@mkdir -p obj_cpu
	${CXX} -c $< -o $@ ${PETSC_CC_INCLUDES} ${CXX_FLAGS} ${CXX_FLAGS_USER}

# --- GPU Build Rules ---
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