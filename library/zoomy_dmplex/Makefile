include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

solver: main.o
	${CLINKER} -o solver main.o ${PETSC_LIB}

clean_solver:
	rm -f solver main.o