include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

solver: zoomy_dmplex.o
	${CLINKER} -o solver zoomy_dmplex.o ${PETSC_LIB}

clean_solver:
	rm -f solver zoomy_dmplex.o