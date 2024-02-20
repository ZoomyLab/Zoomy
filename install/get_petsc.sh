#!/bin/bash

# when changing dependency versions,
# make sure that the related paths in the Makefile are still valid.

get_petsc() {
	# clone PETSC, if no PETSC dir exists
	if [ -d "petsc" ]; then
		echo "Existing directory \"dependencies/petsc\" found. Skipping download..."
	else
		echo "Cloning PETSc..."
		git clone -b release https://gitlab.com/petsc/petsc.git petsc
		cd petsc
		export PETSC_DIR=$PWD
		./configure --with-fortran-bindings=0 --download-fblaslapack
		make all check
	fi
}

get_petsc
