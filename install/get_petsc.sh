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
		# ./configure --with-fortran-bindings=0 --download-fblaslapack --PETSC_ARCH=build_arch --download-mpich --PETSC_DIR=$PWD --prefix=$PWD/build
		./configure --with-fortran-bindings=0 --download-fblaslapack --PETSC_ARCH=build_arch --PETSC_DIR=$PWD --prefix=$PWD/build
		make all check
	fi
}

get_petsc
