#!/bin/bash

# when changing dependency versions,
# make sure that the related paths in the Makefile are still valid.
kokkos_ver="3.7.01"


get_kokkos() {
	local ver="$1"
	# clone KOKKOS, if no kokkos dir exists
	if [ -d "kokkos" ]; then
		echo "Existing directory \"dependencies/kokkos\" found. Skipping download..."
	else
		echo "Cloning kokkos..."
		git clone --depth 1 --branch $ver https://github.com/kokkos/kokkos.git
		# git clone --depth 1 https://github.com/kokkos/kokkos.git
		#TODO manually added. investigate why necessary, but not in serhei (I think due to a MR in kokkos)
		#If I leave this as it is, I probably need to specify CXX and openmp flags to the makefile
		# cd kokkos
		# mkdir build
		# cd build
		# ./../kokkos/generate_makefile.bash
		# cmake  ..  -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=OFF
	fi
}

get_kokkos "$kokkos_ver"
